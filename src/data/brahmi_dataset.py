"""
brahmi_dataset.py  —  CAME Module 2
====================================
Key changes from original
──────────────────────────
1. Akshara-level masking
   Masking now operates on akshara UNITS (C, C+VS, C+virama+C) rather than
   individual Unicode code-points.  One "_" in the noisy string always
   replaces exactly one complete akshara — matching how a physical hole in an
   inscription covers one syllable, not half of one.

   Before:  '𑀢𑀺' split into ['𑀢','𑀺']  → masking '𑀢' leaves orphan '𑀺'
   After:   '𑀢𑀺' is one unit            → masking it gives '_',  '𑀺' gone too

2. extract_aksharas() helper
   Converts a raw Brahmi string into a list of akshara units with perfect
   round-trip fidelity (''.join(units) == original string always).

3. Syllable-token-aware soft_probs
   soft_probs is initialised to uniform across the full (potentially expanded)
   vocab_size that includes composite syllable tokens.  This was already the
   case numerically but is now explicit.

4. val / test split behaviour
   Training split: random akshara-level masking each __getitem__ call
                   (different mask each epoch → implicit augmentation)
   Val / test:     fixed seed per-index so evaluation is deterministic and
                   comparable across runs.

5. Curriculum still works — set_curriculum_stage() controls how many akshara
   units are masked, capped at min(stage, len(units)).
"""

import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.came_model import CAMEModel

# ── Brahmi Unicode character classes ─────────────────────────────────────────
# Keep these here so the dataset has no circular imports with other utils.

_CONSONANTS  = set('𑀓𑀔𑀕𑀖𑀗𑀘𑀙𑀚𑀛𑀜𑀝𑀞𑀟𑀠𑀡𑀢𑀣𑀤𑀥𑀦𑀧𑀨𑀩𑀪𑀫𑀬𑀭𑀮𑀯𑀰𑀱𑀲𑀳𑀴𑀵𑀶𑀷')
_VOWEL_SIGNS = set('𑀸𑀹𑀺𑀻𑀼𑀽𑀾𑀿𑁀𑁁𑁂𑁃𑁄𑁅')
_IND_VOWELS  = set('𑀅𑀆𑀇𑀈𑀉𑀊𑀋𑀌𑀍𑀎𑀏𑀐𑀑𑀒')
_VIRAMA      = '\U00011046'


# ── Akshara extraction ────────────────────────────────────────────────────────

def extract_aksharas(text: str) -> list[str]:
    """
    Split a Brahmi Unicode string into a list of akshara units.

    Rules (applied greedily left-to-right):
      • Consonant + vowel sign            → one unit  (e.g. 𑀢𑀺)
      • Consonant + virama + consonant    → one unit  (e.g. 𑀢𑁆𑀡, conjunct)
      • Bare consonant                    → one unit  (inherent 'a')
      • Independent vowel / vowel sign    → one unit  (standalone)
      • Anything else (modifier, virama   → one unit  (fallback, keeps
        at end, unknown char)                          round-trip lossless)

    Guarantee: ''.join(extract_aksharas(s)) == s  for all valid Brahmi strings.
    """
    units  = []
    chars  = list(text)
    i      = 0

    while i < len(chars):
        ch = chars[i]

        if ch in _CONSONANTS:
            unit = ch
            # C + vowel sign  →  composite syllable
            if i + 1 < len(chars) and chars[i + 1] in _VOWEL_SIGNS:
                unit += chars[i + 1]
                i    += 1
            # C + virama + C  →  conjunct cluster
            elif (i + 1 < len(chars) and chars[i + 1] == _VIRAMA
                  and i + 2 < len(chars) and chars[i + 2] in _CONSONANTS):
                unit += chars[i + 1] + chars[i + 2]
                i    += 2
            units.append(unit)

        elif ch in _IND_VOWELS or ch in _VOWEL_SIGNS:
            # Standalone independent vowel or orphan vowel sign (shouldn't
            # normally appear alone but handle gracefully for robustness)
            units.append(ch)

        else:
            # Virama at word boundary, modifier, punctuation, unknown
            units.append(ch)

        i += 1

    return units


def aksharas_to_noisy(units: list[str], mask_positions: set[int]) -> str:
    """
    Replace aksharas at mask_positions with '_' and join back to a string.
    Each '_' represents exactly one complete akshara unit.
    """
    return ''.join('_' if i in mask_positions else u for i, u in enumerate(units))


# ── Dataset ───────────────────────────────────────────────────────────────────

class BrahmiRestorationDataset(Dataset):
    """
    PyTorch Dataset for self-supervised Brahmi akshara restoration.

    Each sample:
      input_ids      — tokenised noisy string  (masks as '_' tokens)
      attention_mask — padding mask
      labels         — tokenised clean string  (ground truth for loss)
      soft_probs     — placeholder vision soft-probability vectors [seq, vocab]
      confidence     — placeholder per-position confidence scores   [seq, 1]
      clean_text     — raw clean inscription string  (for logging / syllable loss)
      noisy_text     — raw noisy inscription string  (for logging)
      akshara_units  — list of akshara unit strings from the clean text
    """

    def __init__(self, split: str = "train", max_length: int = 512):
        self.split      = split
        self.max_length = max_length
        self.is_train   = (split == "train")

        # ── tokeniser ─────────────────────────────────────────────────────
        # Instantiate the model just to get the tokenizer with Brahmi tokens
        # already added (including composite syllable tokens if came_model.py
        # has been updated).  We do NOT store the model — only the tokenizer.
        _model_for_tok  = CAMEModel()
        self.tokenizer  = _model_for_tok.tokenizer
        self.vocab_size = _model_for_tok.vocab_size
        del _model_for_tok   # free memory immediately

        # ── raw inscriptions ───────────────────────────────────────────────
        if split == "train":
            # Point specifically to our massive new augmented dataset!
            txt_path = Path("data/processed/train_augmented.txt")
        else:
            # Leave validation and testing files exactly as they were
            txt_path = Path(f"data/processed/{split}.txt")

        with open(txt_path, "r", encoding="utf-8") as f:
            raw = [line.strip() for line in f if line.strip()]

        # Pre-compute akshara units for every inscription once at init time.
        # Stored as (clean_string, [unit, unit, ...]) tuples.
        self._inscriptions: list[tuple[str, list[str]]] = [
            (line, extract_aksharas(line)) for line in raw
        ]

        # Curriculum: how many aksharas to mask per sample (1–5)
        self.current_mask_count = 1

        print(
            f"✅ Dataset loaded | split={split} | vocab_size={self.vocab_size} "
            f"| {len(self._inscriptions)} inscriptions"
        )

    # ── curriculum ────────────────────────────────────────────────────────────

    def set_curriculum_stage(self, mask_count: int):
        """
        Called by the trainer at the start of each epoch.
        mask_count is clamped to [1, 5].
        """
        self.current_mask_count = max(1, min(5, mask_count))
        print(f"📈 Curriculum updated → {self.current_mask_count} mask(s) per example")

    # ── length ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._inscriptions)

    # ── sample generation ─────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        clean, units = self._inscriptions[idx]

        # ── masking at akshara level ───────────────────────────────────────
        #
        # ORIGINAL BUG: masking was done on list(clean), i.e. individual
        # Unicode code-points.  Masking position 5 in '𑀢𑀺𑀓' would replace
        # only 𑀢 with '_', leaving the orphan vowel sign '𑀺' at position 6.
        # The model then tried to predict one char but the ground truth had two
        # chars at that position, causing a misalignment and the "invisible
        # character" symptom in evaluation output.
        #
        # FIX: sample mask positions from range(len(units)) — akshara indices.
        # '_' replaces the entire unit ('𑀢𑀺' → '_'), so one blank always
        # corresponds to exactly one complete syllable prediction.

        n_units   = len(units)
        n_masks   = min(self.current_mask_count, n_units)

        if self.is_train:
            # Training: fresh random mask every call → every epoch sees
            # different masking patterns for the same inscription.
            mask_positions = set(random.sample(range(n_units), k=n_masks))
        else:
            # Val / test: deterministic mask derived from idx so evaluation
            # is reproducible and comparable across training runs.
            rng = random.Random(idx * 7919)      # prime seed per index
            mask_positions = set(rng.sample(range(n_units), k=n_masks))

        noisy_text = aksharas_to_noisy(units, mask_positions)

        # ── tokenisation ──────────────────────────────────────────────────
        #
        # Both noisy and clean strings are tokenised to max_length with
        # padding and truncation.  ByT5 processes these at byte level
        # internally, but composite syllable tokens (𑀢𑀺, 𑀮𑁂, …) are mapped
        # to single IDs by the custom vocabulary added in CAMEModel.

        enc_noisy = self.tokenizer(
            noisy_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc_clean = self.tokenizer(
            clean,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # ── vision placeholders ───────────────────────────────────────────
        #
        # When the Vision Module (Component 1) is integrated, soft_probs will
        # be a real probability distribution over the vocab for each position,
        # and confidence will be a per-position certainty score from the vision
        # model.  For now we use:
        #   soft_probs  →  uniform distribution  (no vision prior)
        #   confidence  →  0.8 everywhere         (moderate prior trust)
        #
        # Shape: [max_length, vocab_size] and [max_length, 1] respectively.
        # These match what GatedFusion expects after the batch dimension is
        # added by the DataLoader collate_fn.

        soft_probs = torch.ones(self.max_length, self.vocab_size) / self.vocab_size
        confidence = torch.full((self.max_length, 1), 0.8)

        return {
            # Model inputs
            "input_ids": enc_noisy.input_ids.squeeze(0),  # [max_length]
            "attention_mask": enc_noisy.attention_mask.squeeze(0),  # [max_length]
            "labels": enc_clean.input_ids.squeeze(0),  # [max_length]

            # Vision placeholders
            "soft_probs": soft_probs,  # [max_length, vocab]
            "confidence": confidence,  # [max_length, 1]
        }