import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.came_model import CAMEModel
from src.utils.brahmi_converter import brahmi_to_iast
from src.utils.validity import is_valid_akshara

# ── Brahmi character sets for positional constraint ───────────────────────────
_VOWEL_SIGNS = set('𑀸𑀹𑀺𑀻𑀼𑀽𑀾𑀿𑁀𑁁𑁂𑁃𑁄𑁅')
_CONSONANTS  = set('𑀓𑀔𑀕𑀖𑀗𑀘𑀙𑀚𑀛𑀜𑀝𑀞𑀟𑀠𑀡𑀢𑀣𑀤𑀥𑀦𑀧𑀨𑀩𑀪𑀫𑀬𑀭𑀮𑀯𑀰𑀱𑀲𑀳𑀴𑀵𑀶𑀷')
_IND_VOWELS  = set('𑀅𑀆𑀇𑀈𑀉𑀊𑀋𑀌𑀍𑀎𑀏𑀐𑀑𑀒')
_VIRAMA      = '\U00011046'
_MODIFIERS   = set('𑀀𑀁𑀂')


class BeamSearchRestorer:
    def __init__(self, checkpoint_path: str = "checkpoints/came_latest.pt", beam_size: int = 20):
        self.model = CAMEModel()
        if Path(checkpoint_path).exists():
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            )
            print("✅ Loaded trained checkpoint")
        else:
            print("⚠️ No checkpoint – using random weights (demo mode)")
        self.model.eval()
        self.tokenizer    = self.model.tokenizer
        self.beam_size    = beam_size
        self.lambda_valid = 1.5
        self.mu_conf      = 0.3

        # Build token-ID sets once so the hot loop stays fast
        self._vowel_sign_ids = self._build_token_id_set(_VOWEL_SIGNS)
        self._consonant_ids  = self._build_token_id_set(_CONSONANTS)
        self._ind_vowel_ids  = self._build_token_id_set(_IND_VOWELS)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_token_id_set(self, chars):
        """Map a set of Brahmi chars to their vocab IDs (single-token only)."""
        ids = set()
        for ch in chars:
            encoded = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
        return ids

    def _decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def _build_logit_mask(self, original_input_ids: torch.Tensor, mask_pos: int) -> torch.Tensor:
        """
        POSITIONAL CONSTRAINT — fixes the 'predicted character appears missing' bug.

        WHY this happens
        ────────────────
        𑁂 (BRAHMI VOWEL SIGN E, U+11042) is the #2 most frequent token in the
        training corpus (2 555 occurrences).  After only a few epochs the model
        defaults to it at uncertain positions.

        The existing is_valid_akshara() check does NOT catch this — consonant
        followed by vowel sign IS valid Brahmi, so it even gets the +1.5 bonus
        and scores highest in the beam.

        The visual 'disappearing' effect is a Unicode rendering artifact:
        vowel signs are category Mn (Mark, Nonspacing) — zero advance-width
        combining chars that glue onto the PRECEDING glyph.  𑀟𑁂 renders as a
        single composite glyph, so the restored character looks invisible.

        THE FIX
        ───────
        Inspect the decoded characters immediately before and after the mask.
        If the local context makes vowel signs linguistically impossible, set
        their logits to -1e9 (≈ -∞) BEFORE softmax.  This forces the model to
        choose from the plausible category regardless of raw frequency bias.

        Rules (any one → block vowel signs):
          1. prev char is NOT a consonant  — vowel sign has no base to attach to
          2. next char is a consonant      — would create consonant cluster
                                            with no virama connector
          3. next char is a vowel sign     — consecutive vowel signs are invalid
        """
        seq      = original_input_ids[0]
        vocab_sz = self.model.vocab_size
        block    = torch.zeros(vocab_sz, dtype=torch.bool)

        pad_id    = self.model.byt5.config.pad_token_id
        prev_char = self._decode_token(seq[mask_pos - 1].item()) if mask_pos > 0 else ''

        next_char = ''
        if mask_pos + 1 < len(seq):
            nxt_id = seq[mask_pos + 1].item()
            if nxt_id != pad_id:
                next_char = self._decode_token(nxt_id)

        block_vowel_signs = (
            (prev_char not in _CONSONANTS)   # rule 1
            or (next_char in _CONSONANTS)    # rule 2
            or (next_char in _VOWEL_SIGNS)   # rule 3
        )

        if block_vowel_signs:
            for vid in self._vowel_sign_ids:
                if vid < vocab_sz:
                    block[vid] = True

        return block   # True = force logit to -1e9

    # ── main restore ──────────────────────────────────────────────────────────

    def restore(self, noisy_brahmi: str, soft_probs=None, confidence=None):
        inputs             = self.tokenizer(noisy_brahmi, return_tensors="pt")
        original_input_ids = inputs.input_ids      # never mutated — always sent to encoder
        attention_mask     = inputs.attention_mask

        seq_len = original_input_ids.shape[1]

        if soft_probs is None:
            soft_probs = torch.ones((1, seq_len, self.model.vocab_size)) / self.model.vocab_size
        if confidence is None:
            confidence = torch.ones((1, seq_len, 1)) * 0.8

        mask_token_id = self.tokenizer.encode("_", add_special_tokens=False)[0]
        pad_token_id  = self.model.byt5.config.pad_token_id

        mask_positions = [
            i for i, t in enumerate(original_input_ids[0])
            if t == mask_token_id
        ]

        beams = [(original_input_ids.clone(), 0.0)]

        for mask_pos in mask_positions:

            # Positional constraint mask — built once per mask position
            logit_block = self._build_logit_mask(original_input_ids, mask_pos)

            new_beams = []
            for filled_seq, score in beams:
                with torch.no_grad():
                    # Replace any remaining "_" with pad so decoder never sees
                    # the mask token — it was absent during training (Bug 5 fix)
                    decoder_labels = filled_seq.clone()
                    decoder_labels[decoder_labels == mask_token_id] = pad_token_id

                    outputs = self.model(
                        input_ids=original_input_ids,   # always original (Bug 5A fix)
                        attention_mask=attention_mask,
                        soft_probs=soft_probs,
                        confidence=confidence,
                        labels=decoder_labels           # no "_" in decoder (Bug 5B fix)
                    )

                logits = outputs["restoration_logits"][0, mask_pos].clone()

                # ── POSITIONAL CONSTRAINT ─────────────────────────────────────
                # Block logits for impossible token categories BEFORE softmax.
                logits[logit_block] = -1e9

                probs            = torch.softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(probs, k=self.beam_size)

                for k in range(self.beam_size):
                    new_filled_seq              = filled_seq.clone()
                    new_filled_seq[0, mask_pos] = topk_ids[k]

                    candidate = self.tokenizer.decode(
                        new_filled_seq[0], skip_special_tokens=True
                    )
                    new_score = score + torch.log(topk_probs[k]).item()

                    if is_valid_akshara(candidate):
                        new_score += self.lambda_valid

                    new_score += self.mu_conf * outputs["confidence_score"][0, mask_pos].item()
                    new_beams.append((new_filled_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[: self.beam_size]

        results = []
        for filled_seq, score in beams:
            brahmi = self.tokenizer.decode(filled_seq[0], skip_special_tokens=True)
            # No .replace("_","") — all mask positions filled in the loop above
            iast   = brahmi_to_iast(brahmi)
            conf   = round(torch.sigmoid(torch.tensor(score)).item(), 4)
            results.append((brahmi, iast, conf))

        return sorted(results, key=lambda x: x[2], reverse=True)