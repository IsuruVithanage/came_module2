import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.brahmi_dataset import BrahmiRestorationDataset
from src.models.came_model import CAMEModel
from src.utils.validity import is_valid_akshara
from transformers import get_linear_schedule_with_warmup


class CAMETrainer:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.accelerator = Accelerator()
        set_seed(42)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config["training"]["learning_rate"] = float(self.config["training"]["learning_rate"])

        # Define accumulation steps FIRST — scheduler calc uses it below
        self.gradient_accumulation_steps = 8

        print(f"✅ Config loaded | LR = {self.config['training']['learning_rate']} | "
              f"Batch=1 (accum={self.gradient_accumulation_steps})")

        # ── Model ──────────────────────────────────────────────────────────────
        self.model = CAMEModel(model_name=self.config["model"]["name"])

        # Silence the tie_word_embeddings warning.
        # ByT5 ties shared / lm_head / embed_tokens weights by default.
        # After we resize embeddings for Brahmi tokens those weights diverge,
        # so we explicitly turn off tying to match reality and stop the warning.
        self.model.byt5.config.tie_word_embeddings = False

        # Store tokenizer BEFORE accelerator.prepare() wraps the model so we
        # can always access it as self.tokenizer without .module indirection.
        self.tokenizer = self.model.tokenizer

        # ── Datasets & loaders ─────────────────────────────────────────────────
        self.train_dataset = BrahmiRestorationDataset(split="train")
        self.val_dataset   = BrahmiRestorationDataset(split="val")

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=1, shuffle=True,  num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset,   batch_size=1, shuffle=False, num_workers=0
        )

        # ── Optimizer ──────────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=0.01
        )

        # ── Scheduler ──────────────────────────────────────────────────────────
        # Total optimizer steps = (batches_per_epoch / accum_steps) × num_epochs
        # Using loader length (not dataset length) is correct regardless of batch size.
        # Warmup bumped to 500 because the new syllable token embeddings are
        # randomly initialised and need extra steps to catch up to the pre-trained
        # ByT5 weights before the LR starts decaying.
        total_steps = (len(self.train_loader) // self.gradient_accumulation_steps) * 50
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=500,
            num_training_steps=total_steps
        )

        # ── Frequency weights (built BEFORE accelerator.prepare) ───────────────
        # Must be built here in __init__ so self.freq_weights exists before any
        # call to train_epoch().  We build it on CPU first, then move it to the
        # accelerator device after prepare().
        self.freq_weights = self._build_frequency_weights()

        # ── Accelerator wrapping ───────────────────────────────────────────────
        self.model, self.optimizer, self.train_loader, self.val_loader = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )
        )

        # Move freq_weights to the correct device after accelerator.prepare()
        self.freq_weights = self.freq_weights.to(self.accelerator.device)

        self.global_step = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_frequency_weights(self) -> torch.Tensor:
        """
        Build inverse-frequency weights over the full vocabulary.

        Why this matters
        ────────────────
        Without weighting, cross_entropy treats every token equally.
        𑀟 (DDA) appears ~130× in training while 𑀫 (MA) appears ~1370×.
        The model learns that predicting MA is almost always a safe bet,
        so it defaults to high-frequency consonants at uncertain mask positions.

        Inverse-frequency weighting amplifies the gradient for rare-but-correct
        characters so the model learns to distinguish them from frequent ones.

        Formula
        ───────
        weight[id] = 1.0 / (relative_frequency + 0.05)

        The 0.05 smoothing floor prevents tokens that never appear in training
        from getting infinite weight (which would destabilise training).
        Tokens not found in the training text get weight 1.0 (neutral).
        Pad token always gets weight 0.0 (excluded from loss).
        """
        txt_path = Path("data/processed/train.txt")
        txt      = txt_path.read_text(encoding="utf-8")
        counts   = Counter(txt)
        max_count = max(counts.values()) if counts else 1

        # len(self.tokenizer) = 515 (base 256 + all added Brahmi + syllable tokens)
        # self.tokenizer.vocab_size = 256 (ByT5 base only — ignores added tokens)
        # Using vocab_size here causes IndexError when a Brahmi token ID > 255
        # tries to index into a 256-element tensor.
        weights = torch.ones(len(self.tokenizer))

        for char, count in counts.items():
            ids = self.tokenizer.encode(char, add_special_tokens=False)
            if len(ids) == 1:                         # single-token character
                relative_freq  = count / max_count
                weights[ids[0]] = 1.0 / (relative_freq + 0.05)

        # Also weight composite syllable tokens (𑀮𑁂, 𑀡𑁂, 𑀢𑀺, …)
        # by counting their occurrences as multi-char substrings in the corpus.
        lines = [l.strip() for l in txt.strip().splitlines() if l.strip()]
        syllable_counts: Counter = Counter()
        for line in lines:
            i = 0
            chars = list(line)
            VOWEL_SIGNS = set('𑀸𑀹𑀺𑀻𑀼𑀽𑀾𑀿𑁀𑁁𑁂𑁃𑁄𑁅')
            CONSONANTS  = set('𑀓𑀔𑀕𑀖𑀗𑀘𑀙𑀚𑀛𑀜𑀝𑀞𑀟𑀠𑀡𑀢𑀣𑀤𑀥𑀦𑀧𑀨𑀩𑀪𑀫𑀬𑀭𑀮𑀯𑀰𑀱𑀲𑀳𑀴𑀵𑀶𑀷')
            VIRAMA      = '\U00011046'
            while i < len(chars):
                ch = chars[i]
                if ch in CONSONANTS:
                    if i+1 < len(chars) and chars[i+1] in VOWEL_SIGNS:
                        syllable_counts[ch + chars[i+1]] += 1
                        i += 2; continue
                    elif (i+1 < len(chars) and chars[i+1] == VIRAMA
                          and i+2 < len(chars) and chars[i+2] in CONSONANTS):
                        syllable_counts[ch + chars[i+1] + chars[i+2]] += 1
                        i += 3; continue
                i += 1

        if syllable_counts:
            max_syll = max(syllable_counts.values())
            for syll, count in syllable_counts.items():
                ids = self.tokenizer.encode(syll, add_special_tokens=False)
                if len(ids) == 1:                     # added as a single token
                    relative_freq  = count / max_syll
                    weights[ids[0]] = 1.0 / (relative_freq + 0.05)

        # Pad token contributes nothing to the loss
        weights[self.tokenizer.pad_token_id] = 0.0

        print(f"✅ Frequency weights built | vocab_size={len(weights)} | "
              f"min={weights.min():.3f} max={weights.max():.3f}")
        return weights

    def _build_syllable_labels(self, label_ids: torch.Tensor, device) -> torch.Tensor:
        """
        Build one validity label PER TOKEN (not per sequence).

        Each label token ID is decoded back to its Brahmi character/syllable
        and checked with is_valid_akshara().  Pad positions get label 0 and
        are excluded from the syllable loss via ignore_index.
        """
        pad_id = self.tokenizer.pad_token_id
        labels = []
        for tok_id in label_ids:
            t = tok_id.item()
            if t == pad_id:
                labels.append(-100)                   # ignored by cross_entropy
            else:
                char = self.tokenizer.decode([t])
                labels.append(1 if is_valid_akshara(char) else 0)
        return torch.tensor(labels, dtype=torch.long, device=device)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0    # sum of per-step mean losses (for progress display)
        accum_loss = 0.0    # running sum within one accumulation window

        # Curriculum masking: 1 mask for epochs 0-1, 2 for 2-3, … capped at 5
        mask_stage = min(1 + (epoch // 2), 5)
        self.train_dataset.set_curriculum_stage(mask_stage)

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")

        for batch_idx, batch in enumerate(progress):

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                soft_probs=batch["soft_probs"],
                confidence=batch["confidence"],
                labels=batch["labels"]
            )

            # ── Restoration loss — weighted cross entropy ──────────────────────
            # freq_weights boosts rare characters (e.g. 𑀟 DDA at 130×) so the
            # model stops defaulting to common ones (e.g. 𑀫 MA at 1370×).
            loss_rest = torch.nn.functional.cross_entropy(
                outputs["restoration_logits"].view(-1, outputs["restoration_logits"].size(-1)),
                batch["labels"].view(-1),
                weight=self.freq_weights,
                ignore_index=self.tokenizer.pad_token_id
            )

            # ── Syllable validity loss — per token ────────────────────────────
            label_ids    = batch["labels"][0]                        # [seq_len]
            valid_labels = self._build_syllable_labels(
                label_ids, device=self.accelerator.device
            )
            loss_syll = torch.nn.functional.cross_entropy(
                outputs["syllable_logits"].view(-1, 2),
                valid_labels,
                ignore_index=-100
            )

            # ── Confidence calibration loss ────────────────────────────────────
            loss_conf = torch.nn.functional.mse_loss(
                outputs["confidence_score"].squeeze(-1),
                batch["confidence"].squeeze(-1)
            )

            # ── Combined loss ──────────────────────────────────────────────────
            loss = loss_rest + 0.4 * loss_syll + 0.2 * loss_conf

            # ── Gradient accumulation (Bug 1 fix) ─────────────────────────────
            # Scale loss BEFORE backward so each of the 8 mini-steps contributes
            # exactly 1/8 of the total gradient.  Track unscaled loss for logging.
            accum_loss += loss.item()
            scaled_loss = loss / self.gradient_accumulation_steps
            self.accelerator.backward(scaled_loss)

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                mean_step_loss = accum_loss / self.gradient_accumulation_steps
                total_loss    += mean_step_loss
                accum_loss     = 0.0

                num_opt_steps = (batch_idx + 1) // self.gradient_accumulation_steps
                progress.set_postfix({"loss": f"{total_loss / num_opt_steps:.4f}"})

            self.global_step += 1

        # Handle leftover batches that didn't fill a full accumulation window
        if accum_loss > 0.0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            remainder = len(self.train_loader) % self.gradient_accumulation_steps
            total_loss += accum_loss / max(remainder, 1)

        avg_loss = total_loss / max(
            len(self.train_loader) // self.gradient_accumulation_steps, 1
        )
        print(f"✅ Epoch {epoch + 1} finished — Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self):
        Path("checkpoints").mkdir(exist_ok=True)
        self.accelerator.save(self.model.state_dict(), "checkpoints/came_latest.pt")
        print("💾 Checkpoint saved")


if __name__ == "__main__":
    trainer = CAMETrainer()
    for epoch in range(50):
        trainer.train_epoch(epoch)
        if epoch % 5 == 0:
            trainer.save_checkpoint()
    print("🎉 Training finished!")