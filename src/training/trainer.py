import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from pathlib import Path
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

        # ── define accumulation steps FIRST so scheduler calc can use it ──────
        self.gradient_accumulation_steps = 8

        print(f"✅ Config loaded | LR = {self.config['training']['learning_rate']} | Batch=1 (accum={self.gradient_accumulation_steps})")

        self.model = CAMEModel(model_name=self.config["model"]["name"])

        # Store tokenizer reference BEFORE accelerator.prepare() wraps the model,
        # so we can safely call self.tokenizer throughout training without worrying
        # about model.module vs model attribute access on multi-GPU setups.
        self.tokenizer = self.model.tokenizer

        self.train_dataset = BrahmiRestorationDataset(split="train")
        self.val_dataset   = BrahmiRestorationDataset(split="val")

        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True,  num_workers=0)
        self.val_loader   = DataLoader(self.val_dataset,   batch_size=1, shuffle=False, num_workers=0)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=0.01
        )

        # ── BUG 3 FIX ─────────────────────────────────────────────────────────
        # Original: (len(self.train_dataset) // 8) * 50
        # With batch_size=1 those are numerically equal, but using the loader
        # length is semantically correct and stays correct if batch_size changes.
        # Total optimizer steps = batches_per_epoch / accum_steps × num_epochs
        total_steps = (len(self.train_loader) // self.gradient_accumulation_steps) * 50
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=200,
            num_training_steps=total_steps
        )

        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        self.global_step = 0

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_syllable_labels(self, label_ids: torch.Tensor, device) -> torch.Tensor:
        """
        BUG 2 FIX — build one validity label per TOKEN, not per sequence.

        Original code:
            valid_labels = [1 if is_valid_akshara(t) else 0 for t in batch["clean_text"]]
        This called is_valid_akshara() on the FULL inscription string, produced
        one label for the whole batch item, then broadcast it to every token
        position.  The syllable head is supposed to judge individual characters,
        not whole inscriptions.

        Fix: decode each label token ID back to its Brahmi character and call
        is_valid_akshara() on that single character.  Pad positions receive
        label 0 (invalid) so they contribute zero weight if ignored_index is
        used, but here we simply exclude them via the mask below.
        """
        pad_id = self.tokenizer.pad_token_id
        labels = []
        for tok_id in label_ids:
            t = tok_id.item()
            if t == pad_id:
                labels.append(0)   # pad → treated as invalid
            else:
                char = self.tokenizer.decode([t])
                labels.append(1 if is_valid_akshara(char) else 0)
        return torch.tensor(labels, dtype=torch.long, device=device)

    # ── training loop ─────────────────────────────────────────────────────────

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0   # sum of mean losses per optimizer step (for logging)
        accum_loss = 0.0   # running sum of raw losses inside one accumulation window

        # Curriculum: add one mask every 2 epochs, cap at 5
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

            # ── Restoration loss (main objective) ─────────────────────────────
            loss_rest = torch.nn.functional.cross_entropy(
                outputs["restoration_logits"].view(-1, outputs["restoration_logits"].size(-1)),
                batch["labels"].view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )

            # ── Syllable validity loss (BUG 2 FIX) ────────────────────────────
            # batch_size=1, so labels has shape [1, seq_len]; take the first row.
            label_ids  = batch["labels"][0]                            # [seq_len]
            valid_labels = self._build_syllable_labels(                # [seq_len]
                label_ids, device=self.accelerator.device
            )

            # syllable_logits: [1, seq_len, 2]  →  view: [seq_len, 2]
            loss_syll = torch.nn.functional.cross_entropy(
                outputs["syllable_logits"].view(-1, 2),
                valid_labels,
                ignore_index=-100  # keep pad positions out of the loss
            )

            # ── Confidence calibration loss ────────────────────────────────────
            loss_conf = torch.nn.functional.mse_loss(
                outputs["confidence_score"].squeeze(-1),
                batch["confidence"].squeeze(-1)
            )

            # ── Combined loss ──────────────────────────────────────────────────
            loss = loss_rest + 0.4 * loss_syll + 0.2 * loss_conf

            # ── BUG 1 FIX: gradient accumulation ──────────────────────────────
            #
            # Original (broken):
            #   oss = loss / self.gradient_accumulation_steps   ← 'oss' never used
            #   self.accelerator.backward(loss)                 ← backward on FULL loss
            #   accum_loss += loss.item() * gradient_accum_steps ← 8× inflated logging
            #
            # With the original code every backward call pushed gradients 8× too
            # large, making training diverge or oscillate from the very first step.
            #
            # Fix: divide BEFORE backward so each of the 8 mini-steps contributes
            # exactly 1/8 of the gradient.  Accumulate the unscaled loss for
            # logging so the displayed value is still human-readable.
            accum_loss += loss.item()                                  # track unscaled
            scaled_loss = loss / self.gradient_accumulation_steps      # scale for backward
            self.accelerator.backward(scaled_loss)                     # ← correct

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()          # ← was missing in original; advance LR
                self.optimizer.zero_grad()

                # Average loss over the accumulation window
                mean_step_loss = accum_loss / self.gradient_accumulation_steps
                total_loss += mean_step_loss
                accum_loss  = 0.0

                num_opt_steps = (batch_idx + 1) // self.gradient_accumulation_steps
                progress.set_postfix({"loss": f"{total_loss / num_opt_steps:.4f}"})

            self.global_step += 1

        # Handle any leftover batches that didn't fill a complete accumulation window
        if accum_loss > 0.0:
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            total_loss += accum_loss / (len(self.train_loader) % self.gradient_accumulation_steps)

        avg_loss = total_loss / max(len(self.train_loader) // self.gradient_accumulation_steps, 1)
        print(f"✅ Epoch {epoch + 1} finished — Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self):
        Path("checkpoints").mkdir(exist_ok=True)
        self.accelerator.save(self.model.state_dict(), "checkpoints/came_latest.pt")
        print("💾 Checkpoint saved")


if __name__ == "__main__":
    trainer = CAMETrainer()
    for epoch in range(10):
        trainer.train_epoch(epoch)
        if epoch % 5 == 0:
            trainer.save_checkpoint()
    print("🎉 Training finished!")