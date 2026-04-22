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


class CAMETrainer:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.accelerator = Accelerator()
        set_seed(42)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config["training"]["learning_rate"] = float(self.config["training"]["learning_rate"])

        print(f"✅ Config loaded | LR = {self.config['training']['learning_rate']} | Batch=1 (accum=8)")

        self.model = CAMEModel(model_name=self.config["model"]["name"])

        self.train_dataset = BrahmiRestorationDataset(split="train")
        self.val_dataset = BrahmiRestorationDataset(split="val")

        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=0)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["training"]["learning_rate"],
                                           weight_decay=0.01)

        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        self.gradient_accumulation_steps = 8
        self.global_step = 0

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        accum_loss = 0.0

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

            # Restoration loss
            loss_rest = torch.nn.functional.cross_entropy(
                outputs["restoration_logits"].view(-1, outputs["restoration_logits"].size(-1)),
                batch["labels"].view(-1),
                ignore_index=self.model.tokenizer.pad_token_id
            )

            # === FIXED: Syllable validity loss with correct shape ===
            valid_labels = torch.tensor(
                [1 if is_valid_akshara(t) else 0 for t in batch["clean_text"]],
                dtype=torch.long, device=self.accelerator.device
            )
            # Expand label to match every token in the sequence (512 tokens)
            seq_len = outputs["syllable_logits"].shape[1]
            valid_labels = valid_labels.unsqueeze(1).expand(-1, seq_len).reshape(-1)

            loss_syll = torch.nn.functional.cross_entropy(
                outputs["syllable_logits"].view(-1, 2), valid_labels
            )

            loss_conf = torch.nn.functional.mse_loss(
                outputs["confidence_score"].squeeze(-1),
                batch["confidence"].squeeze(-1)
            )

            loss = loss_rest + 0.4 * loss_syll + 0.2 * loss_conf

            loss = loss / self.gradient_accumulation_steps
            self.accelerator.backward(loss)
            accum_loss += loss.item() * self.gradient_accumulation_steps

            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_loss += accum_loss
                accum_loss = 0.0
                progress.set_postfix(
                    {"loss": f"{total_loss / ((batch_idx + 1) // self.gradient_accumulation_steps + 1):.4f}"})

            self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)
        print(f"✅ Epoch {epoch + 1} finished - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self):
        self.accelerator.save(self.model.state_dict(), "checkpoints/came_latest.pt")
        print("💾 Checkpoint saved")


if __name__ == "__main__":
    trainer = CAMETrainer()
    for epoch in range(20):  # continue training
        trainer.train_epoch(epoch)
        if epoch % 5 == 0:
            trainer.save_checkpoint()
    print("🎉 Extra training finished!")