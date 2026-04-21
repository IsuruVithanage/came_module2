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


class CAMETrainer:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        self.accelerator = Accelerator()
        set_seed(42)

        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config["training"]["learning_rate"] = float(self.config["training"]["learning_rate"])

        print(f"✅ Config loaded | LR = {self.config['training']['learning_rate']}")

        self.model = CAMEModel(model_name=self.config["model"]["name"])

        self.train_dataset = BrahmiRestorationDataset(split="train")
        self.val_dataset = BrahmiRestorationDataset(split="val")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=0
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=0
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"]
        )

        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )

        self.global_step = 0

    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0

        mask_stage = min(1 + (epoch // 2), 5)
        self.train_dataset.set_curriculum_stage(mask_stage)

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for batch_idx, batch in enumerate(progress):
            if batch_idx == 0:
                print(
                    f"   Debug shapes - soft_probs: {batch['soft_probs'].shape} | confidence: {batch['confidence'].shape} | h_text expected: (batch, seq, 768)")

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                soft_probs=batch["soft_probs"],
                confidence=batch["confidence"],
                labels=batch["labels"]
            )

            loss_rest = torch.nn.functional.cross_entropy(
                outputs["restoration_logits"].view(-1, outputs["restoration_logits"].size(-1)),
                batch["labels"].view(-1),
                ignore_index=self.model.tokenizer.pad_token_id
            )

            loss_syll = torch.tensor(0.0, device=self.accelerator.device)
            loss_conf = torch.nn.functional.mse_loss(
                outputs["confidence_score"].squeeze(-1),
                batch["confidence"].squeeze(-1)
            )

            loss = loss_rest + 0.3 * loss_syll + 0.2 * loss_conf

            self.accelerator.backward(loss)
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            self.global_step += 1

        avg_loss = total_loss / len(self.train_loader)
        print(f"✅ Epoch {epoch + 1} finished - Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self):
        self.accelerator.save(self.model.state_dict(), "checkpoints/came_latest.pt")
        print("💾 Checkpoint saved → checkpoints/came_latest.pt")


if __name__ == "__main__":
    trainer = CAMETrainer()
    for epoch in range(3):
        trainer.train_epoch(epoch)
        if epoch % 2 == 0:
            trainer.save_checkpoint()
    print("🎉 Training test complete!")