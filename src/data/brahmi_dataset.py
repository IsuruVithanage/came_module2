import torch
from torch.utils.data import Dataset
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.came_model import CAMEModel


class BrahmiRestorationDataset(Dataset):
    def __init__(self, split: str = "train", max_length: int = 512):
        self.split = split
        self.max_length = max_length

        model_for_tokenizer = CAMEModel()
        self.tokenizer = model_for_tokenizer.tokenizer
        self.vocab_size = self.tokenizer.vocab_size

        txt_path = Path(f"data/processed/{split}.txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            self.clean_texts = [line.strip() for line in f if line.strip()]

        self.current_mask_count = 1
        print(f"✅ Dataset loaded | vocab_size = {self.vocab_size} | {len(self.clean_texts)} examples")

    def set_curriculum_stage(self, mask_count: int):
        self.current_mask_count = max(1, min(5, mask_count))
        print(f"📈 Curriculum updated → {self.current_mask_count} masks per example")

    def __len__(self):
        return len(self.clean_texts)

    def __getitem__(self, idx):
        clean = self.clean_texts[idx]

        noisy = list(clean)
        mask_positions = random.sample(range(len(noisy)), k=min(self.current_mask_count, len(noisy)))
        for pos in mask_positions:
            noisy[pos] = "<mask>"

        noisy_text = "".join(noisy)

        inputs = self.tokenizer(noisy_text, padding="max_length", truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        targets = self.tokenizer(clean, padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")

        # === CORRECT SHAPE: (seq_len, ...) - DataLoader will add batch dim automatically ===
        soft_probs = torch.zeros((self.max_length, self.vocab_size))
        confidence = torch.ones((self.max_length, 1)) * 0.8

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": targets.input_ids.squeeze(0),
            "soft_probs": soft_probs,
            "confidence": confidence,
            "clean_text": clean,
            "noisy_text": noisy_text
        }