import torch
from torch.utils.data import Dataset
import random
import sys
from pathlib import Path

from src.utils.build_syllable_vocab import extract_aksharas

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.came_model import CAMEModel

class BrahmiRestorationDataset(Dataset):
    def __init__(self, split: str = "train", max_length: int = 512):
        self.split = split
        self.max_length = max_length

        model_for_tokenizer = CAMEModel()
        self.tokenizer = model_for_tokenizer.tokenizer
        self.vocab_size = model_for_tokenizer.vocab_size

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

        # 1. Split into physical akshara units, mask whole units
        units = extract_aksharas(clean)

        # 2. Pick random syllables to hide
        mask_positions = random.sample(
            range(len(units)),
            k=min(self.current_mask_count, len(units))
        )

        for pos in mask_positions:
            units[pos] = "_"

        # 3. Join back into a string. The model will now predict exactly ONE token for each "_"
        noisy_text = "".join(units)

        # 4. Tokenize the input and the clean label
        inputs = self.tokenizer(
            noisy_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = self.tokenizer(
            clean,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels.input_ids.squeeze(0),
            "clean_text": clean,
            # (Generate your dummy soft_probs and confidence here just like before)
            "soft_probs": torch.ones((self.max_length, self.vocab_size)) / self.vocab_size,
            "confidence": torch.ones((self.max_length, 1)) * 0.8
        }