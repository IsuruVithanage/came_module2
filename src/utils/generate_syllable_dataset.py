import json
import random
import sys
from pathlib import Path

# === FIX: Make sure Python can find the 'src' folder ===
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.validity import is_valid_akshara, CONSONANTS, VOWEL_SIGNS


def generate_syllable_validity_dataset(
    train_txt: str = "data/processed/train.txt",
    output_jsonl: str = "data/processed/syllable_validity_train.jsonl",
    num_examples: int = 80000
):
    with open(train_txt, "r", encoding="utf-8") as f:
        clean_texts = [line.strip() for line in f if line.strip()]

    dataset = []
    for text in clean_texts:
        for length in range(3, min(16, len(text) + 1)):
            prefix = text[:length]
            if is_valid_akshara(prefix):
                dataset.append({"partial_sequence": prefix, "label": 1})

            # Generate one controlled negative example
            if len(prefix) > 2:
                corrupt = list(prefix)
                i = random.randint(0, len(corrupt) - 2)
                if corrupt[i] in CONSONANTS and corrupt[i + 1] in CONSONANTS:
                    corrupt[i + 1] = random.choice(list(VOWEL_SIGNS))
                else:
                    corrupt[i] = random.choice(list(VOWEL_SIGNS))
                corrupt_str = "".join(corrupt)
                dataset.append({"partial_sequence": corrupt_str, "label": 0})

    random.shuffle(dataset)
    dataset = dataset[:num_examples]  # cap at requested size

    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")

    print(f"✅ Generated {len(dataset):,} syllable validity examples")
    print(f"   → {output_path}")
    print(f"   Positive: {sum(1 for d in dataset if d['label'] == 1)}")
    print(f"   Negative: {sum(1 for d in dataset if d['label'] == 0)}")


if __name__ == "__main__":
    generate_syllable_validity_dataset()