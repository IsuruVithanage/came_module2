import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # ← This fixes the import

from src.inference.beam_search import BeamSearchRestorer
from src.data.brahmi_dataset import BrahmiRestorationDataset
import torch
from tqdm import tqdm


def evaluate_came(num_samples=20):
    print("🔥 Starting evaluation on test set...")

    # Load restorer (uses latest checkpoint if available)
    restorer = BeamSearchRestorer()

    dataset = BrahmiRestorationDataset(split="test")

    correct_top1 = 0
    total = min(num_samples, len(dataset))

    print(f"Evaluating {total} test examples...")
    for i in tqdm(range(total)):
        item = dataset[i]
        noisy = item["noisy_text"]
        ground_truth = item["clean_text"]

        try:
            results = restorer.restore(noisy)
            top1_iast = results[0][0].replace(" ", "").strip()
            truth = ground_truth.replace(" ", "").strip()

            if top1_iast == truth:
                correct_top1 += 1
        except Exception as e:
            print(f"⚠️  Skipped example {i} due to error: {e}")
            continue

    accuracy = (correct_top1 / total) * 100 if total > 0 else 0
    print(f"\n✅ Top-1 Accuracy on test set: {accuracy:.1f}% ({correct_top1}/{total})")
    return accuracy


if __name__ == "__main__":
    evaluate_came(num_samples=20)  # change number if you want more/fewer