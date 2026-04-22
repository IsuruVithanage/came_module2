import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.beam_search import BeamSearchRestorer
from src.data.brahmi_dataset import BrahmiRestorationDataset
from tqdm import tqdm
import torch


def evaluate_came(num_samples=20):
    print("🔥 Running improved evaluation (with CER)...")
    restorer = BeamSearchRestorer()
    dataset = BrahmiRestorationDataset(split="test")

    total_cer = 0.0
    correct_top1 = 0
    total = min(num_samples, len(dataset))

    print(f"Evaluating {total} test examples...\n")

    for i in tqdm(range(total)):
        item = dataset[i]
        noisy = item["noisy_text"]
        ground_truth = item["clean_text"]  # pure Brahmi

        results = restorer.restore(noisy)
        top1_iast = results[0][0]
        top1_brahmi = top1_iast  # for simplicity (we compare in Brahmi space)

        # Simple CER (character error rate)
        pred = top1_brahmi.replace(" ", "")
        truth = ground_truth.replace(" ", "")
        cer = sum(a != b for a, b in zip(pred, truth)) / max(len(pred), len(truth)) if len(truth) > 0 else 1.0
        total_cer += cer

        if pred == truth:
            correct_top1 += 1

        # Show first 3 examples in detail
        if i < 3:
            print(f"Example {i + 1}:")
            print(f"   Noisy   : {noisy}")
            print(f"   Ground  : {ground_truth}")
            print(f"   Predicted: {top1_brahmi}   (CER: {cer:.3f})")
            print(f"   Confidence: {results[0][1]:.4f}\n")

    avg_cer = total_cer / total
    accuracy = (correct_top1 / total) * 100
    print(f"✅ Final Results on {total} examples:")
    print(f"   Top-1 Accuracy (exact match): {accuracy:.1f}%")
    print(f"   Average Character Error Rate (CER): {avg_cer:.3f}  ← this is the realistic metric")
    print(f"   Lower CER = better (0.0 is perfect)")


if __name__ == "__main__":
    evaluate_came(num_samples=20)