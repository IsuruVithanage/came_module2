import sys
from pathlib import Path
import editdistance

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
        ground_truth = item["clean_text"]

        results = restorer.restore(noisy)
        top1_brahmi = results[0][0]
        top1_iast = results[0][1]
        confidence = results[0][2]

        pred = top1_brahmi.replace(" ", "")
        truth = ground_truth.replace(" ", "")

        # Calculate true Levenshtein Character Error Rate
        if len(truth) > 0:
            cer = editdistance.eval(pred, truth) / max(len(pred), len(truth))
        else:
            cer = 1.0

        total_cer += cer

        if pred == truth:
            correct_top1 += 1

        if i < 3:
            print(f"Example {i + 1}:")
            print(f"   Noisy     : {noisy}")
            print(f"   Ground    : {ground_truth}")
            print(f"   Predicted : {top1_brahmi}")
            print(f"   IAST      : {top1_iast}   (CER: {cer:.3f})")
            print(f"   Confidence: {confidence:.4f}\n")

    avg_cer = total_cer / total
    accuracy = (correct_top1 / total) * 100
    print(f"✅ Final Results on {total} examples:")
    print(f"   Top-1 Accuracy (exact match): {accuracy:.1f}%")
    print(f"   Average Character Error Rate (CER): {avg_cer:.3f} ")
    print(f"   Lower CER = better (0.0 is perfect)")


if __name__ == "__main__":
    evaluate_came(num_samples=20)