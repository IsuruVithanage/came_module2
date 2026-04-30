import sys
from pathlib import Path
import editdistance

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.beam_search import BeamSearchRestorer
from src.data.brahmi_dataset import BrahmiRestorationDataset
from tqdm import tqdm
import torch


def evaluate_came(num_samples=20):
    print("🔥 Running improved evaluation (with CER & Top-3 Accuracy)...")
    restorer = BeamSearchRestorer()
    dataset = BrahmiRestorationDataset(split="test")

    total_cer = 0.0
    correct_top1 = 0
    correct_top3 = 0  # <--- Added tracker for Top 3
    total = min(num_samples, len(dataset))

    print(f"Evaluating {total} test examples...\n")

    for i in tqdm(range(total)):
        item = dataset[i]
        noisy = item["noisy_text"]
        ground_truth = item["clean_text"]

        # Get all predictions sorted by confidence
        results = restorer.restore(noisy)

        # Grab the absolute best guess for Top-1 and CER printing
        top1_brahmi = results[0][0]
        top1_iast = results[0][1]
        confidence = results[0][2]

        pred_top1 = top1_brahmi.replace(" ", "")
        truth = ground_truth.replace(" ", "")

        # <--- NEW: Extract the Top 3 predictions safely
        # results[:3] grabs up to the first 3 items in the list
        top3_preds = [res[0].replace(" ", "") for res in results[:3]]

        # Calculate true Levenshtein Character Error Rate (based on Top 1)
        if len(truth) > 0:
            cer = editdistance.eval(pred_top1, truth) / max(len(pred_top1), len(truth))
        else:
            cer = 1.0

        total_cer += cer

        # Grade Top-1
        if pred_top1 == truth:
            correct_top1 += 1

        # Grade Top-3
        if truth in top3_preds:
            correct_top3 += 1

        if i < 3:
            print(f"Example {i + 1}:")
            print(f"   Noisy     : {noisy}")
            print(f"   Ground    : {ground_truth}")
            print(f"   Predicted : {top1_brahmi}")
            print(f"   IAST      : {top1_iast}   (CER: {cer:.3f})")
            print(f"   Confidence: {confidence:.4f}\n")

    avg_cer = total_cer / total
    acc_top1 = (correct_top1 / total) * 100
    acc_top3 = (correct_top3 / total) * 100  # <--- Calculate Top 3 Percentage

    print(f"✅ Final Results on {total} examples:")
    print(f"   Top-1 Accuracy (exact match): {acc_top1:.1f}%")
    print(f"   Top-3 Accuracy              : {acc_top3:.1f}%")  # <--- Print Top 3
    print(f"   Average Character Error Rate: {avg_cer:.3f} ")
    print(f"   Lower CER = better (0.0 is perfect)")


if __name__ == "__main__":
    # 9999 is much larger than your test set, so your min() function
    # will automatically restrict it to exactly 128 test items!
    evaluate_came(num_samples=9999)