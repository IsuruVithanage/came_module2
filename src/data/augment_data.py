import sys
from pathlib import Path

# 1. Dynamically find the absolute project root folder
project_root = Path(__file__).parent.parent.parent

# Ensure Python can find your custom modules
sys.path.insert(0, str(project_root))

from src.utils.build_syllable_vocab import extract_aksharas


def augment_training_data(
        window_sizes=[5, 8],  # The lengths of the stone fragments we want to simulate
        step_size=3  # How many syllables to slide forward for each chunk
):
    print("🔨 Starting Data Augmentation (Sliding Window Cropping)...")

    # 2. Use the absolute paths so it never gets lost!
    input_path = project_root / "data" / "processed" / "train.txt"
    output_path = project_root / "data" / "processed" / "train_augmented.txt"

    with open(input_path, "r", encoding="utf-8") as f:
        original_lines = [line.strip() for line in f if line.strip()]

    augmented_lines = set()  # Use a set to automatically remove exact duplicates

    for line in original_lines:
        # 1. Always keep the original, complete sentence
        augmented_lines.add(line)

        # 2. Break the sentence into safe, unbreakable physical syllables
        syllables = extract_aksharas(line)
        length = len(syllables)

        # 3. If the sentence is long enough, shatter it into overlapping chunks
        for window in window_sizes:
            if length > window:
                for start_idx in range(0, length - window + 1, step_size):
                    chunk_syllables = syllables[start_idx: start_idx + window]
                    chunk_text = "".join(chunk_syllables)
                    augmented_lines.add(chunk_text)

    # 4. Save the massive new dataset
    with open(output_path, "w", encoding="utf-8") as f:
        for line in sorted(augmented_lines):
            f.write(line + "\n")

    print(f"✅ Augmentation Complete!")
    print(f"   Original Sentences : {len(original_lines)}")
    print(f"   New Dataset Size   : {len(augmented_lines)} fragments")
    print(f"   Multiplier         : {len(augmented_lines) / len(original_lines):.1f}x larger!")


if __name__ == "__main__":
    augment_training_data()