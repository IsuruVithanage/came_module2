import json
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def prepare_brahmi_data(raw_json_path: str = "data/raw/brahmi_training_data_new.json",
                        processed_dir: str = "data/processed",
                        test_size: float = 0.2,
                        random_seed: int = 42):
    """
    Production-grade data preparation for CAME Module 2.
    Extracts brahmi_unicode_without_spaces, performs 80/10/10 split,
    and saves clean text files ready for training.
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the raw JSON
    logger.info(f"Loading dataset from {raw_json_path}")
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 2. Extract clean Brahmi Unicode (without spaces)
    clean_texts = []
    skipped = 0
    for idx, item in enumerate(raw_data):
        if isinstance(item, dict) and "brahmi_unicode_without_spaces" in item:
            text = str(item["brahmi_unicode_without_spaces"]).strip()
            if text:  # skip empty strings
                clean_texts.append(text)
            else:
                skipped += 1
        else:
            skipped += 1

    logger.info(f"✅ Extracted {len(clean_texts)} clean Brahmi strings")
    logger.info(f"   Skipped {skipped} invalid/missing entries")

    if len(clean_texts) < 10:
        raise ValueError("Not enough valid Brahmi data! Check your JSON.")

    # 3. Train / Val / Test split (80 / 10 / 10)
    train_texts, temp = train_test_split(
        clean_texts, test_size=test_size, random_state=random_seed
    )
    val_texts, test_texts = train_test_split(
        temp, test_size=0.5, random_state=random_seed
    )

    # 4. Save splits as simple .txt files (one string per line)
    splits = {
        "train": train_texts,
        "val": val_texts,
        "test": test_texts
    }

    for split_name, texts in splits.items():
        output_path = processed_dir / f"{split_name}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(texts))
        logger.info(f"   Saved {len(texts)} examples → {output_path}")

    # 5. Optional: Save a small JSONL version for quick inspection
    sample_jsonl = processed_dir / "sample_brahmi.jsonl"
    with open(sample_jsonl, "w", encoding="utf-8") as f:
        for text in clean_texts[:50]:
            f.write(json.dumps({"brahmi_unicode": text}) + "\n")
    logger.info(f"   Sample JSONL created for inspection → {sample_jsonl}")

    # Final statistics
    logger.info("\n" + "="*60)
    logger.info("🎉 DATA PREPARATION COMPLETE")
    logger.info(f"Train : {len(train_texts):,} examples")
    logger.info(f"Val   : {len(val_texts):,} examples")
    logger.info(f"Test  : {len(test_texts):,} examples")
    logger.info(f"Total : {len(clean_texts):,} Brahmi strings")
    logger.info("="*60)

    return {
        "train": len(train_texts),
        "val": len(val_texts),
        "test": len(test_texts),
        "total": len(clean_texts)
    }


if __name__ == "__main__":
    stats = prepare_brahmi_data()
    print(f"\n✅ Ready for training! Total clean Brahmi strings: {stats['total']}")