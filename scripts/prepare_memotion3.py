"""
Memotion 3 Dataset Preparation for Humor LoRA Fine-Tuning.

Reads the already-downloaded Memotion 3 CSVs from data/processed/,
converts humor labels to binary HUMOR/NON-HUMOR, excludes the 20
controlled evaluation memes, validates image existence, and outputs
clean JSONL files for training.

This script is CPU-safe and does not require a GPU.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

# Binary label mapping: Memotion 3 humour → binary
HUMOR_LABEL_MAP = {
    "not_funny": "NON-HUMOR",
    "funny": "HUMOR",
    "very_funny": "HUMOR",
    "hilarious": "HUMOR",
}

# Instruction template for Qwen2.5-VL SFT
SYSTEM_MSG = "You are a multimodal humor classifier for Hindi/Hinglish memes. Respond with only HUMOR or NON-HUMOR."

USER_TEMPLATE = (
    "Analyze this Hindi/Hinglish meme using the image and text.\n"
    "Text: {ocr_text}\n\n"
    "Is this meme humorous? Respond with only HUMOR or NON-HUMOR."
)


def load_eval_meme_filenames(eval_csv_path: str) -> set:
    """Load the 20 controlled evaluation meme filenames to exclude."""
    if not os.path.exists(eval_csv_path):
        logger.warning("Evaluation CSV not found at %s — cannot exclude eval memes!", eval_csv_path)
        return set()

    eval_df = pd.read_csv(eval_csv_path)
    if "image_filename" not in eval_df.columns:
        logger.warning("Evaluation CSV does not contain 'image_filename' column. Columns: %s", list(eval_df.columns))
        return set()

    filenames = set(eval_df["image_filename"].tolist())
    logger.info("Loaded %d evaluation meme filenames for exclusion.", len(filenames))
    return filenames


def validate_and_load_split(csv_path: str, images_dir: str, split_name: str) -> pd.DataFrame:
    """Load a single split CSV and validate its contents."""
    if not os.path.exists(csv_path):
        logger.error("CSV file not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("[%s] Loaded %d rows from %s", split_name, len(df), csv_path)

    # Validate required columns
    required_cols = {"humour", "ocr", "image_filename"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        logger.error("[%s] Missing required columns: %s. Available: %s", split_name, missing_cols, list(df.columns))
        sys.exit(1)

    # Check for missing values
    for col in required_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            logger.warning("[%s] %d missing values in column '%s'", split_name, n_missing, col)

    # Drop rows with missing essential values
    initial_len = len(df)
    df = df.dropna(subset=["humour", "ocr", "image_filename"])
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info("[%s] Dropped %d rows with missing essential values.", split_name, dropped)

    # Validate humor label values
    unknown_labels = set(df["humour"].unique()) - set(HUMOR_LABEL_MAP.keys())
    if unknown_labels:
        logger.error("[%s] Unknown humor label values: %s. Expected: %s", split_name, unknown_labels, list(HUMOR_LABEL_MAP.keys()))
        sys.exit(1)

    # Validate image file existence
    missing_images = []
    for fname in df["image_filename"]:
        if not os.path.exists(os.path.join(images_dir, str(fname))):
            missing_images.append(fname)

    if missing_images:
        logger.warning("[%s] %d images not found on disk.", split_name, len(missing_images))
        df = df[~df["image_filename"].isin(missing_images)]
        logger.info("[%s] Removed %d rows with missing images. Remaining: %d", split_name, len(missing_images), len(df))
    else:
        logger.info("[%s] All %d images verified on disk.", split_name, len(df))

    return df


def convert_to_binary(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Apply binary label mapping."""
    df = df.copy()
    df["binary_label"] = df["humour"].map(HUMOR_LABEL_MAP)

    # Verify no unmapped labels
    unmapped = df["binary_label"].isna().sum()
    if unmapped > 0:
        logger.error("[%s] %d rows could not be mapped to binary labels!", split_name, unmapped)
        sys.exit(1)

    # Cross-validate with existing is_humorous column if present
    if "is_humorous" in df.columns:
        expected_binary = df["is_humorous"].map({1: "HUMOR", 0: "NON-HUMOR"})
        mismatches = (df["binary_label"] != expected_binary).sum()
        if mismatches > 0:
            logger.error("[%s] %d mismatches between binary_label and is_humorous!", split_name, mismatches)
            sys.exit(1)
        else:
            logger.info("[%s] Binary labels cross-validated against is_humorous: OK", split_name)

    return df


def exclude_eval_memes(df: pd.DataFrame, eval_filenames: set, split_name: str) -> pd.DataFrame:
    """Remove evaluation memes from the dataset."""
    if not eval_filenames:
        return df

    overlap = set(df["image_filename"].tolist()) & eval_filenames
    if overlap:
        logger.info("[%s] Excluding %d evaluation memes: %s", split_name, len(overlap), sorted(overlap))
        df = df[~df["image_filename"].isin(eval_filenames)]
    else:
        logger.info("[%s] No evaluation memes found in this split.", split_name)

    return df


def report_distribution(df: pd.DataFrame, split_name: str):
    """Log the class distribution for a split."""
    dist = Counter(df["binary_label"])
    total = len(df)
    logger.info("[%s] Class Distribution (total=%d):", split_name, total)
    for label in ["HUMOR", "NON-HUMOR"]:
        count = dist.get(label, 0)
        pct = count / total * 100 if total > 0 else 0
        logger.info("  %s: %d (%.1f%%)", label, count, pct)

    imbalance_ratio = max(dist.values()) / min(dist.values()) if min(dist.values()) > 0 else float("inf")
    if imbalance_ratio > 3.0:
        logger.warning("[%s] Significant class imbalance detected (ratio %.1f:1). Consider class weighting during training.", split_name, imbalance_ratio)


def check_text_duplicates(df: pd.DataFrame, split_name: str):
    """Check for duplicate OCR text entries."""
    dupes = df["ocr"].duplicated().sum()
    if dupes > 0:
        logger.warning("[%s] %d duplicate OCR text entries found.", split_name, dupes)
    else:
        logger.info("[%s] No duplicate OCR text entries.", split_name)


def save_jsonl(df: pd.DataFrame, images_dir: str, output_path: str, split_name: str):
    """Save the processed dataset as JSONL with instruction-tuning format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    abs_images_dir = os.path.abspath(images_dir)
    records = []

    for _, row in df.iterrows():
        image_path = os.path.join(abs_images_dir, str(row["image_filename"]))
        ocr_text = str(row["ocr"]).strip()
        binary_label = row["binary_label"]
        original_label = str(row["humour"])

        record = {
            "image": image_path,
            "image_filename": str(row["image_filename"]),
            "ocr_text": ocr_text,
            "label": binary_label,
            "original_label": original_label,
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": USER_TEMPLATE.format(ocr_text=ocr_text)},
                {"role": "assistant", "content": binary_label},
            ],
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("[%s] Saved %d records to %s", split_name, len(records), output_path)


def print_samples(df: pd.DataFrame, split_name: str, n: int = 3):
    """Print a few representative samples."""
    logger.info("[%s] Sample records:", split_name)
    for i, (_, row) in enumerate(df.head(n).iterrows()):
        logger.info("  #%d: image=%s, humour=%s -> %s, ocr=%s",
                     i + 1, row["image_filename"], row["humour"], row["binary_label"],
                     str(row["ocr"])[:80].replace("\n", " ").replace("\r", ""))


def main():
    parser = argparse.ArgumentParser(description="Prepare Memotion 3 dataset for humor LoRA training.")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing Memotion 3 CSVs and images/")
    parser.add_argument("--output-dir", type=str, default="data/memotion3_processed",
                        help="Directory to write the processed JSONL files")
    parser.add_argument("--eval-csv", type=str, default="results/evaluation_1790249336/comparison_results.csv",
                        help="Path to the controlled evaluation CSV for exclusion")
    args = parser.parse_args()

    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")
    output_dir = args.output_dir

    logger.info("=" * 60)
    logger.info("MEMOTION 3 DATASET PREPARATION")
    logger.info("=" * 60)
    logger.info("Data directory: %s", data_dir)
    logger.info("Images directory: %s", images_dir)
    logger.info("Output directory: %s", output_dir)

    # Verify images directory exists
    if not os.path.isdir(images_dir):
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    # Phase 4: Load evaluation meme filenames for exclusion
    eval_filenames = load_eval_meme_filenames(args.eval_csv)

    # Phase 1-2: Load and validate each split
    splits = {
        "train": os.path.join(data_dir, "train.csv"),
        "validation": os.path.join(data_dir, "validation.csv"),
        "test": os.path.join(data_dir, "test.csv"),
    }

    processed = {}
    for split_name, csv_path in splits.items():
        logger.info("")
        logger.info("-" * 40)
        logger.info("Processing: %s", split_name.upper())
        logger.info("-" * 40)

        # Load and validate
        df = validate_and_load_split(csv_path, images_dir, split_name)

        # Phase 3: Convert to binary labels
        df = convert_to_binary(df, split_name)

        # Phase 4: Exclude evaluation memes
        df = exclude_eval_memes(df, eval_filenames, split_name)

        # Phase 5: Report distribution
        report_distribution(df, split_name)
        check_text_duplicates(df, split_name)

        # Print samples
        print_samples(df, split_name)

        processed[split_name] = df

    # Phase 5: Overall summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DATASET SUMMARY (after exclusion)")
    logger.info("=" * 60)

    total_humor = 0
    total_non_humor = 0

    for split_name, df in processed.items():
        h = (df["binary_label"] == "HUMOR").sum()
        nh = (df["binary_label"] == "NON-HUMOR").sum()
        total_humor += h
        total_non_humor += nh
        logger.info("  %s: %d total (HUMOR=%d, NON-HUMOR=%d)", split_name, len(df), h, nh)

    total = total_humor + total_non_humor
    logger.info("")
    logger.info("  TOTAL: %d", total)
    logger.info("  HUMOR: %d (%.1f%%)", total_humor, total_humor / total * 100)
    logger.info("  NON-HUMOR: %d (%.1f%%)", total_non_humor, total_non_humor / total * 100)
    logger.info("  Imbalance ratio: %.1f:1", total_humor / total_non_humor if total_non_humor > 0 else float("inf"))
    logger.info("  Evaluation memes excluded: %d", 20 - sum(len(set(df["image_filename"]) & eval_filenames) for df in processed.values()))

    # Save JSONL files
    logger.info("")
    logger.info("=" * 60)
    logger.info("SAVING PROCESSED DATASET")
    logger.info("=" * 60)

    for split_name, df in processed.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        save_jsonl(df, images_dir, output_path, split_name)

    # Save preprocessing report
    report = {
        "dataset": "Memotion 3",
        "source_dir": os.path.abspath(data_dir),
        "images_dir": os.path.abspath(images_dir),
        "output_dir": os.path.abspath(output_dir),
        "eval_memes_excluded": len(eval_filenames),
        "label_mapping": HUMOR_LABEL_MAP,
        "splits": {},
    }
    for split_name, df in processed.items():
        dist = Counter(df["binary_label"])
        report["splits"][split_name] = {
            "total": int(len(df)),
            "HUMOR": int(dist.get("HUMOR", 0)),
            "NON-HUMOR": int(dist.get("NON-HUMOR", 0)),
        }
    report["total"] = int(total)
    report["total_HUMOR"] = int(total_humor)
    report["total_NON_HUMOR"] = int(total_non_humor)

    report_path = os.path.join(output_dir, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Preprocessing report saved to %s", report_path)

    logger.info("")
    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
