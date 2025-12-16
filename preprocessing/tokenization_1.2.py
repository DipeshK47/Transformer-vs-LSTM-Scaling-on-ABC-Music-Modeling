import os
import json
import random
from pathlib import Path
from collections import Counter
from tqdm import tqdm

# Input and output paths
ABC_ROOT = "/scratch/dk5288/data/abc_midi2abc"
OUT_DIR = "/scratch/dk5288/data/abc_char_corpus"

TRAIN_PATH = os.path.join(OUT_DIR, "train.txt")
VAL_PATH = os.path.join(OUT_DIR, "val.txt")
VOCAB_PATH = os.path.join(OUT_DIR, "vocab.json")
STATS_PATH = os.path.join(OUT_DIR, "stats.json")

# Hyperparameters for splitting and filtering
VAL_FRACTION = 0.1
MIN_CHARS = 64          # drop very tiny tunes
MAX_FILE_BYTES = 2_000_000  # skip huge or corrupted ABC files (> 2 MB)

def load_abc_text(path: Path) -> str:
    """
    Load an ABC file as text, dropping comment lines starting with %.
    Does not hold anything except this file in memory.
    """
    lines = []
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            if line.startswith("%"):
                continue
            lines.append(line)
    # Keep original newlines, plus a blank line at end to separate tunes
    return "".join(lines).rstrip() + "\n\n"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Collect all ABC files
    paths = sorted(Path(ABC_ROOT).rglob("*.abc"))
    n_total = len(paths)
    print(f"Found {n_total} ABC files under {ABC_ROOT}")

    if n_total == 0:
        print("No ABC files found. Check ABC_ROOT.")
        return

    # Shuffle and split into train and val at file level
    random.seed(1337)
    random.shuffle(paths)
    n_val = int(VAL_FRACTION * n_total)
    val_set = set(paths[:n_val])

    print(f"Train files: {n_total - n_val}, Val files: {n_val}")
    print(f"Skipping files larger than {MAX_FILE_BYTES} bytes")

    char_counter = Counter()
    lengths = []

    num_train_tunes = 0
    num_val_tunes = 0
    num_skipped_short = 0
    num_skipped_large = 0
    num_failed_read = 0
    total_chars = 0

    with open(TRAIN_PATH, "w", encoding="utf8") as f_train, \
         open(VAL_PATH, "w", encoding="utf8") as f_val:

        for path in tqdm(paths, desc="Tokenizing ABC tunes (char level)"):
            try:
                size_bytes = path.stat().st_size
                if size_bytes > MAX_FILE_BYTES:
                    # Likely corrupted or extremely long, skip to avoid OOM
                    num_skipped_large += 1
                    continue

                text = load_abc_text(path)
            except Exception as e:
                print(f"Failed to read {path}: {e}")
                num_failed_read += 1
                continue

            n_chars = len(text)
            if n_chars < MIN_CHARS:
                num_skipped_short += 1
                continue

            # Update vocab and stats
            char_counter.update(text)
            lengths.append(n_chars)
            total_chars += n_chars

            # Write to train or val
            if path in val_set:
                f_val.write(text)
                num_val_tunes += 1
            else:
                f_train.write(text)
                num_train_tunes += 1

    # Build vocab mapping char -> id
    vocab_chars = sorted(char_counter.keys())
    vocab = {ch: i for i, ch in enumerate(vocab_chars)}

    # Compute basic stats
    if lengths:
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)

        def perc(p):
            idx = int(p * (n - 1))
            return lengths_sorted[idx]

        stats = {
            "num_tunes_total": num_train_tunes + num_val_tunes,
            "num_tunes_train": num_train_tunes,
            "num_tunes_val": num_val_tunes,
            "num_tunes_skipped_short": num_skipped_short,
            "num_tunes_skipped_large": num_skipped_large,
            "num_failed_read": num_failed_read,
            "total_chars": total_chars,
            "avg_chars_per_tune": total_chars / max(1, len(lengths_sorted)),
            "p50_chars": perc(0.50),
            "p90_chars": perc(0.90),
            "p95_chars": perc(0.95),
            "p99_chars": perc(0.99),
            "vocab_size": len(vocab),
            "example_chars": vocab_chars[:50],
        }
    else:
        stats = {
            "num_tunes_total": 0,
            "num_tunes_train": 0,
            "num_tunes_val": 0,
            "num_tunes_skipped_short": num_skipped_short,
            "num_tunes_skipped_large": num_skipped_large,
            "num_failed_read": num_failed_read,
            "total_chars": 0,
            "avg_chars_per_tune": 0,
            "vocab_size": 0,
            "example_chars": [],
        }

    # Save vocab and stats
    with open(VOCAB_PATH, "w", encoding="utf8") as f:
        json.dump(vocab, f, indent=2)

    with open(STATS_PATH, "w", encoding="utf8") as f:
        json.dump(stats, f, indent=2)

    print("Done.")
    print(f"Train tunes: {num_train_tunes}")
    print(f"Val tunes:   {num_val_tunes}")
    print(f"Skipped tiny tunes:  {num_skipped_short}")
    print(f"Skipped large tunes: {num_skipped_large}")
    print(f"Failed reads:        {num_failed_read}")
    print(f"Total chars:         {total_chars}")
    print(f"Vocab size:          {len(vocab)}")
    print(f"Stats written to {STATS_PATH}")
    print(f"Train file: {TRAIN_PATH}")
    print(f"Val file:   {VAL_PATH}")

if __name__ == "__main__":
    main()