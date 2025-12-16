import os
import json
from pathlib import Path
from tqdm import tqdm

# Root where all abc files from midi2abc live
ABC_ROOT = "/scratch/dk5288/data/abc_midi2abc"

# Where to write the cleaning report
OUT_DIR = "/scratch/dk5288/data/abc_char_corpus_98_1_1"
REPORT_PATH = os.path.join(OUT_DIR, "cleaning_report.json")

# Length thresholds for "short" and "long" pieces in characters
MIN_LEN_SHORT = 64       # below this is "very short"
MAX_LEN_LONG = 8192      # above this is "very long"

def main():
    abc_files = sorted(Path(ABC_ROOT).rglob("*.abc"))
    n_total = len(abc_files)
    print(f"Found {n_total} ABC files under {ABC_ROOT}")
    if n_total == 0:
        print("No ABC files found. Check ABC_ROOT.")
        return

    lengths = []
    empty_count = 0
    short_count = 0
    long_count = 0
    failed_reads = 0

    non_ascii_files = 0
    non_ascii_chars = set()

    for path in tqdm(abc_files, desc="Scanning ABC files"):
        try:
            with open(path, "r", encoding="utf8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            failed_reads += 1
            print(f"Failed to read {path}: {e}")
            continue

        L = len(text)
        lengths.append(L)

        if L == 0:
            empty_count += 1
        if L < MIN_LEN_SHORT:
            short_count += 1
        if L > MAX_LEN_LONG:
            long_count += 1

        # Check for non ascii chars
        if not text.isascii():
            non_ascii_files += 1
            for ch in text:
                if not ch.isascii():
                    non_ascii_chars.add(ch)

    if lengths:
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)

        def perc(p):
            idx = int(p * (n - 1))
            return lengths_sorted[idx]

        stats = {
            "num_files_total": n_total,
            "failed_reads": failed_reads,
            "empty_files": empty_count,
            "short_files_len_lt_%d" % MIN_LEN_SHORT: short_count,
            "long_files_len_gt_%d" % MAX_LEN_LONG: long_count,
            "min_len": lengths_sorted[0],
            "max_len": lengths_sorted[-1],
            "p50_len": perc(0.50),
            "p90_len": perc(0.90),
            "p95_len": perc(0.95),
            "p99_len": perc(0.99),
            "non_ascii_files": non_ascii_files,
            "non_ascii_chars_sample": sorted(list(non_ascii_chars))[:50],
        }
    else:
        stats = {
            "num_files_total": n_total,
            "failed_reads": failed_reads,
            "empty_files": empty_count,
            "short_files_len_lt_%d" % MIN_LEN_SHORT: short_count,
            "long_files_len_gt_%d" % MAX_LEN_LONG: long_count,
            "min_len": 0,
            "max_len": 0,
            "p50_len": 0,
            "p90_len": 0,
            "p95_len": 0,
            "p99_len": 0,
            "non_ascii_files": non_ascii_files,
            "non_ascii_chars_sample": [],
        }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf8") as f:
        json.dump(stats, f, indent=2)

    print("Step 1.4 cleaning analysis done.")
    print(f"Report written to {REPORT_PATH}")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()