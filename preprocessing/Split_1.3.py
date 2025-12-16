import os
import json
import random
from pathlib import Path
from tqdm import tqdm

# Root folder with your abc files from midi2abc
ABC_ROOT = "/scratch/dk5288/data/abc_midi2abc"

# Output folder for the final 98/1/1 char level corpus
OUT_DIR = "/scratch/dk5288/data/abc_char_corpus_98_1_1"

TRAIN_PATH = os.path.join(OUT_DIR, "train.txt")
VAL_PATH   = os.path.join(OUT_DIR, "val.txt")
TEST_PATH  = os.path.join(OUT_DIR, "test.txt")
VOCAB_PATH = os.path.join(OUT_DIR, "vocab.json")
STATS_PATH = os.path.join(OUT_DIR, "stats.json")

SPLIT_TRAIN = 0.98
SPLIT_VAL   = 0.01   # test gets the rest

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Collect all abc files
    abc_files = sorted(Path(ABC_ROOT).rglob("*.abc"))
    n_total = len(abc_files)
    print(f"Found {n_total} ABC files under {ABC_ROOT}")
    if n_total == 0:
        print("No ABC files found. Check ABC_ROOT.")
        return

    # 2. Shuffle and split file list into 98/1/1
    random.seed(1337)
    random.shuffle(abc_files)

    n_train = int(SPLIT_TRAIN * n_total)
    n_val   = int(SPLIT_VAL * n_total)
    n_test  = n_total - n_train - n_val

    train_files = abc_files[:n_train]
    val_files   = abc_files[n_train:n_train + n_val]
    test_files  = abc_files[n_train + n_val:]

    print(f"Train files: {len(train_files)}")
    print(f"Val files:   {len(val_files)}")
    print(f"Test files:  {len(test_files)}")

    # 3. Stream files, concatenate contents, write to train/val/test
    #    Char level tokenization here just means "characters are the tokens"
    #    so we only need to count characters as we write.

    train_chars = 0
    val_chars   = 0
    test_chars  = 0

    train_vocab_chars = set()  # vocab from train only
    failed_reads = 0

    # Helper to stream one split
    def write_split(files, out_path, count_chars, update_vocab=False):
        nonlocal failed_reads, train_vocab_chars

        with open(out_path, "w", encoding="utf8") as fout:
            for path in tqdm(files, desc=f"Writing {os.path.basename(out_path)}"):
                try:
                    with open(path, "r", encoding="utf8", errors="ignore") as fin:
                        text = fin.read()
                except Exception as e:
                    failed_reads += 1
                    print(f"Failed to read {path}: {e}")
                    continue

                # Append the tune plus a double newline separator
                fout.write(text)
                fout.write("\n\n")

                # Count characters (tokens)
                count_chars += len(text) + 2  # include the two newlines

                # Update train vocab from this split if requested
                if update_vocab:
                    train_vocab_chars.update(set(text))

        return count_chars

    # Train split (also builds vocab)
    train_chars = write_split(train_files, TRAIN_PATH, train_chars, update_vocab=True)

    # Val split
    val_chars = write_split(val_files, VAL_PATH, val_chars, update_vocab=False)

    # Test split
    test_chars = write_split(test_files, TEST_PATH, test_chars, update_vocab=False)

    # 4. Build vocab mapping from train chars
    vocab_list = sorted(train_vocab_chars)
    vocab = {ch: i for i, ch in enumerate(vocab_list)}

    # 5. Collect stats
    stats = {
        "num_files_total": n_total,
        "num_files_train": len(train_files),
        "num_files_val": len(val_files),
        "num_files_test": len(test_files),
        "train_chars": train_chars,
        "val_chars": val_chars,
        "test_chars": test_chars,
        "total_chars": train_chars + val_chars + test_chars,
        "vocab_size": len(vocab),
        "failed_reads": failed_reads,
    }

    # 6. Save vocab and stats
    with open(VOCAB_PATH, "w", encoding="utf8") as f:
        json.dump(vocab, f, indent=2)

    with open(STATS_PATH, "w", encoding="utf8") as f:
        json.dump(stats, f, indent=2)

    # 7. Print summary
    print("Done step 1.3.")
    print(f"Train chars: {train_chars}")
    print(f"Val chars:   {val_chars}")
    print(f"Test chars:  {test_chars}")
    print(f"Total chars: {train_chars + val_chars + test_chars}")
    print(f"Vocab size (train chars): {len(vocab)}")
    print(f"Stats written to: {STATS_PATH}")
    print(f"Train file: {TRAIN_PATH}")
    print(f"Val file:   {VAL_PATH}")
    print(f"Test file:  {TEST_PATH}")
    print(f"Failed reads: {failed_reads}")

if __name__ == "__main__":
    main()