
import os
import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


MIDI_ROOT = "/scratch/dk5288/data/lmd_full"  
ABC_ROOT = "/scratch/dk5288/data/abc_midi2abc"  
CORPUS_DIR = "/scratch/dk5288/data/abc_char_corpus_98_1_1"

STATS_JSON_PATH = os.path.join(CORPUS_DIR, "stats.json")
VOCAB_JSON_PATH = os.path.join(CORPUS_DIR, "vocab.json")
LENGTH_STATS_PATH = os.path.join(CORPUS_DIR, "dataset_length_stats.json")
HIST_PATH_LINEAR = os.path.join(CORPUS_DIR, "tune_length_hist.png")
HIST_PATH_LOGY = os.path.join(CORPUS_DIR, "tune_length_hist_logy.png")
SAMPLE_ABC_PATH = os.path.join(CORPUS_DIR, "sample_abc_snippet.abc")


def load_corpus_stats():
    print(f"Loading corpus stats from {STATS_JSON_PATH}")
    with open(STATS_JSON_PATH, "r", encoding="utf8") as f:
        stats = json.load(f)
    return stats


def load_vocab():
    print(f"Loading vocab from {VOCAB_JSON_PATH}")
    with open(VOCAB_JSON_PATH, "r", encoding="utf8") as f:
        vocab = json.load(f)
    
    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    
    sample_ids = sorted(inv_vocab.keys())[:50]
    sample_tokens = [inv_vocab[i] for i in sample_ids]
    return vocab, vocab_size, sample_tokens


def compute_conversion_success():
    print("Counting original MIDI files...")
    midi_files = list(Path(MIDI_ROOT).rglob("*.mid"))
    n_midi = len(midi_files)
    print(f"Total MIDI files under {MIDI_ROOT}: {n_midi}")

    print("Counting converted ABC files...")
    abc_files = list(Path(ABC_ROOT).rglob("*.abc"))
    n_abc = len(abc_files)
    print(f"Total ABC files under {ABC_ROOT}: {n_abc}")

    success_rate = 100.0 * n_abc / n_midi if n_midi > 0 else 0.0
    return n_midi, n_abc, success_rate, abc_files


def compute_tune_lengths(abc_files):
    lengths = []
    for path in tqdm(abc_files, desc="Measuring tune lengths"):
        try:
            text = path.read_text(encoding="utf8", errors="ignore")
            lengths.append(len(text))
        except Exception as e:
            print(f"Failed to read {path}: {e}")
    return np.array(lengths, dtype=np.int64)


def summarize_lengths(lengths):
    n = int(len(lengths))
    if n == 0:
        return None

    stats = {
        "num_tunes": n,
        "min_len": int(lengths.min()),
        "max_len": int(lengths.max()),
        "mean_len": float(lengths.mean()),
        "median_len": float(np.median(lengths)),
        "p90_len": float(np.percentile(lengths, 90)),
        "p95_len": float(np.percentile(lengths, 95)),
        "p99_len": float(np.percentile(lengths, 99)),
    }
    return stats


def plot_histograms(lengths):
    
    max_for_plot = np.percentile(lengths, 99.5)
    clipped = np.clip(lengths, 0, max_for_plot)

    plt.figure(figsize=(8, 5))
    plt.hist(clipped, bins=100)
    plt.xlabel("Tune length (characters)")
    plt.ylabel("Count")
    plt.title("ABC tune length distribution (clipped at 99.5 percentile)")
    plt.tight_layout()
    plt.savefig(HIST_PATH_LINEAR, dpi=150)
    plt.close()
    print(f"Saved length histogram (linear y) to {HIST_PATH_LINEAR}")

    
    plt.figure(figsize=(8, 5))
    plt.hist(clipped, bins=100, log=True)
    plt.xlabel("Tune length (characters)")
    plt.ylabel("Count (log scale)")
    plt.title("ABC tune length distribution (log y, clipped at 99.5 percentile)")
    plt.tight_layout()
    plt.savefig(HIST_PATH_LOGY, dpi=150)
    plt.close()
    print(f"Saved length histogram (log y) to {HIST_PATH_LOGY}")


def save_sample_abc(abc_files):
    if not abc_files:
        print("No ABC files available to sample from.")
        return
    sample_path = random.choice(abc_files)
    print(f"Sampling ABC snippet from {sample_path}")
    try:
        text = sample_path.read_text(encoding="utf8", errors="ignore")
        
        lines = text.splitlines()
        snippet = "\n".join(lines[:40])
        with open(SAMPLE_ABC_PATH, "w", encoding="utf8") as f:
            f.write(snippet)
        print(f"Saved sample ABC snippet to {SAMPLE_ABC_PATH}")
    except Exception as e:
        print(f"Failed to write sample ABC snippet: {e}")


def main():
    os.makedirs(CORPUS_DIR, exist_ok=True)

    
    corpus_stats = load_corpus_stats()

    
    vocab, vocab_size, sample_tokens = load_vocab()

    
    n_midi, n_abc, success_rate, abc_files = compute_conversion_success()

    
    lengths = compute_tune_lengths(abc_files)
    length_stats = summarize_lengths(lengths)

    
    if length_stats is not None:
        plot_histograms(lengths)

    
    save_sample_abc(abc_files)

    
    summary = {
        "vocab_size": vocab_size,
        "vocab_example_tokens": sample_tokens,
        "train_val_test_char_stats": corpus_stats,
        "midi_file_count": n_midi,
        "abc_file_count": n_abc,
        "conversion_success_rate_percent": success_rate,
        "tune_length_stats": length_stats,
    }

    with open(LENGTH_STATS_PATH, "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2)

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print(f"\nFull dataset statistics written to {LENGTH_STATS_PATH}")


if __name__ == "__main__":
    main()