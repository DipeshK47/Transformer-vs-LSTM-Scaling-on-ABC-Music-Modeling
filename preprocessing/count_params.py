# check_param_counts.py
import json
import os
import torch

from gpt_model import GPT, get_model_config

VOCAB_PATH = "/scratch/dk5288/data/abc_char_corpus_98_1_1/vocab.json"
BLOCK_SIZE = 512

def main():
    print("[check_param_counts] Starting parameter count check...")

    with open(VOCAB_PATH, "r", encoding="utf8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    print(f"Vocab size from file: {vocab_size}")

    for name in ["tiny", "small", "medium", "large", "xl"]:
        print(f"[check_param_counts] Building model '{name}'...")
        cfg = get_model_config(name, vocab_size=vocab_size, block_size=BLOCK_SIZE)
        model = GPT(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:6s}: {n_params / 1e6:.2f}M parameters")

    print("[check_param_counts] Done. All model sizes computed successfully.")

if __name__ == "__main__":
    main()