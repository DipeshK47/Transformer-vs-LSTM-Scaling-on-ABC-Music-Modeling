# train_gpt_part4.py
import os
import json
import math
import time
import argparse
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from gpt_model import GPT, get_model_config

# suppress all warnings (including the AMP FutureWarning)
warnings.filterwarnings("ignore")

# -----------------------------
# Paths (same as Part 2)
# -----------------------------
DATA_DIR = "/scratch/dk5288/data/abc_char_corpus_98_1_1"
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT = os.path.join(DATA_DIR, "val.txt")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

# -----------------------------
# Hyperparameters (same style as Part 2)
# -----------------------------
TARGET_TOKENS = 200_000_000        # 200M tokens per epoch
VAL_TOKENS = 5_000_000
BLOCK_SIZE = 256
BATCH_SIZE = 64

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_FRACTION = 0.05
MAX_GRAD_NORM = 1.0

EPOCHS = 5
EVAL_INTERVAL = 500       # global steps
EVAL_ITERS = 200
SEED = 1337

# -----------------------------
# Data utilities
# -----------------------------
def load_text(path, max_chars=None):
    with open(path, "r", encoding="utf8") as f:
        if max_chars is None:
            return f.read()

        chunks = []
        remaining = max_chars
        chunk_size = 1024 * 1024

        while remaining > 0:
            chunk = f.read(min(chunk_size, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)

        return "".join(chunks)


def encode_text(text, stoi):
    ids = [stoi.get(ch, 0) for ch in text]
    return torch.tensor(ids, dtype=torch.long)


class CharDataset:
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size
        self.num_tokens = len(ids)

    def get_batch(self, batch_size, device):
        ix = torch.randint(0, self.num_tokens - self.block_size - 1, (batch_size,))
        x = torch.stack([self.ids[i:i + self.block_size] for i in ix])
        y = torch.stack([self.ids[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)


# -----------------------------
# LR schedule (same as Part 2)
# -----------------------------
def get_lr(step, max_steps):
    warmup_steps = int(WARMUP_FRACTION * max_steps)

    if step < warmup_steps:
        return LEARNING_RATE * (step + 1) / max(1, warmup_steps)

    decay_steps = max_steps - warmup_steps
    if decay_steps <= 0:
        return LEARNING_RATE

    progress = (step - warmup_steps) / decay_steps
    return LEARNING_RATE * max(0.0, 1.0 - progress)


# -----------------------------
# Evaluation
# -----------------------------
def estimate_val_loss(model, val_data, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(EVAL_ITERS):
            xb, yb = val_data.get_batch(BATCH_SIZE, device)
            with autocast():
                _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# -----------------------------
# Main training loop
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/scratch/dk5288/models/part4_best",
        help="Where to save checkpoints and logs",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Load vocab
    with open(VOCAB_PATH, "r", encoding="utf8") as f:
        vocab = json.load(f)
    stoi = {ch: int(i) for ch, i in vocab.items()}
    vocab_size = len(stoi)

    # Load data
    print("Loading data...")
    train_text = load_text(TRAIN_TXT, max_chars=TARGET_TOKENS)
    val_text = load_text(VAL_TXT, max_chars=VAL_TOKENS)

    train_ids = encode_text(train_text, stoi)
    val_ids = encode_text(val_text, stoi)

    train_data = CharDataset(train_ids, BLOCK_SIZE)
    val_data = CharDataset(val_ids, BLOCK_SIZE)

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    steps_per_epoch = TARGET_TOKENS // tokens_per_step

    print(f"tokens_per_step = {tokens_per_step}")
    print(f"steps_per_epoch = {steps_per_epoch}")

    # Build XL model (same config as Part 2)
    cfg = get_model_config("xl", vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model = GPT(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"XL model has {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler()

    best_val = float("inf")
    global_step = 0
    total_start = time.time()

    # -----------------------------
    # Epoch loop with tqdm
    # -----------------------------
    for epoch in range(1, EPOCHS + 1):
        print(f"\n========== EPOCH {epoch}/{EPOCHS} ==========")
        epoch_start = time.time()

        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch}/{EPOCHS}",
            ncols=100,
            leave=True,
        )

        for step_in_epoch in pbar:
            # LR schedule reset each epoch
            lr = get_lr(step_in_epoch, steps_per_epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            xb, yb = train_data.get_batch(BATCH_SIZE, device)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                _, loss = model(xb, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            # update tqdm bar with loss and lr every few steps
            if global_step % 20 == 0:
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{lr:.2e}",
                )

            # periodic eval across epochs
            if global_step % EVAL_INTERVAL == 0 and global_step > 0:
                val_loss = estimate_val_loss(model, val_data, device)
                print(f"\n[eval step {global_step}] val_loss={val_loss:.4f}")

                if val_loss < best_val:
                    best_val = val_loss
                    ckpt = os.path.join(args.out_dir, "best_xl.pt")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "val_loss": val_loss,
                            "epoch": epoch,
                            "global_step": global_step,
                            "config": cfg.__dict__,
                        },
                        ckpt,
                    )
                    print("[checkpoint] Saved new best XL model")

            global_step += 1

        epoch_time = time.time() - epoch_start
        if device == "cuda":
            mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[epoch {epoch}] GPU peak memory: {mem_gb:.2f} GB")
        print(f"[epoch {epoch}] Duration: {epoch_time/60:.2f} minutes")

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time/3600:.2f} hours")

    # Save final model
    final_path = os.path.join(args.out_dir, "final_xl.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
        },
        final_path,
    )
    print("Saved final XL model")


if __name__ == "__main__":
    main()