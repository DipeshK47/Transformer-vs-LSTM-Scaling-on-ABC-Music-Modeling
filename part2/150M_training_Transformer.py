# train_gpt_scaling.py
import os
import json
import math
import time
import argparse
import random

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from gpt_model import GPT, get_model_config

# Paths to your processed data
DATA_DIR = "/scratch/dk5288/data/abc_char_corpus_98_1_1"
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT = os.path.join(DATA_DIR, "val.txt")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

# Training regime (same for all model sizes)
TARGET_TOKENS = 150_000_000     # tokens seen in one epoch
VAL_TOKENS = 5_000_000          # subset of val text for eval

# Heavier training but still reasonable
BLOCK_SIZE = 256                # context length
BATCH_SIZE = 64                 # sequences per batch

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
WARMUP_FRACTION = 0.05          # fraction of steps used for LR warmup
MAX_GRAD_NORM = 1.0
EVAL_INTERVAL = 500             # steps between full val eval
EVAL_ITERS = 200                # val batches per eval
SEED = 1337


def load_text(path, max_chars=None):
    """Load up to max_chars characters from a text file."""
    print(f"[load_text] Loading from {path} (max_chars={max_chars})")
    with open(path, "r", encoding="utf8") as f:
        if max_chars is None:
            text = f.read()
        else:
            chunks = []
            remaining = max_chars
            chunk_size = 1024 * 1024
            while remaining > 0:
                chunk = f.read(min(chunk_size, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            text = "".join(chunks)
    print(f"[load_text] Loaded {len(text)} characters")
    return text


def encode_text(text, stoi):
    """Map characters to integer token ids."""
    print("[encode_text] Encoding text to ids...")
    ids = [stoi.get(ch, 0) for ch in text]
    ids_t = torch.tensor(ids, dtype=torch.long)
    print(f"[encode_text] Encoded {len(ids_t)} tokens")
    return ids_t


class CharDataset:
    """Simple random crop dataset over a long sequence of token ids."""
    def __init__(self, ids, block_size):
        self.ids = ids
        self.block_size = block_size
        self.num_tokens = len(ids)

    def get_batch(self, batch_size, device):
        ix = torch.randint(
            0,
            self.num_tokens - self.block_size - 1,
            (batch_size,),
        )
        x = torch.stack([self.ids[i:i + self.block_size] for i in ix])
        y = torch.stack([self.ids[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)


def get_lr(step, max_steps):
    """Linear warmup then linear decay to zero."""
    warmup_steps = int(WARMUP_FRACTION * max_steps)
    if step < warmup_steps:
        return LEARNING_RATE * (step + 1) / max(1, warmup_steps)
    else:
        decay_steps = max_steps - warmup_steps
        if decay_steps <= 0:
            return LEARNING_RATE
        progress = (step - warmup_steps) / decay_steps
        return LEARNING_RATE * max(0.0, 1.0 - progress)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium", "large", "xl"],
        help="Which model configuration to train",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/scratch/dk5288/models/abc_scaling",
        help="Where to save checkpoints and logs",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[train] Starting training for model_size={args.model_size}")
    print(f"[train] Output directory: {args.out_dir}")

    torch.manual_seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Using device: {device}")

    # reset GPU peak memory stats so we get clean numbers for this run
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Load vocab
    with open(VOCAB_PATH, "r", encoding="utf8") as f:
        vocab = json.load(f)
    stoi = {ch: int(idx) for ch, idx in vocab.items()}
    vocab_size = len(stoi)
    print(f"[train] Vocab size: {vocab_size}")

    # Load subset of train and val text for one epoch regime
    train_text = load_text(TRAIN_TXT, max_chars=TARGET_TOKENS)
    val_text = load_text(VAL_TXT, max_chars=VAL_TOKENS)

    # Encode to ids
    train_ids = encode_text(train_text, stoi)
    val_ids = encode_text(val_text, stoi)

    train_data = CharDataset(train_ids, BLOCK_SIZE)
    val_data = CharDataset(val_ids, BLOCK_SIZE)

    tokens_per_step = BATCH_SIZE * BLOCK_SIZE
    max_steps = TARGET_TOKENS // tokens_per_step
    print(f"[train] tokens_per_step = {tokens_per_step}")
    print(f"[train] max_steps (one epoch over ~{TARGET_TOKENS} tokens) = {max_steps}")

    # Build model
    cfg = get_model_config(args.model_size, vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model = GPT(cfg).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model {args.model_size} has {n_params / 1e6:.2f}M parameters")

    # Optimizer and scaler for mixed precision
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = GradScaler()

    # Training loop
    print("[train] Beginning training loop...")
    start_time = time.time()

    best_val = float("inf")
    for step in range(max_steps):
        # Update LR
        lr = get_lr(step, max_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get a batch
        xb, yb = train_data.get_batch(BATCH_SIZE, device)

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            _, loss = model(xb, yb)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"[train] step {step}/{max_steps} "
                f"train_loss={loss.item():.4f} lr={lr:.2e} "
                f"elapsed={elapsed/60:.2f} min"
            )

        if step % EVAL_INTERVAL == 0 or step == max_steps - 1:
            val_loss = estimate_val_loss(model, val_data, device)
            print(f"[eval] step {step} val_loss={val_loss:.4f}")

            # Save best checkpoint for this model size
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(args.out_dir, f"{args.model_size}_best.pt")
                torch.save(
                    {
                        "model_size": args.model_size,
                        "model_state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "step": step,
                        "config": cfg.__dict__,
                    },
                    ckpt_path,
                )
                print(f"[ckpt] Saved new best model to {ckpt_path}")

    total_time = time.time() - start_time
    print(f"[train] Done. Total training time: {total_time/60:.2f} minutes")

    # report peak GPU memory
    if device == "cuda":
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"[train] Peak GPU memory allocated: {max_mem_gb:.2f} GB")

    # Save final checkpoint too
    final_path = os.path.join(args.out_dir, f"{args.model_size}_final.pt")
    torch.save(
        {
            "model_size": args.model_size,
            "model_state_dict": model.state_dict(),
            "config": cfg.__dict__,
        },
        final_path,
    )
    print(f"[ckpt] Saved final model to {final_path}")
    print("[train] Training script finished cleanly.")


if __name__ == "__main__":
    main()