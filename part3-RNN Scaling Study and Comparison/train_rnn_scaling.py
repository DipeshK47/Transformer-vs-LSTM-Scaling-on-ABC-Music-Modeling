import sys
import time
import json
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
# ---------------------------------------------------------
# Paths and global training setup
# ---------------------------------------------------------

DATA_DIR = "/scratch/dk5288/data/abc_char_corpus_98_1_1"
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VAL_TXT = os.path.join(DATA_DIR, "val.txt")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

OUT_DIR = "/scratch/dk5288/code/my_project/part3"
os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_CSV = Path(OUT_DIR) / "rnn_scaling_results.csv"

BLOCK_SIZE = 256          # context length
BATCH_SIZE = 64           # sequences per batch
TOKENS_PER_STEP = BLOCK_SIZE * BATCH_SIZE

LR = 3e-4
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
DROPOUT = 0.1

WARMUP_FRACTION = 0.05    # first part of steps are warmup
EVAL_EVERY = 500          # steps between full val evals


# ---------------------------------------------------------
# RNN model size configs
# ---------------------------------------------------------

MODEL_CONFIGS = {
    "tiny":   dict(n_layers=1, d_model=128),
    "small":  dict(n_layers=2, d_model=256),
    "medium": dict(n_layers=2, d_model=384),
    "large":  dict(n_layers=3, d_model=512),
}


# ---------------------------------------------------------
# Data helpers
# ---------------------------------------------------------

def load_text(path: str, max_chars: int):
    print(f"[data] Loading from {path} (max_chars={max_chars})")
    with open(path, "r", encoding="utf8") as f:
        text = f.read(max_chars)
    print(f"[data] Loaded {len(text)} characters from {path}")
    return text


def encode_text(text: str, stoi: dict):
    print("[data] Encoding text to ids...")
    ids = [stoi.get(ch, 0) for ch in text]
    ids_t = torch.tensor(ids, dtype=torch.long)
    print(f"[data] Encoded {ids_t.numel()} tokens")
    return ids_t


class TokenDatasetTXT:
    """
    Streaming style dataset over a flat tensor of token ids.
    Equivalent pattern to the memmap TokenDataset in your friend's code.
    """

    def __init__(self, ids: torch.Tensor, max_effective_tokens=None):
        # ids is a 1D tensor of token ids
        total_tokens = ids.shape[0] - 1
        if max_effective_tokens is None:
            self.effective_tokens = total_tokens
        else:
            self.effective_tokens = min(total_tokens, max_effective_tokens)
        # clip underlying buffer to a safe range
        self.tokens = ids[: self.effective_tokens + 1]

    def num_steps_per_epoch(self) -> int:
        return self.effective_tokens // TOKENS_PER_STEP

    def get_batch(self, step_idx: int, device: torch.device):
        assert 0 <= step_idx < self.num_steps_per_epoch()
        base_index = step_idx * TOKENS_PER_STEP
        B = BATCH_SIZE
        T = BLOCK_SIZE

        # use numpy arrays then convert to torch for speed
        x = np.empty((B, T), dtype=np.int64)
        y = np.empty((B, T), dtype=np.int64)

        for i in range(B):
            start = base_index + i * T
            end = start + T
            x[i] = self.tokens[start:end]
            y[i] = self.tokens[start + 1 : end + 1]

        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        return x, y


# ---------------------------------------------------------
# RNN language model
# ---------------------------------------------------------

class RNNLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(vocab_size, d_model)

        self.rnn = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # tie weights with embedding
        self.head.weight = self.token_emb.weight

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.token_emb.weight, -0.1, 0.1)
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, idx, targets=None):
        # idx: (B, T)
        x = self.token_emb(idx)             # (B, T, C)
        out, _ = self.rnn(x)                # (B, T, C)
        out = self.ln_f(out)
        logits = self.head(out)             # (B, T, V)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T),
                reduction="mean",
            )
        return logits, loss


# ---------------------------------------------------------
# Helpers shared with transformer script
# ---------------------------------------------------------

def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf8") as f:
        vocab = json.load(f)
    # vocab here is char -> index
    stoi = {ch: int(idx) for ch, idx in vocab.items()}
    vocab_size = len(stoi)
    return stoi, vocab_size


def create_model(model_name: str, vocab_size: int, device: torch.device):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")
    cfg = MODEL_CONFIGS[model_name]
    print(f"[model] RNN config for {model_name}: {cfg}")
    model = RNNLM(
        vocab_size=vocab_size,
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        dropout=DROPOUT,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] RNN {model_name} has {n_params:,} trainable parameters")
    return model, n_params


def create_optimizer(model: nn.Module):
    return torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )


def create_scheduler(optimizer, total_steps: int):
    warmup_steps = int(WARMUP_FRACTION * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 1e-8), 1.0)
        # cosine from 1 down to 0
        return 0.5 * (1.0 + float(torch.cos(torch.tensor(progress * 3.1415926535))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_full(model: nn.Module, dataset: TokenDatasetTXT, device: torch.device):
    model.eval()
    steps = dataset.num_steps_per_epoch()
    total_loss = 0.0
    for step_idx in range(steps):
        x, y = dataset.get_batch(step_idx, device)
        _, loss = model(x, y)
        total_loss += loss.item()
    avg_loss = total_loss / max(1, steps)
    model.train()
    return avg_loss


def append_results_csv(row: dict):
    header = [
        "model_name",
        "mode",
        "n_params",
        "steps_per_epoch",
        "total_train_tokens",
        "final_val_loss",
        "wall_clock_seconds",
        "gpu_mem_start_mb",
        "gpu_mem_end_mb",
        "gpu_mem_peak_mb",
    ]
    file_exists = RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(",".join(header) + "\n")
        vals = [str(row.get(col, "")) for col in header]
        f.write(",".join(vals) + "\n")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="one of tiny, small, medium, large",
    )
    args = parser.parse_args()
    model_name = args.model_size

    print("=" * 30)
    print(f"Running LSTM training for {model_name}")
    print("=" * 30)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")
    print(f"[train] Output directory: {OUT_DIR}")

    stoi, vocab_size = load_vocab()
    print(f"[train] Vocab size: {vocab_size}")

    # Load about 100M train tokens and 1M val tokens, plus BLOCK_SIZE for shifting
    max_train_tokens = 100_000_000
    max_val_tokens = 1_000_000

    train_text = load_text(TRAIN_TXT, max_train_tokens + BLOCK_SIZE)
    val_text = load_text(VAL_TXT, max_val_tokens + BLOCK_SIZE)

    train_ids_full = encode_text(train_text, stoi)
    val_ids_full = encode_text(val_text, stoi)

    train_ds = TokenDatasetTXT(train_ids_full, max_effective_tokens=max_train_tokens)
    val_ds = TokenDatasetTXT(val_ids_full, max_effective_tokens=max_val_tokens)

    steps_per_epoch = train_ds.num_steps_per_epoch()
    total_train_tokens = steps_per_epoch * TOKENS_PER_STEP

    print(f"[train] Train effective tokens: {train_ds.effective_tokens:,}")
    print(f"[train] Val effective tokens:   {val_ds.effective_tokens:,}")
    print(f"[train] Tokens per step:        {TOKENS_PER_STEP:,}")
    print(f"[train] Steps per epoch:        {steps_per_epoch:,}")
    print(f"[train] Total train tokens:     {total_train_tokens:,}")

    model, n_params = create_model(model_name, vocab_size, device)

    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer, total_steps=steps_per_epoch)

    gpu_mem_before = None
    gpu_mem_after = None
    gpu_mem_peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        gpu_mem_before = torch.cuda.memory_allocated() / 1024**2
        print(f"[train] GPU memory at start: {gpu_mem_before:.2f} MB")

    print("[train] Training for exactly one epoch over streaming batches")
    start_time = time.time()

    best_val_loss = float("inf")
    global_step = 0

    for step_idx in range(steps_per_epoch):
        x, y = train_ds.get_batch(step_idx, device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % 50 == 0 or global_step == 1:
            lr = scheduler.get_last_lr()[0]
            elapsed = (time.time() - start_time) / 60.0
            print(
                f"[train] step {global_step}/{steps_per_epoch} "
                f"loss={loss.item():.4f} lr={lr:.2e} elapsed={elapsed:.2f} min"
            )

        if global_step % EVAL_EVERY == 0 or global_step == steps_per_epoch:
            val_loss = evaluate_full(model, val_ds, device)
            print(f"[eval] step {global_step}  val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_best = os.path.join(OUT_DIR, f"rnn_{model_name}_best.pt")
                torch.save(
                    {
                        "model_name": model_name,
                        "model_state_dict": model.state_dict(),
                        "val_loss": float(best_val_loss),
                    },
                    ckpt_best,
                )
                print(f"[ckpt] Saved new best model to {ckpt_best}")

    end_time = time.time()
    elapsed = end_time - start_time

    final_val_loss = best_val_loss

    if device.type == "cuda":
        gpu_mem_after = torch.cuda.memory_allocated() / 1024**2
        gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024**2

    print("===========================================")
    print(f"[train] RNN model name:     {model_name}")
    print(f"[train] Parameters:         {n_params:,}")
    print(f"[train] Best val loss:      {final_val_loss:.4f}")
    print(f"[train] Steps in 1 epoch:   {steps_per_epoch}")
    print(f"[train] Total train tokens: {total_train_tokens:,}")
    print(f"[train] Wall clock seconds: {elapsed:.1f}")
    print(f"[train] Wall clock minutes: {elapsed / 60.0:.1f}")
    if device.type == "cuda":
        print(f"[train] GPU memory start:   {gpu_mem_before:.2f} MB")
        print(f"[train] GPU memory end:     {gpu_mem_after:.2f} MB")
        print(f"[train] GPU memory peak:    {gpu_mem_peak:.2f} MB")
    else:
        print("[train] GPU memory tracking unavailable (CPU).")
    print("===========================================")

    ckpt_final = os.path.join(OUT_DIR, f"rnn_{model_name}_final.pt")
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "final_val_loss": float(final_val_loss),
        },
        ckpt_final,
    )
    print(f"[ckpt] Saved final model to {ckpt_final}")

    row = dict(
        model_name=f"rnn_{model_name}",
        mode="train",
        n_params=n_params,
        steps_per_epoch=steps_per_epoch,
        total_train_tokens=total_train_tokens,
        final_val_loss=round(float(final_val_loss), 6),
        wall_clock_seconds=round(float(elapsed), 3),
        gpu_mem_start_mb=round(gpu_mem_before or 0.0, 3),
        gpu_mem_end_mb=round(gpu_mem_after or 0.0, 3),
        gpu_mem_peak_mb=round(gpu_mem_peak or 0.0, 3),
    )
    append_results_csv(row)

if __name__ == "__main__":
    main()