# gpt_model.py
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    n_ff: int


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        # causal mask kept as a buffer so we do not recreate it every time
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        # shape (1, 1, T, T) for broadcasting over batch and heads
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.size()

        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=2)

        # reshape for multi head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)

        # scaled dot product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)

        # apply causal mask: only allow attention to current and previous positions
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, T, hd)

        # merge heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.proj(y)  # final projection
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_ff)
        self.fc2 = nn.Linear(config.n_ff, config.n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        # pre norm transformer block
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying (optional but standard)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        # progress print: total parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"[GPT] Initialized model: layers={config.n_layer}, heads={config.n_head}, "
            f"emb={config.n_embd}, block_size={config.block_size}, "
            f"params={total_params / 1e6:.2f}M"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: LongTensor of shape (B, T) with token indices
        targets: optional LongTensor of shape (B, T) for loss
        """
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block size"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)

        tok_emb = self.transformer.wte(idx)          # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)[None]    # (1, T, n_embd)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                     # (B, T, vocab_size)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )
        return logits, loss


# Convenience configs for scaling study

def get_model_config(name: str, vocab_size: int, block_size: int = 512) -> GPTConfig:
    name = name.lower()
    if name == "tiny":
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=2,
            n_head=6,
            n_embd=192,
            n_ff=4 * 192,
        )
    elif name == "small":
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=6,
            n_head=8,
            n_embd=256,
            n_ff=4 * 256,
        )
    elif name == "medium":
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=8,
            n_head=8,
            n_embd=448,
            n_ff=4 * 448,
        )
    elif name == "large":
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=10,
            n_head=8,
            n_embd=640,
            n_ff=4 * 640,
        )
    elif name == "xl":
        cfg = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=12,
            n_head=8,
            n_embd=832,
            n_ff=4 * 832,
        )
    else:
        raise ValueError(f"Unknown model name: {name}")

    print(
        f"[get_model_config] Built config '{name}' "
        f"(layers={cfg.n_layer}, heads={cfg.n_head}, emb={cfg.n_embd}, "
        f"block_size={cfg.block_size}, vocab_size={cfg.vocab_size})"
    )
    return cfg


# Optional small self test to show progress when you run this file directly
if __name__ == "__main__":
    vocab_size = 99   # set from your Part 1 stats
    block_size = 512  # or 1024 if you want

    for name in ["tiny", "small", "medium", "large", "xl"]:
        print(f"\n[main] Building model '{name}'...")
        cfg = get_model_config(name, vocab_size=vocab_size, block_size=block_size)
        model = GPT(cfg)
        print(f"[main] Done building '{name}'.\n")

    print("[main] All model configs built successfully.")