# rnn_model.py
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RNNConfig:
    vocab_size: int
    block_size: int
    emb_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float


class CharLSTM(nn.Module):
    """
    Simple LSTM language model for character sequences
    similar spirit to your GPT model but recurrent.
    """

    def __init__(self, config: RNNConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.emb_dim)

        self.lstm = nn.LSTM(
            input_size=config.emb_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.ln_f = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # optional weight tying (like GPT)
        if config.emb_dim == config.hidden_dim:
            self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters())
        print(
            f"[RNN] Initialized LSTM: layers={config.num_layers}, "
            f"emb_dim={config.emb_dim}, hidden_dim={config.hidden_dim}, "
            f"block_size={config.block_size}, params={total_params / 1e6:.2f}M"
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

    def forward(self, idx, targets=None):
        """
        idx: LongTensor (B, T) with token ids
        targets: optional LongTensor (B, T)
        """
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block size"

        x = self.token_emb(idx)             # (B, T, emb_dim)
        out, _ = self.lstm(x)               # (B, T, hidden_dim)
        out = self.ln_f(out)
        logits = self.head(out)             # (B, T, vocab_size)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )
        return logits, loss


def get_rnn_config(name: str, vocab_size: int, block_size: int = 256) -> RNNConfig:
    """
    Define RNN model sizes so parameter counts are in the same ballpark
    as your tiny/small/medium/large transformer models.
    """

    name = name.lower()

    if name == "tiny":
        cfg = RNNConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            emb_dim=192,
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
        )
    elif name == "small":
        cfg = RNNConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            emb_dim=256,
            hidden_dim=384,
            num_layers=2,
            dropout=0.1,
        )
    elif name == "medium":
        cfg = RNNConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            emb_dim=384,
            hidden_dim=512,
            num_layers=3,
            dropout=0.1,
        )
    elif name == "large":
        cfg = RNNConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            emb_dim=512,
            hidden_dim=640,
            num_layers=3,
            dropout=0.1,
        )
    else:
        raise ValueError(f"Unknown RNN model name: {name}")

    print(
        f"[get_rnn_config] Built RNN config '{name}' "
        f"(layers={cfg.num_layers}, emb_dim={cfg.emb_dim}, "
        f"hidden_dim={cfg.hidden_dim}, block_size={cfg.block_size}, "
        f"vocab_size={cfg.vocab_size})"
    )
    return cfg


if __name__ == "__main__":
    # quick self test
    vocab_size = 99
    block_size = 256
    for name in ["tiny", "small", "medium", "large"]:
        print(f"\n[main] Building LSTM model '{name}'...")
        cfg = get_rnn_config(name, vocab_size=vocab_size, block_size=block_size)
        model = CharLSTM(cfg)
        x = torch.randint(0, vocab_size, (2, block_size))
        logits, loss = model(x, x)
        print(f"[main] logits shape: {logits.shape}, loss: {loss.item():.4f}")
    print("\n[main] All RNN configs built successfully.")