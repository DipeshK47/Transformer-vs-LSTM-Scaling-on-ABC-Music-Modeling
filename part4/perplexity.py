#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore


# ============================================================
# Paths required (edit only here)
# ============================================================
ABC_SAMPLES_DIR = Path("/scratch/dk5288/code/my_project/part4/samples")
OUT_DIR = Path("/scratch/dk5288/code/my_project/part4/sample_eval_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_JSON = Path("/scratch/dk5288/code/my_project/part4/vocab.json")
CHECKPOINT_PATH = Path("/scratch/dk5288/code/my_project/part4/checkpoints/best.pt")

# This must point to your real model builder
# Format: "package.module:function"
MODEL_BUILD_FN = "my_project.models.transformer:build_model"

# These kwargs must match what your build_model expects
# Make sure vocab_size and block_size match training
MODEL_KWARGS_JSON = r"""
{
  "vocab_size": 0,
  "block_size": 256
}
""".strip()

BLOCK_SIZE = 256
BATCH_SIZE = 16
DEVICE = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
# ============================================================


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(f"MODEL_BUILD_FN must look like package.module:function, got: {spec}")
    mod_name, fn_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Could not import callable {fn_name} from {mod_name}")
    return fn


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    obj = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("vocab.json must be a dict mapping token to id (or id to token)")

    # If keys are digits, it is likely id to token, invert it
    keys = list(obj.keys())
    if keys and all(k.isdigit() for k in keys[: min(50, len(keys))]):
        inv: Dict[str, int] = {}
        for k, v in obj.items():
            inv[str(v)] = int(k)  # type: ignore[arg-type]
        # This inversion guess is not reliable if original format is unusual
        raise ValueError("vocab.json looks like id to token. Please provide token to id mapping.")

    # token to id
    tok2id: Dict[str, int] = {}
    for k, v in obj.items():
        if not isinstance(v, int):
            raise ValueError("vocab.json values must be ints (token to id).")
        tok2id[str(k)] = int(v)
    return tok2id


def guess_char_level(tok2id: Dict[str, int]) -> bool:
    toks = list(tok2id.keys())
    if not toks:
        return True
    sample = toks[: min(200, len(toks))]
    one_char = sum(1 for t in sample if len(t) == 1)
    return (one_char / max(1, len(sample))) > 0.8


def encode_text(text: str, tok2id: Dict[str, int]) -> np.ndarray:
    # Assumes char level vocab, which is typical for ABC tokenization in small projects
    # If your project uses a different tokenizer, replace this function with your real one
    if not guess_char_level(tok2id):
        raise ValueError("vocab does not look char level. Replace encode_text with your real tokenizer.")

    unk_id = tok2id.get("<unk>", None)
    ids: List[int] = []
    for ch in text:
        if ch in tok2id:
            ids.append(tok2id[ch])
        elif unk_id is not None:
            ids.append(unk_id)
        else:
            # skip unknown char if no unk token
            continue
    return np.asarray(ids, dtype=np.int64)


def load_checkpoint_state_dict(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")  # type: ignore[attr-defined]
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        # maybe it is already a state dict
        if any(isinstance(k, str) and ("weight" in k or "bias" in k or "." in k) for k in ckpt.keys()):
            return ckpt
    raise ValueError(f"Unrecognized checkpoint format: {path}")


@torch.no_grad()  # type: ignore[misc]
def perplexity_for_token_ids(model: Any, token_ids: np.ndarray) -> float:
    model.eval()

    if token_ids.ndim != 1:
        token_ids = token_ids.reshape(-1)

    if token_ids.size < 2:
        return float("inf")

    x_all = torch.from_numpy(token_ids.astype(np.int64)).to(DEVICE)  # type: ignore[attr-defined]

    total_nll = 0.0
    total_tokens = 0

    # Score as a sequence by sliding chunks of length BLOCK_SIZE
    # We compute next token loss for each position that has a previous context inside the chunk
    stride = BLOCK_SIZE
    n = x_all.numel()
    starts = list(range(0, max(1, n - 1), stride))

    for s in starts:
        # Need at least 2 tokens to form x and y
        end = min(n, s + BLOCK_SIZE + 1)
        chunk = x_all[s:end]
        if chunk.numel() < 2:
            continue

        x = chunk[:-1].unsqueeze(0)  # [1,T]
        y = chunk[1:].unsqueeze(0)   # [1,T]

        out = None
        loss = None

        try:
            out = model(x, y)
        except TypeError:
            out = model(x)

        if isinstance(out, tuple) and len(out) >= 2:
            logits, loss = out[0], out[1]
        elif torch.is_tensor(out):
            logits, loss = out, None
        else:
            raise RuntimeError("Unsupported model forward output")

        if loss is None:
            B, T, V = logits.shape
            nll = F.cross_entropy(logits.reshape(B * T, V), y.reshape(B * T), reduction="sum")
            total_nll += float(nll.item())
            total_tokens += int(B * T)
        else:
            B, T = y.shape
            total_nll += float(loss.item()) * float(B * T)
            total_tokens += int(B * T)

    if total_tokens == 0:
        return float("inf")

    avg_nll = total_nll / total_tokens
    return float(math.exp(avg_nll))


def build_and_load_model(tok2id: Dict[str, int]) -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is not available in this environment")

    build_fn = import_callable(MODEL_BUILD_FN)
    kwargs = json.loads(MODEL_KWARGS_JSON)

    # Fill vocab_size if left as 0
    if int(kwargs.get("vocab_size", 0)) == 0:
        kwargs["vocab_size"] = len(tok2id)
    # Ensure block_size matches
    kwargs["block_size"] = int(kwargs.get("block_size", BLOCK_SIZE))

    model = build_fn(**kwargs)
    state = load_checkpoint_state_dict(CHECKPOINT_PATH)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE)
    return model


ABC_TUNE_SPLIT_RE = re.compile(r"(?m)^(?=X:\s*\d+)")
REQUIRED_HEADERS = ["X:", "T:", "K:"]


def split_abc_tunes(txt: str) -> List[str]:
    s = txt.strip()
    if not s:
        return []
    parts = [p.strip() for p in ABC_TUNE_SPLIT_RE.split(s) if p.strip()]
    if len(parts) <= 1:
        return [s]
    tunes: List[str] = []
    for p in parts:
        if not p.lstrip().startswith("X:"):
            p = "X:1\n" + p
        tunes.append(p)
    return tunes


def quick_header_check(abc: str) -> bool:
    a = abc.strip()
    for h in REQUIRED_HEADERS:
        if h not in a:
            return False
    return True


def have_abc2midi() -> bool:
    return shutil.which("abc2midi") is not None


def try_convert_with_abc2midi(abc: str, out_mid: Path) -> Tuple[bool, str]:
    if not have_abc2midi():
        return False, "abc2midi not found"

    out_mid.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        in_abc = Path(td) / "tune.abc"
        in_abc.write_text(abc, encoding="utf-8", errors="ignore")
        cmd = ["abc2midi", str(in_abc), "-o", str(out_mid)]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "").strip()
            return False, msg[:500]
        if out_mid.exists() and out_mid.stat().st_size > 0:
            return True, ""
        return False, "abc2midi wrote empty midi"


def try_convert_with_music21(abc: str, out_mid: Path) -> Tuple[bool, str]:
    try:
        from music21 import converter  # type: ignore
    except Exception as e:
        return False, f"music21 import failed: {e}"

    try:
        score = converter.parseData(abc)
        out_mid.parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=str(out_mid))
        if out_mid.exists() and out_mid.stat().st_size > 0:
            return True, ""
        return False, "music21 wrote empty midi"
    except Exception as e:
        return False, str(e)


def is_syntactically_valid_abc(abc: str) -> Tuple[bool, str]:
    if not quick_header_check(abc):
        return False, "missing headers (need X:, T:, K:)"

    # Best check: conversion to midi works
    with tempfile.TemporaryDirectory() as td:
        tmp_mid = Path(td) / "tmp.mid"
        ok, err = try_convert_with_abc2midi(abc, tmp_mid)
        if ok:
            return True, ""
        ok2, err2 = try_convert_with_music21(abc, tmp_mid)
        if ok2:
            return True, ""
        return False, (err2 or err).strip()


@dataclass
class Row:
    file: str
    tune_idx: int
    n_tokens: int
    ppl: float
    syntactic_valid: int
    midi_ok: int
    midi_path: str
    error: str


def main() -> None:
    if not VOCAB_JSON.exists():
        raise FileNotFoundError(f"Missing vocab.json at {VOCAB_JSON}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint at {CHECKPOINT_PATH}")
    if not ABC_SAMPLES_DIR.exists():
        raise FileNotFoundError(f"Missing samples dir at {ABC_SAMPLES_DIR}")

    tok2id = load_vocab(VOCAB_JSON)
    model = build_and_load_model(tok2id)

    midi_out = OUT_DIR / "midi"
    midi_out.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in ABC_SAMPLES_DIR.glob("**/*") if p.is_file()])

    rows: List[Row] = []

    total_tunes = 0
    syntactic_ok = 0
    midi_ok = 0

    total_nll_tokens = 0
    total_nll_sum = 0.0

    for fp in files:
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        tunes = split_abc_tunes(txt)
        if not tunes:
            continue

        for j, tune in enumerate(tunes):
            total_tunes += 1

            ok_syn, err_syn = is_syntactically_valid_abc(tune)
            if ok_syn:
                syntactic_ok += 1

            out_mid = midi_out / f"{fp.stem}_t{j+1}.mid"
            ok_mid = False
            err_mid = ""

            if ok_syn:
                ok_mid, err_mid = try_convert_with_abc2midi(tune, out_mid)
                if not ok_mid:
                    ok_mid, err_mid = try_convert_with_music21(tune, out_mid)
                if ok_mid:
                    midi_ok += 1

            token_ids = encode_text(tune, tok2id)
            ppl = perplexity_for_token_ids(model, token_ids) if token_ids.size >= 2 else float("inf")

            # Accumulate overall ppl in token space by summing NLL
            # We approximate overall ppl from per tune ppl by reconstructing NLL
            # avg_nll = log(ppl), total_nll = avg_nll * n_tokens
            if np.isfinite(ppl) and ppl > 0 and token_ids.size >= 2:
                avg_nll = math.log(ppl)
                total_nll_sum += avg_nll * int(token_ids.size - 1)
                total_nll_tokens += int(token_ids.size - 1)

            rows.append(
                Row(
                    file=str(fp.relative_to(ABC_SAMPLES_DIR)),
                    tune_idx=j + 1,
                    n_tokens=int(token_ids.size),
                    ppl=float(ppl),
                    syntactic_valid=1 if ok_syn else 0,
                    midi_ok=1 if ok_mid else 0,
                    midi_path=str(out_mid) if ok_mid else "",
                    error=(err_mid or err_syn)[:500],
                )
            )

    syntactic_pct = 0.0 if total_tunes == 0 else 100.0 * syntactic_ok / total_tunes
    midi_pct = 0.0 if total_tunes == 0 else 100.0 * midi_ok / total_tunes

    overall_ppl = float("inf")
    if total_nll_tokens > 0:
        overall_ppl = float(math.exp(total_nll_sum / total_nll_tokens))

    details_csv = OUT_DIR / "sample_eval_with_ppl.csv"
    with open(details_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "tune_idx", "n_tokens", "ppl", "syntactic_valid", "midi_ok", "midi_path", "error"])
        for r in rows:
            w.writerow([r.file, r.tune_idx, r.n_tokens, f"{r.ppl:.6f}", r.syntactic_valid, r.midi_ok, r.midi_path, r.error])

    summary = {
        "timestamp": now_str(),
        "samples_dir": str(ABC_SAMPLES_DIR),
        "out_dir": str(OUT_DIR),
        "checkpoint": str(CHECKPOINT_PATH),
        "vocab_json": str(VOCAB_JSON),
        "device": DEVICE,
        "total_tunes": total_tunes,
        "syntactic_valid_pct": syntactic_pct,
        "abc_to_midi_success_pct": midi_pct,
        "overall_sample_perplexity": overall_ppl,
        "details_csv": str(details_csv),
        "used_abc2midi": bool(have_abc2midi()),
    }

    (OUT_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote details: {details_csv}")
    print(f"Wrote summary: {OUT_DIR / 'eval_summary.json'}")


if __name__ == "__main__":
    main()