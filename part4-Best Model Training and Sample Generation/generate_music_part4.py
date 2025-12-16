import os
import json
import argparse
import random

import torch
import torch.nn.functional as F

from gpt_model import GPT, get_model_config


DATA_DIR = "/scratch/dk5288/data/abc_char_corpus_98_1_1"
TRAIN_TXT = os.path.join(DATA_DIR, "train.txt")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.json")

MODEL_DIR = "/scratch/dk5288/models/part4_best"
BEST_CKPT = os.path.join(MODEL_DIR, "best_xl.pt")
FINAL_CKPT = os.path.join(MODEL_DIR, "final_xl.pt")

OUTPUT_DIR = "/scratch/dk5288/code/my_project/part4/samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BLOCK_SIZE = 256


def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf8") as f:
        vocab = json.load(f)
    stoi = {ch: int(i) for ch, i in vocab.items()}
    itos = {int(i): ch for ch, i in vocab.items()}
    return stoi, itos


def load_model(device, stoi):
    vocab_size = len(stoi)
    cfg = get_model_config("xl", vocab_size=vocab_size, block_size=BLOCK_SIZE)
    model = GPT(cfg).to(device)

    if os.path.exists(BEST_CKPT):
        print(f"Loading checkpoint {BEST_CKPT}")
        state_dict = torch.load(BEST_CKPT, map_location=device)
    elif os.path.exists(FINAL_CKPT):
        print(f"Loading checkpoint {FINAL_CKPT}")
        state_dict = torch.load(FINAL_CKPT, map_location=device)
    else:
        raise FileNotFoundError("No best_xl.pt or final_xl.pt found in MODEL_DIR")

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.eval()
    return model


def sample_sequence(
    model,
    start_ids,
    device,
    max_new_tokens=400,
    temperature=1.0,
    top_k=5,
    top_p=None,
):
    model.eval()
    idx = torch.tensor(start_ids, dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]

        with torch.no_grad():
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :]
        logits = logits / temperature

        if top_k is not None:
            v, ix = torch.topk(logits, top_k)
            mask = torch.ones_like(logits, dtype=torch.bool)
            mask.scatter_(1, ix, False)
            logits = logits.masked_fill(mask, float("-inf"))

        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)

            cutoff = cumprobs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False

            sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
            logits = torch.zeros_like(logits).scatter_(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return idx[0].tolist()


def ids_to_text(ids, itos):
    return "".join(itos[i] for i in ids)


def load_small_train_prefix(max_chars=200_000):
    with open(TRAIN_TXT, "r", encoding="utf8") as f:
        text = f.read(max_chars)
    return text


def build_conditional_prefixes(train_text, n_prefixes, prefix_len=200):
    starts = []
    start_idx = 0
    while True:
        idx = train_text.find("X:", start_idx)
        if idx == -1:
            break
        starts.append(idx)
        start_idx = idx + 1
        if len(starts) >= n_prefixes * 3:
            break

    if not starts:
        return [train_text[:prefix_len]] * n_prefixes

    prefixes = []
    for _ in range(n_prefixes):
        s = random.choice(starts)
        snippet = train_text[s : s + prefix_len]
        prefixes.append(snippet)

    return prefixes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_uncond", type=int, default=25)
    parser.add_argument("--num_cond", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, use this seed for reproducible sampling",
    )
    args = parser.parse_args()

    if args.seed is not None:
        print(f"Using seed {args.seed} for sampling")
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    stoi, itos = load_vocab()
    model = load_model(device, stoi)

    uncond_prompts = [
        "X:1\nT:AI Composition 1\nM:4/4\nK:C\n",
        "X:2\nT:AI Jig\nM:6/8\nK:G\n",
        "X:3\nT:AI Reel\nM:4/4\nK:D\n",
        "X:4\nT:AI Waltz\nM:3/4\nK:F\n",
        "X:5\nT:AI Tune\nM:4/4\nK:Am\n",
    ]

    print("\nGenerating unconditional samples")
    for i in range(args.num_uncond):
        prompt_text = uncond_prompts[i % len(uncond_prompts)]
        start_ids = [stoi.get(ch, 0) for ch in prompt_text]

        ids = sample_sequence(
            model,
            start_ids,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=(args.top_p if args.top_p > 0 else None),
        )
        out_text = ids_to_text(ids, itos)

        abc_path = os.path.join(OUTPUT_DIR, f"uncond_{i+1}.abc")
        with open(abc_path, "w", encoding="utf8") as f:
            f.write(out_text)
        print(f"Saved unconditional ABC to {abc_path}")

    print("\nGenerating conditional samples")

    small_train = load_small_train_prefix(max_chars=200_000)
    cond_prefixes = build_conditional_prefixes(
        small_train,
        n_prefixes=args.num_cond,
        prefix_len=200,
    )

    for i in range(args.num_cond):
        prompt_text = cond_prefixes[i]
        start_ids = [stoi.get(ch, 0) for ch in prompt_text]

        ids = sample_sequence(
            model,
            start_ids,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=(args.top_p if args.top_p > 0 else None),
        )
        out_text = ids_to_text(ids, itos)

        abc_path = os.path.join(OUTPUT_DIR, f"cond_{i+1}.abc")
        with open(abc_path, "w", encoding="utf8") as f:
            f.write(out_text)
        print(f"Saved conditional ABC to {abc_path}")

    print("\nDone generating samples")


if __name__ == "__main__":
    main()