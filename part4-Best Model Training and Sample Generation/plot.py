

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple




RNN_LOG_PATH = "/scratch/dk5288/code/my_project/part2/150M_Training_Log.out"
TRANSFORMER_LOG_PATH = "/scratch/dk5288/code/my_project/part3/final_model_logs.out"
OUT_PNG_PATH = "/scratch/dk5288/code/my_project/part4/scaling_plot.png"


SIZE_TO_PARAMS_MANUAL: Dict[str, int] = {}


def _unit_to_multiplier(unit: str) -> float:
    unit = (unit or "").strip().upper()
    if unit == "K":
        return 1e3
    if unit == "M":
        return 1e6
    if unit == "B":
        return 1e9
    return 1.0


def _parse_size_name(segment: str) -> Optional[str]:
    
    pats = [
        r"\bStarting training for\s+(\w+)\b",
        r"\bmodel_size\s*[:=]\s*(\w+)\b",
        r"\bsize\s*[:=]\s*(\w+)\b",
        r"\bvariant\s*[:=]\s*(\w+)\b",
        r"\brun_name\s*[:=]\s*(\w+)\b",
        
        r"^\[model\]\s+\w+\s+(\w+)\s+has\b",
    ]
    for pat in pats:
        ms = list(re.finditer(pat, segment, flags=re.IGNORECASE | re.MULTILINE))
        if ms:
            return ms[-1].group(1)
    return None


def _parse_params(segment: str) -> Optional[int]:
    patterns = [
        
        r"\bModel\s+\w+\s+has\s+([0-9]*\.?[0-9]+)\s*([KMB])?\s*parameters\b",
        
        r"^\[model\]\s+\w+\s+\w+\s+has\s+([0-9][0-9,]*)\s+trainable\s+parameters\b",
        r"^\[model\]\s+\w+\s+\w+\s+has\s+([0-9]*\.?[0-9]+)\s*([KMB])?\s+trainable\s+parameters\b",
        
        r"^\[train\]\s+Parameters:\s*([0-9][0-9,]*)\b",
        
        r"\b(trainable\s+)?params?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*([KMB])\b",
        r"\b(trainable\s+)?params?\s*[:=]\s*([0-9][0-9,]*)\b",
        r"\b(n_params|num_params|num_parameters|total_params|total_parameters)\s*[:=]\s*([0-9]*\.?[0-9]+)\s*([KMB])\b",
        r"\b(n_params|num_params|num_parameters|total_params|total_parameters)\s*[:=]\s*([0-9][0-9,]*)\b",
        r"\bTotal\s+parameters?\s*:\s*([0-9][0-9,]*)\b",
        r"\bNum\s+parameters?\s*:\s*([0-9][0-9,]*)\b",
    ]

    best: Optional[int] = None

    for pat in patterns:
        for m in re.finditer(pat, segment, flags=re.IGNORECASE | re.MULTILINE):
            gs = [g for g in m.groups() if g is not None]
            if not gs:
                continue

            
            if len(gs) >= 2 and re.fullmatch(r"[KMB]", str(gs[-1]).strip(), flags=re.IGNORECASE):
                try:
                    val = float(str(gs[-2]))
                    unit = str(gs[-1])
                    best = int(round(val * _unit_to_multiplier(unit)))
                    continue
                except Exception:
                    pass

            
            for token in reversed(gs):
                tok = str(token).replace(",", "").strip()
                if re.fullmatch(r"[0-9]+", tok):
                    try:
                        best = int(tok)
                        break
                    except Exception:
                        pass

    return best


def _parse_best_val_loss(segment: str) -> Optional[float]:
    
    m = re.search(r"^\[train\]\s+Best\s+val\s+loss:\s*([0-9]*\.?[0-9]+)\b", segment, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        return float(m.group(1))
    return None


def _parse_min_or_last_val_loss(segment: str) -> Optional[float]:
    
    vals = re.findall(r"\bval_loss\s*[:=]\s*([0-9]*\.?[0-9]+)\b", segment, flags=re.IGNORECASE)
    if not vals:
        vals = re.findall(r"\bval\s+loss\s*[:=]\s*([0-9]*\.?[0-9]+)\b", segment, flags=re.IGNORECASE)
    if not vals:
        vals = re.findall(r"\bvalidation\s+loss\s*[:=]\s*([0-9]*\.?[0-9]+)\b", segment, flags=re.IGNORECASE)
    if not vals:
        return None

    nums = [float(x) for x in vals]
    
    return min(nums)


def _split_into_runs(text: str) -> List[Tuple[int, int]]:
    
    starts = [m.start() for m in re.finditer(r"Starting training for\s+\w+", text, flags=re.IGNORECASE)]
    if starts:
        return [(starts[i], starts[i + 1] if i + 1 < len(starts) else len(text)) for i in range(len(starts))]

    
    model_starts = [m.start() for m in re.finditer(r"^\[model\].*has.*parameters", text, flags=re.IGNORECASE | re.MULTILINE)]
    if len(model_starts) >= 2:
        return [(model_starts[i], model_starts[i + 1] if i + 1 < len(model_starts) else len(text)) for i in range(len(model_starts))]

    
    return [(0, len(text))]


def parse_scaling_points(
    log_path: str,
    size_to_params_fallback: Optional[Dict[str, int]] = None,
) -> List[Dict]:
    text = Path(log_path).read_text(errors="ignore")

    segments = _split_into_runs(text)
    points: List[Dict] = []

    for i, (a, b) in enumerate(segments):
        seg = text[a:b]
        size_name = _parse_size_name(seg) or f"run{i+1}"

        n_params = _parse_params(seg)
        if n_params is None and size_to_params_fallback is not None:
            n_params = size_to_params_fallback.get(size_name)

        best_loss = _parse_best_val_loss(seg)
        if best_loss is None:
            best_loss = _parse_min_or_last_val_loss(seg)

        if n_params is None or best_loss is None:
            continue

        points.append({"size": size_name, "params": int(n_params), "val_loss_1epoch": float(best_loss)})

    if not points:
        lines = text.splitlines()
        interesting = []
        for ln in lines:
            l = ln.lower()
            if ("[model]" in l) or ("val_loss" in l) or ("best val loss" in l) or ("parameters" in l):
                interesting.append(ln.strip())
            if len(interesting) >= 40:
                break

        hint = "\n".join(interesting[:40]) if interesting else "(no obvious model, loss, or parameter lines found)"
        raise RuntimeError(
            f"Could not parse any (params, val_loss) points from: {log_path}\n"
            f"First useful-looking lines:\n{hint}\n"
            f"If this log does not print parameter counts, fill SIZE_TO_PARAMS_MANUAL at top."
        )

    points.sort(key=lambda d: d["params"])
    return points


def main() -> None:
    import matplotlib.pyplot as plt

    print(f"RNN log: {RNN_LOG_PATH}")
    print(f"Transformer log: {TRANSFORMER_LOG_PATH}")
    print(f"Output: {OUT_PNG_PATH}")

    rnn_pts = parse_scaling_points(RNN_LOG_PATH, size_to_params_fallback=SIZE_TO_PARAMS_MANUAL)

    inferred_map: Dict[str, int] = dict(SIZE_TO_PARAMS_MANUAL)
    for p in rnn_pts:
        inferred_map[p["size"]] = int(p["params"])

    tr_pts = parse_scaling_points(TRANSFORMER_LOG_PATH, size_to_params_fallback=inferred_map)

    rnn_x = [p["params"] for p in rnn_pts]
    rnn_y = [p["val_loss_1epoch"] for p in rnn_pts]

    tr_x = [p["params"] for p in tr_pts]
    tr_y = [p["val_loss_1epoch"] for p in tr_pts]

    plt.figure()
    plt.plot(rnn_x, rnn_y, marker="o", linestyle="-", label="RNN")
    plt.plot(tr_x, tr_y, marker="o", linestyle="-", label="Transformer")
    plt.xscale("log")
    plt.xlabel("Parameters (log scale)")
    plt.ylabel("Best val loss (1 epoch)")
    plt.title("Scaling curve")
    plt.legend()
    plt.tight_layout()

    Path(OUT_PNG_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG_PATH, dpi=200)
    print(f"Saved: {OUT_PNG_PATH}")

    print("\nParsed points (RNN):")
    for p in rnn_pts:
        print(f"  {p['size']}: params={p['params']}, best_val_loss={p['val_loss_1epoch']}")

    print("\nParsed points (Transformer):")
    for p in tr_pts:
        print(f"  {p['size']}: params={p['params']}, best_val_loss={p['val_loss_1epoch']}")


if __name__ == "__main__":
    main()