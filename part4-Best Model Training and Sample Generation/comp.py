import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv

BASE = Path("/scratch/dk5288/code/my_project/part4")

# These are "summary .txt" files (not true CSV files)
TRANS_SUMMARY = Path("/scratch/dk5288/code/my_project/part2/150M_scaling_analysis_summary.txt")
RNN_SUMMARY = Path("/scratch/dk5288/code/my_project/part3/rnn_scaling_analysis_summary.txt")

OUT_DIR = BASE / "compare"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_SIZES = ["tiny", "small", "medium", "large", "xl"]


def parse_summary_table(path: Path) -> dict:
    """
    Extracts the embedded table that starts with:
      model_size, params, val_loss_after_1_epoch, epoch_time_min, epoch_time_sec, peak_gpu_mem_gb
    Returns dict keyed by model_size.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")

    lines = path.read_text(encoding="utf-8").splitlines()

    header_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("model_size,"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"Could not find embedded table header in: {path}")

    header = [h.strip() for h in lines[header_idx].split(",")]

    rows = {}
    for j in range(header_idx + 1, len(lines)):
        line = lines[j].strip()
        if not line:
            break
        if line.lower().startswith("power law fit"):
            break
        if "," not in line:
            break

        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(header):
            break

        r = dict(zip(header, parts))
        size = r.get("model_size")
        if size:
            rows[size] = {
                "model_size": size,
                "n_params": int(float(r["params"])),
                "final_val_loss": float(r["val_loss_after_1_epoch"]),
                "wall_clock_seconds": float(r["epoch_time_sec"]),
                "wall_clock_minutes": float(r["epoch_time_min"]),
                "gpu_mem_peak_gb": float(r["peak_gpu_mem_gb"]),
            }

    if not rows:
        raise RuntimeError(f"No table rows parsed from: {path}")
    return rows


def parse_powerlaw_fit(path: Path):
    """
    Parses:
      a ≈ ...
      alpha ≈ ...
      c ≈ ...
    Returns (a, alpha, c) or (None, None, None) if not found.
    """
    txt = path.read_text(encoding="utf-8")
    m_a = re.search(r"\ba\s*≈\s*([0-9.]+)", txt)
    m_alpha = re.search(r"\balpha\s*≈\s*([0-9.]+)", txt)
    m_c = re.search(r"\bc\s*≈\s*([0-9.]+)", txt)

    a = float(m_a.group(1)) if m_a else None
    alpha = float(m_alpha.group(1)) if m_alpha else None
    c = float(m_c.group(1)) if m_c else None
    return a, alpha, c


def load_training_curve(path: Path):
    """
    Path = *.out file, extract: Step X/Y loss Z
    """
    if not path.exists():
        return [], []
    txt = path.read_text(encoding="utf-8").splitlines()
    steps, losses = [], []
    pat = re.compile(r"Step\s+(\d+)/(\d+)\s+loss\s+([0-9.]+)")
    for line in txt:
        m = pat.search(line)
        if m:
            steps.append(int(m.group(1)))
            losses.append(float(m.group(3)))
    return steps, losses


def fit_powerlaw_grid(N, L):
    """
    Simple grid fit: L = a * N^(-alpha) + c
    Returns (a, alpha, c).
    """
    alpha_grid = np.linspace(0.01, 1.0, 200)
    c_max = float(min(L)) * 0.9
    c_grid = np.linspace(0.0, c_max, 200)

    best = (None, None, None, float("inf"))
    N = N.astype(float)
    L = L.astype(float)

    for alpha in alpha_grid:
        term = N ** (-alpha)
        denom = float(np.dot(term, term))
        if denom == 0:
            continue
        for c in c_grid:
            y = L - c
            a = float(np.dot(term, y) / denom)
            pred = a * term + c
            err = float(np.mean((pred - L) ** 2))
            if err < best[3]:
                best = (a, alpha, c, err)

    return best[0], best[1], best[2]


def make_combined_scaling_plot(trans_rows, rnn_rows, trans_fit=None, rnn_fit=None):
    # Use whatever sizes exist in each file
    sizes_t = [s for s in ALL_SIZES if s in trans_rows]
    sizes_r = [s for s in ALL_SIZES if s in rnn_rows]

    N_t = np.array([trans_rows[s]["n_params"] for s in sizes_t], dtype=float)
    L_t = np.array([trans_rows[s]["final_val_loss"] for s in sizes_t], dtype=float)

    N_r = np.array([rnn_rows[s]["n_params"] for s in sizes_r], dtype=float)
    L_r = np.array([rnn_rows[s]["final_val_loss"] for s in sizes_r], dtype=float)

    # Prefer parsed fit values from summary; fallback to grid fit
    if trans_fit and all(v is not None for v in trans_fit):
        a_t, alpha_t, c_t = trans_fit
    else:
        a_t, alpha_t, c_t = fit_powerlaw_grid(N_t, L_t)

    if rnn_fit and all(v is not None for v in rnn_fit):
        a_r, alpha_r, c_r = rnn_fit
    else:
        a_r, alpha_r, c_r = fit_powerlaw_grid(N_r, L_r)

    plt.figure()
    plt.scatter(N_t, L_t, color="blue", label="Transformer")
    plt.scatter(N_r, L_r, color="red", label="RNN")

    N_min = min(float(N_t.min()), float(N_r.min()))
    N_max = max(float(N_t.max()), float(N_r.max()))
    N_fit = np.logspace(np.log10(N_min), np.log10(N_max), 200)

    plt.plot(N_fit, a_t * N_fit ** (-alpha_t) + c_t, color="blue", linestyle="-")
    plt.plot(N_fit, a_r * N_fit ** (-alpha_r) + c_r, color="red", linestyle="-")

    for s in sizes_t:
        plt.text(trans_rows[s]["n_params"], trans_rows[s]["final_val_loss"], f"T-{s}", color="blue", fontsize=8)
    for s in sizes_r:
        plt.text(rnn_rows[s]["n_params"], rnn_rows[s]["final_val_loss"], f"R-{s}", color="red", fontsize=8)

    plt.xscale("log")
    plt.xlabel("Model size N (parameters, log scale)")
    plt.ylabel("Validation loss after 1 epoch")
    plt.title("Transformer vs RNN Scaling Comparison")
    plt.legend()
    plt.tight_layout()

    out = OUT_DIR / "compare_scaling_plot.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)

    print(f"Transformer alpha ≈ {alpha_t:.6f}")
    print(f"RNN alpha ≈ {alpha_r:.6f}")


def make_pairwise_curves(size):
    trans_log = BASE / "transformer" / f"trans_{size}.out"
    rnn_log = BASE / "rnn" / f"rnn_{size}.out"

    t_steps, t_loss = load_training_curve(trans_log)
    r_steps, r_loss = load_training_curve(rnn_log)

    if not t_steps or not r_steps:
        print(f"Skipping {size}: missing curve data")
        return

    plt.figure()
    plt.plot(t_steps, t_loss, label=f"Transformer-{size}", linewidth=2)
    plt.plot(r_steps, r_loss, label=f"RNN-{size}", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title(f"Training Curve Comparison: {size}")
    plt.legend()
    plt.tight_layout()

    out = OUT_DIR / f"compare_curve_{size}.png"
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def write_compare_table(trans_rows, rnn_rows):
    """
    Writes a CSV with blanks for missing sizes (ex: RNN may not have xl).
    Note: peak_gpu_mem_gb column in your files might actually be MB for RNN
    based on the magnitudes. This function preserves the numeric values as-is.
    """
    out = OUT_DIR / "compare_table.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "size",
            "trans_params", "rnn_params",
            "trans_val_loss", "rnn_val_loss",
            "trans_time_sec", "rnn_time_sec",
            "trans_peak_gpu_mem_gb", "rnn_peak_gpu_mem_gb",
        ])

        for s in ALL_SIZES:
            t = trans_rows.get(s)
            r = rnn_rows.get(s)
            w.writerow([
                s,
                "" if t is None else t["n_params"],
                "" if r is None else r["n_params"],
                "" if t is None else t["final_val_loss"],
                "" if r is None else r["final_val_loss"],
                "" if t is None else t["wall_clock_seconds"],
                "" if r is None else r["wall_clock_seconds"],
                "" if t is None else t["gpu_mem_peak_gb"],
                "" if r is None else r["gpu_mem_peak_gb"],
            ])
    print("Saved:", out)


if __name__ == "__main__":
    trans_rows = parse_summary_table(TRANS_SUMMARY)
    rnn_rows = parse_summary_table(RNN_SUMMARY)

    trans_fit = parse_powerlaw_fit(TRANS_SUMMARY)  # (a, alpha, c)
    rnn_fit = parse_powerlaw_fit(RNN_SUMMARY)      # (a, alpha, c)

    write_compare_table(trans_rows, rnn_rows)
    make_combined_scaling_plot(trans_rows, rnn_rows, trans_fit=trans_fit, rnn_fit=rnn_fit)

    for size in ALL_SIZES:
        make_pairwise_curves(size)