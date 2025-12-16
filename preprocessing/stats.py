#!/usr/bin/env python

import re
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Path to your combined training log (slurm output or tee output)
LOG_FILE = "/scratch/dk5288/code/my_project/slurm-132965.out"

# Output directory for plots and summary (will be created next to log file)
OUT_DIR = Path(LOG_FILE).parent


MODEL_SIZES = ["tiny", "small", "medium", "large", "xl"]


def extract_param_counts(lines):
    """
    Parse lines like:
      [train] Model tiny has 1.01M parameters
    Returns dict: { "tiny": int_param_count, ... }
    """
    param_counts = {}
    pattern = re.compile(r"\[train\]\s+Model\s+(\w+)\s+has\s+([0-9.]+)M parameters")

    for line in lines:
        m = pattern.search(line)
        if m:
            name = m.group(1)
            millions = float(m.group(2))
            param_counts[name] = int(round(millions * 1_000_000))
    return param_counts


def extract_val_losses(lines):
    """
    Track current model by:
      [train] Starting training for model_size=tiny

    For each model, store the last seen val_loss:
      [eval] step 6102 val_loss=1.2026

    Returns dict: { "tiny": 1.2026, ... }
    """
    val_losses = {}
    current_model = None

    pattern_start = re.compile(r"\[train\]\s+Starting training for model_size=(\w+)")
    pattern_val = re.compile(r"val_loss=([0-9.]+)")

    for line in lines:
        m_start = pattern_start.search(line)
        if m_start:
            current_model = m_start.group(1)
            continue

        m_val = pattern_val.search(line)
        if m_val and current_model is not None:
            val_losses[current_model] = float(m_val.group(1))

    return val_losses


def extract_training_curves_and_times(lines):
    """
    From lines like
      [train] Starting training for model_size=tiny
      [train] step 500/6103 train_loss=0.6452 lr=2.90e-04 elapsed=0.09 min
      [train] Done. Total training time: 0.85 minutes
      [train] Peak GPU memory allocated: 0.70 GB

    Return:
      train_steps, train_losses, epoch_time_min, epoch_time_sec, peak_mem_gb
      each is a dict keyed by model size
    """
    train_steps = {s: [] for s in MODEL_SIZES}
    train_losses = {s: [] for s in MODEL_SIZES}
    epoch_time_min = {s: None for s in MODEL_SIZES}
    epoch_time_sec = {s: None for s in MODEL_SIZES}
    peak_mem_gb = {s: None for s in MODEL_SIZES}

    current_size = None

    # patterns
    start_model_re1 = re.compile(
        r"\[train\]\s+Starting training for model_size=(tiny|small|medium|large|xl)"
    )
    start_model_re2 = re.compile(
        r"Starting training for (tiny|small|medium|large|xl)"
    )
    train_line_re = re.compile(
        r"\[train\]\s+step\s+(\d+)/\d+\s+train_loss=([0-9.]+)"
    )
    time_line_re = re.compile(
        r"\[train\]\s+Done\. Total training time:\s+([0-9.]+)\s+minutes"
    )
    mem_line_re = re.compile(
        r"\[train\]\s+Peak GPU memory allocated:\s+([0-9.]+)\s+GB"
    )

    for line in lines:
        m = start_model_re1.search(line)
        if m:
            current_size = m.group(1)
            continue

        m = start_model_re2.search(line)
        if m:
            current_size = m.group(1)
            continue

        m = train_line_re.search(line)
        if m and current_size is not None:
            step = int(m.group(1))
            loss = float(m.group(2))
            train_steps[current_size].append(step)
            train_losses[current_size].append(loss)
            continue

        m = time_line_re.search(line)
        if m and current_size is not None:
            minutes = float(m.group(1))
            epoch_time_min[current_size] = minutes
            epoch_time_sec[current_size] = minutes * 60.0
            continue

        m = mem_line_re.search(line)
        if m and current_size is not None:
            gb = float(m.group(1))
            peak_mem_gb[current_size] = gb
            continue

    # convert to numpy arrays
    for s in MODEL_SIZES:
        steps = np.array(train_steps[s], dtype=np.int64) if train_steps[s] else np.array([])
        losses = np.array(train_losses[s], dtype=np.float32) if train_losses[s] else np.array([])
        train_steps[s] = steps
        train_losses[s] = losses

    return train_steps, train_losses, epoch_time_min, epoch_time_sec, peak_mem_gb


def fit_power_law_with_c(Ns, Ls):
    """
    Fit L = a * N^(-alpha) + c using a simple grid search over c and
    linear regression on log(L - c) vs log N.

    Ns, Ls are numpy arrays.
    Returns (a, alpha, c).
    """

    # small grid around the minimum loss
    L_min = float(Ls.min())
    # ensure c < min(L) so that L - c > 0
    c_candidates = np.linspace(0.0, 0.95 * L_min, 200)

    best_err = float("inf")
    best_a = None
    best_alpha = None
    best_c = None

    logN = np.log(Ns)

    for c in c_candidates:
        shifted = Ls - c
        if np.any(shifted <= 0):
            continue
        logL_shifted = np.log(shifted)
        m, b = np.polyfit(logN, logL_shifted, 1)
        alpha = -m
        a = np.exp(b)

        # compute squared error in log space
        L_pred = a * (Ns ** (-alpha)) + c
        err = np.mean((np.log(Ls) - np.log(L_pred)) ** 2)

        if err < best_err:
            best_err = err
            best_a = a
            best_alpha = alpha
            best_c = c

    return best_a, best_alpha, best_c


def plot_training_curves(train_steps, train_losses):
    plt.figure(figsize=(8, 6))
    any_points = False
    for s in MODEL_SIZES:
        steps = train_steps[s]
        losses = train_losses[s]
        if steps.size == 0:
            continue
        any_points = True
        order = np.argsort(steps)
        plt.plot(steps[order], losses[order], label=s)

    plt.xlabel("Training step")
    plt.ylabel("Training loss")
    plt.title("Training loss vs steps for each model size")
    if any_points:
        plt.legend()
    plt.grid(True, which="both", linestyle=":")
    plt.tight_layout()

    out_path = OUT_DIR / "150M_training_loss_curves_all_models.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_scaling_law(Ns, Ls, model_order, a, alpha, c):
    plt.figure(figsize=(8, 6))
    plt.xscale("log")
    plt.yscale("log")

    plt.scatter(Ns, Ls, label="Models")
    for n, l, name in zip(Ns, Ls, model_order):
        plt.text(n, l, name, fontsize=8, ha="center", va="bottom")

    N_line = np.logspace(np.log10(Ns.min()), np.log10(Ns.max()), 200)
    L_line = a * (N_line ** (-alpha)) + c
    plt.plot(N_line, L_line, "--", label=f"Fit: alpha ≈ {alpha:.3f}")

    plt.xlabel("Model size N (parameters)")
    plt.ylabel("Validation loss after 1 epoch")
    plt.title("Scaling of validation loss with model size")
    plt.legend()
    plt.tight_layout()

    out_path = OUT_DIR / "150M_scaling_law_val_loss_vs_params.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def write_summary_txt(
    model_order,
    param_counts,
    val_losses,
    epoch_time_min,
    epoch_time_sec,
    peak_mem_gb,
    a,
    alpha,
    c,
):
    """
    Create a human readable summary text file containing:
      - model size, N, val loss, epoch time, peak GPU memory
      - fitted power law parameters and a short comment
    """
    out_path = OUT_DIR / "150M_scaling_analysis_summary.txt"

    with open(out_path, "w") as f:
        f.write("Scaling analysis summary\n")
        f.write("========================\n\n")

        f.write("Per model metrics (ordered by size):\n\n")
        f.write(
            "model_size, params, val_loss_after_1_epoch, "
            "epoch_time_min, epoch_time_sec, peak_gpu_mem_gb\n"
        )
        for m in model_order:
            N = param_counts.get(m)
            L = val_losses.get(m)
            tmin = epoch_time_min.get(m)
            tsec = epoch_time_sec.get(m)
            mem = peak_mem_gb.get(m)

            f.write(
                f"{m}, {N}, {L}, {tmin}, {tsec}, {mem}\n"
            )

        f.write("\nPower law fit: L = a * N^(-alpha) + c\n")
        f.write(f"a ≈ {a:.6f}\n")
        f.write(f"alpha ≈ {alpha:.6f}\n")
        f.write(f"c ≈ {c:.6f}\n\n")

        # short implication text
        f.write("Implications:\n")
        f.write(
            "The exponent alpha controls how fast validation loss decreases as "
            "model size increases. For example, doubling the number of parameters "
            "changes the loss by roughly a factor of 2^(-alpha). A smaller alpha "
            "means diminishing returns from increasing model size, while a larger "
            "alpha would indicate stronger gains from scaling the model.\n"
        )

    print(f"Wrote summary to {out_path}")


def write_extra_metrics_csv(
    model_order, epoch_time_min, epoch_time_sec, peak_mem_gb
):
    out_path = OUT_DIR / "150M_scaling_extra_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["model_size", "epoch_time_min", "epoch_time_sec", "peak_gpu_mem_gb"]
        )
        for s in model_order:
            writer.writerow([s, epoch_time_min[s], epoch_time_sec[s], peak_mem_gb[s]])
    print(f"Saved {out_path}")


def main(log_file):
    log_file = Path(log_file)
    global OUT_DIR
    OUT_DIR = log_file.parent

    with open(log_file, "r") as f:
        lines = f.readlines()

    # parse different parts
    param_counts = extract_param_counts(lines)
    val_losses = extract_val_losses(lines)
    (
        train_steps,
        train_losses,
        epoch_time_min,
        epoch_time_sec,
        peak_mem_gb,
    ) = extract_training_curves_and_times(lines)

    print("Parameter counts found:")
    for k, v in param_counts.items():
        print(f"  {k}: {v}")

    print("\nValidation losses found:")
    for k, v in val_losses.items():
        print(f"  {k}: {v}")

    print("\nWall clock time per epoch (minutes):")
    for s in MODEL_SIZES:
        print(f"  {s:6s}: {epoch_time_min[s]}")

    print("\nPeak GPU memory usage (GB):")
    for s in MODEL_SIZES:
        print(f"  {s:6s}: {peak_mem_gb[s]}")

    # model order that actually exists
    model_order = [
        m for m in MODEL_SIZES if m in param_counts and m in val_losses
    ]
    if len(model_order) < 2:
        raise ValueError("Not enough models found in log to fit a scaling law.")

    Ns = np.array([param_counts[m] for m in model_order], dtype=float)
    Ls = np.array([val_losses[m] for m in model_order], dtype=float)

    # fit power law with c
    a, alpha, c = fit_power_law_with_c(Ns, Ls)
    print("\nPower law fit L = a * N^(-alpha) + c:")
    print(f"  a ≈ {a:.6f}")
    print(f"  alpha ≈ {alpha:.6f}")
    print(f"  c ≈ {c:.6f}")

    # plots
    plot_training_curves(train_steps, train_losses)
    plot_scaling_law(Ns, Ls, model_order, a, alpha, c)

    # summary files
    write_summary_txt(
        model_order,
        param_counts,
        val_losses,
        epoch_time_min,
        epoch_time_sec,
        peak_mem_gb,
        a,
        alpha,
        c,
    )
    write_extra_metrics_csv(model_order, epoch_time_min, epoch_time_sec, peak_mem_gb)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        LOG_FILE = sys.argv[1]
    main(LOG_FILE)