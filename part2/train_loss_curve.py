import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

log_path = Path("/scratch/dk5288/code/my_project/model_training_logs.out")

sizes = ["tiny", "small", "medium", "large", "xl"]

# storage
train_steps = {s: [] for s in sizes}
train_losses = {s: [] for s in sizes}
epoch_time_min = {s: None for s in sizes}
epoch_time_sec = {s: None for s in sizes}
peak_mem_gb = {s: None for s in sizes}  # not available in this log

current_size = None

# patterns
# examples:
# [train] Starting training for model_size=tiny
start_model_re1 = re.compile(
    r"\[train\]\s+Starting training for model_size=(tiny|small|medium|large|xl)"
)
# also handle the banner line:
# Starting training for tiny
start_model_re2 = re.compile(
    r"Starting training for (tiny|small|medium|large|xl)"
)

# [train] step 500/6103 train_loss=0.6452 lr=2.90e-04 elapsed=0.09 min
train_line_re = re.compile(
    r"\[train\]\s+step\s+(\d+)/\d+\s+train_loss=([0-9.]+)"
)

# [train] Done. Total training time: 0.85 minutes
time_line_re = re.compile(
    r"\[train\]\s+Done\. Total training time:\s+([0-9.]+)\s+minutes"
)

with open(log_path, "r") as f:
    for line in f:
        # detect which model size we are in
        m = start_model_re1.search(line)
        if m:
            current_size = m.group(1)
            continue

        m = start_model_re2.search(line)
        if m:
            current_size = m.group(1)
            continue

        # training loss line
        m = train_line_re.search(line)
        if m and current_size is not None:
            step = int(m.group(1))
            loss = float(m.group(2))
            train_steps[current_size].append(step)
            train_losses[current_size].append(loss)
            continue

        # epoch total time
        m = time_line_re.search(line)
        if m and current_size is not None:
            minutes = float(m.group(1))
            epoch_time_min[current_size] = minutes
            epoch_time_sec[current_size] = minutes * 60.0
            continue

# convert to numpy arrays and report counts
for s in sizes:
    steps = np.array(train_steps[s], dtype=np.int64) if train_steps[s] else np.array([])
    losses = np.array(train_losses[s], dtype=np.float32) if train_losses[s] else np.array([])
    train_steps[s] = steps
    train_losses[s] = losses
    print(f"{s}: {len(steps)} training loss points")

# 1) training loss curves
plt.figure(figsize=(8, 6))
any_points = False
for s in sizes:
    if train_steps[s].size == 0:
        continue
    any_points = True
    # sort by step just in case
    order = np.argsort(train_steps[s])
    plt.plot(train_steps[s][order], train_losses[s][order], label=s)

plt.xlabel("Training step")
plt.ylabel("Training loss")
plt.title("Training loss vs steps for each model size")
if any_points:
    plt.legend()
plt.grid(True, which="both", linestyle=":")
plt.tight_layout()
plt.savefig("training_loss_curves_all_models.png")
print("Saved training_loss_curves_all_models.png")

# 2) epoch times
print("\nWall clock time per epoch:")
for s in sizes:
    if epoch_time_min[s] is None:
        print(f"  {s:6s}: None")
    else:
        print(f"  {s:6s}: {epoch_time_min[s]:.2f} min ({epoch_time_sec[s]:.2f} sec)")

# 3) peak GPU memory (not in this log)
print("\nPeak GPU memory usage (GB):")
for s in sizes:
    print(f"  {s:6s}: {peak_mem_gb[s]}")

# save csv
with open("scaling_extra_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["model_size", "epoch_time_min", "epoch_time_sec", "peak_gpu_mem_gb"]
    )
    for s in sizes:
        writer.writerow([s, epoch_time_min[s], epoch_time_sec[s], peak_mem_gb[s]])

print("Saved scaling_extra_metrics.csv")