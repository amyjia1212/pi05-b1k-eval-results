"""Regeneratable training-progress plot for pi05_b1k_joint_90_93_96.
Reads step/loss/grad_norm straight from the tee'd training log (no wandb dependency).
Usage: .venv/bin/python plot_training_progress.py
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = Path("/workspace/openpi/train_joint_90_93_96.log")
OUT_PATH = Path("/workspace/openpi/training_progress_joint_90_93_96.png")

PATTERN = re.compile(r"Step (\d+): grad_norm=([\d.]+), loss=([\d.]+), param_norm=([\d.]+)")

steps, losses, grad_norms = [], [], []
for line in LOG_PATH.read_text(errors="ignore").splitlines():
    m = PATTERN.search(line)
    if m:
        steps.append(int(m.group(1)))
        grad_norms.append(float(m.group(2)))
        losses.append(float(m.group(3)))

if not steps:
    sys.exit(f"No 'Step N: ...' lines found in {LOG_PATH}")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ax1.plot(steps, losses, color="#2563eb", linewidth=1.5, marker="o", markersize=3)
ax1.set_ylabel("loss")
ax1.set_title(f"pi05_b1k_joint_90_93_96 training progress ({len(steps)} logged steps, latest step {steps[-1]})")
ax1.grid(alpha=0.3)

ax2.plot(steps, grad_norms, color="#dc2626", linewidth=1.5, marker="o", markersize=3)
ax2.set_ylabel("grad_norm")
ax2.set_xlabel("step")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Wrote {OUT_PATH} ({len(steps)} points, steps {steps[0]}-{steps[-1]})")
