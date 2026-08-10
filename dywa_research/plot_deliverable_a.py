#!/usr/bin/env python3
"""Deliverable A — single-arm object-size frontier (scaling law) + crossover s*."""
import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/bruce/DyWA/dywa_research/results/phase_a"
rows = []
with open(f"{ROOT}/scale_sweep.csv") as f:
    for r in csv.DictReader(f):
        if r["suc_rate"] not in ("NA", ""):
            rows.append((float(r["scale"]), float(r["suc_rate"]),
                         float(r["balanced_suc_rate"]), int(r["episode_count"])))
rows.sort()
xs   = [r[0] for r in rows]
sr   = [r[1] for r in rows]
bsr  = [r[2] for r in rows]

# crossover s*: linear-interpolate where suc_rate crosses 0.5
sstar = None
for (x0, y0, *_), (x1, y1, *_) in zip(rows, rows[1:]):
    if y0 >= 0.5 > y1:
        sstar = x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
        break

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.axvspan(0.05, 0.10, color="tab:green", alpha=0.10, label="train (in-dist)")
ax.plot(xs, sr,  "o-", lw=2, color="tab:blue",   label="suc_rate")
ax.plot(xs, bsr, "s--", lw=1.4, color="tab:orange", label="balanced_suc_rate")
ax.axhline(0.5, color="grey", ls=":", lw=1)
if sstar:
    ax.axvline(sstar, color="tab:red", ls="-.", lw=1.6)
    ax.annotate(f"s* ≈ {sstar:.3f} m", (sstar, 0.5),
                textcoords="offset points", xytext=(8, 20), color="tab:red",
                fontsize=11, fontweight="bold")
ax.set_xlabel("object scale s (m)")
ax.set_ylabel("success rate")
ax.set_ylim(-0.02, 1.02)
ax.set_title("Phase A — Single-arm non-prehensile size frontier\n"
             "Trossen teacher (validated, anchor 94.5%), held-out objects, goal_radius=0.05")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
out = f"{ROOT}/deliverable_A_scaling_law.png"
fig.savefig(out, dpi=150)
print("wrote", out)
print("s* =", sstar)
