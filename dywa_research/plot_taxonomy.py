#!/usr/bin/env python3
"""Deliverable B — failure-mode taxonomy vs object scale, and raw vs feasible-only frontier."""
import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/bruce/DyWA/dywa_research/results/phase_b_taxonomy"
S, succ, oob, to, imm, done = [], [], [], [], [], []
for r in csv.DictReader(open(f"{ROOT}/taxonomy.csv")):
    if r["term_done"] == "NA":
        continue
    S.append(float(r["scale"])); done.append(int(r["term_done"]))
    succ.append(float(r["frac_success"])); oob.append(float(r["frac_oob"]))
    to.append(float(r["frac_timeout"])); imm.append(float(r["frac_immediate"]))
# feasible-only success = success_count / (done - immediate)
feas = []
for i in range(len(S)):
    d = done[i]; sc = round(succ[i]*d); im = round(imm[i]*d)
    feas.append(sc/(d-im) if (d-im) > 0 else float("nan"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# left: termination-cause composition (stacked)
ax1.stackplot(S, succ, oob, to,
              labels=["success", "oob (lost object)", "timeout"],
              colors=["tab:green", "tab:red", "tab:orange"], alpha=0.85)
ax1.plot(S, imm, "k--o", lw=1.5, label="immediate (≤3 steps)")
ax1.set_xlabel("object scale s (m)"); ax1.set_ylabel("fraction of episodes")
ax1.set_title("Failure-mode composition vs scale\n(OOB dominates at collapse; immediate spikes ≥ 0.20)")
ax1.set_ylim(0, 1); ax1.legend(loc="center left"); ax1.grid(alpha=0.3)

# right: raw vs feasible-only frontier
ax2.axvspan(0.05, 0.10, color="tab:green", alpha=0.10, label="train (in-dist)")
ax2.plot(S, succ, "o-", lw=2, color="tab:blue", label="raw suc_rate")
ax2.plot(S, feas, "s--", lw=2, color="tab:purple", label="feasible-only (excl. immediate)")
ax2.axhline(0.5, color="grey", ls=":", lw=1)
ax2.set_xlabel("object scale s (m)"); ax2.set_ylabel("success rate")
ax2.set_ylim(-0.02, 1.02)
ax2.set_title("Raw vs feasible-only frontier\n(collapse survives the infeasible-spawn correction)")
ax2.legend(); ax2.grid(alpha=0.3)

fig.tight_layout()
out = f"{ROOT}/deliverable_B_taxonomy.png"
fig.savefig(out, dpi=150)
print("wrote", out)
print("feasible-only:", [f"{s}:{f:.3f}" for s, f in zip(S, feas)])
