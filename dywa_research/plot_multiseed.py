import csv, statistics as st, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
ROOT="/home/bruce/DyWA/dywa_research/results/phase_c_multiseed"
raw=defaultdict(list); fe=defaultdict(list)
for r in csv.DictReader(open(f"{ROOT}/multiseed.csv")):
    if r["frac_success"]=="NA": continue
    raw[float(r["scale"])].append(float(r["frac_success"])); fe[float(r["scale"])].append(float(r["feasible_succ"]))
S=sorted(raw); m=[st.mean(raw[s]) for s in S]; sd=[st.pstdev(raw[s]) for s in S]
fm=[st.mean(fe[s]) for s in S]
fig,ax=plt.subplots(figsize=(7.5,5))
ax.axvspan(0.05,0.10,color="tab:green",alpha=0.10,label="train (in-dist)")
ax.errorbar(S,m,yerr=sd,fmt="o-",lw=2,capsize=4,color="tab:blue",label="raw suc_rate (mean±std, 3 seeds)")
ax.plot(S,fm,"s--",lw=1.6,color="tab:purple",label="feasible-only (mean)")
ax.axhline(0.5,color="grey",ls=":",lw=1)
ax.axvspan(0.17,0.20,color="tab:red",alpha=0.08)
ax.annotate("fixed-table placement\nregion collapses (s≈0.19)",(0.19,0.55),color="tab:red",fontsize=9,ha="center")
ax.set_xlabel("object scale s (m)"); ax.set_ylabel("success rate"); ax.set_ylim(-0.02,1.02)
ax.set_title("Single-arm size frontier — multi-seed\n(transition s≈0.20 is high-variance; coincides with fixed-table limit)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f"{ROOT}/deliverable_A_v2_multiseed.png",dpi=150); print("wrote plot")
