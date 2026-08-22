#!/usr/bin/env python3
"""Deliverable E (cross-sim, illustrative): scripted BIMANUAL push success vs object bounding-radius,
overlaid on the single-arm frontier. NOTE: scripted primitive only (no trained residual policy) => a
LOWER BOUND on two-arm capability, not a fair 'two arms win' number."""
import csv, math, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/home/bruce/DyWA/dywa_research/results"

# box half-extents (m) at scale 1.0, from runbook cat_box_subset_10_residual_push.md §1
HALF={
 "Crayola_Crayons_120_crayons":(0.151,0.091,0.045),
 "Deskstar_Desk_Top_Hard_Drive_1_TB":(0.118,0.151,0.058),
 "Granimals_20_Wooden_ABC_Blocks_Wagon":(0.145,0.149,0.141),
 "Kotex_U_Barely_There_Liners_Thin_60_count":(0.094,0.151,0.081),
 "Marc_Anthony_Strictly_Curls":(0.058,0.151,0.039),
 "Nestle_Pure_Life_Exotics":(0.152,0.080,0.073),
 "Office_Depot_Dell_Series_9":(0.111,0.151,0.064),
 "Office_Depot_HP_74XL75":(0.153,0.113,0.111),
 "Pepsi_Cola_Wild_Cherry":(0.151,0.056,0.048),
 "Whey_Protein_Chocolate_12_Packets":(0.151,0.149,0.101),
}
def radius(name, scale):
    for k,(hx,hy,hz) in HALF.items():
        if name.startswith(k): return math.sqrt(hx*hx+hy*hy+hz*hz)*scale
    return None

bx,by=[],[]
for r in csv.DictReader(open(f"{R}/bimanual/bimanual_results.csv")):
    rad=radius(r["object"], float(r["scale"]))
    if rad: bx.append(rad); by.append(float(r["success_rate"]))

# single-arm frontier (multiseed mean+-std) — DyWA scales objects so s == bounding radius
import statistics as st; from collections import defaultdict
fx=defaultdict(list)
for r in csv.DictReader(open(f"{R}/phase_c_multiseed/multiseed.csv")):
    if r["frac_success"]!="NA": fx[float(r["scale"])].append(float(r["frac_success"]))
S=sorted(fx); mean=[st.mean(fx[s]) for s in S]; sd=[st.pstdev(fx[s]) for s in S]

fig,ax=plt.subplots(figsize=(8,5.5))
ax.errorbar(S,mean,yerr=sd,fmt="o-",lw=2,capsize=4,color="tab:blue",label="single-arm (DyWA, Isaac Gym)")
ax.scatter(bx,by,s=55,color="tab:red",edgecolor="k",lw=.4,alpha=.8,
           label="bimanual SCRIPTED primitive (mjlab) — lower bound")
ax.axhline(0.5,color="grey",ls=":",lw=1)
ax.set_xlabel("object bounding radius (m)"); ax.set_ylabel("success rate"); ax.set_ylim(-0.02,1.02)
ax.set_title("Deliverable E (CROSS-SIM, ILLUSTRATIVE): single vs double arm\n"
             "Bimanual = SCRIPTED primitive only (no trained residual policy) => LOWER BOUND")
ax.legend(loc="upper right"); ax.grid(alpha=0.3); fig.tight_layout()
out=f"{R}/bimanual/deliverable_E_single_vs_double_SCRIPTED.png"; fig.savefig(out,dpi=150)
print("wrote",out); print("bimanual scripted: n=",len(by),"mean=",round(sum(by)/len(by),3),
      "radius range",round(min(bx),3),"-",round(max(bx),3))
