import csv, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R="/home/bruce/DyWA/dywa_research/results"
def load(p,sc,su):
    d={}
    for r in csv.DictReader(open(p)):
        if r.get(su,'NA')=='NA': continue
        d[float(r[sc])]=r
    return d
si=load(f"{R}/phase_d_sizeinv/sizeinv.csv","scale","frac_success")
# fixed = multiseed mean
import statistics as st; from collections import defaultdict
fx=defaultdict(list)
for r in csv.DictReader(open(f"{R}/phase_c_multiseed/multiseed.csv")):
    if r["frac_success"]!="NA": fx[float(r["scale"])].append(float(r["frac_success"]))
S=sorted(si)
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
ax1.axvspan(0.05,0.10,color="tab:green",alpha=0.10,label="train (in-dist)")
ax1.plot(S,[float(si[s]["frac_success"]) for s in S],"s-",lw=2,color="tab:purple",label="size-invariant table (constant room)")
fS=sorted(fx); ax1.plot(fS,[st.mean(fx[s]) for s in fS],"o--",lw=2,color="tab:blue",label="fixed table (raw)")
ax1.axhline(0.5,color="grey",ls=":"); ax1.set_xlabel("object scale s (m)"); ax1.set_ylabel("success rate")
ax1.set_ylim(-0.02,1.02); ax1.set_title("Frontier SURVIVES removing the table artifact\n(collapse is a real control limit, not placement)")
ax1.legend(); ax1.grid(alpha=0.3)
# failure-mode flip on size-invariant table
ax2.stackplot(S,[float(si[s]["frac_success"]) for s in S],[float(si[s]["frac_oob"]) for s in S],
              [float(si[s]["frac_timeout"]) for s in S],
              labels=["success","oob (lost object)","timeout (can't reach goal)"],
              colors=["tab:green","tab:red","tab:orange"],alpha=0.85)
ax2.set_xlabel("object scale s (m)"); ax2.set_ylabel("fraction of episodes"); ax2.set_ylim(0,1)
ax2.set_title("With room: failure is TIMEOUT, not OOB\n=> single-contact controllability limit (H2)")
ax2.legend(loc="center left"); ax2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{R}/phase_d_sizeinv/deliverable_D_control_limit.png",dpi=150); print("wrote plot")
