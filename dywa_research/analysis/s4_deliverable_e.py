#!/usr/bin/env python3
"""Session 4 (CPU-only): what the SCRIPTED bimanual lower bound does and does not show.

Inputs (committed):  results/bimanual/bimanual_results.csv (mjlab, 10 boxes x scales {0.7,1.0,1.3} x 30 ep)
                     results/phase_c_multiseed/multiseed.csv, results/phase_d_sizeinv/sizeinv.csv,
                     results/phase_a/scale_sweep.csv (Isaac Gym single arm)
Box half-extents: copied from plot_deliverable_e.py (source there: Aloha_real runbook
cat_box_subset_10_residual_push.md section 1; NOT re-verified in this session).
Outputs: results/session4/deliverable_E_*.{csv,json,png}
"""
import json, math, os
import numpy as np, pandas as pd
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(R, "session4"); os.makedirs(OUT, exist_ok=True)
HALF = {  # identical to plot_deliverable_e.py
    "Crayola_Crayons_120_crayons": (0.151, 0.091, 0.045),
    "Deskstar_Desk_Top_Hard_Drive_1_TB": (0.118, 0.151, 0.058),
    "Granimals_20_Wooden_ABC_Blocks_Wagon": (0.145, 0.149, 0.141),
    "Kotex_U_Barely_There_Liners_Thin_60_count": (0.094, 0.151, 0.081),
    "Marc_Anthony_Strictly_Curls": (0.058, 0.151, 0.039),
    "Nestle_Pure_Life_Exotics": (0.152, 0.080, 0.073),
    "Office_Depot_Dell_Series_9": (0.111, 0.151, 0.064),
    "Office_Depot_HP_74XL75": (0.153, 0.113, 0.111),
    "Pepsi_Cola_Wild_Cherry": (0.151, 0.056, 0.048),
    "Whey_Protein_Chocolate_12_Packets": (0.151, 0.149, 0.101),
}


def half(name):
    return next(v for k, v in HALF.items() if name.startswith(k))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


b = pd.read_csv(f"{R}/bimanual/bimanual_results.csv")
b["short"] = b.object.str.slice(0, 22)
b["radius"] = [math.sqrt(sum(h * h for h in half(o))) * s for o, s in zip(b.object, b.scale)]
b["min_half"] = [min(half(o)) * s for o, s in zip(b.object, b.scale)]
b[["wilson_lo", "wilson_hi"]] = [wilson(k, n) for k, n in zip(b.n_succ, b.n_tot)]

rep = dict(n_variants=len(b), episodes=int(b.n_tot.sum()), successes=int(b.n_succ.sum()),
           macro_mean=float(b.success_rate.mean()), radius_range=[float(b.radius.min()), float(b.radius.max())])
# object-cluster bootstrap of the macro mean
rng = np.random.default_rng(0); objs = b.object.unique()
bm = [pd.concat([b[b.object == o] for o in rng.choice(objs, len(objs), replace=True)]).success_rate.mean()
      for _ in range(4000)]
rep["macro_mean_object_bootstrap_95"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]


def fit_logit(x, k, n):
    x, k, n = map(np.asarray, (x, k, n))
    def nll(t):
        p = np.clip(1 / (1 + np.exp(-(t[0] + t[1] * x))), 1e-9, 1 - 1e-9)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()
    return optimize.minimize(nll, [-1.5, 0.0], method="Nelder-Mead").x


a, slope = fit_logit(b.radius, b.n_succ, b.n_tot)
sl = []
for _ in range(2000):
    bb = pd.concat([b[b.object == o] for o in rng.choice(objs, len(objs), replace=True)])
    sl.append(fit_logit(bb.radius, bb.n_succ, bb.n_tot)[1])
rep["logit_success_vs_radius"] = dict(intercept=float(a), slope_per_m=float(slope),
                                      slope_object_bootstrap_95=[float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))])
# within-object scale effect (paired, same box)
pv = b.pivot_table(index="object", columns="scale", values="success_rate")
rep["within_object_scale_effect"] = dict(
    mean_by_scale={str(k): float(v) for k, v in pv.mean().items()},
    paired_1p3_minus_0p7=dict(mean=float((pv[1.3] - pv[0.7]).mean()), n_up=int(((pv[1.3] - pv[0.7]) > 0).sum()),
                              n_down=int(((pv[1.3] - pv[0.7]) < 0).sum())))
# heterogeneity across objects: chi-square of pooled-by-object counts
po = b.groupby("object")[["n_succ", "n_tot"]].sum()
from scipy.stats import chi2_contingency
chi = chi2_contingency(np.c_[po.n_succ, po.n_tot - po.n_succ])
rep["between_object_heterogeneity"] = dict(chi2=float(chi[0]), dof=int(chi[2]), p=float(chi[1]),
                                           per_object=(po.n_succ / po.n_tot).round(3).to_dict())

# ------------- single-arm curves (Isaac Gym) at matched bounding radius (DyWA: radius == scale s)
ms = pd.read_csv(f"{R}/phase_c_multiseed/multiseed.csv")
ms["n"] = ms.term_done; ms["k"] = (ms.frac_success * ms.n).round(); ms["nn"] = ms.n - (ms.frac_immediate * ms.n).round()
raw = ms.assign(p=ms.k / ms.n).groupby("scale").p.mean()
feas = ms.assign(p=ms.k / ms.nn).groupby("scale").p.mean()
si = pd.read_csv(f"{R}/phase_d_sizeinv/sizeinv.csv").set_index("scale").frac_success
pa = pd.read_csv(f"{R}/phase_a/scale_sweep.csv").set_index("scale")
curves = {"single raw (fixed table, 3 seeds)": raw, "single feasible-only (fixed table, 3 seeds)": feas,
          "single size-inv table v1 (seed0)": si, "single object-balanced (Phase A, seed0)": pa.balanced_suc_rate}


def at(curve, r):
    xs, ys = np.array(curve.index, float), np.array(curve.values, float)
    return float(np.interp(r, xs, ys)) if r <= xs.max() else float("nan")  # no extrapolation past 0.30


cmp = b[["object", "scale", "radius", "success_rate", "wilson_lo", "wilson_hi"]].copy()
for nm, c in curves.items():
    cmp[nm] = [at(c, r) for r in cmp.radius]
cmp.to_csv(f"{OUT}/deliverable_E_matched_radius.csv", index=False)
# radius bins
bins = [0.10, 0.15, 0.20, 0.25, 0.35]
cmp["bin"] = pd.cut(cmp.radius, bins)
binned = []
for iv, g in cmp.groupby("bin", observed=True):
    sub = b.loc[g.index]; K, N = int(sub.n_succ.sum()), int(sub.n_tot.sum()); lo, hi = wilson(K, N)
    row = dict(radius_bin=str(iv), variants=len(g), bimanual_scripted=K / N, wilson_lo=lo, wilson_hi=hi)
    for nm in curves:
        row[nm] = float(np.nanmean(g[nm])) if g[nm].notna().any() else float("nan")
    binned.append(row)
binned = pd.DataFrame(binned); binned.to_csv(f"{OUT}/deliverable_E_binned.csv", index=False)
rep["binned"] = binned.round(3).to_dict("records")
# where does the fitted scripted-bimanual curve exceed each single-arm curve?
rr = np.linspace(0.10, 0.30, 401); pb = 1 / (1 + np.exp(-(a + slope * rr)))
cross = {}
for nm, c in curves.items():
    ps = np.array([at(c, r) for r in rr]); above = np.where(pb > ps)[0]
    cross[nm] = float(rr[above[0]]) if len(above) else float("nan")
rep["radius_where_scripted_bimanual_fit_exceeds_single"] = cross
json.dump(rep, open(f"{OUT}/deliverable_E_summary.json", "w"), indent=1, default=float)
print(json.dumps(rep, indent=1, default=lambda x: round(float(x), 4)))

# ------------- figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BG, INK, MUTED = "#fcfcfb", "#0b0b0b", "#52514e"
fig, ax = plt.subplots(figsize=(8.4, 5.4), facecolor=BG); ax.set_facecolor(BG)
for nm, c, col, ls in ((list(curves)[0], raw, "#2a78d6", "-"), (list(curves)[1], feas, "#2a78d6", "--"),
                       (list(curves)[2], si, "#1baf7a", "-")):
    ax.plot(c.index, c.values, ls, color=col, lw=2, marker="o", ms=5, label=nm)
ax.errorbar(b.radius, b.success_rate, yerr=[b.success_rate - b.wilson_lo, b.wilson_hi - b.success_rate],
            fmt="o", ms=5, color="#eb6834", ecolor="#eb6834", elinewidth=1, alpha=0.75, capsize=0,
            label="bimanual SCRIPTED primitive, per box x scale (Wilson 95%, n=30)")
ax.plot(rr, pb, color="#eb6834", lw=2, label=f"scripted-bimanual logistic fit (slope {slope:+.1f}/m)")
ax.axvspan(0.05, 0.10, color="#e9e8e4", alpha=0.6, lw=0); ax.text(0.052, 0.96, "single-arm\ntrain range", fontsize=7, color=MUTED, va="top")
ax.set_xlim(0.05, 0.34); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("object bounding radius (m)", color=MUTED); ax.set_ylabel("success rate", color=MUTED)
ax.set_title("Deliverable E, CROSS-SIM & SCRIPTED (lower bound): single arm (Isaac Gym) vs two arms (mjlab)\n"
             "different simulators, object sets and success criteria; NOT a fair comparison", fontsize=9.5, color=INK)
ax.legend(fontsize=7.5, frameon=False, loc="upper right"); ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/deliverable_E_v2_scripted_ci.png", dpi=150)
print("wrote", f"{OUT}/deliverable_E_v2_scripted_ci.png")
