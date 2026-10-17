#!/usr/bin/env python3
"""Session 4 (CPU-only) re-analysis of the single-arm frontier CSVs.

Reads ONLY committed CSVs under dywa_research/results/ and writes dywa_research/results/session4/.
No simulation is run. Answers:
  1. How much of the between-seed variance is binomial vs over-dispersed?
  2. Are the s=0.20 "immediate" terminations an ADDITIVE artifact (a few looping envs) rather than
     a change in the behaviour of the other envs?  (test: non-immediate episode counts across seeds)
  3. s* with uncertainty: per-seed linear interpolation + a binomial-logistic fit, raw vs feasible.
  4. Are the Phase-B single-knob deltas at s=0.20 larger than seed noise?
Run:  python3 dywa_research/analysis/s4_frontier_stats.py
"""
import json, math, os
import numpy as np, pandas as pd
from scipy import stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(R, "session4"); os.makedirs(OUT, exist_ok=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def interp_cross(xs, ys, level=0.5):
    """first downward crossing of `level` by piecewise-linear interpolation"""
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if y0 >= level > y1:
            return x0 + (x1 - x0) * (y0 - level) / (y0 - y1)
    return float("nan")


def logistic_fit(s, k, n):
    """binomial MLE for p(s) = top / (1 + exp((s - s50)/w)); top fixed by the data's plateau ~ max.
    Returns s_half (where p = 0.5) and w."""
    s, k, n = map(np.asarray, (s, k, n))
    def nll(th):
        s50, logw, logit_top = th
        w = math.exp(logw); top = 1 / (1 + math.exp(-logit_top))
        p = np.clip(top / (1 + np.exp((s - s50) / w)), 1e-9, 1 - 1e-9)
        return -(k * np.log(p) + (n - k) * np.log(1 - p)).sum()
    best = min((optimize.minimize(nll, x0, method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-6, fatol=1e-8))
                for x0 in ([0.17, math.log(0.02), 3.0], [0.15, math.log(0.04), 2.0], [0.19, math.log(0.01), 4.0])),
               key=lambda r: r.fun)
    s50, logw, lt = best.x; w = math.exp(logw); top = 1 / (1 + math.exp(-lt))
    s_half = s50 + w * math.log(2 * top - 1) if top > 0.5 else float("nan")  # p(s_half)=0.5
    return dict(s50=s50, w=w, top=top, s_half=s_half)


rep = {}

# ---------------------------------------------------------------- 1+2. multi-seed frontier
ms = pd.read_csv(f"{R}/phase_c_multiseed/multiseed.csv")
ms["n"] = ms.term_done.astype(int)
ms["k"] = (ms.frac_success * ms.n).round().astype(int)
ms["n_imm"] = (ms.frac_immediate * ms.n).round().astype(int)
ms["n_nonimm"] = ms.n - ms.n_imm
ms["feas"] = ms.k / ms.n_nonimm
rows = []
for s, g in ms.groupby("scale"):
    for col, kcol, ncol in (("raw", "k", "n"), ("feasible", "k", "n_nonimm")):
        p = (g[kcol] / g[ncol]).values
        mean, sd = p.mean(), p.std(ddof=1)
        tci = stats.t.ppf(0.975, len(p) - 1) * sd / math.sqrt(len(p))
        binom_var = np.mean(p * (1 - p) / g[ncol].values)
        K, N = int(g[kcol].sum()), int(g[ncol].sum())
        lo, hi = wilson(K, N)
        rows.append(dict(scale=s, metric=col, seeds=len(p), mean=mean, sd_between_seeds=sd,
                         t95_lo=mean - tci, t95_hi=mean + tci, pooled_k=K, pooled_n=N,
                         pooled_wilson_lo=lo, pooled_wilson_hi=hi,
                         overdispersion=(sd ** 2 / binom_var) if binom_var > 0 else float("nan")))
fs = pd.DataFrame(rows); fs.to_csv(f"{OUT}/frontier_seed_stats.csv", index=False)
rep["frontier_seed_stats"] = fs.round(4).to_dict("records")

loop = ms[["scale", "seed", "n", "n_imm", "n_nonimm", "k", "feas"]]
loop.to_csv(f"{OUT}/immediate_loop_check.csv", index=False)
rep["immediate_loop_check"] = loop.round(4).to_dict("records")

# ---------------------------------------------------------------- 3. s*
scales = sorted(ms.scale.unique())
sstar = {}
for metric, ncol in (("raw", "n"), ("feasible", "n_nonimm")):
    per_seed = []
    for seed, g in ms.groupby("seed"):
        g = g.sort_values("scale")
        per_seed.append(interp_cross(list(g.scale), list(g.k / g[ncol])))
    mean_curve = [float((ms[ms.scale == s].k / ms[ms.scale == s][ncol]).mean()) for s in scales]
    fit = logistic_fit(ms.scale.values, ms.k.values, ms[ncol].values)
    # parametric bootstrap of the logistic s_half, resampling SEEDS (cluster) within each scale
    rng = np.random.default_rng(0); boots = []
    for _ in range(400):
        idx = []
        for s in scales:
            sub = ms[ms.scale == s].index.values
            idx.extend(rng.choice(sub, size=len(sub), replace=True))
        b = ms.loc[idx]
        boots.append(logistic_fit(b.scale.values, b.k.values, b[ncol].values)["s_half"])
    boots = np.array([b for b in boots if np.isfinite(b)])
    sstar[metric] = dict(interp_seed_mean_curve=interp_cross(scales, mean_curve),
                         interp_per_seed=per_seed,
                         interp_per_seed_mean=float(np.mean(per_seed)),
                         interp_per_seed_sd=float(np.std(per_seed, ddof=1)),
                         logistic=fit,
                         logistic_seed_bootstrap_95=[float(np.percentile(boots, 2.5)),
                                                     float(np.percentile(boots, 97.5))])

# Phase-A single-seed (7 scales) and size-invariant table (single seed): binomial only
pa = pd.read_csv(f"{R}/phase_a/scale_sweep.csv")
sstar["phaseA_seed0_raw"] = dict(interp=interp_cross(list(pa.scale), list(pa.suc_rate)),
                                 interp_balanced=interp_cross(list(pa.scale), list(pa.balanced_suc_rate)),
                                 logistic=logistic_fit(pa.scale.values, pa.success_count.values,
                                                       pa.episode_count.values))
si = pd.read_csv(f"{R}/phase_d_sizeinv/sizeinv.csv")
si["k"] = (si.frac_success * si.term_done).round().astype(int)
rng = np.random.default_rng(1); bs = []
for _ in range(2000):
    kk = rng.binomial(si.term_done.values, si.k.values / si.term_done.values)
    bs.append(interp_cross(list(si.scale), list(kk / si.term_done.values)))
bs = np.array([b for b in bs if np.isfinite(b)])
sstar["sizeinv_seed0"] = dict(interp=interp_cross(list(si.scale), list(si.frac_success)),
                              binomial_bootstrap_95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                              note="binomial-only CI: understates uncertainty (single seed; see overdispersion)")
rep["s_star"] = sstar

# ---------------------------------------------------------------- 4. Phase-B deltas vs seed noise @ s=0.20
s20 = fs[(fs.scale == 0.20) & (fs.metric == "raw")].iloc[0]
seed_sd = float(s20.sd_between_seeds)
seed_vals = (ms[ms.scale == 0.20].k / ms[ms.scale == 0.20].n).tolist()
bat = []
mech = pd.read_csv(f"{R}/phase_b_mechanism/mechanism_s020.csv")
thr = pd.read_csv(f"{R}/phase_b_threshold/threshold.csv")
base = float(mech[mech.tag == "baseline"].suc_rate.iloc[0])
for _, r in mech.iterrows():
    if r.tag in ("baseline", "goal_0.10"):  # goal_0.10 was a no-op (force_rad clobber; progress log §13)
        continue
    bat.append(dict(source="mechanism", tag=r.tag, suc_rate=r.suc_rate, delta_vs_seed0=r.suc_rate - base,
                    delta_in_seed_sd=(r.suc_rate - base) / seed_sd,
                    inside_3seed_range=min(seed_vals) - 1e-9 <= r.suc_rate <= max(seed_vals) + 1e-9))
for _, r in thr[thr.scale == 0.20].iterrows():
    bat.append(dict(source="threshold", tag=r.tag, suc_rate=r.suc_rate, delta_vs_seed0=r.suc_rate - base,
                    delta_in_seed_sd=(r.suc_rate - base) / seed_sd,
                    inside_3seed_range=min(seed_vals) - 1e-9 <= r.suc_rate <= max(seed_vals) + 1e-9))
bat = pd.DataFrame(bat); bat.to_csv(f"{OUT}/phaseB_vs_seed_noise.csv", index=False)
rep["phaseB_vs_seed_noise"] = dict(seed_sd_raw_s020=seed_sd, seed_values=seed_vals,
                                   rows=bat.round(4).to_dict("records"))

json.dump(rep, open(f"{OUT}/frontier_stats.json", "w"), indent=1, default=float)
pd.set_option("display.width", 200)
print(fs.round(4).to_string(index=False)); print()
print(loop.round(3).to_string(index=False)); print()
print(json.dumps(sstar, indent=1, default=lambda x: round(float(x), 4))); print()
print(bat.round(3).to_string(index=False))
