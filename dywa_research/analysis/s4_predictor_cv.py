#!/usr/bin/env python3
"""Session 4 (CPU-only): leakage-free re-evaluation of the Deliverable-C competence predictor.

The original phase_c_predictor.py used cross_val_score(cv=5) on 192 rows = 64 objects x 3 scales:
unshuffled StratifiedKFold over rows sorted by scale, with the SAME object in train and test folds
(object-identity leakage). Here: GroupKFold by object (repeated with shuffled group assignment),
AUC / Brier / accuracy, a majority baseline, a scale-only baseline, a within-scale (scale-fixed)
shape test, and a paired permutation test of "shape adds beyond size".
Input: results/phase_c_predictor/competence_dataset.csv  Output: results/session4/predictor_*.{csv,json,png}
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.normpath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(R, "session4"); os.makedirs(OUT, exist_ok=True)
d = pd.read_csv(f"{R}/phase_c_predictor/competence_dataset.csv")
d["y"] = (d.succ >= 0.5).astype(int)
d["log_aspect"] = np.log(d.aspect)
# yaw-only goal confound: scene.py:68-70 is_thin(obb) = middle extent >= 2.5 x smallest -> such objects get
# YAW-ONLY goals (push_task.py: only ~is_yaw_only envs get a new stable-pose goal) = an easier task.
# Proxy from the cloud AABB extents we have (M/S = aspect/elong); the sim uses the OBB, so this is approximate.
d["yaw_only_proxy"] = ((d.aspect / d.elong) >= 2.5).astype(int)
groups = d.key.values

FEATS = {
    "scale_only": ["scale"],
    "size(scale+scaled_L)": ["scale", "scaled_L"],
    "size+shape(orig: +aspect,flatness)": ["scale", "scaled_L", "aspect", "flatness"],
    "scaled_S_only": ["scaled_S"],
    "scale+scaled_S": ["scale", "scaled_S"],
    "scale+yaw_only_proxy": ["scale", "yaw_only_proxy"],
    "scale+yaw_only_proxy+scaled_S": ["scale", "yaw_only_proxy", "scaled_S"],
    "all7": ["scale", "scaled_L", "scaled_M", "scaled_S", "aspect", "flatness", "elong"],
}


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def grouped_oof(feats, model_fn, rep_seed):
    """out-of-fold probabilities with 5 object-disjoint folds; group->fold assignment shuffled by seed"""
    rng = np.random.default_rng(rep_seed)
    uk = np.unique(groups); rng.shuffle(uk)
    fold_of = {k: i % 5 for i, k in enumerate(uk)}
    f = np.array([fold_of[k] for k in groups])
    X = d[feats].values; y = d.y.values; p = np.zeros(len(d))
    for i in range(5):
        tr, te = f != i, f == i
        m = model_fn().fit(X[tr], y[tr]); p[te] = m.predict_proba(X[te])[:, 1]
    return p


def metrics(p, y=None):
    y = d.y.values if y is None else y
    return dict(auc=roc_auc_score(y, p), brier=brier_score_loss(y, p), acc=accuracy_score(y, p >= 0.5))


REPS = 50
res, oof = [], {}
for name, feats in FEATS.items():
    for mname, mfn in (("logistic", logit),
                       ("tree_d3", lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0))):
        ms = [metrics(grouped_oof(feats, mfn, r)) for r in range(REPS)]
        oof[(name, mname)] = np.mean([grouped_oof(feats, mfn, r) for r in range(5)], axis=0)
        res.append(dict(features=name, model=mname,
                        **{f"{k}_mean": float(np.mean([m[k] for m in ms])) for k in ("auc", "brier", "acc")},
                        **{f"{k}_sd_over_reps": float(np.std([m[k] for m in ms])) for k in ("auc", "acc")}))
maj = d.y.mean()
res.append(dict(features="majority_class", model="-", auc_mean=0.5, brier_mean=float(((d.y - maj) ** 2).mean()),
                acc_mean=float(max(maj, 1 - maj))))
res = pd.DataFrame(res); res.to_csv(f"{OUT}/predictor_grouped_cv.csv", index=False)

# --- leaky original protocol reproduced for comparison (unshuffled stratified 5-fold over rows)
from sklearn.model_selection import cross_val_score
leaky = {n: float(cross_val_score(logit(), d[f].values, d.y.values, cv=5).mean()) for n, f in FEATS.items()}

# --- paired object-level bootstrap of AUC(size+shape) - AUC(size) on fixed OOF predictions
pa, pb = oof[("size+shape(orig: +aspect,flatness)", "logistic")], oof[("size(scale+scaled_L)", "logistic")]
pc = oof[("scale+scaled_S", "logistic")]; p0 = oof[("scale_only", "logistic")]
uk = np.unique(groups); idx_of = {k: np.where(groups == k)[0] for k in uk}
rng = np.random.default_rng(0); dl, ds = [], []
for _ in range(2000):
    ii = np.concatenate([idx_of[k] for k in rng.choice(uk, len(uk), replace=True)])
    y = d.y.values[ii]
    if y.min() == y.max():
        continue
    dl.append(roc_auc_score(y, pa[ii]) - roc_auc_score(y, pb[ii]))
    ds.append(roc_auc_score(y, pc[ii]) - roc_auc_score(y, p0[ii]))
boot = dict(shape_minus_size_auc=[float(np.mean(dl)), float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
            scaledS_minus_scaleonly_auc=[float(np.mean(ds)), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))])

# --- within-scale (scale held fixed): does shape predict the per-object success RATE? (Spearman)
within = []
for s, g in d.groupby("scale"):
    for f in ("scaled_S", "scaled_L", "aspect", "flatness", "elong"):
        rho, pv = stats.spearmanr(g[f], g.succ)
        within.append(dict(scale=s, feature=f, spearman_rho=rho, p=pv, n=len(g)))
within = pd.DataFrame(within)
# object-level: average success over the 3 scales vs (scale-free) shape
obj = d.groupby("key").agg(succ=("succ", "mean"), flatness=("flatness", "first"), aspect=("aspect", "first"),
                           elong=("elong", "first")).reset_index()
objc = {f: [float(x) for x in stats.spearmanr(obj[f], obj.succ)] for f in ("flatness", "aspect", "elong")}
# cross-scale consistency of per-object success (object identity effect, partly spawn pose; see log)
pv = d.pivot_table(index="key", columns="scale", values="succ")
xs = {f"{a}-{b}": float(stats.spearmanr(pv[a], pv[b])[0]) for a, b in ((0.125, 0.15), (0.15, 0.175), (0.125, 0.175))}
within.to_csv(f"{OUT}/predictor_within_scale.csv", index=False)
yaw = []
for (yo, s), g in d.groupby(["yaw_only_proxy", "scale"]):
    rho, pv_ = stats.spearmanr(g.scaled_S, g.succ)
    yaw.append(dict(yaw_only_proxy=yo, scale=s, n=len(g), mean_succ=g.succ.mean(), rho_scaledS=rho, p=pv_))
yaw = pd.DataFrame(yaw); yaw.to_csv(f"{OUT}/predictor_yaw_only_confound.csv", index=False)
pyo = oof[("scale+yaw_only_proxy+scaled_S", "logistic")]; pyb = oof[("scale+yaw_only_proxy", "logistic")]
dy = []
for _ in range(2000):
    ii = np.concatenate([idx_of[k] for k in rng.choice(uk, len(uk), replace=True)])
    y = d.y.values[ii]
    if y.min() != y.max():
        dy.append(roc_auc_score(y, pyo[ii]) - roc_auc_score(y, pyb[ii]))
boot["scaledS_beyond_scale+yawonly_auc"] = [float(np.mean(dy)), float(np.percentile(dy, 2.5)), float(np.percentile(dy, 97.5))]

summary = dict(n_rows=len(d), n_objects=int(d.key.nunique()), pos_rate=float(maj),
               pos_rate_by_scale=d.groupby("scale").y.mean().round(3).to_dict(),
               leaky_original_protocol_acc=leaky, grouped_cv=res.round(4).to_dict("records"),
               paired_object_bootstrap_auc_delta=boot,
               yaw_only_confound=yaw.round(4).to_dict("records"),
               object_level_spearman_mean_succ_vs_shape=objc, cross_scale_rank_consistency=xs)
json.dump(summary, open(f"{OUT}/predictor_summary.json", "w"), indent=1, default=float)

pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False)); print("\nleaky original protocol acc:", {k: round(v, 3) for k, v in leaky.items()})
print("\npaired bootstrap AUC deltas [mean, 2.5, 97.5]:", {k: [round(x, 3) for x in v] for k, v in boot.items()})
print("\nwithin-scale Spearman:\n", within.round(3).to_string(index=False))
print("\nobject-level (mean over scales) Spearman [rho,p]:", {k: [round(x, 3) for x in v] for k, v in objc.items()})
print("\nyaw-only confound:\n", yaw.round(3).to_string(index=False))
print("cross-scale rank consistency:", {k: round(v, 3) for k, v in xs.items()})

# --- figure: grouped-CV ROC for scale-only vs scale+scaled_S vs size+shape (logistic)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(5.6, 5.0), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
for (nm, key), col in ((("scale only", ("scale_only", "logistic")), "#8a8985"),
                       (("scale + shortest extent", ("scale+scaled_S", "logistic")), "#2a78d6"),
                       (("size + aspect/flatness (orig.)", ("size+shape(orig: +aspect,flatness)", "logistic")), "#eb6834")):
    fpr, tpr, _ = roc_curve(d.y, oof[key])
    auc = res[(res.features == key[0]) & (res.model == "logistic")].auc_mean.iloc[0]
    ax.plot(fpr, tpr, lw=2, color=col, label=f"{nm}  (AUC {auc:.2f})")
ax.plot([0, 1], [0, 1], ls=":", lw=1, color="#b5b4ae")
ax.set_xlabel("false-positive rate (predicted single-arm OK, actually failed)", color="#52514e")
ax.set_ylabel("true-positive rate", color="#52514e")
ax.set_title("Deliverable C, object-disjoint 5-fold CV (logistic)\n64 objects x 3 scales, fixed-table, seed 0", fontsize=10)
ax.legend(loc="lower right", fontsize=8, frameon=False); ax.grid(alpha=0.25)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/deliverable_C_grouped_cv_roc.png", dpi=150)
print("wrote", f"{OUT}/deliverable_C_grouped_cv_roc.png")
