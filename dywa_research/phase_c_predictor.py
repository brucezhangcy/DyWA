#!/usr/bin/env python3
"""Phase-C competence predictor: 'can ONE arm do THIS object?' from object geometry.
Trains an interpretable model on per-object single-arm success (transition scales) vs geometric
features. Answers whether SHAPE (aspect/flatness) predicts failure beyond SIZE alone -- the basis of
an explicit single-vs-bimanual trigger's single-arm branch."""
import pickle, glob, os, numpy as np
ROOT = "/home/bruce/DyWA/dywa_research/results/phase_c_predictor"
MESH = "/home/bruce/DyWA/data/meta-v8/cloud"

rows = []
for pk in sorted(glob.glob(f"{ROOT}/s*/per_obj.pkl")):
    scale = float(pk.split("/s")[-1].split("/")[0])
    d = pickle.load(open(pk, "rb"))
    for key, succ in d.items():
        cp = os.path.join(MESH, key + ".npy")
        if not os.path.exists(cp):
            continue
        c = np.load(cp).astype(np.float64)
        ext = np.sort(c.max(0) - c.min(0))[::-1]           # L >= M >= S (canonical)
        L, M, S = ext
        r0 = np.linalg.norm(c - c.mean(0), axis=1).max()   # canonical bounding radius
        f = s = scale
        sc = scale / r0                                    # eval rescales so radius ~= scale
        rows.append(dict(key=key, scale=scale, succ=float(succ),
                         scaled_L=L*sc, scaled_M=M*sc, scaled_S=S*sc,
                         aspect=L/max(S,1e-9), flatness=S/max(L,1e-9),
                         elong=L/max(M,1e-9)))
import statistics as st
print(f"n=({len(rows)}) points over scales", sorted({r['scale'] for r in rows}))

feat = ["scale", "scaled_L", "scaled_M", "scaled_S", "aspect", "flatness", "elong"]
X = np.array([[r[k] for k in feat] for r in rows]); y = np.array([r["succ"] for r in rows])
yb = (y >= 0.5).astype(int)

# correlations (Pearson) of each feature with success
print("\n== feature vs success (Pearson r) ==")
for j, k in enumerate(feat):
    xj = X[:, j]
    r = np.corrcoef(xj, y)[0, 1]
    print(f"  {k:10s} r={r:+.3f}")

# interpretable models via sklearn if available
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=1000).fit(Xs, yb)
    acc = cross_val_score(LogisticRegression(max_iter=1000), Xs, yb, cv=5).mean()
    print(f"\n== Logistic regression (standardized) CV acc={acc:.3f} ==")
    for k, co in sorted(zip(feat, lr.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {k:10s} coef={co:+.3f}")
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5).fit(X, yb)
    dacc = cross_val_score(DecisionTreeClassifier(max_depth=3, min_samples_leaf=5), X, yb, cv=5).mean()
    print(f"\n== Decision tree (depth 3) CV acc={dacc:.3f} ==")
    print("  feature importances:", {k: round(float(i),3) for k,i in zip(feat, dt.feature_importances_)})
    print(export_text(dt, feature_names=feat))
    # shape-beyond-size test: does aspect/flatness add over size (scaled_L, scale)?
    base = cross_val_score(LogisticRegression(max_iter=1000),
                           StandardScaler().fit_transform(X[:, [0,1]]), yb, cv=5).mean()
    full = cross_val_score(LogisticRegression(max_iter=1000),
                           StandardScaler().fit_transform(X[:, [0,1,4,5]]), yb, cv=5).mean()
    print(f"\n== SHAPE-beyond-SIZE: size-only acc={base:.3f}  size+shape acc={full:.3f}  (delta={full-base:+.3f}) ==")
except Exception as e:
    print("sklearn unavailable, correlations only:", e)

# plot: scaled_L vs aspect colored by success
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5,5.5))
    scv = ax.scatter([r["scaled_L"] for r in rows], [r["aspect"] for r in rows],
                     c=y, cmap="RdYlGn", vmin=0, vmax=1, s=45, edgecolor="k", lw=0.3)
    ax.set_xlabel("scaled longest extent (m)  [size]")
    ax.set_ylabel("aspect ratio L/S  [shape]")
    ax.set_title("Single-arm competence per object\n(green=success, red=fail) -- trigger's single-arm branch")
    plt.colorbar(scv, label="single-arm success rate")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(f"{ROOT}/deliverable_C_competence.png", dpi=150)
    print("\nwrote", f"{ROOT}/deliverable_C_competence.png")
except Exception as e:
    print("plot skipped:", e)

# save dataset
import csv
with open(f"{ROOT}/competence_dataset.csv","w",newline="") as fp:
    w=csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote competence_dataset.csv")
