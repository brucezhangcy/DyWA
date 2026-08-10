#!/usr/bin/env python3
"""Aggregate Phase-A scale-sweep results into a scaling law (Deliverable A).

Walks the per-cell output dirs produced by phase_a_scale_sweep.sh, extracts
per-cell success rate and termination-cause breakdown, writes scale_sweep.csv,
plots success-vs-scale (one curve per goal-radius regime), and locates the
crossover / collapse scale s*.

STATUS: the exact on-disk schema written by CountCategoricalSuccess.save() must
be confirmed on the first real run (GPU was occupied at build time). The reader
below tries the known locations and falls back to parsing run.log; the parse
points are marked CONFIRM-ON-DRYRUN. Kept dependency-light (stdlib + optional
matplotlib) so it runs inside pkm1:v0.
"""
import argparse
import csv
import json
import re
from pathlib import Path

# CountCategoricalSuccess.save() writes under output/test_rma/{name}/result/{set}/
# (dev_log.md:1391). We search each cell's out_dir for result artifacts.
RESULT_GLOBS = ["**/result/**/*.json", "**/result/**/*.pkl", "**/*success*.json"]

# CONFIRM-ON-DRYRUN: regex fallbacks against run.log if no structured result file.
RE_SUCC = re.compile(r"(?:cur_)?suc(?:cess)?_rate[^0-9]*([0-9]*\.?[0-9]+)", re.I)
RE_OOB = re.compile(r"\boob[^0-9]*([0-9]*\.?[0-9]+)", re.I)
RE_TIMEOUT = re.compile(r"timeout[^0-9]*([0-9]*\.?[0-9]+)", re.I)


def read_cell(cell_dir: Path):
    """Return dict(success, oob, timeout, source) for one sweep cell, or None."""
    # 1) structured result file (preferred)
    for pat in RESULT_GLOBS:
        for f in cell_dir.glob(pat):
            if f.suffix == ".json":
                try:
                    data = json.loads(f.read_text())
                except Exception:
                    continue
                succ = _dig(data, ("success", "suc_rate", "cur_suc_rate", "success_rate"))
                if succ is not None:
                    return dict(success=succ,
                                oob=_dig(data, ("oob", "oob_rate")),
                                timeout=_dig(data, ("timeout", "timeout_rate")),
                                source=str(f.relative_to(cell_dir)))
    # 2) fallback: last matching numbers in run.log
    log = cell_dir / "run.log"
    if log.exists():
        text = log.read_text(errors="ignore")
        succ = _last(RE_SUCC, text)
        if succ is not None:
            return dict(success=succ, oob=_last(RE_OOB, text),
                        timeout=_last(RE_TIMEOUT, text), source="run.log")
    return None


def _dig(d, keys):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and isinstance(d[k], (int, float)):
            return float(d[k])
    # one level deep
    for v in d.values():
        if isinstance(v, dict):
            r = _dig(v, keys)
            if r is not None:
                return r
    return None


def _last(rx, text):
    m = rx.findall(text)
    return float(m[-1]) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="OUT_ROOT of the sweep")
    ap.add_argument("--collapse-thresh", type=float, default=0.5,
                    help="success below this = 'collapsed' when locating s*")
    args = ap.parse_args()
    root = Path(args.root)

    manifest = root / "manifest.csv"
    rows = []
    if manifest.exists():
        with open(manifest) as fp:
            cells = list(csv.DictReader(fp))
    else:  # discover dirs directly
        cells = [dict(scale=d.name, regime="?", goal_radius="?",
                      run_name=d.name, out_dir=str(d), status="?")
                 for d in sorted(root.iterdir()) if d.is_dir()]

    for c in cells:
        cell_dir = Path(c["out_dir"])
        res = read_cell(cell_dir) or {}
        rows.append({**c,
                     "success": res.get("success"),
                     "oob": res.get("oob"),
                     "timeout": res.get("timeout"),
                     "result_source": res.get("source", "MISSING")})

    out_csv = root / "scale_sweep.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else
                           ["scale", "regime", "success"])
        w.writeheader()
        w.writerows(rows)
    print(f"[agg] wrote {out_csv} ({len(rows)} cells)")

    # locate s* per regime: first scale where success drops below threshold
    by_regime = {}
    for r in rows:
        if r.get("success") is None:
            continue
        by_regime.setdefault(r.get("regime", "?"), []).append(
            (float(r["scale"]), float(r["success"])))
    for regime, pts in by_regime.items():
        pts.sort()
        s_star = next((s for s, v in pts if v < args.collapse_thresh), None)
        print(f"[agg] regime={regime}: s* (success<{args.collapse_thresh}) = {s_star}"
              f"  curve={['%.3f:%.2f' % (s, v) for s, v in pts]}")

    # optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for regime, pts in sorted(by_regime.items()):
            pts.sort()
            ax.plot([s for s, _ in pts], [v for _, v in pts],
                    marker="o", label=regime)
        ax.set_xlabel("object scale s (m)")
        ax.set_ylabel("success rate")
        ax.set_title("Phase A — single-arm size frontier")
        ax.axvspan(0.05, 0.10, alpha=0.1, label="train (in-dist)")
        ax.legend()
        fig.tight_layout()
        out_png = root / "scaling_law.png"
        fig.savefig(out_png, dpi=150)
        print(f"[agg] wrote {out_png}")
    except Exception as e:
        print(f"[agg] plot skipped ({e}); CSV is the source of truth")


if __name__ == "__main__":
    main()
