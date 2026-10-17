#!/usr/bin/env python3
"""Write bimanual_scales_radius_grid.yaml: per-box mjlab scale factors that put each box's bounding radius
(half-diagonal, half-extents from plot_deliverable_e.py / runbook section 1) on the SAME radius grid as the
single-arm sweep, so a trained-bimanual eval can be read at matched radius. Factors outside [0.4, 1.9] are
dropped (untested asset scaling range in mjlab)."""
import math, os, yaml
HERE = os.path.dirname(os.path.abspath(__file__))
src = yaml.safe_load(open(os.path.join(HERE, "..", "bimanual_scales.yaml")))
HALF = {}
for line in open(os.path.join(HERE, "s4_deliverable_e.py")):
    if line.strip().startswith('"') and "): " not in line and line.strip().endswith("),"):
        k, v = line.strip().rstrip(",").split(": ", 1); HALF[k.strip('"')] = eval(v)
GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
out = {"object_names": src["object_names"], "scales": {}}
for o in src["object_names"]:
    h = next(v for k, v in HALF.items() if o.startswith(k)); r1 = math.sqrt(sum(x * x for x in h))
    out["scales"][o] = [round(r / r1, 3) for r in GRID if 0.4 <= r / r1 <= 1.9]
    print(f"{o[:40]:40s} r(1.0)={r1:.3f}  scales={out['scales'][o]}")
yaml.safe_dump(out, open(os.path.join(HERE, "..", "bimanual_scales_radius_grid.yaml"), "w"), sort_keys=False)
