#!/usr/bin/env python3
"""Write bimanual_scales_radius_grid.yaml: per-box mjlab scale factors that put each box's bounding radius
(half-diagonal, half-extents from s4_deliverable_e.py / runbook section 1) on the SAME radius grid as the
single-arm sweep, so a trained-bimanual eval can be read at matched radius. Factors outside [0.4, 1.9] are
dropped (untested asset scaling range in mjlab)."""
import math, os, yaml, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
src = yaml.safe_load(open(os.path.join(HERE, "..", "bimanual_scales.yaml")))
import ast
module = ast.parse(open(os.path.join(HERE, "s4_deliverable_e.py")).read())
assignment = next(node for node in module.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'HALF' for t in node.targets))
HALF = ast.literal_eval(assignment.value)
GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
out = {"object_names": src["object_names"], "scales": {}}
for o in src["object_names"]:
    h = next(v for k, v in HALF.items() if o.startswith(k)); r1 = math.sqrt(sum(x * x for x in h))
    out["scales"][o] = [round(r / r1, 3) for r in GRID if 0.4 <= r / r1 <= 1.9]
    print(f"{o[:40]:40s} r(1.0)={r1:.3f}  scales={out['scales'][o]}")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', default=os.path.join(HERE, "..", "bimanual_scales_radius_grid.yaml"))
args = parser.parse_args()
yaml.safe_dump(out, open(args.output, "w"), sort_keys=False)
