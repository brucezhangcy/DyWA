#!/usr/bin/env python3
"""CPU-only syntax, raw-table and proposed-table geometry audit; never imports simulation code."""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd


def syntax_audit(base):
    errors=[]; files=list(base.rglob('*.py')); shells=list(base.rglob('*.sh'))
    for path in files:
        try: ast.parse(path.read_text(errors='replace'),filename=str(path))
        except SyntaxError as exc: errors.append({'path':str(path.relative_to(base)),'error':str(exc)})
    for path in shells:
        result=subprocess.run(['bash','-n',str(path)],capture_output=True,text=True)
        if result.returncode: errors.append({'path':str(path.relative_to(base)),'error':result.stderr})
    return dict(python_files=len(files),shell_files=len(shells),errors=errors)


def geometry():
    # Exactly the constants presently in sizeinv_v2.sh. Analytical support
    # certificates for a radius-s enclosing sphere, NOT simulated fall rates.
    table=np.array([.821,1.026]); spawn=.5*.95*.4/.821*table
    goal=.5*.8*.4/.821*table
    rows=[]
    for s in [.05,.10,.125,.15,.175,.20,.25,.30]:
        def support(half):
            allowed=np.maximum(table/2-s,0)
            return float(np.prod(np.minimum(allowed/half,1)))
        rows.append(dict(scale=s,spawn_support_clearance_xy_m=(table/2-spawn-s).tolist(),
            goal_support_clearance_xy_m=(table/2-goal-s).tolist(),
            fraction_uniform_spawn_centres_failing_sphere_support_certificate=1-support(spawn),
            fraction_uniform_goal_centres_failing_sphere_support_certificate=1-support(goal)))
    return dict(table_xy=table.tolist(),spawn_half_xy=spawn.tolist(),goal_half_xy=goal.tolist(),
        minimum_table_xy_for_all_radius_0p30_spawns=(2*(spawn+.30)).tolist(),
        caution='Bounding-sphere support is sufficient, not necessary; fractions are NOT measured failures. Exact shape/orientation support must be checked on GPU.',rows=rows)


def raw_tables(root):
    rows=[]
    for path in sorted((root/'results').glob('phase*/*.csv')):
        d=pd.read_csv(path); problems=[]
        keys=next((k for k in [['tag','scale'],['scale','seed'],['key','scale'],['scale']] if all(x in d for x in k)),None)
        if keys and d.duplicated(keys).any(): problems.append('duplicate rows under key '+str(keys))
        for col in ['suc_rate','balanced_suc_rate','frac_success','frac_oob','frac_timeout','frac_immediate','feasible_succ','succ']:
            if col in d:
                v=pd.to_numeric(d[col],errors='coerce')
                if ((v<0)|(v>1)|~np.isfinite(v)).any(): problems.append('invalid probability '+col)
        if 'status' in d and (d.status!='OK').any(): problems.append('non-OK rows')
        rows.append(dict(path=str(path.relative_to(root)),rows=len(d),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),problems=problems))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--core',type=Path)
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    out=root/'results/session5';out.mkdir(parents=True,exist_ok=True)
    report=dict(research_syntax=syntax_audit(root),core_syntax=syntax_audit(a.core) if a.core else None,
        raw_tables=raw_tables(root),proposed_v2_geometry=geometry(),
        scope='Static parsing, shell syntax and CPU data/geometry checks. Not a simulation or hardware validation.')
    (out/'repository_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
