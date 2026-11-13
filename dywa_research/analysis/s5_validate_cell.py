#!/usr/bin/env python3
"""Reject incomplete/stale diagnostic cells before they enter a sweep CSV (CPU only)."""
import argparse
import ast
import re
from pathlib import Path
import numpy as np


def validate(log, status='OK'):
    if status != 'OK':
        raise ValueError('simulation process failed')
    log = Path(log)
    text = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(errors='replace'))
    if re.search(r'DIAG_ERR|TAXONOMY_ERR|Traceback \(most recent call last\)', text):
        raise ValueError('diagnostic/entry-point error in log')
    matches = re.findall(r'TAXONOMY (\{[^\n]*\})', text)
    if not matches:
        raise ValueError('missing TAXONOMY output')
    counts = ast.literal_eval(matches[-1])
    if counts['term_done'] <= 0:
        raise ValueError('zero completed episodes')
    with np.load(log.with_name('diag.npz'), allow_pickle=False) as data:
        if int(data['schema_version']) != 2:
            raise ValueError('outdated diagnostic schema')
        if data['errors'].size:
            raise ValueError('diagnostic errors: ' + str(data['errors'].tolist()))
        n = len(data['env'])
        if any(len(data[key]) != n for key in ('len', 'suc', 'timeout', 'start', 'goal')):
            raise ValueError('inconsistent diagnostic array lengths')
        if n != counts['term_done'] or int(data['suc'].sum()) != counts['term_success']:
            raise ValueError('diagnostic and taxonomy counts disagree')
        if data['start'].shape != (n, 3) or data['goal'].shape != (n, 3):
            raise ValueError('invalid pose array shape')
        if not np.isfinite(data['start']).all() or not np.isfinite(data['goal']).all():
            raise ValueError('unknown/nonfinite episode start or goal')
        if (data['len'] <= 0).any():
            raise ValueError('nonpositive episode length')
    return counts, text


def append_row(log, prefix, csv, status):
    try:
        d, text = validate(log, status)
        n, im, k = d['term_done'], d['term_immediate_le3'], d['term_success']
        # Historical feasible_succ removes ALL immediate episodes from the
        # denominator but retains immediate successes. Keep that legacy column;
        # the NPZ reporter computes the correctly filtered sensitivity metric.
        feasible = k / (n - im) if n > im else float('nan')
        balanced = re.findall(r"'valid/balanced_suc_rate': ([0-9.eE+-]+)", text)
        if not balanced:
            raise ValueError('missing balanced success metric')
        row = f"{prefix},{n},{d['frac_success']},{d['frac_oob']},{d['frac_timeout']},{d['frac_immediate']},{feasible:.4f},{balanced[-1]},OK\n"
    except (ValueError, KeyError, OSError, SyntaxError) as exc:
        with open(csv, 'a') as stream:
            stream.write(prefix + ',NA,NA,NA,NA,NA,NA,NA,FAILED\n')
        raise ValueError(str(exc)) from exc
    with open(csv, 'a') as stream:
        stream.write(row)
    print(row.strip())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log'); p.add_argument('prefix'); p.add_argument('csv')
    p.add_argument('status', nargs='?', default='OK')
    args = p.parse_args()
    try:
        append_row(args.log, args.prefix, args.csv, args.status)
    except ValueError as exc:
        p.exit(1, 'INVALID CELL: ' + str(exc) + '\n')
