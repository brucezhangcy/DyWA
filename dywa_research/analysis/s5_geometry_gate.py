#!/usr/bin/env python3
"""CPU-only conservative support check. Does not change any experiment setting."""
import argparse
import sys


def check(table, radius, scene_margin):
    half_spawn = [0.5 * scene_margin * d for d in table[:2]]
    return [0.5 * d - h - radius for d, h in zip(table[:2],half_spawn)]


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--radius',type=float,required=True)
    p.add_argument('--table',type=float,nargs=2,default=[.821,1.026])
    p.add_argument('--scene-margin',type=float,default=.95*.4/.821)
    args=p.parse_args()
    if args.radius<=0 or not 0<args.scene_margin<=1 or min(args.table)<=0:
        p.error('radius/table must be positive and scene margin in (0,1]')
    clearance=check(args.table,args.radius,args.scene_margin)
    print('bounding-sphere support clearance x/y (m):',clearance)
    if min(clearance)<-1e-6:
        sys.exit('BLOCKED: proposed table does not guarantee support for every spawn at this radius. Review exact shape/orientation support or revise geometry before approving this experiment. No configuration was changed.')
