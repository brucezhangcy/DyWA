#!/usr/bin/env python3
"""Generate valid_ppo_arm_diag.py from valid_ppo_arm_taxonomy.py by adding a PerEnvDiag wrapper.
Session 4: written on a CPU-only machine, NOT run yet. Intended to be behaviour-neutral: the wrapper only
reads tensors, everything is inside try/except, and the existing TAXONOMY/per-object outputs are
unchanged. It dumps $DIAG_OUT (an .npz) containing:
  static : table_dims, robot root xyz as seen in env.tensors['root'] (the tensor value, which may differ
           from the welded physical pose; see trossen.py:356-376), scene._valid_poses z vs
           scene.stable_poses[:, 0] z (valid-mode spawn z possibly un-rescaled; see CRITIQUES #13), cur_radii
  per-episode : env id, start object pose (xyz), goal xyz, init->goal xy distance, length, success, timeout
Run: python3 dywa_research/patch_diag.py   (paths are relative to this file)"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "valid_ppo_arm_taxonomy.py")
DST = os.path.join(HERE, "valid_ppo_arm_diag.py")
s = open(SRC).read()

WRAPPER = '''class PerEnvDiag(WrapperEnv):
    """Session-4 diagnostics (read-only). Records per-episode start pose / goal / outcome per env."""
    def __init__(self, env):
        super().__init__(env)
        n, dev = env.num_env, env.device
        self._sc = th.zeros(n, dtype=th.long, device=dev)
        self._start = th.full((n, 3), float('nan'), device=dev)
        self._goal = th.full((n, 3), float('nan'), device=dev)
        self.rec = {k: [] for k in ('env', 'len', 'suc', 'timeout', 'start', 'goal')}
        self.static = {}
        self._first = True

    def _obj_goal(self):
        root = self.env.tensors['root']
        obj = root[self.env.scene.cur_ids.long(), :3]
        goal = self.env.task.goal[:, :3]
        return obj, goal

    def _static(self):
        sc = self.env.scene
        def put(k, f):
            try:
                self.static[k] = dcn(f())
            except Exception as e:  # noqa
                self.static[k + '_ERR'] = str(e)
        put('table_dims', lambda: sc.table_dims[:1])
        put('table_pos', lambda: sc.table_pos[:1])
        put('cur_radii', lambda: sc.cur_radii)
        put('robot_root_tensor_xyz', lambda: self.env.tensors['root'][self.env.robot.indices.long(), :3])
        put('valid_pose_z', lambda: sc._valid_poses[..., 0, 2])
        put('stable_pose0_z_rescaled', lambda: sc.stable_poses[..., 0, 2])
        put('valid_pose_xy', lambda: sc._valid_poses[..., 0, :2])

    def step(self, *a, **k):
        if self._first:
            self._first = False
            try:
                self._static()
                o, g = self._obj_goal(); self._start[:] = o; self._goal[:] = g
            except Exception as e:
                print('DIAG_ERR(init)', e)
        obs, rew, done, info = super().step(*a, **k)
        try:
            with th.no_grad():
                self._sc += 1
                d = done.bool()
                if d.any():
                    suc = info.get('success', th.zeros_like(d)).bool()
                    to = info.get('timeout', th.zeros_like(d)).bool()
                    ids = th.nonzero(d).squeeze(-1)
                    self.rec['env'].append(dcn(ids)); self.rec['len'].append(dcn(self._sc[ids]))
                    self.rec['suc'].append(dcn(suc[ids])); self.rec['timeout'].append(dcn(to[ids]))
                    self.rec['start'].append(dcn(self._start[ids])); self.rec['goal'].append(dcn(self._goal[ids]))
                    self._sc[d] = 0
                    o, g = self._obj_goal()  # post-reset state = start of the next episode
                    self._start[d] = o[d]; self._goal[d] = g[d]
        except Exception as e:
            print('DIAG_ERR(step)', e)
        return obs, rew, done, info

    def dump(self, path):
        out = {k: np.concatenate(v) if len(v) else np.zeros(0) for k, v in self.rec.items()}
        out.update({'static_' + k: np.asarray(v) for k, v in self.static.items() if not k.endswith('_ERR')})
        errs = {k: v for k, v in self.static.items() if k.endswith('_ERR')}
        np.savez_compressed(path, **out)
        print('DIAG wrote', path, 'episodes=', len(out['env']), 'static_errors=', errs)


'''
anchor = "class TrackTerminationCause(WrapperEnv):"
assert anchor in s
s = s.replace(anchor, WRAPPER + anchor, 1)
wrap_line = "        env = TrackTerminationCause(env)\n"
assert wrap_line in s
s = s.replace(wrap_line, wrap_line + "        env = PerEnvDiag(env)\n", 1)
tax = "            print('TAXONOMY', _tc.summary())\n"
assert tax in s
s = s.replace(tax, tax + """        _dg = env.unwrap(target=PerEnvDiag)
        if isinstance(_dg, PerEnvDiag):
            _dg.dump(os.environ.get('DIAG_OUT', '/tmp/diag.npz'))
""", 1)
if "\nimport os\n" not in s:
    s = s.replace("import torch as th\n", "import torch as th\nimport os\n", 1)
open(DST, "w").write(s)
print("wrote", DST)
