"""Read-only episode diagnostics for DyWA's delayed-reset environment.

Loaded by patch_diag.py. No scene, action, reward or training configuration is changed.
A reset hook observes assigned poses after reset_indexed returns, before physics.
"""
import numpy as np
import torch as th
from env.env.wrap.base import WrapperEnv


class PerEnvDiag(WrapperEnv):
    def __init__(self, env):
        super().__init__(env)
        # reset_indexed must be observed on the actual environment: wrappers delegate
        # reset(), and EnvBase invokes self.reset_indexed internally on the next step.
        base = env
        while isinstance(base, WrapperEnv):
            base = base.env
        self._base = base
        n, dev = env.num_env, env.device
        self._sc = th.zeros(n, dtype=th.long, device=dev)
        self._start = th.full((n, 3), float('nan'), device=dev)
        self._goal = th.full((n, 3), float('nan'), device=dev)
        self._known = th.zeros(n, dtype=th.bool, device=dev)
        self.rec = {k: [] for k in ('env', 'len', 'suc', 'timeout', 'start', 'goal')}
        self.static = {}
        self.errors = []
        self._reset_original = base.reset_indexed
        self._reset_hook = self._observe_reset
        base.reset_indexed = self._reset_hook

    @staticmethod
    def _array(x):
        if isinstance(x, th.Tensor):
            return x.detach().cpu().numpy().copy()
        return np.array(x, copy=True)

    def _obj_goal(self):
        base = self._base
        return (base.tensors['root'][base.scene.cur_ids.long(), :3],
                base.task.goal[:, :3])

    def _observe_reset(self, indices=None, *args, **kwargs):
        out = self._reset_original(indices, *args, **kwargs)
        try:
            with th.no_grad():
                ids = (th.arange(self.num_env, device=self.device) if indices is None
                       else th.as_tensor(indices, device=self.device, dtype=th.long))
                if ids.numel():
                    obj, goal = self._obj_goal()
                    self._start[ids] = obj[ids]
                    self._goal[ids] = goal[ids]
                    self._sc[ids] = 0
                    self._known[ids] = True
        except Exception as exc:
            self.errors.append('reset: ' + repr(exc))
        return out

    def _static(self):
        sc = self._base.scene
        def put(key, fn):
            try:
                self.static[key] = self._array(fn())
            except Exception as exc:
                self.errors.append(key + ': ' + repr(exc))
        put('table_dims', lambda: sc.table_dims[:1])
        put('table_pos', lambda: sc.table_pos[:1])
        put('cur_radii', lambda: sc.cur_radii)
        put('robot_root_tensor_xyz', lambda: self._base.tensors['root'][self._base.robot.indices.long(), :3])
        # Pose tables are asset-indexed, episode env IDs are env-indexed. Export the
        # mapping rather than silently indexing asset arrays using env IDs.
        put('env_asset_index', lambda: th.arange(self.num_env, device=self.device) * sc.cfg.num_obj_per_env + sc._per_env_offsets % sc.cfg.num_obj_per_env)
        put('valid_pose_z', lambda: sc._valid_poses[:, 0, 2])
        put('stable_pose0_z_rescaled', lambda: sc.stable_poses[:, 0, 2])
        put('valid_pose_xy', lambda: sc._valid_poses[:, 0, :2])

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        try:
            with th.no_grad():
                self._sc += 1
                ids = th.nonzero(done.bool(), as_tuple=False).flatten()
                if ids.numel():
                    if not self._known[ids].all():
                        self.errors.append('episode ended without an observed reset')
                    suc = info.get('success', th.zeros_like(done)).bool()
                    timeout = info.get('timeout', th.zeros_like(done)).bool()
                    for key, value in (('env', ids), ('len', self._sc[ids]),
                                       ('suc', suc[ids]), ('timeout', timeout[ids]),
                                       ('start', self._start[ids]), ('goal', self._goal[ids])):
                        self.rec[key].append(self._array(value))
                    # The returned root is TERMINAL. Next start is recorded only
                    # when reset_indexed runs at the beginning of the next step.
                    self._known[ids] = False
        except Exception as exc:
            self.errors.append('step: ' + repr(exc))
        return obs, rew, done, info

    def dump(self, path):
        self._static()
        out = {key: np.concatenate(value) if value else np.empty((0, 3) if key in ('start', 'goal') else (0,))
               for key, value in self.rec.items()}
        out.update({'static_' + key: value for key, value in self.static.items()})
        out['schema_version'] = np.asarray(2)
        out['errors'] = np.asarray(self.errors, dtype=str)
        out['pending_env'] = self._array(th.nonzero(self._known, as_tuple=False).flatten())
        out['pending_len'] = self._array(self._sc[self._known])
        np.savez_compressed(path, **out)
        print('DIAG wrote', path, 'episodes=', len(out['env']), 'errors=', self.errors)
        if self.errors:
            raise RuntimeError('Incomplete diagnostic; inspect errors in ' + str(path))

    def close_diagnostics(self):
        if self._base.reset_indexed is self._reset_hook:
            self._base.reset_indexed = self._reset_original
