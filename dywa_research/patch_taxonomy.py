#!/usr/bin/env python3
"""Generate valid_ppo_arm_taxonomy.py from the validated valid_ppo_arm_patched.py by inserting a
termination-cause wrapper. Behavior-neutral (only adds tracking + a TAXONOMY print)."""
import io, sys
SRC = "/home/bruce/DyWA/dywa_research/valid_ppo_arm_patched.py"
DST = "/home/bruce/DyWA/dywa_research/valid_ppo_arm_taxonomy.py"
s = open(SRC).read()

WRAPPER = '''class TrackTerminationCause(WrapperEnv):
    """Accumulate per-episode termination causes: success / oob / timeout / immediate(<=3 steps),
    plus mean episode length. oob is derived: done & ~success & ~timeout (push_task sets
    info['success'], info['timeout']; done = suc|oob|timeout)."""
    def __init__(self, env):
        super().__init__(env)
        n, dev = env.num_env, env.device
        self._sc = th.zeros(n, dtype=th.long, device=dev)
        self.c = {k: th.zeros((), dtype=th.long, device=dev)
                  for k in ('done', 'success', 'timeout', 'oob', 'immediate', 'eplen')}

    def step(self, *a, **k):
        obs, rew, done, info = super().step(*a, **k)
        with th.no_grad():
            self._sc += 1
            d = done.bool()
            suc = info.get('success', None)
            suc = suc.bool() if suc is not None else th.zeros_like(d)
            to = info.get('timeout', None)
            to = to.bool() if to is not None else th.zeros_like(d)
            oob = d & (~suc) & (~to)
            self.c['done'] += d.sum()
            self.c['success'] += (d & suc).sum()
            self.c['timeout'] += (d & to).sum()
            self.c['oob'] += oob.sum()
            self.c['immediate'] += (d & (self._sc <= 3)).sum()
            self.c['eplen'] += self._sc[d].sum()
            self._sc[d] = 0
        return obs, rew, done, info

    def summary(self):
        nd = int(self.c['done']) or 1
        return {'term_done': int(self.c['done']),
                'term_success': int(self.c['success']),
                'term_oob': int(self.c['oob']),
                'term_timeout': int(self.c['timeout']),
                'term_immediate_le3': int(self.c['immediate']),
                'term_mean_eplen': round(float(self.c['eplen']) / nd, 2),
                'frac_success': round(float(self.c['success']) / nd, 4),
                'frac_oob': round(float(self.c['oob']) / nd, 4),
                'frac_timeout': round(float(self.c['timeout']) / nd, 4),
                'frac_immediate': round(float(self.c['immediate']) / nd, 4)}


'''

PRINT = '''    try:
        _tc = env.unwrap(target=TrackTerminationCause)
        if isinstance(_tc, TrackTerminationCause):
            print('TAXONOMY', _tc.summary())
    except Exception as _e:
        print('TAXONOMY_ERR', _e)
'''

a = "@dataclass\nclass Config(TrainConfig):"
assert s.count(a) == 1, "anchor1"
s = s.replace(a, WRAPPER + a, 1)

b = "        env = TrackPerObjectSucRate(env)\n"
assert s.count(b) == 1, "anchor2"
s = s.replace(b, b + "        env = TrackTerminationCause(env)\n", 1)

c = "    # == RETURN ==\n"
assert s.count(c) == 1, "anchor3"
s = s.replace(c, c + PRINT, 1)

open(DST, "w").write(s)
print("wrote", DST)
