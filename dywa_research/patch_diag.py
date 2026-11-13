#!/usr/bin/env python3
"""Generate valid_ppo_arm_diag.py from valid_ppo_arm_taxonomy.py by adding a PerEnvDiag wrapper.
Session 4: written on a CPU-only machine, NOT run yet. Intended to be behaviour-neutral: the wrapper only
reads tensors, errors are preserved in the dump and make the launcher fail, and the existing TAXONOMY/per-object outputs are
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

# Keep the collector independently testable on CPU. Embed its class into the
# generated entry so the entry remains directly executable on the lab machine.
from pathlib import Path
collector = Path(HERE, "per_env_diag.py").read_text()
WRAPPER = collector[collector.index("class PerEnvDiag("):] + "\n\n"

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
s = s.replace("open('/tmp/per-cat-suc.pkl', 'wb')",
              "open(os.environ.get('PER_OBJECT_OUT', '/tmp/per-cat-suc.pkl'), 'wb')")
open(DST, "w").write(s)
print("wrote", DST)
