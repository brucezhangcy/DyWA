# Progress Report — DyWA

**Date:** 2026-05-18
**Project:** DyWA — Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation
**Working directory:** `/data/bruce`

---

## 0. DyWA Paper Reading

**Paper:** *DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile
Manipulation* — Lyu et al., arXiv [2503.16806](https://arxiv.org/abs/2503.16806) (2025).
Authors are PKU EPIC + Galbot, the same group that did CORN.

### Problem

**Non-prehensile manipulation** — pushing / nudging an object on a table to a target 6D
pose without grasping it. Hard because contact is intermittent and the dynamics depend
on physics (mass, friction, inertia) you don't observe directly.

Generalization target: unseen object shapes, unknown physical parameters, and (in the
hardest setting) only a **single-view** depth observation. Headline numbers on
DexGraspNet (Table 1, success rate %):

| Observation | Seen | Unseen |
|---|---|---|
| Known state, 3-view | 87.9 | 85.0 |
| Unknown state, 3-view | 85.8 | 82.3 |
| Unknown state, **1-view** | 82.2 | **75.0** |

The bottom-right cell — 1-view, unknown state, unseen objects — is the regime a real
robot faces (one depth camera, no privileged mass/friction, novel object). Baselines on
that row: HACMan 2.9%, CORN 29.8%, CORN-PN++ 49.4%. DyWA's 75.0% is the contribution.

### Background concepts the paper composes

The architecture isn't novel from scratch — it composes five existing ideas in a way
that's new for non-prehensile manipulation:

- **Teacher-student distillation** — train a teacher with privileged sim info (object
  pose, physics), then distill into a student that only sees realistic observations.
- **RMA (Rapid Motor Adaptation)** — Kumar et al. 2021. A policy that doesn't see
  dynamics parameters can still *infer* them from a short history of (obs, action)
  pairs. Originally for quadruped locomotion; DyWA inherits the idea for manipulation.
- **FiLM (Feature-wise Linear Modulation)** — Perez et al. 2018. Instead of
  concatenating a conditioning vector `z` into features `x`, do `x' = γ(z) * x + β(z)`.
  Multiplicative gating is a sharper inductive bias than concatenation when "context
  changes behavior" (here: dynamics conditions action).
- **DAgger** — Ross et al. 2011. Student drives the env on-policy, teacher labels each
  visited state. Fixes the distribution-shift problem in naive imitation learning.
- **PPO** — Standard model-free RL for the teacher (not novel; included for completeness).

### Pipeline

Two paper-described stages plus one code-only extra:

| Stage | Algorithm | Envs | Steps | Lives in |
|---|---|---|---|---|
| 1 — Teacher PPO | PPO with privileged state | 4,096 | 200K | Paper + code |
| 1.5 — Phase-2 fine-tune | PPO at low LR (2e-6) | 2,048 | ~50–100k | Code only |
| 2 — Student DAgger | DAgger distillation | 1,024 | 500K | Paper + code |

Phase 1.5 produces a "smoother, more deterministic" teacher that's easier to distill —
not described in the paper, only in the repo.

### The novel student architecture

The five ingredients that make DyWA different from CORN:

1. **Simplified PointNet++ vision encoder** — two grouping layers (64 keypoints, then
   16), producing 16 patch tokens × 128-D. Per-paper this is the student's actual
   visual backbone, trained from scratch.
2. **5-step temporal history conv** — buffer the last 5 (proprio, action) tuples,
   aggregate with Conv1d + MaxPool to produce a dynamics embedding `zₜ`. This is the
   RMA idea compressed into the main encoder, no separate adapter network.
3. **FiLM-conditioned decoder** — `zₜ` modulates the action MLP via 3 FiLM blocks in
   the early layers (γ, β scale-and-shift per feature). Final layers unconditioned.
4. **One-step world-model auxiliary head** — predict next task state `Sₜ₊₁ =
   (translation, rotation)` from the encoded feature. L2 on translation, L1 on the
   flattened 9-D rotation matrix.
5. **Adaptation-latent regression** — supervise `zₜ` to match the teacher's
   concatenated geometry + physics features: `‖zₜ − concat(f^Geo, f^Phy)‖²`. This is
   the RMA-style adapter loss.

Total loss: `L_imitation + L_world + L_adapt`, equally weighted per the paper.

### Why the name parses

- **Dynamics-adaptive** → history conv produces `zₜ`, FiLM lets the policy condition
  on it. Online dynamics inference + dynamics-conditioned action head.
- **World** → one-step prediction of the next task state. The encoder is forced to
  retain enough physical-state info to forecast where the object ends up.
- **Action Model** → the FiLM-modulated MLP that outputs the 20-D Cartesian-impedance
  action (6D Δpose + 7D positional gains + 7D damping factors — no gripper, since
  non-prehensile means the gripper stays closed).

Contribution over **plain CORN**: the FiLM modulation + history conv + world-model
head + adaptation latent. Contribution over **plain RMA**: replacing the standalone
adapter with the temporal-conv + FiLM mechanism, plus adding the one-step world model.

### Two flags worth raising

Two places where the YAML config diverges from the paper's described loss structure
and would be worth confirming in code before relying on them:

- The paper writes the adaptation loss as L2 regression `‖zₜ − concat(f^Geo, f^Phy)‖²`,
  but the config uses `loss_type: contrastive` with `margin: 1`. Either the code uses a
  contrastive variant of the same idea, or the config name is legacy.
- The world-model head's config name is `vision_pose_predictor`, which reads like
  "predicts current pose." Per the paper it predicts **next** pose. Worth checking the
  training target the loss is computed against.

Full study notes are in [DyWA/dywa_paper_notes.md](DyWA/dywa_paper_notes.md).

---

## TL;DR

- Full DyWA pipeline reproduced on the new 3× RTX 6000 Ada lab box. Pretrained student
  eval: **71.40%** (paper Table 1, Unknown-State 1-view, unseen objects: 75.0%).
- Added a **Trossen WidowX AI** 6-DOF arm option alongside the original 7-DOF Franka.
  End-to-end Stage 1 PPO smoke test (50k steps, `num_env=1024`) passes — loss converges,
  no NaNs, geometry verified.
- Discovered a previously unnoticed **placement bug** affecting both Franka and Trossen:
  `fix_base_link=True` actors are welded at world origin in PhysX, while the reset-time
  body-tensor write is silently ignored. With the symmetric fix applied to Franka, the
  same pretrained checkpoint scores **78.79%** (+7.4 pp over the bugged baseline).
- Ready to launch Trossen Stage 1 production-scale training; Stage 2 / DAgger / student
  eval for Trossen are not yet validated.

---

## 1. Reproducing the Paper Baseline

### Why Docker, not conda — the machine-change problem

On the old machine I had a conda env (`corn-deploy2`) with about a dozen patches to
the source to remap `/input/DGN/` and similar hardcoded paths to my home directory. On
the new machine with CUDA 13.0 host drivers, the codebase's pinned `flash-attn==1.0.4`
and `torch 1.11+cu113` would have needed even more patching to compile against the
host toolchain. The README's Docker image bundles CUDA 11.3 plus the pinned wheels and
exposes the codebase-expected paths (`/input/DGN`, `/home/user/DyWA`) inside the
container, so the dozen prior patches all become unnecessary. **Docker isolates the
OS-level state that conda can't**, which is exactly the source of friction when moving
between machines.

The trade-off accepted: ~6 GB image to build, a Dockerfile to maintain, and a one-time
`docker` group permission from the sysadmin. In return: zero source patches needed for
path/version mismatches, and the setup is reproducible if the lab box changes again.

### Docker setup on the new box

Three Dockerfile patches were needed during the build:

1. **`bgithub.xyz` mirror unreachable.** The original Dockerfile pulls `nvdiffrast`,
   `mvp`, and `pytorch-cosine-annealing` from `bgithub.xyz` (a Chinese GitHub mirror)
   which doesn't resolve from this US lab box. Patched all three to `github.com`.
2. **`nvdiffrast` needs `--no-build-isolation`.** Otherwise pip creates a build env
   that lacks PyTorch, and `setup.py` exits before compiling the CUDA extension.
3. **`line_profiler` / `cachetools` / `nvtx` need the same flag.** Their
   `pyproject.toml` files use the old `license = {text=...}` form, which
   `setuptools ≥ 77` rejects. Forcing `--no-build-isolation` uses the pinned
   `setuptools 59.5.0` which doesn't enforce SPDX.

Other setup outcomes:

| Step | Status |
|---|---|
| Sysadmin added `yunshuan` to `docker` group | ✅ unblocked |
| Image build (`pkm1:v0`, 6.13 GB content / 19.8 GB on disk) | ✅ after the 3 patches above |
| Container `dywa_1` launched with mounts from `docker/run.sh` | ✅ |
| `setup.sh` inside container (Isaac Gym copy, package install, Eigen 3.4.0 fetch) | ✅ — CUDA extensions `franka_kin_cuda.so` and `ur5_kin_cuda.so` built fine on sm_89 via PTX |
| Pretrained student checkpoints from HuggingFace | ✅ |

**Mount layout** (`docker/run.sh`):

| Host path | Container mount | Purpose |
|---|---|---|
| `/data/bruce/DyWA/isaacgym` | `/opt/isaacgym` | Isaac Gym Preview 4 |
| `/data/bruce/DyWA/.cache/dywa` | `/home/user/.cache/pkm` | Pretrained encoder cache |
| `/data/bruce/DyWA/datasets` | `/input` | DGN dataset |
| `/data/bruce/DyWA/tmp/docker` | `/tmp/docker` | Scratch |
| `/data/bruce/DyWA` | `/home/user/DyWA` | The repo itself |

This is what makes the codebase's hardcoded paths (`/input/DGN/`, `/home/user/DyWA`)
"just work" inside the container.

### Hardware / runtime

| Aspect | Value |
|---|---|
| GPUs | 3× RTX 6000 Ada (49 GB each, sm_89) |
| Driver / host CUDA | 580.76.05 / CUDA 13.0 |
| Container | `pkm1:v0` (Py 3.8, torch 1.11+cu113, flash-attn 1.0.4, pytorch3d 0.7.2) |
| Datasets | DGN (26,738 URDFs, 3.4 GB) at `/data/bruce/DyWA/datasets/DGN/` |
| Isaac Gym | Preview 4 at `/data/bruce/DyWA/isaacgym/` |

**Pretrained student eval** (`Steve3zz/Dywa_abs_1view`, 60 envs × 3000 test_steps,
1-view, unseen objects):

| Source | Success rate |
|---|---|
| Paper Table 1 (Unknown-State, 1-view, unseen) | 75.0% |
| Our reproduction on new machine | **71.40%** |

Difference vs the paper is within the run-to-run noise observed during reproduction
(71.40% vs an earlier 71.84% rerun). Pipeline is healthy end-to-end.

The path to this number was not free: nine eval attempts surfaced real bugs — committed
local-machine paths (`/home/brucezhang/...`), a stale teacher checkpoint pointing at a
non-existent Stage 2 dir, the `nvdiffrast` install silently producing an `UNKNOWN-0.0.0`
wheel against setuptools 59.5.0, the NumPy 1.24 `np.float` removal, and missing HF stat
files. All resolved; fixes committed to GitHub `main`.

---

## 2. Trossen WidowX AI Integration

The lab's downstream sim-to-real plan calls for a 6-DOF Trossen WXAI arm, so I added it
alongside the existing 7-DOF Franka. Only the **environment layer** changed; policy
network, training scripts, and task code are untouched.

| Aspect | Franka | Trossen WXAI |
|---|---|---|
| Arm DOF | 7 | 6 |
| Gripper | Parallel jaw, position-driven | Prismatic carriage, position-driven |
| Control | Cartesian Impedance via OSC (effort mode) | Cartesian position + numerical IK (position mode) |
| Action space | 6D Δpose (+ optional gains) | 6D Δpose (Δpos + Δori axis-angle) |

**Files added:** new robot class `dywa/src/env/robot/trossen.py` (FK / geometric
Jacobian / damped-LS IK in pure PyTorch — no CUDA extension), URDF + mesh assets, parallel
Hydra configs (`trossen_icra_base.yaml`, `trossen_icra_ours_abs_rel.yaml`), and a
headless sanity-check script `test_trossen_env.py`.

**Stage 1 PPO smoke test (2026-05-16):** 200 outer steps with `num_env=128`,
`base_height='table'`. Action / observation shapes match the modified configs
(`ctx_dim=36`, `dim_out=2084`). No NaN losses, no shape mismatches. Per-iteration
throughput ≈ 2.5–3.0 it/s on a single RTX 6000 Ada.

**Stage 1 50k-step run (2026-05-17):** Full pipeline validation at `num_env=1024`,
~7 h 11 min wall-clock. Loss converged from 5.09 → 0.024; episode return trended up
(-0.452 → -0.226); no NaN. Success rate at this scale is ~0.07% — expected, since the
paper trains at `num_env=4096` for 200–500k steps. The point of this run was pipeline
validation, not a learned policy.

**Throughput caveat:** Trossen is ~16× slower per-step than Franka because the 30-iter
damped-LS numerical IK runs in pure PyTorch, where Franka uses OSC (a single matrix
inverse). A production Stage 1 run on this hardware will take ~3–4 days.

---

## 3. Placement Bug Discovery (Both Arms)

This was the largest finding of the period. Triggered by Trossen visuals; turned out
to affect Franka identically and shifted the pretrained eval by +7.4 pp once fixed.

### 3.1 Initial trigger

While debugging Trossen visuals, the rendered frame showed the arm "embedded" inside
the table block, even though the body-tensor read-back claimed the base was at
`(−0.5, 0, 0.4)` (correct — 30 cm behind the table back edge, at table-top height).

### 3.2 Mechanism — `fix_base_link` is what makes this a bug

The bug only occurs on **`fix_base_link=True`** actors. That flag *is* the mechanism.

When you call `gym.create_actor(env, asset, transform, ...)`:

- **`fix_base_link=True`** → PhysX **welds** the actor's base rigidly at whatever
  pose was in the `transform` argument. The base never moves after that. Forever.
- **`fix_base_link=False`** → the actor is free-floating. Its root pose can be moved
  at runtime by writing into `env.tensors['root']` and calling
  `set_actor_root_state_tensor_indexed`.

Both Franka and Trossen are mounted with `fix_base_link=True` (they're robot arms
bolted to a workbench — they don't fly around in the air). The codebase's logic was:

1. At `create_actor` time, pass identity `gymapi.Transform()` → PhysX welds the base
   at world origin `(0, 0, 0)`.
2. At `reset()` time, compute the desired pose `(−0.5, 0, 0.4)` and write it into
   `env.tensors['root']`, then call `set_actor_root_state_tensor_indexed`.
3. Assume step 2 moves the base.

Step 2 **silently doesn't work** because the base is welded. PhysX ignores the root
tensor write for welded actors. The body-tensor read-back still echoes the written
value — it reads from the same buffer that was just written — which is why the bug
stayed invisible: the code "saw" the correct number, but PhysX had the actor stuck
at origin.

**The fix:** bake the desired pose into the `gymapi.Transform()` passed at
`create_actor` time, so PhysX welds at the *correct* pose from the start. After that,
the reset-time tensor write is still a no-op, but it doesn't matter because the
welded pose is already right.

If the arms were `fix_base_link=False`, the original reset-time write would have
worked fine — no bug. The welding is what creates the asymmetry between "what the
code wrote" and "what PhysX has."

### 3.3 Evidence

Five experiments, in the order they were run:

**Experiment 1 — Body-tensor read-back vs rendered frame for Trossen.**
Called `gym.find_actor_rigid_body_index` and printed world positions of every link
after reset, then rendered the same frame from a raw Isaac Gym GPU camera with
`base_height='table'`. Read-back claimed `base_link: (−0.500, 0, 0.400)`,
`link_4: (−0.323, 0, 0.865)`, etc. — all visibly above the table top per the numbers.
Rendered frame showed the arm at world origin, embedded inside the table.
**Conclusion:** the body tensor disagrees with what the renderer actually sees in
PhysX. One of them is lying.

**Experiment 2 — `base_height='0.80'` debug override on Trossen.**
Bumped the base 40 cm above the table top via a numeric `base_height` string. The
read-back updated to `base_link: (0.800, ...)` — the tensor moved. The rendered frame
didn't budge — arm still at world origin.
**Conclusion:** the reset-time write into `env.tensors['root']` is a no-op for the
welded pose. PhysX has the arm fixed wherever the create-time `gymapi.Transform()`
put it (identity → origin).

**Aside — what I did *not* do as a fix:** the obvious "easy" fix would be to flip
`fix_base_link` from `True` to `False`, which would make the reset-time root-tensor
write actually work (no weld to block it). I deliberately rejected this, because the
arm in the real world **is bolted to a workbench** — `fix_base_link=True` is the
correct semantics. Setting it to `False` would make the base a free-floating rigid
body that falls under gravity unless I continuously apply pose-holding forces, and
that would wobble from numerical noise and from contact reactions transmitted up
the kinematic chain. So instead of changing the flag, I kept the arm welded and
moved the placement to the only step PhysX actually honors for welded actors —
the `gymapi.Transform()` passed at `create_actor` time (see Experiment 3). The
reset-time no-op write is *still* a no-op after the fix; I just route around it
rather than try to make it work.

**Experiment 3 — Create-time `gymapi.Transform()` fix for Trossen.**
Patched [`trossen.create_actors`](DyWA/dywa/src/env/robot/trossen.py) to bake the
`base_height`-driven placement into the `gymapi.Transform()` passed to
`gym.create_actor`. Re-ran the same headless render. Read-back numbers unchanged
(still `(−0.500, 0, 0.400)`); rendered frame now shows the orange arm sitting behind
the white table at table-top height, reaching forward, casting a shadow on the floor.
**Conclusion:** PhysX honors the create-time pose but not the reset-time write for
`fix_base_link=True` actors. The fix is at create time; the reset-time write stays as
a defensive fallback. A polished A/B recording for Trossen is in Exp 3b below.

**Experiment 3b — Side-view A/B for Trossen (companion to the Franka comparison).**
Same approach as Exp 4b below but for Trossen — toggling
[`test_trossen_env.py`](DyWA/dywa/exp/train/test_trossen_env.py) via the new
`--base_height` CLI flag: `--base_height=origin` reproduces the pre-fix bug (arm
welded at world origin), `--base_height=table` is the production fix (arm at
`(−0.5, 0, 0.4)`). Both use the same script and same scene seed; the bug-vs-fix
toggle is a one-line config change, no `trossen.py` modification needed.

| Pass | Clip | Still |
|---|---|---|
| Bugged | [trossen_sideview_bugged_h264.mp4](DyWA/dywa/output/trossen_sideview_bugged_h264.mp4) | [PNG](DyWA/dywa/output/trossen_sideview_bugged.png) — arm almost entirely **inside** the white table block, only a small orange sliver visible because Trossen's shorter ~50 cm reach keeps most of the body at z<0.4 |
| Fixed | [trossen_sideview_fixed_h264.mp4](DyWA/dywa/output/trossen_sideview_fixed_h264.mp4) | [PNG](DyWA/dywa/output/trossen_sideview_fixed.png) — arm clearly behind the table at table-top height, casting separate shadow |
| Side-by-side | [trossen_sideview_compare.mp4](DyWA/dywa/output/trossen_sideview_compare.mp4) | [PNG](DyWA/dywa/output/trossen_sideview_compare.png) — labeled `BUGGED` / `FIXED` |

The Trossen comparison is even more dramatic than Franka's: in the bugged frame, the
arm is **almost completely occluded** by the table mesh, because Trossen's shorter
reach means very little of the body pokes above z=0.4. This matches the
visibility-vs-reach analysis in §3.4.

**Experiment 4 — Symmetry check on Franka (does the same bug exist?).**
Fixed [`test_franka_env.py`](DyWA/dywa/exp/train/test_franka_env.py) (was segfaulting
from missing collision meshes — Franka URDF references `meshes/collision/*.obj` but
asset dir only has `.stl`; converted via trimesh). Then rendered Franka from a raw
Isaac Gym GPU camera at `eye=(0.001, 0, 3), at=(0, 0, 0)` (straight-down) with
`init_type='home'`, `base_height='table'`. Read-back:
`panda_link0: (−0.500, 0, 0.400)` — same value Trossen had. Rendered frame:
**arm visually embedded at table center**, not at the back-edge offset the body tensor
claims. Saved as
[`franka_phys_buried_evidence.png`](DyWA/dywa/output/franka_phys_buried_evidence.png).
**Conclusion:** Franka has the identical `fix_base_link`-weld bug. The DyWA paper's
production code path has been running with Franka welded at world origin all along.

**Experiment 4b — Side-view A/B recording (the headline evidence for the mentor).**
Top-down view from Exp 4 confirms X-Y placement but collapses the z-axis, which is
exactly where the burial shows. Recorded a side-view A/B against the same scene/seed
to expose vertical burial. Script:
[`test_franka_env_side.py`](DyWA/dywa/exp/train/test_franka_env_side.py), camera at
`eye=(1.6, −1.6, 0.9)` looking at `(−0.25, 0, 0.3)`. Two passes: (a) un-modified
`franka.create_actors` with identity `gymapi.Transform()` → bugged; (b) symmetric
create-time Transform fix applied → fixed. Body-tensor read-back was *identical*
across both passes (the tensor lies regardless). Rendered frames diverged
dramatically — see the comparison artifact below. Franka was then reverted to the
identity Transform to preserve 71.40% reproducibility.

| Pass | Clip | Still |
|---|---|---|
| Bugged | [franka_sideview_bugged.mp4](DyWA/dywa/output/franka_sideview_bugged_h264.mp4) | [PNG](DyWA/dywa/output/franka_sideview_bugged.png) — arm centered on/inside the white table block, lower joints embedded in the mesh |
| Fixed | [franka_sideview_fixed.mp4](DyWA/dywa/output/franka_sideview_fixed_h264.mp4) | [PNG](DyWA/dywa/output/franka_sideview_fixed.png) — arm clearly **behind** the table at table-top height, casting separate shadow |
| Side-by-side | [franka_sideview_compare.mp4](DyWA/dywa/output/franka_sideview_compare.mp4) | [PNG](DyWA/dywa/output/franka_sideview_compare.png) — labeled `BUGGED` / `FIXED` |

**Conclusion:** the side-view composite is the cleanest single artifact for the
mentor. Same arm, same scene, only the placement fix differs, burial is unambiguous.

**Experiment 5 — Source-level confirmation of the rendering split.**
Read
[`NvdrCameraWrapper._wrap_obs`](DyWA/dywa/src/env/env/wrap/nvdr_camera_wrapper.py)
directly. Confirmed it reads body poses from `self.env.tensors['body']` (line 520) —
i.e. the lying tensor. nvdiffrast then paints meshes at those written positions,
bypassing PhysX entirely. So the production rendering path (used by
`show_ppo_arm.py`'s `NvdrRecordEpisode`) **renders the intended pose, not the actual
welded pose** — which is why the bug has been invisible in every paper/figure render
that went through this path.
**Conclusion:** there are two parallel renderers in the codebase, and they disagree
precisely when the tensor lies (which is whenever `fix_base_link=True` + reset-time
root write).

| Renderer | API | Pose source | Shows |
|---|---|---|---|
| Raw Isaac Gym GPU camera | `gym.create_camera_sensor` + `gym.render_all_camera_sensors` | PhysX directly | Actual welded pose |
| nvdiffrast via `NvdrCameraWrapper` | nvdiffrast | `env.tensors['body']` (lying) | Written-but-ignored pose |

### 3.4 Why Trossen looks fully buried but Franka still pokes above the table

It's a **reach difference, not a placement difference**. Both arms are welded at
world origin under the un-modified code path — exact table center horizontally, floor
level vertically. The table mesh occupies `x ∈ [−0.2, 0.2]`, `y ∈ [−0.5, 0.5]`,
`z ∈ [0, 0.4]` and surrounds both bases identically.

- **Franka** has a ~85 cm reach. Its upper links (link4–link7, hand) push above
  `z = 0.4`, so part of the arm sticks up out of the table mesh and is visible from a
  top-down camera. The top-down render shows the gripper and a few of the upper links
  poking out of the table top — which is what
  [`franka_phys_buried_evidence.png`](DyWA/dywa/output/franka_phys_buried_evidence.png)
  captures.
- **Trossen** is 6-DOF with shorter reach (~50 cm). More of its body stays at
  `z < 0.4` — i.e. inside the table block's vertical extent — so more of it is
  occluded by the table mesh from the same camera. The arm looks "fully buried"
  because most of it geometrically *is* inside the table block, not because it's
  positioned differently from Franka.

Same bug, same world position. Different visual signature simply because Franka has
more arm above the table line.

This is partly why the bug wasn't caught earlier on Franka: enough of the arm sticks
out that renders going through `NvdrCameraWrapper` (the production path, which reads
from the lying tensor) looked plausible, and the visible-above-table portion in the
raw-camera path also looked roughly correct at a glance. Trossen has so much less arm
visible that the discrepancy became impossible to miss.

### 3.5 Empirical effect on the published baseline

Applied the symmetric create-time `gymapi.Transform()` fix to
[`franka.create_actors`](DyWA/dywa/src/env/robot/franka.py) (base placed at
`(−0.5, 0, 0.4)` via `gymapi.Transform()` baked at create time), re-ran the
pretrained-student eval against the **same checkpoint** and **same `test_set.json`** —
60 envs × 3000 test_steps, output at `output/eval_franka_fixed/`:

| Setup | Geometry | Success rate |
|---|---|---|
| Baseline (bugged) | Franka welded at world origin | 71.40% |
| **Fixed** | **Franka at (−0.5, 0, 0.4)** — paper-intended layout | **78.79%** |

Δ = +7.4 pp, well above the ~3% run-to-run noise observed in baseline reproduction
(71.40% vs an earlier 71.84% rerun).

**Eval rollout videos were not produced** (`show_ppo_arm.py`'s `NvdrRecordEpisode`
reads from the same lying body tensor and would hide the bug; a clean implementation
would require writing a new entry point mirroring `test_rma.py`'s pretrained-student
loading and adding a raw GPU camera, ~45 min of work plus CUDA-wedge risk). The
geometric side-view A/B (Experiment 4b above) carries the same evidence and is
faster to verify visually.

**Why the policy is invariant in principle but improves in practice:** the policy's
inputs are joint angles, robot-relative point cloud features, relative goals
(object-relative), and Δ-action history. None reference absolute world position — so
the welded-at-origin layout vs the correctly-placed layout *should* look identical to
the input pipeline. The +7.4 pp gain most plausibly comes from removing **geometric
interference**: the welded-at-origin arm's lower body intersects the table block,
occasionally routing through unwanted internal contacts. Moving the base behind the
table back edge gives clean reach, so the same learned policy applies more cleanly.

### 3.6 Action taken

[`franka.py`](DyWA/dywa/src/env/robot/franka.py) was **reverted** back to the
`gymapi.Transform()` identity to preserve 71.40% reproducibility against the
paper-original geometry. The Trossen create-time Transform fix remains in place (no
published Trossen number to preserve).

Going forward:
- Anyone running the pretrained eval reproduces ~71.40% as before.
- Anyone training a **new** Franka policy may want to apply the symmetric fix for
  ~7 pp upside and sim-to-real fidelity.

### 3.7 Why the bug wasn't fixed at the PhysX level

The underlying Isaac Gym behavior — "for `fix_base_link=True` actors, the reset-time
root-tensor write is a no-op" — is a property of Isaac Gym that I can't change from
the user side. What I did is **work around** it by moving the placement to create
time. The faulty no-op write is still in both
[`Trossen.reset()`](DyWA/dywa/src/env/robot/trossen.py) and
[`Franka.reset()`](DyWA/dywa/src/env/robot/franka.py); PhysX still silently ignores
it. If NVIDIA ever ships an Isaac Gym update that makes
`set_actor_root_state_tensor_indexed` actually move `fix_base_link` actors, both
reset-time writes would suddenly start working — at which point Trossen would
double-apply the placement (still correct) and Franka would jump from world origin to
the back-edge pose, breaking the 71.40% checkpoint until retrained. Worth flagging on
any Isaac Gym upgrade.

### 3.8 Why the paper's figures look fine despite the bug

A natural follow-up question: if Franka has been welded at world origin in PhysX
since CORN, **why don't the paper's figures show a buried arm?** Three reasons,
which together explain why this bug stayed invisible for years:

**Reason 1 — The paper's renders are not Isaac Gym screenshots.** The
[DyWA project page](https://pku-epic.github.io/DyWA/) figures (`sim_setup_v1.png`,
the demo videos under `medias/videos/sim/`) are **offline renders** — wooden tables
with thin legs and wood-grain texture, soft ambient lighting, multiple tables in
the background, polished Franka model. The actual Isaac Gym sim has a flat
checkerboard floor and a plain white box table (confirmed by every render in
[DyWA/dywa/output/](DyWA/dywa/output/) on this machine). So the paper figures
depict an *idealized* layout that the authors set up in Blender or similar — they
don't constrain what PhysX is doing.

**Reason 2 — When figures *are* generated from Isaac Gym, they use
`NvdrCameraWrapper`.** This is the production rendering path used by
`show_ppo_arm.py`'s `NvdrRecordEpisode`. It reads body poses from
`env.tensors['body']`
([nvdr_camera_wrapper.py:520](DyWA/dywa/src/env/env/wrap/nvdr_camera_wrapper.py#L520))
— i.e. the **lying tensor** — and renders meshes itself via nvdiffrast at the
written-but-ignored positions. So any author-side debug render or training-time
recording paints the arm at the *intended* `(−0.5, 0, 0.4)` regardless of where
PhysX actually has it. The bug is masked by construction. The only way to see it
is to use the raw Isaac Gym GPU camera (`gym.create_camera_sensor`), which is
what `test_franka_env.py` and `test_franka_env_side.py` do — neither is in the
project's normal rendering path.

**Reason 3 — Franka's reach disguises the burial.** Franka has ~85 cm reach; its
upper links push above z=0.4 even from a base buried in the table, so a casual
glance at a top-down or 3/4 render shows "an arm sticking up above a table" —
which is geometrically what you'd expect for a correctly-placed arm. The
*lower-body* burial is only visible from a side angle (which the paper figures
and the production renderer don't show). Trossen, with its shorter ~50 cm reach,
stays mostly below z=0.4 and gets almost entirely occluded by the table mesh —
which is exactly why this bug was finally caught when I started visualizing
Trossen.

So the bug-vs-paper-figures discrepancy isn't actually a discrepancy: the paper
shows what the authors *meant* the sim to look like, the production renderer
shows what was *written* to the tensors, and only the raw GPU camera shows what
PhysX is actually simulating. The first two agree with each other and disagree
with the third; the bug lives in the gap.

---

## 4. What's Validated vs. What Isn't

| Pipeline stage | Franka | Trossen |
|---|---|---|
| Sim env / reset / step | ✅ | ✅ (geometry fixed) |
| Stage 1 PPO (teacher) | ✅ (paper-trained ckpt) | ✅ smoke (50k steps) |
| Stage 2 PPO | ✅ (paper-trained ckpt) | ❌ not run |
| DAgger distillation | ✅ (paper-trained ckpt) | ❌ not run |
| Student eval | ✅ (71.40% reproduced) | ❌ not run |
| Visualization | ✅ (NvdrRecordEpisode) | ⚠ workaround (`show_trossen_record.py`) |

The first surface that will need work for Trossen Stage 2 / eval is `NvdrCameraWrapper`,
which currently hardcodes Franka URDF paths and link names. The workaround
`show_trossen_record.py` handles Stage 1 playback only.

---

## 5. Questions

### Q1. Is the GPU CUDA context wedge I hit normal?

After accumulating ~12 abrupt `kill -9`s of Trossen visualization processes during one
afternoon of debugging, every subsequent Isaac Gym sim-init started hanging at "create
simulation + actors" (immediately after `create_object_assets: 100%`). The Python
process is in **S state with 0 CPU time** — sleeping forever on an event from
PhysX/CUDA that never fires. `nvidia-smi` shows lingering allocated memory (~5.3 GB
per GPU) with no owning process.

Symptom didn't go away with `docker restart`, didn't move when I switched
`CUDA_VISIBLE_DEVICES`, and was identical across `test_trossen_env.py` variants. Best
inference: each killed PhysX process left an inconsistent CUDA context in the NVIDIA
driver. That state lives in **kernel memory**, so the container can't touch it. New
PhysX sims try to allocate via the wedged context and block on a CUDA stream sync
that never returns.

What clears it:

| Method | Effect | Permission |
|---|---|---|
| `sudo nvidia-smi --gpu-reset -i <id>` | Forcibly clears CUDA contexts. ~1 sec. | sudo |
| Switch GPU index | May work if not all GPUs are wedged | none |
| Wait several hours | Driver eventually garbage-collects orphaned contexts | none |
| Host reboot | Definitive | sudo + extreme |

**Questions:** Is this a known failure mode in the lab — i.e. should I just expect to
need a sysadmin `gpu-reset` every so often? Or am I doing something wrong upstream
(e.g. should I never `kill -9` Isaac Gym processes, always SIGTERM with a grace
period)? What's the lab's normal recovery workflow when this happens?

### Q2. Next-step training — what should I run, in what order?

Stage 1 PPO pipeline is verified for Trossen. The natural production-scale next step
is `num_env=4096`, ≥200k steps, `base_height='table'`, single RTX 6000 Ada — extrapolated
from the 50k smoke run, that's ~3–4 days of unattended compute. Stage 2 PPO and DAgger
distillation for Trossen have **not** been smoke-tested. The first failure surface to
expect is `NvdrCameraWrapper` (used by `show_ppo_arm.py`'s `NvdrRecordEpisode`),
which hardcodes the Franka URDF/link layout (~half a day to extend).

Concrete decision points I'd like to align on:

a. **Launch Stage 1 now, or speed up IK first?** Should I launch Trossen Stage 1
   production now (unattended ~3–4 days at the current ~16× Franka slowdown), or
   first invest in IK speedup (CUDA extension or batched analytical IK — see Q4) to
   close the gap before committing days of compute?

b. **Order of operations.** Is the right sequence (i) Trossen Stage 1 production →
   (ii) extend `NvdrCameraWrapper` for Trossen → (iii) Trossen Stage 2 PPO / DAgger /
   student eval? Or do you want me to smoke-test the full pipeline end-to-end before
   committing 3–4 days to Stage 1?

c. **Revisit Franka with the placement fix?** The +7.4 pp result (78.79% vs 71.40%) is
   one eval run, not a multi-seed comparison. A multi-seed retrain at paper scale
   would settle whether the fix is worth landing as the new Franka default — at the
   cost of a Stage 1 + Stage 2 + DAgger pass on Franka. Current default is "document
   the bug, leave 71.40% as the reference baseline." Worth retraining or not?

### Q3. Trossen base placement — how should I determine the right position?

Currently I weld Trossen at world `(−0.5, 0, 0.4)` via a create-time
`gymapi.Transform()`. This position isn't paper-prescribed; it's **computed from
three codebase defaults inherited from CORN**:

1. [`tabletop_scene.py`](DyWA/dywa/src/env/scene/tabletop_scene.py):
   `table_dims = (0.4, 1.0, 0.4)`, `table_pos = (0.0, 0.0, 0.2)`
2. [`trossen.py`](DyWA/dywa/src/env/robot/trossen.py): `keepout_radius = 0.3`
   (same default as `franka.py`)
3. `base_height` config knob: `'table'` (production) → z = `table_dims[2]` = 0.4

Arithmetic mirrored in the patched `create_actors`:

```
x = table_pos[0] − 0.5 × table_dims[0] − keepout_radius
  = 0 − 0.5 × 0.4 − 0.3
  = −0.5

y = table_pos[1] = 0

z (base_height='table') = table_dims[2] = 0.4
```

So `(−0.5, 0, 0.4)` decodes as "table is 40 cm wide centered at x=0, stand 30 cm
behind it, mount base at table-top height (40 cm)." The paper describes the *kind* of
layout (arm mounted behind table workspace) but doesn't pin the 40/30/40 numbers.

**Questions:**

a. Are these CORN-inherited defaults the right placement for our use case, or should
   the Trossen mount match a specific real-world rig? (e.g. should I set
   `keepout_radius=0.0` so the base sits *on* the table back edge instead of 30 cm
   behind it?)

b. For sim-to-real, what's the actual mount you have in mind — table-top height,
   floor with a stand, ceiling-suspended? The geometry fix bakes the choice in at
   sim init, and the policy gets ~7 pp better in the geometrically-correct case, so
   it's worth matching the eventual real-world setup before committing to a
   multi-day training run.

c. Should the y-offset stay at 0 (centered along the table's long axis) or shift to
   one side?

### Q4. Why is Trossen ~16× slower than Franka per simulation step?

This came up in Q2a — including the mechanism here so the IK-speedup decision is
informed.

**Measured:** ~1.94 it/s on the Trossen 50k smoke run at `num_env=1024` (per the
2026-05-17 dev_log entry). The Franka equivalent on the prior RTX 4090 machine was
roughly 30 it/s — ~16× the per-iteration throughput.

**Mechanism — what the two arms do per simulation step:**

| Aspect | Franka (OSC, Cartesian Impedance) | Trossen (numerical IK) |
|---|---|---|
| Code path | [`franka.step_controller:1015-1031`](DyWA/dywa/src/env/robot/franka.py#L1015-L1031) | [`trossen.apply_actions:451-455`](DyWA/dywa/src/env/robot/trossen.py#L451-L455) |
| Kinematics | Body Jacobian `self.j_eef` already maintained by Isaac Gym (free, from the body tensor); FK is a CUDA extension in [`dywa/c_src/`](DyWA/dywa/c_src/) for the 7-DOF Franka | FK + Jacobian computed in **pure PyTorch** each iteration ([`wxai_fk`](DyWA/dywa/src/env/robot/trossen_kin.py), [`wxai_jacobian`](DyWA/dywa/src/env/robot/trossen_kin.py)); no CUDA extension |
| Inner loop | **One pass.** Compute pose error → run the OSC controller (single matmul + small matrix decomposition for the mass-matrix scaling) → emit joint efforts | **30-iteration** damped-least-squares loop. Each iter: FK → Jacobian → 6×6 solve `(JJᵀ + λ²I)⁻¹` → joint update → clamp to joint limits |
| Output | Joint efforts (`set_dof_actuation_force_tensor_indexed`) — PhysX runs effort mode | Joint position targets (`set_dof_position_target_tensor_indexed`) — PhysX's built-in PD runs position mode |

So per env per step, Trossen runs **30 × (FK + Jacobian + 6×6 linear solve)** in
PyTorch where Franka runs **1 × (Jacobian lookup + OSC compute)** with the heavy
kinematics in CUDA C++. The 30 iterations alone account for the bulk of the gap; the
PyTorch dispatch overhead per op (~5+ tiny kernel launches per iter × 30 iters)
amplifies it further, because each launch carries fixed overhead that doesn't scale
with batch size at our `num_env`.

**What could close the gap:**

| Option | Effort | Expected speedup |
|---|---|---|
| Write a Trossen CUDA kinematics extension mirroring `dywa/c_src/{franka,ur5}_kin_cuda.so` | ~1–2 days (analytical FK + Jacobian from the URDF DH parameters, fused into one kernel) | Most of the 16× — should bring Trossen within 2–3× of Franka |
| Replace damped-LS with **batched analytical IK** for the 6-DOF arm (closed-form for WXAI's geometry) | ~2–3 days (need to derive and verify the closed-form solution) | Eliminates the 30-iter loop entirely — could match or beat Franka |
| Cap `ik_n_iter` (currently 30) — e.g. drop to 10–15 with adaptive termination on error tolerance | ~30 min (one config tweak + a tolerance check) | Linear in iter count: ~2× speedup, but with quality risk if early termination misses convergence |
| Switch from per-step IK to a learned IK head (one feed-forward pass) | ~half day to set up, plus training time | Single forward pass per step, similar to OSC, but adds policy-stack complexity |

**Question:** Is it worth a 1–2 day detour to write the CUDA extension before
launching Trossen Stage 1 production training? Likely answer is yes — saving 8×
on a 3–4 day run pays for itself many times over and helps Stage 2 / DAgger as
well. But it's also a real engineering investment with debugging risk; happy to defer
if you want the policy data sooner than the speedup.

---

*Working notes and verification details are in [DyWA/dev_log.md](DyWA/dev_log.md).*
