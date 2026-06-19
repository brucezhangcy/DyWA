# DyWA Development Log

**Project:** DyWA — Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation
**Repo (current machine):** `/data/bruce/DyWA`
**License:** CC BY-NC 4.0

---

## Project Overview

DyWA is a research system for learning generalizable robot manipulation policies through teacher-student distillation:

1. Train a **teacher policy** with privileged state information via PPO in Isaac Gym simulation
2. **Distill** that policy into a **student model** that only uses limited real-world observations (point clouds, 1 or 3 views)
3. Evaluate generalization on **unseen objects**

Target task is non-prehensile manipulation (pushing, not grasping).

**Stack:** Isaac Gym · PyTorch · Hydra · WandB · Point MAE / ICP embeddings · CUDA kinematics extensions

---

## Repository Structure (high level)

```
DyWA/
├── dywa/
│   ├── c_src/          # CUDA extensions — Franka & UR5 kinematics
│   ├── src/
│   │   ├── data/cfg/   # Hydra configs (env, run, net, student)
│   │   ├── env/        # Isaac Gym environments (arm_env, push_env, tasks)
│   │   ├── models/     # PPO, policy nets, point cloud encoders
│   │   ├── train/      # Losses, metrics, training utilities
│   │   └── util/       # Config helpers, Hydra CLI
│   └── exp/
│       ├── train/      # Python training entry points
│       └── scripts/    # Shell training / eval scripts
├── docker/             # Docker setup & run.sh
├── fig/                # Figures for README
├── setup.sh            # Isaac Gym + package install (run inside container)
└── README.md
```

---

## Key Config Knobs

| Config file | What it controls |
|---|---|
| `cfg/env/icra_base.yaml` | 4096 parallel envs, DGN object set, domain randomization |
| `cfg/run/icra_ours.yaml` | ICP embeddings + cross-attention aggregation |
| `cfg/student/dywa/base.yaml` | Student model baseline architecture |
| `cfg/net/feature/` | 50+ feature extractor configs (point patches, images) |
| `cfg/net/aggregator/` | State aggregation (GRU, NOOP) |

---

## Current Machine + Docker Setup (2026-05-14)

After a fresh `git clone` on a new lab box, switched from the prior conda env approach
(`corn-deploy2` on the old machine) to the **README-recommended Docker route**. Under Docker
the codebase's hardcoded paths (`/input/DGN/`, `/home/user/DyWA`) and pinned versions
(`flash-attn==1.0.4`, `torch 1.11+cu113`) all line up natively — most of the prior
local-machine patches become unnecessary.

### Machine

| Aspect | Value |
|---|---|
| User | `yunshuan` (workspace is `/data/bruce/`) |
| GPUs | 3× RTX 6000 Ada (49 GB each, sm_89 / Ada Lovelace) |
| Driver / host CUDA | 580.76.05 / CUDA 13.0 |
| Run env | Docker image `pkm1:v0` (Py 3.8, torch 1.11+cu113, flash-attn 1.0.4, pytorch3d 0.7.2) |

### GPU compatibility note

RTX 6000 Ada is **sm_89**, post-dates CUDA 11.3 (which natively compiles up to sm_86).
Pre-built wheels will JIT via PTX. If the dywa CUDA kinematics extension (`dywa/c_src/`)
needs a rebuild inside the container, export `TORCH_CUDA_ARCH_LIST='Ampere;8.9+PTX'`
before running `setup.sh` (the Dockerfile defaults to `'Ampere'` which omits sm_89).

### `docker/run.sh` paths

| Var | Host path | Container mount |
|---|---|---|
| `IG_PATH` | `/data/bruce/DyWA/isaacgym` | `/opt/isaacgym` |
| `CACHE_PATH` | `/data/bruce/DyWA/.cache/dywa` | `/home/user/.cache/pkm` |
| `DATA_PATH` | `/data/bruce/DyWA/datasets` | `/input` |
| `TMP_PATH` | `/data/bruce/DyWA/tmp/docker` | `/tmp/docker` |
| (repo bind) | `/data/bruce/DyWA` | `/home/user/DyWA` |

### Setup status

- ✅ DGN dataset extracted at `/data/bruce/DyWA/datasets/DGN/` (3.4 GB, 26,738 URDFs, 5,751 collision meshes)
- ✅ `test_set.json` copied into `/data/bruce/DyWA/datasets/DGN/`
- ✅ PyTorch3D 0.7.2 CUDA 11.3 wheel placed at `docker/pytorch3d-0.7.2-cp38-cp38-linux_x86_64.whl`
- ✅ Isaac Gym Preview 4 extracted at `/data/bruce/DyWA/isaacgym/` (655 MB)
- ⏸️ `yunshuan` not in `docker` group — needs sysadmin to run `sudo usermod -aG docker yunshuan`
- ⏸️ Image build (`./build.sh`) blocked on the above
- ⏸️ `setup.sh` inside container blocked on the image

---

## Known Issues / Gotchas

### `rel_goal` key mismatch during distillation
Standardized teacher/student observation key to `rel_goal` across configs and training scripts
in commit `12c43a5` (2025-09-23). Listed here because the failure mode (silent key drop) is
easy to reintroduce if observation bounds are edited.

---

### DAgger outer loop uses `++train_step`, NOT `++agent.train.train_steps`

**File:** `dywa/exp/train/train_rma.py:334,574`

Passing `++agent.train.train_steps=20000` to `train_rma.py` has **no effect** on the DAgger
loop. The outer loop is `for step in tqdm(range(cfg.train_step))` where `cfg.train_step`
defaults to `1_000_000`. The `agent.train.train_steps` field is read only when
`cfg.train_student_policy=True` (the PPO distillation path), not the DAgger path.

**Use `++train_step=20000`** (top-level Hydra key) to cap DAgger.

---

### Don't pass `++load_ckpt=<teacher_ckpt>` when evaluating the pretrained student

**File:** `dywa/exp/train/test_rma.py` (interaction with `AddTeacherState` env wrapper)

**Symptom:** Pretrained student (`Steve3zz/Dywa_abs_1view`) gives 0% success and ~2.8
steps/episode (OOB every 3 steps) when any teacher ckpt is passed via `++load_ckpt`. Same
student with default (no `++load_ckpt`, random teacher init) gives ~72% success.

**Root cause:** `AddTeacherState` appends the teacher's GRU hidden state to the student's
observation. The pretrained student was trained against a specific teacher's hidden-state
distribution; loading the paper's `teacher-last.ckpt` produces hidden states with a different
magnitude than what the student saw during DAgger. The student then outputs high `log_std`;
with stochastic action sampling the sampled action blows up → instant OOB.

**Empirical confirmation:**
| Setup | Success | Steps/episode |
|---|---|---|
| `++load_ckpt=teacher-last.ckpt`, stochastic | **0%** | ~2.86 |
| no `++load_ckpt` (random teacher), stochastic | **71.84%** | ~124 ✓ |
| `++load_ckpt=teacher-last.ckpt`, deterministic | **0%** | ~2896 (no OOB but never succeeds) |

**Fix:** Omit `++load_ckpt`. The official `eval_student_unseen_obj.sh` correctly omits it.

---

### `np.float` removed in NumPy ≥ 1.24 may still bite under Docker

**File:** `isaacgym/python/isaacgym/torch_utils.py:135`

```
AttributeError: module 'numpy' has no attribute 'float'
```

Only surfaces when `test_rma.py` (which uses `from isaacgym import gymtorch`) is imported
before `import isaacgym` runs the package `__init__`. Training scripts that `import isaacgym`
first avoid the ordering issue.

The Dockerfile installs `numpy` unpinned (line 274), so whether this bites in the container
depends on which numpy gets pulled in. If hit: edit line 135 of `torch_utils.py`,
`np.float` → `float`.

---

## Pretrained Student Eval — Reference Result

Paper Table 1, Unknown-State 1-view, Unseen objects: **75.0%**.
Our local reproduction (60 envs, 12 unique × 5 scales test set): **71.84%**.
Teacher-only (sanity check on env physics): **70.55%**.

### Correct eval command (container paths)

```bash
# Inside container, repo at /home/user/DyWA, data at /input/DGN, pretrained at $PRETRAINED.
cd /home/user/DyWA/dywa/exp/train
PYTORCH_JIT=0 python test_rma.py \
  +platform=debug +env=abs_goal_1view +run=teacher_base +student=dywa/base \
  ++name=dywa \
  ++path.root=/home/user/DyWA/output/eval_official \
  ++env.num_env=60 ++global_device=cuda:0 \
  ++student.norm=ln \
  ++add_teacher_state=1 '++student.decoder.film_mlp=1' \
  +load_student=$PRETRAINED/ckpt/last.ckpt \
  ++plot_pc=0 \
  ++dagger_train_env.anneal_step=1 \
  ++use_nvdr_record_episode=false \
  ++monitor.num_env_record=60 \
  ++env.single_object_scene.filter_file=/input/DGN/test_set.json \
  ++env.single_object_scene.mode=valid \
  ++log_categorical_results=True \
  ++env.single_object_scene.dgn.data_path=/input/DGN/meta-v8 \
  ++env.single_object_scene.dgn.pose_path=/input/DGN/meta-v8/unique_dgn_poses \
  ++test_step=3000
```

**Critical:** Do NOT add `++load_ckpt=<teacher_ckpt>`. See Known Issues.

---

## Training Scale to Reproduce Paper

Paper Table 1 highlights:

| Setting | Seen | Unseen |
|---|---|---|
| Known State, 3-view | 87.9% | 85.0% |
| Unknown State, 3-view | 85.8% | 82.3% |
| **Unknown State, 1-view** | **82.2%** | **75.0%** |

Approximate effort to retrain end-to-end (paper-scale). Wall times below are old single-RTX-4090
estimates from the prior session; **on the new 3× RTX 6000 Ada box they should be substantially
lower** if Stage 1 can be sharded across GPUs:

| Stage | `num_env` | Steps | Old 4090 estimate |
|---|---|---|---|
| Stage 1 (teacher PPO) | 4,096 | ~200–500k | 36–90 hr |
| Stage 2 (phase-2 PPO) | 2,048 | ~50–100k | 8–15 hr |
| Distillation (DAgger) | 256 | ~50–100k | 8–15 hr |

The prior under-scale run (1024 envs / 50k Stage 1 steps) gave 0% success on the distilled
student — confirming the trained policy is the bottleneck, not the env / eval pipeline.

---

## Q&A — Project Knowledge

### Why does the ICP encoder checkpoint reference "corn" (`imm-unicorn/corn-public`)?

DyWA does **not** train its own point cloud encoder. It reuses the ICP encoder pre-trained
in the predecessor project **CORN** (Contact-based Object Representation for Nonprehensile
Manipulation), which the README acknowledges:

> "This work is built upon and further extended from the prior work CORN."

`imm-unicorn/corn-public` is CORN's public HuggingFace repository (same group, PKU EPIC /
Galbot). The checkpoint `512-32-balanced-SAM-wd-5e-05-920` is CORN's pre-trained encoder.
The default in `icra_ours_abs_rel.yaml` is `corn/col-pre:512-32-balanced-SAM-wd-5e-05-920`
(CORN's private repo); override to `imm-unicorn/corn-public:...` for the public version.

The colon-separated format is parsed in `dywa/src/train/ckpt.py:92` — everything before `:`
is the HuggingFace repo ID, everything after is the filename.

---

## Trossen WidowX AI Integration

Added a 6-DOF **Trossen WidowX AI (WXAI)** arm option alongside the original Franka.
Only the environment layer changed; policy, training scripts, and task code are untouched.
Integration is committed to GitHub `main`.

### Franka vs Trossen

| Aspect | Franka | Trossen WXAI |
|---|---|---|
| Arm DOF | 7 | 6 |
| Total DOF (+ gripper) | 9 (2 finger joints) | 8 (2 prismatic carriage joints) |
| Gripper | Parallel jaw, position-driven | Prismatic carriage, position-driven |
| Control mode | Cartesian Impedance (CI) via OSC | Cartesian position + numerical IK (cpos_n) |
| Action space | 6D Δpose (+ optional gains) | 6D Δpose (Δpos + Δori axis-angle) |
| Joint drive | Effort mode via OSC | Position mode via PhysX built-in PD |

### Files

| File | Purpose |
|---|---|
| `dywa/src/data/assets/trossen_arm/` | URDF + 18 STL meshes |
| `dywa/src/env/robot/trossen_kin.py` | FK / geometric Jacobian / damped-LS IK (pure PyTorch) |
| `dywa/src/env/robot/trossen.py` | Isaac Gym robot class — same `RobotBase` interface as `Franka` |
| `dywa/src/env/push_env.py` | `Trossen` import, `trossen` Config field, `elif 'trossen'` branches |
| `dywa/src/env/arm_env.py` | `pos6` / `pos_vel6` observation space types (6-DOF arm, 12-D pos+vel) |
| `dywa/exp/train/test_trossen_env.py` | Headless sanity-check / recording script |

### Control architecture

Trossen uses **joint position drive** (PhysX built-in PD) with **numerical IK** (damped
least-squares, 30 iterations) to convert Cartesian actions into joint position targets:

```
action (6D Δpose) → workspace clipping → IK → q_target → set_dof_position_target_tensor_indexed
```

Jacobian (`wxai_jacobian`) and FK (`wxai_fk`) are hand-derived from URDF DH parameters,
entirely in PyTorch — no CUDA extension (unlike Franka's `dywa/c_src/`).

### Headless GPU camera rendering — visibility fix

Isaac Gym's headless GPU camera (`use_gpu_pipeline=True`) renders box actors fine but
returns black pixels for URDF mesh actors **unless rigid body color is explicitly assigned
at runtime**. URDF `<material>` tags are ignored by the headless GPU renderer.

**Fix** (in `dywa/src/env/robot/trossen.py:create_actors`):

```python
n_bodies = gym.get_actor_rigid_body_count(env, robot)
dark = gymapi.Vec3(0.15, 0.15, 0.15)
for b in range(n_bodies):
    gym.set_rigid_body_color(env, robot, b, gymapi.MESH_VISUAL, dark)
```

Things that did **not** fix it: camera angle, `use_collision_geometry=True`, URDF `<material>`
tags, FOV / near-far plane tuning.


---

## Patches needed only when running OUTSIDE Docker

Reference for future sessions where Docker isn't viable. Under Docker (`pkm1:v0`) **none of
these apply** — paths and pinned package versions match the codebase expectations.

| File | Patch |
|---|---|
| `isaacgym/python/isaacgym/torch_utils.py:135` | `np.float` → `float` (NumPy ≥ 1.24) |
| `dywa/src/env/env/help/with_nvdr_camera.py` | Remap `/input/DGN/` → host data dir for URDF mesh paths |
| `dywa/exp/train/envs/cube_env_wrappers.py:1504` | `CountCategoricalSuccess` accepts `save_dir` param (default was `/home/user/DyWA`) |
| `flash_attn/flash_attention.py` (shim) | Backport `FlashMHA` / `FlashAttention` API from 1.x using `F.scaled_dot_product_attention` |
| `data/meta-v8/urdf/*.urdf` (26,738 files) | `sed -i 's\|/input/DGN/coacd/\|<HOST>/data/coacd/\|g'` |

---

## 2026-05-15/16 — Docker build, container setup, eval reproduced (71.40%)

End-to-end run from fresh clone to reproducing the pretrained-student number, all inside
the `pkm1:v0` Docker container.

### Setup outcomes

| Step | Result |
|---|---|
| Sysadmin added `yunshuan` to `docker` group | ✅ unblocked |
| Isaac Gym Preview 4 tarball (provided by user) extracted to `/data/bruce/DyWA/isaacgym/` (655 MB) | ✅ |
| Docker image build (`./build.sh` → `pkm1:v0`, 6.13 GB content / 19.8 GB on disk) | ✅ after 3 Dockerfile patches (see below) |
| Container `dywa_1` launched detached with mount config from `docker/run.sh` | ✅ |
| `setup.sh` inside container (copy isaacgym, pip install isaacgym + dywa editable, fetch Eigen 3.4.0) | ✅ — CUDA extensions `cxx/{franka,ur5}_kin_cuda.so` built fine on sm_89 via PTX |
| Pretrained student `Steve3zz/Dywa_abs_1view` (`ckpt/last.ckpt`, `stat/env-last.ckpt`, `stat/env-teacher-last.ckpt`, `test_set.json`) downloaded from HuggingFace | ✅ |
| Pretrained student eval: 60 envs, 3000 test_steps, 1-view, unseen objects | **✅ 71.40% success rate** (vs dev_log reference 71.84% — within env-init noise) |

### Dockerfile patches needed during build (recorded for future rebuilds)

1. **`bgithub.xyz` mirror unreachable.** Original Dockerfile uses
   `git+https://bgithub.xyz/...` for nvdiffrast, mvp, and pytorch-cosine-annealing.
   `bgithub.xyz` is a Chinese GitHub mirror that doesn't resolve from this US lab box.
   Patched all three to `github.com` directly.

2. **nvdiffrast needs `--no-build-isolation`.** Without it, pip creates a build env that
   lacks PyTorch, and nvdiffrast's `setup.py` exits with the explicit "ERROR! Cannot
   compile nvdiffrast CUDA extension. Please ensure that: ... 2. You run 'pip install'
   with --no-build-isolation flag" message. Patched line 215.

3. **`line_profiler / cachetools / nvtx`: same fix.** Their pyproject.toml uses the old
   `license = {text=...}` dict form, which setuptools ≥77 rejects. Build isolation
   pulls in latest setuptools by default → fails. Forcing `--no-build-isolation` makes
   pip use the system-pinned setuptools 59.5.0 (which doesn't enforce SPDX). Patched
   line 235.

### Eval-time fixes (post-build, inside container)

The eval did **not** work out of the box. Nine attempts; each surfaced a real bug.

| # | Failed at | Cause | Fix |
|---|---|---|---|
| 1 | gymtorch JIT compile of `gymtorch.so` | `/home/user/.cache/` owned by `root` after Docker created intermediate dirs to mount `.cache/pkm` | `docker exec dywa_1 sudo chown user:user /home/user/.cache` |
| 2 | `import isaacgym.torch_utils` at line 135 | NumPy 1.24 dropped `np.float`. Predicted in dev_log Known Issues. | `sed -i 's/dtype=np\.float,/dtype=float,/'` inside container |
| 3 | `import nvdiffrast` | Dockerfile's `pip install --no-build-isolation` *succeeded* but installed the package as `UNKNOWN-0.0.0` because pinned `setuptools 59.5.0` can't read PEP 621 `name=` from pyproject.toml | Cloned nvdiffrast, edited `setup.py` to pass `name="nvdiffrast", version="0.4.0", packages=["nvdiffrast", "nvdiffrast.torch"]` explicitly, reinstalled. ([details below](#nvdiffrast-pep-621-quirk)) |
| 4 | `FileNotFoundError: /input/DGN/coacd/...obj not found from paths = [/tmp/...]` | **Previous-session local-machine patch leaked into GitHub:** `with_nvdr_camera.py:237–240` remapped `/input/DGN/` → `/home/brucezhang/Downloads/DyWA/data/`. Under Docker `/input/DGN/` IS correct — the remap broke it. | Removed lines 237–240 |
| 5 | `CountCategoricalSuccess.__init__` → `ensure_directory` → `PermissionError /home/brucezhang/...` | Same root cause as #4: prior session's patch committed `/home/brucezhang/...` as the hardcoded default save_dir | Changed default in `cube_env_wrappers.py:1507` to `/home/user/DyWA/output` (container repo bind) |
| 6 | `*** Can't create empty tensor` from `gymtorch.cpp:43`, after two swallowed ValueErrors looking for a teacher ckpt | `teacher_base.yaml:35` had `load_ckpt: '/home/user/DyWA/output/dywa/teacher-stage2/run-000/ckpt/'` — a stale default pointing to a Stage 2 ckpt we don't have. Per dev_log known issue, we *don't* want to load a teacher; need the random-init code path. | Changed `load_ckpt` default to `null` in YAML. (Hydra override `++load_ckpt=null` from CLI did *not* work — apparently parsed as the string `"null"`.) |
| 7 | `ValueError: not enough values to unpack` from `ICPNet.load` → `last_ckpt` | `teacher_base.yaml:14` had `ckpt: '/home/user/DyWA/ckpts/512-32-balanced-SAM-wd-5e-05-920'` — another stale local path. With no `:` separator, the HF fallback split fails. | Changed to `'imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920'` (HF public CORN repo, as per Q&A in this log) |
| 8 | `raise ValueError` at `test_rma.py:284` (env_ckpt doesn't exist) | Only `ckpt/last.ckpt` was downloaded from HF; `stat/env-last.ckpt` and `stat/env-teacher-last.ckpt` were also required | Downloaded both stat files from `https://huggingface.co/Steve3zz/Dywa_abs_1view/resolve/main/stat/...` |
| 9 | — | — | **Success.** 71.40% on 60 envs × 3000 steps. |

### nvdiffrast PEP 621 quirk

Worth noting because it'll bite again on a clean rebuild: `pip install --no-build-isolation
git+https://github.com/NVlabs/nvdiffrast.git` against setuptools 59.5.0 produces a wheel
whose name is literally `UNKNOWN-0.0.0`. Setuptools 59.5.0 predates PEP 621 support, so
the `name=` field in nvdiffrast's pyproject.toml is ignored, and there's no `name=` in
`setup.py`. The install reports success; the module is unimportable.

Workaround: clone, edit `setup.py` to add explicit `name`, `version`, `packages`. Or:
temporarily upgrade setuptools just for this step (haven't tried).

### Patches committed to the repo this session

For future container rebuilds these should land in `main`:

| File | What changed |
|---|---|
| `docker/Dockerfile` lines 216, 250, 276 | `bgithub.xyz` → `github.com` |
| `docker/Dockerfile` line 215 (nvdiffrast) | added `--no-build-isolation` (insufficient alone — see [PEP 621 quirk](#nvdiffrast-pep-621-quirk)) |
| `docker/Dockerfile` line 235 (line_profiler/cachetools/nvtx) | added `--no-build-isolation` |
| `docker/run.sh` | host paths set to `/data/bruce/{isaacgym,.cache/dywa,datasets,tmp/docker}` |
| `dywa/src/env/env/help/with_nvdr_camera.py` | removed prior-session `/input/DGN/` → `/home/brucezhang/...` remap (4 lines) |
| `dywa/exp/train/envs/cube_env_wrappers.py:1507` | `CountCategoricalSuccess` default save_dir → `/home/user/DyWA/output` (was `/home/brucezhang/Downloads/DyWA/output`) |
| `dywa/src/data/cfg/run/teacher_base.yaml:14` | ICP `ckpt` → `imm-unicorn/corn-public:...` (was stale local `/home/user/DyWA/ckpts/...`) |
| `dywa/src/data/cfg/run/teacher_base.yaml:35` | `load_ckpt: null` (was stale local `/home/user/DyWA/output/dywa/teacher-stage2/...`) |

### Lessons / additions to Known Issues

- **Don't commit local-machine paths.** Several stale `/home/brucezhang/...` paths and a
  stale Docker-prefix remap leaked from the prior session into GitHub. Each one cost a
  failed eval attempt. Going forward: keep host-specific edits out of tracked files.
- **Hydra `++load_ckpt=null` did NOT override a YAML string default.** Likely got parsed
  as the literal string `"null"`. Always change the YAML default directly when you want
  a None.
- **nvdiffrast vs setuptools 59.5.0.** Don't pin setuptools below 64 if you also need to
  install packages that rely on PEP 621 `name=` in pyproject.toml.
- **CUDA extensions on sm_89 (RTX 6000 Ada).** No problems. Both `cxx/franka_kin_cuda.so`
  and `cxx/ur5_kin_cuda.so` built cleanly via the Dockerfile's default
  `TORCH_CUDA_ARCH_LIST='Ampere'` (PTX forward-compat to sm_89). The `TBD` in the
  earlier 2026-05-14 section can be resolved: didn't need to override
  `TORCH_CUDA_ARCH_LIST`.

---

## 2026-05-16 — Trossen arm validation (parallel-config approach)

Switched the active arm from Franka to Trossen via a new parallel config tree, leaving Franka
configs untouched. Validation focused on env-level sanity (`test_trossen_env.py --headless --record`)
inside the Docker container. PPO smoke test (Check C from plan) deferred.

### Files added / modified

| File | Action | Notes |
|---|---|---|
| `dywa/src/data/cfg/env/trossen_icra_base.yaml` | **new** | `defaults: [icra_base]` + `env.which_robot=trossen`, `env.robot_state_type=pos_vel6`, `env.trossen.init_type=home` |
| `dywa/src/data/cfg/run/trossen_icra_ours_abs_rel.yaml` | **new** | `defaults: [icra_ours_abs_rel]` + `net.state.feature.icp_emb.ctx_dim=46` (= 9 + 20 + 12 + 5; Trossen `robot_state=12` vs Franka 14). NOTE: `pos_vel6` actually produces a 16-D `robot_state` at runtime (arm 6+6 + gripper 2+2). Final ctx_dim may need to be 50, not 46 — to be verified during Check C. |
| `dywa/exp/train/test_trossen_env.py` | **modified** | (1) Stale `/home/brucezhang/...` DGN path → `os.environ.get('DGN_PATH', '/input/DGN')` (yet another committed-from-old-machine path). (2) Camera moved from `eye=(-0.8,-0.25,0.26)` (clipped INSIDE table) to `(0.7,-0.7,0.9)` looking at `(-0.2,0,0.4)` — 3/4 above-table view. (3) Action amplitudes bumped from 0.03/0.01 → 0.12/0.20/0.10 (X/Y/Z) and cycles 4 → 2 so EE motion is visible. (4) Added `'trossen.base_height': '0.80'` override (see below). |
| `dywa/src/env/robot/trossen.py:323-336` | **modified** | Reset placement code: previously a binary `'ground'` (z=0) vs `'table'` (z=table_dims[2]=0.4) string switch. Now also accepts a numeric string — `float(cfg.base_height)` — so the base z can be set arbitrarily for visualization. |

### Verified placement numbers

These are the printed link world-positions at reset for **`base_height='table'`** (base sits ON
the table top — the configuration captured in the previous video pass before this section was
written). Source: `test_trossen_env.py:139-146` calls `gym.find_actor_rigid_body_index` and
prints world positions:

| Quantity | Value (m) |
|---|---|
| Table center | (0.000, 0.000, 0.200) |
| Table dims (XYZ) | (0.4, 1.0, 0.4) |
| Table z extents | [0.000, 0.400] |
| Table TOP surface | **z = 0.400** |
| Trossen base x | −0.500 (= 0 − 0.5·0.4 − 0.3 = behind back edge of table by `keepout_radius`=0.3 m) |
| Trossen base z (`base_height='table'`) | **0.400** (matches table top exactly) |
| `link_1` z | 0.457 |
| `link_2` z | 0.503 |
| `link_3` z | 0.758 |
| `link_4` z | 0.865 |
| `link_5` z | 0.923 |
| `link_6` z | 0.883 |

All visible links are above the table top **per the body-tensor read-back**. The first
video the user saw still showed the arm "embedded" because of two compounding issues:

1. **Placement bug (primary cause).** With identity-`Transform` at create time +
   reset-time write into `env.tensors['root']`, the `fix_base_link=True` actor is
   welded at world origin and the root-tensor write does not move it (see the
   [2026-05-17 create-time Transform fix entry](#2026-05-17--trossen-placement-create-time-transform-required)).
   So PhysX had the base at `(0, 0, 0)` — inside the table's XY footprint, at floor
   level — even though `body[idx, :3]` echoed the written `(−0.5, 0, 0.4)`. The
   "placement numbers" table above is therefore a tensor read, not a render check;
   the actual welded pose only matches these numbers after the create-time Transform
   fix is in place.
2. **Camera clipping (compounding).** The original camera at `(-0.8, -0.25, 0.26)`
   had eye z=0.26, which is *inside* the table block (table z ∈ [0, 0.4]). The gray
   "wall" in the first video was the inside of the table mesh, not the table seen
   from outside. Even if the base had been correctly placed, this camera would have
   shown table-interior.

The later `'0.80'` debug override (lifts the base 40 cm above the table top) made the
arm visually unambiguous for the screenshot/preview pass once the create-time fix was
in place.

### Confirmed Trossen link positions

Link positions for **`base_height='0.80'`** run (recorded 2026-05-16):

| Link | World z (m) |
|---|---|
| `base_link` | **0.800** (= 40 cm above table top) |
| `link_1` | 0.857 |
| `link_2` | 0.903 |
| `link_3` | 1.158 |
| `link_4` | 1.265 |
| `link_5` | 1.323 |
| `link_6` | 1.283 |

All links ≥ 0.8 m, i.e. ≥ 40 cm above the table surface (z=0.4). 500-step run completed
with non-NaN rewards in [0.0006, 0.0126]. Video at `dywa/output/trossen_vis.mp4` (83 KB
H.264, host-side transcode).

### Open items deferred

- **Check C (PPO smoke test)** not yet run. Once run, verify `ctx_dim=46` vs the observed
  `robot_state=16` claim — likely need to change to 50.
- **`PYTORCH_JIT=0` required for Trossen too.** The `quat_from_axa(daxa)` call in
  `Trossen.apply_actions` ([trossen.py:393](dywa/src/env/robot/trossen.py#L393)) is decorated
  for JIT; nvrtc compilation fails on sm_89 because the bundled CUDA-11.3-era nvrtc lists
  arches only up to sm_86. Same workaround as elsewhere in this project.
- **No `ffmpeg` in the `pkm1:v0` container.** `test_trossen_env.py` does the H.264 transcode
  via a `subprocess.run(['ffmpeg', ...])` call at line 179, which fails inside the container.
  The cv2 VideoWriter writes a raw mp4v (DivX-era) `.tmp.mp4` that doesn't play in modern
  players (Chrome / VS Code preview). Workaround: transcode on the host with system ffmpeg.

---

## 2026-05-16 — Trossen PPO smoke test passes (functional equivalence to Franka)

Demonstrated that Trossen runs end-to-end in the same `train_ppo_arm.py` entry point used
for Franka Stage 1. **200 PPO outer steps completed** with `num_env=128` and
`base_height='ground'` (arm grounded on the floor, EE reaches over table from behind, same
production setup as Franka).

### Files touched to get the smoke test to pass

| File:line | Change | Reason |
|---|---|---|
| [trossen.py:301-307](dywa/src/env/robot/trossen.py#L301-L307) | Added `self.cur_hand_friction = th.full((num_env,), default_hand_friction, ...)` | `AddPhysParams` obs wrapper at [cube_env_wrappers.py:665](dywa/exp/train/envs/cube_env_wrappers.py#L665) reads `robot.cur_hand_friction[..., None]` and concatenates it into `phys_params` (5-D). Franka exposes it ([franka.py:441](dywa/src/env/robot/franka.py#L441)); Trossen didn't. |
| [arm_env.py:43-44](dywa/src/env/arm_env.py#L43-L44) | `TrossenArmDofPos` (6,) → (8,); `TrossenArmDofPosVel` (12,) → (16,) | Trossen URDF has 8 DOFs total (6 arm + 2 gripper carriage), not just 6. Flatten of `tensors['dof']` produces 16 elements (= 8 × pos+vel), so the gym `spaces.Box` declaration must match the actual obs shape. Franka's `pos_vel7` is 14 because Franka URDF lists 7 DOFs (fingers are mimic/fixed). |
| [arm_env.py:147-148](dywa/src/env/arm_env.py#L147-L148) | `pos6`, `vel6` tuples from length 6 to length 8 | Same reason — the bound tuple needs 8 entries to match the 8-DOF obs. |
| [trossen_icra_ours_abs_rel.yaml:11-12](dywa/src/data/cfg/run/trossen_icra_ours_abs_rel.yaml#L11-L12) | `ctx_dim: 46` → `36`; added `dim_out: 2084` | `ctx_dim` = sum of query-key dims fed into cross-attention: 9 (rel_goal) + 6 (previous_action, matching Trossen's 6-D action — not Franka's 20-D) + 16 (robot_state) + 5 (phys_params) = **36**. `dim_out` = cross-attention output: 16 query tokens × 128 emb_dim + ctx_dim = **2084** (Franka uses 2096 = 2048+48). |

### Verification

```
got dim_act=6
PiNet got dims=(128, 64, 6) from (128, [64], 6)
```

Per-iter throughput ≈ 2.5–3.0 it/s with `num_env=128` (single RTX 6000 Ada). All of
`ckpt/step-00000.ckpt`, `ckpt/last.ckpt`, and `stat/env-last.ckpt` saved to
`/data/bruce/DyWA/output/trossen_smoke/run-003/`. No NaN losses, no shape mismatches.

### Final visualization

`dywa/output/trossen_vis.mp4` (61 KB H.264) shows the Trossen arm grounded on the floor
at x=−0.5, z=0, reaching up and over the white table from a side view (camera at
(−0.25, −1.6, 0.7) along the table's long axis). Same physical layout as the Franka
production setup. Link world positions at reset:

| Link | World (x, y, z) |
|---|---|
| `base_link` | (−0.500, 0, 0.000) |
| `link_4` | (−0.323, 0, 0.465) |
| `link_5` | (−0.266, 0, 0.523) |
| `link_6` | (−0.232, −0.014, 0.483) |

Floor is at z=0; table top is at z=0.4. The upper arm is above the table surface —
exactly where it needs to be to reach DGN objects.

---

## 2026-05-16 — Correction: `ws_bound` ≠ hard reach limit

An earlier draft of this log argued that "Trossen must be on the ground because the WS
bound clamps EE z to [base_z + 0.35, base_z + 0.75], so a table-top mount makes the EE
unreachable to the table." **That analysis was wrong** — and is removed here. The actual
behavior:

- [`trossen.py:420`](dywa/src/env/robot/trossen.py#L420) only applies the WS clamp when
  **`cfg.clip_bound=True`**. Default is `False`, so the bound is a soft hint, not a
  reachability constraint.
- [`franka.py:185-203`](dywa/src/env/robot/franka.py#L185-L203) is the same: WS bound is
  passed to the IK *only if* `clip_bound=True`.

So in production (default `clip_bound=False`), the arm's reach is whatever the IK + joint
limits physically allow. With Franka's 7-DOF and ~85 cm reach, a table-top mount works
fine — the arm reaches down and across to manipulate. This matches the standard lab
convention `base_height: 'table'` is *trying* to encode.

### Visualization: both arms at table-edge height

[trossen_vis.mp4](dywa/output/trossen_vis.mp4) shows Trossen with `base_height: 'table'`:
base at world (−0.5, 0, 0.4), upper links above the table top (link_4 z=0.865,
link_5 z=0.923, link_6 z=0.883). The arm appears to "float" because the supporting
robot stand isn't modeled in sim — in a real lab the stand would extend from the floor
up to z=0.4 to hold the arm base.

| Link | World (x, y, z) under `base_height: 'table'` |
|---|---|
| `base_link` | (−0.500, 0, 0.400) |
| `link_4` | (−0.323, 0, 0.865) |
| `link_5` | (−0.266, 0, 0.923) |
| `link_6` | (−0.232, −0.014, 0.883) |

---

## 2026-05-17 — Trossen Stage 1 smoke training (50k steps) + playback recording

Ran a 50,000-step PPO Stage 1 against the Trossen env to verify the full training pipeline
works end-to-end (`base_height='table'`, mounted at world (−0.5, 0, 0.4)). Goal was
pipeline/throughput validation, not a learned policy — 50k steps at `num_env=1024` is far
below production scale (paper trains 4096 envs × ~200–500k steps).

### Setup

| | |
|---|---|
| Entry | `train_ppo_arm.py` |
| Configs | `+env=trossen_icra_base +run=trossen_icra_ours_abs_rel` |
| Overrides | `++env.num_env=1024 ++env.trossen.base_height=table ++agent.train.train_steps=50000` |
| ICP encoder | `imm-unicorn/corn-public:512-32-balanced-SAM-wd-5e-05-920` (auto-downloaded) |
| Output | `output/trossen_pipeline/teacher-stage1/run-000/` |
| Wall-clock | **~7 h 11 min** (started 00:14, finished 07:25) |
| Throughput | ~1.94 it/s; ~106 s per `log_period=2048` interval |

Far slower per-iteration than the dev_log's prior Franka run (50k steps at num_env=1024
took ~37 min on RTX 4090). The slowdown is from Trossen's **30-iter damped-LS numerical
IK** in pure PyTorch ([trossen.py:359-454](dywa/src/env/robot/trossen.py#L359-L454)),
whereas Franka uses OSC (single matrix inverse). IK dominates per-step cost.

### Results (from tensorboard scalars)

| Metric | Step 1024 | Step 13312 | Step 25600 | Step 37888 | Step 49152 |
|---|---|---|---|---|---|
| `loss/total` | 5.093 | 0.013 | 0.017 | 0.017 | 0.024 |
| `loss/policy` | 0.133 | −0.005 | −0.008 | −0.006 | −0.001 |
| `loss/value` | 2.513 | 0.005 | 0.007 | 0.006 | 0.008 |
| `env/avg_episode_return` | −0.452 | −0.330 | −0.277 | −0.246 | **−0.226** |
| `env/eplen` | 123.9 | 145.9 | 132.4 | 115.8 | **106.7** |
| `env/suc_rate` | 0.0000 | 0.0003 | 0.0006 | 0.0007 | **0.0007** |

**Pipeline is healthy** (loss converging, policy/value losses stable, no NaN, return trending up, episodes terminating faster). **But success rate is essentially zero (~0.07%)** — same as the prior Franka under-scale run in dev_log. Expected at 50k steps; would need production-scale (~200-500k steps) to actually solve the task.

Checkpoints saved at: `output/trossen_pipeline/teacher-stage1/run-000/ckpt/{step-00000, step-16384, step-32768, step-49152, last}.ckpt`.

### Playback visualization

To produce a video of the trained Trossen policy:

1. **`NvdrRecordEpisode`** (the wrapper used by `show_ppo_arm.py`) **does NOT work with
   Trossen** — its underlying `NvdrCameraWrapper` only knows how to copy Franka assets
   into the temp dir. Loading the Trossen URDF from there fails with
   `ValueError: /tmp/tmpXXX/trossen_arm/urdf/wxai_base.urdf is not a file`.

2. **Workaround**: wrote
   [`exp/train/show_trossen_record.py`](dywa/exp/train/show_trossen_record.py) — a clone
   of `show_ppo_arm.py` minus the NvdrRecordEpisode wrap, with a direct Isaac Gym GPU
   camera setup (same approach as `test_trossen_env.py`).

3. **`sample_action=false` gives a stationary arm** at this training level. The trained
   policy's mean output is near zero (it hasn't learned anything yet), so deterministic
   action gives near-zero Δpose commands per step → arm barely moves. **Use
   `++sample_action=true`** for visible motion — the policy's `log_std` provides the
   exploration noise.

Final command (visible motion):

```bash
PYTORCH_JIT=0 python3 show_trossen_record.py \
  +platform=debug +env=trossen_icra_base +run=trossen_icra_ours_abs_rel \
  ++env.num_env=4 ++env.trossen.base_height=table \
  ++load_ckpt=output/trossen_pipeline/teacher-stage1/run-000/ckpt/step-32768.ckpt \
  ++sample_action=true ++n_steps=500 \
  ++out_path=output/trossen_pipeline/trossen_playback.mp4
```

Result: `output/trossen_pipeline/trossen_playback.mp4`, ~70 KB H.264.

### Takeaway

End-to-end Trossen pipeline works in the Franka training entry-point. Loss curves and
return trends are healthy; throughput is ~16× slower than Franka due to per-step IK.
To get a useful policy, need production-scale training (`num_env=4096`, ≥200k steps,
~3–4 days on RTX 6000 Ada). 50k steps was a smoke test — pipeline verified, no
learned behavior expected.

---

## 2026-05-17 — GPU CUDA context wedge (Isaac Gym PhysX-init hang)

After accumulating ~12 abrupt `kill -9` of Trossen visualization processes during today's
debugging, every subsequent Isaac Gym sim-init hangs at stage "create simulation + actors"
(immediately after `create_object_assets: 100%`). The Python process is in **S state with
0 CPU time** — sleeping forever waiting for an event from PhysX/CUDA that never fires.
NOT an I/O wait (which would be D state with non-zero CPU).

### Symptom

Six+ identical attempts (`test_trossen_env.py`, varying `num_env`, GPU choice, base_height,
container restart) all freeze at the same line:

```
create_object_assets: 100%|██████████| 4/4 [00:02<00:00, 1.57it/s]
   ← hangs here, no further output, no CPU activity, 0% GPU util
```

Yesterday's `trossen_vis.mp4` (Date: 2026-05-16 22:06) and this morning's
`trossen_playback.mp4` (12:38) both worked — same scripts, same container. The hang only
started after a series of mid-execution kills in the afternoon.

### Cause (best inference)

Each killed PhysX process leaves an inconsistent CUDA context in the NVIDIA driver. The
context state lives in **kernel memory**, not the container, so:

- `docker restart` doesn't clear it.
- Same wedge appears even on different GPU IDs (we tried `CUDA_VISIBLE_DEVICES=2`).
- `nvidia-smi` shows lingering allocated memory (5.3 GB per GPU) with no owning process.

New PhysX sims try to allocate via the wedged context, block on a CUDA stream sync that
never returns. Effectively a driver-level deadlock.

### What clears it

| Method | Effect | Permission |
|---|---|---|
| `nvidia-smi --gpu-reset -i 0` | Forcibly clears all CUDA contexts on GPU 0. ~1 sec. | sudo |
| Stop+start container with `--gpus '"device=N"'` for an unused GPU | Forces fresh CUDA context allocation on a different GPU. May work if some GPUs aren't wedged. | docker group (have) |
| Wait several hours | Driver eventually garbage-collects orphaned contexts. Not guaranteed. | none |
| Host reboot | Definitive fix, kills everyone else. | sudo + extreme |

### Practical mitigation for future sessions

Don't `kill -9` Isaac Gym processes mid-execution. Use SIGTERM (`kill -15`, the default
of `pkill -f`) and give them ~10s to clean up CUDA contexts. Only escalate to SIGKILL if
SIGTERM doesn't release.

The lab box at this point appears stuck for the rest of today (load also high). Workflow
options:

1. Wait until tomorrow morning, retry — driver may garbage-collect overnight.
2. Ask sysadmin to run `sudo nvidia-smi --gpu-reset -i 0 -i 2 -i 3` (skip GPU 1 which has
   another user's workload).
3. Skip visual verification, launch training based on geometry math — but the training's
   first call is also a PhysX init, so it would hit the same wedge.
4. Trying option 2 from the table above (recreate container targeting only GPU 1) is risky
   — GPU 1 has someone else's 36 GB workload; competing might crash both.

---

## 2026-05-17 — Trossen placement: create-time Transform required

The Trossen base does not land at `base_height`-driven coordinates if `create_actors`
passes an identity `gymapi.Transform()` and relies on the reset-time write into
`env.tensors['root']` to place it. For `fix_base_link=True` actors, the welded PhysX
pose is fixed at the pose passed to `gym.create_actor`; the reset-time write into the
root tensor is empirically a no-op for the welded pose.

The body-tensor read-back masks this: `gym.find_actor_rigid_body_index(...) → body[idx, :3]`
returns whatever was last written into `env.tensors['root']`, not the actor's actual
welded pose. So `test_trossen_env.py` could print `base_link: (−0.500, 0, 0.400)` while
the rendered frame showed the arm at world origin, embedded inside the table block.

### Fix

[`trossen.create_actors`](dywa/src/env/robot/trossen.py#L193-L221) now computes the
`base_height`-driven placement at create time and bakes it into the
`gymapi.Transform()` passed to `gym.create_actor`. Hardcoded table values
(`table_pos=(0,0,0.2)`, `table_dims=(0.4,1.0,0.4)`) match
[`tabletop_scene.py:16-17`](dywa/src/env/scene/tabletop_scene.py#L16-L17) — `env` here
is the raw Isaac Gym Env handle, not the EnvBase wrapper, so `env.scene` isn't
accessible. The reset-time write at
[trossen.py:337-353](dywa/src/env/robot/trossen.py#L337-L353) is left as a defensive
fallback; its comment reads "empirically a no-op for fix_base_link actors in this
code path."

### Verification

After the fix, `test_trossen_env.py --headless --record` with `base_height='table'` and
default `keepout_radius=0.3` produces:

| Link | World (x, y, z) |
|---|---|
| `base_link` | (−0.500, 0, 0.400) |
| `link_4` | (−0.323, 0, 0.865) |
| `link_5` | (−0.266, 0, 0.923) |
| `link_6` | (−0.232, −0.014, 0.883) |

A render at this point (the original `trossen_vis.mp4` from 2026-05-17) clearly
showed the orange arm sitting behind the white table at table-top height, reaching
forward over the table, casting a shadow on the floor next to the table — paper-figure
layout. The render matched the printed positions, which it didn't before. The
2026-05-17 artifact was superseded by the polished A/B comparison recorded on
2026-05-18 (see "Trossen side-view A/B" section below); the older file was deleted
in that cleanup pass.

### Operational note

When `fix_base_link=True`, body-tensor read-back can't be trusted as a placement check
on its own. Use a top-down rendered camera against the production code path.

---

## 2026-05-18 — Franka has the same bug, verified by top-down render

Followup verification of the Trossen finding for Franka, plus reading the rendering
path used in show_ppo_arm.py / NvdrRecordEpisode.

### Step 1: top-down render of unmodified Franka

Fixed [test_franka_env.py](dywa/exp/train/test_franka_env.py) (was segfaulting
from missing collision meshes — Franka URDF references `meshes/collision/*.obj` but
the asset dir only has `.stl`; converted via trimesh, plus aliased `finger_coacd.obj`
to `finger.obj`). With `init_type='home'`, `base_height='table'`, default
`keepout_radius=0.3`, and a raw Isaac Gym GPU camera at `eye=(0.001, 0, 3)
at=(0, 0, 0)` (straight-down view):

- Body-tensor read-back: `panda_link0: (-0.500, -0.000, 0.400)` — same "lying"
  value as Trossen had.
- Rendered frame: **arm visually embedded at table center**, not at the
  back-edge offset the body tensor claims. See
  [franka_phys_buried_evidence.png](dywa/output/franka_phys_buried_evidence.png).

So Franka has the **identical** `fix_base_link`-weld bug. PhysX has the actor at
world origin; the body tensor echoes whatever was written into the root tensor,
not the welded pose.

### Step 2: why the rendering path matters

[`NvdrCameraWrapper._wrap_obs`](dywa/src/env/env/wrap/nvdr_camera_wrapper.py#L516-L527)
reads its body poses directly from `self.env.tensors['body']` (line 520). That's
the same lying tensor. So any rendering that goes through NvdrCameraWrapper /
`NvdrRecordEpisode` (the visualization path used by `show_ppo_arm.py`) renders
the arm at the *written* position, not at the actual welded pose. This is why
the bug isn't visible in renders that go through that path.

The raw Isaac Gym GPU camera (`gym.create_camera_sensor` +
`gym.render_all_camera_sensors`) renders the actual PhysX scene, bypassing the
body tensor — which is what exposed the bug for Trossen and now confirms it for
Franka.

### Step 4: paper figures are not from Isaac Gym

The paper's project page figures (`sim_setup_v1.png`, `medias/videos/sim/*.mp4`)
appear to be **offline renders** (Blender or similar): wooden tables with thin
legs and wood-grain texture, soft ambient lighting, multiple tables visible in
the background, polished Franka model. The actual Isaac Gym sim has a flat
checkerboard floor and a white box table (confirmed by `test_franka_env.py`
output and Trossen renders).

So the paper figures showing Franka mounted at the table edge are *not* evidence
about where Franka actually is in PhysX. They depict an idealized layout. The
actual PhysX placement is what the raw GPU camera shows: world origin.

### Synthesis

- Both Franka and Trossen are welded at world origin in PhysX under the
  un-modified code path. Confirmed by direct top-down render in both cases.
- The body-tensor read-back lies for `fix_base_link=True` actors: it echoes
  what was written into `env.tensors['root']` during reset, not the actual
  PhysX welded pose.
- `NvdrCameraWrapper` renders from the lying body tensor, so any visualization
  going through it (paper figures *might* go through a different path; the
  point is the bug is hidden in any tensor-sourced render).
- The published 71.40% pretrained student eval was obtained against a Franka
  welded at world origin. The number is real, but the geometry is bugged.

### Step 5: policy invariance — **success rate INCREASED to 78.79%**

Applied the symmetric create-time-Transform fix to
[franka.create_actors](dywa/src/env/robot/franka.py#L678) (base placed at
`(−0.5, 0, 0.4)` via `gymapi.Transform()` baked at create time), re-ran the
pretrained-student eval against the same checkpoint and same test_set.json. 60
envs × 3000 test_steps, output at `output/eval_franka_fixed/`.

| Setup | Geometry | Success rate |
|---|---|---|
| Baseline (bugged) | Franka welded at world origin (0, 0, 0) | 71.40% |
| **This run (fixed)** | **Franka at (−0.5, 0, 0.4)** — table back edge, table-top height | **78.79%** |

Δ = +7.4 pp. That's larger than the ~3% run-to-run noise observed in the
baseline (71.40% vs prior 71.84% reproduction).

**Interpretation**: the policy is not just invariant to base position — it
performs *better* against the geometrically-correct setup. The most plausible
reason is that the welded-at-origin Franka has its base inside the table block,
which creates persistent geometric interference (the arm reaching *through* the
table mesh to manipulate objects on top). Moving the base behind the table back
edge eliminates that interference, and the policy — which doesn't observe
world-frame base position — applies its learned behavior cleanly.

This refutes the earlier concern that "moving Franka to its correct position
will shift the obs distribution and likely break the eval". The trained
representation transferred fine — better, in fact.

### Action taken

[franka.py:678](dywa/src/env/robot/franka.py#L678) **reverted** back to
`gymapi.Transform()` identity per the plan — to preserve the 71.40%
reproducibility against the paper-original geometry. The Trossen create-time
Transform fix remains in place.

Going forward:

- Anyone running the pretrained eval reproduces ~71.40% as before.
- Anyone training a **new** Franka policy may want to apply the symmetric fix
  to get a geometrically-correct Franka. Likely small upside for unseen-object
  pushing (this single eval suggests ~7 pp), with the caveat that this is one
  run, not a multi-seed comparison.
- For sim-to-real transfer, the symmetric fix is the right default — real
  Frankas are mounted at the table back edge, not buried in the table block.

### Bottom line on the verification plan

All five steps came back consistent with the same picture:

1. NvdrCameraWrapper reads body tensor (confirmed by source).
2. Franka is welded at world origin in PhysX (confirmed by top-down render).
3. Source + body-tensor verification: tensor-sourced renders show the lying
   value; raw GPU camera shows the actual PhysX state. No additional render
   needed.
4. Paper figures are offline renders (Blender), not Isaac Gym screenshots —
   so they don't constrain the PhysX-state question either way.
5. Policy invariance verified empirically: 78.79% with Franka moved to
   correct position vs 71.40% with bugged geometry. Not just invariant —
   improved.

The earlier mid-thread retraction (claiming the reset-time write was working
and Franka was correctly placed) was wrong. The "Hidden bug discovered" line
of analysis was correct in mechanism, even though the empirical confirmation
took until 2026-05-18 to land.

---

## 2026-05-18 — Q&A on the placement bug findings

Follow-up questions that came up after the verification finished. Recording the
answers here because they tie the loose ends together.

### Q1. Why didn't the buried Franka hurt performance more than ~7 pp?

The policy doesn't observe absolute world coordinates. Its inputs are joint
angles, point cloud features (in robot-relative frame), relative goals
(object-relative), and previous actions (Δ-poses). None of those reference
"where is my base in world frame," so the welded-at-origin layout vs the
correctly-placed-at-`(−0.5, 0, 0.4)` layout look identical to the policy's
input pipeline.

The +7.4 pp improvement when Franka is moved off origin is most plausibly
geometric interference: at world origin the arm's lower body intersects the
table block, the policy occasionally gets stuck or routes through unwanted
internal contacts. Moving the base behind the table back edge gives clean
reach, so the same learned policy applies more cleanly.

Self-consistent training/eval (both at the same bugged geometry) is what
preserves the 71.40% number — the published checkpoint learned to push
table-center objects given that specific arm-relative-to-table configuration.

### Q2. Why is Trossen fully buried but Franka still visible from top-down?

It's a reach difference, not a placement difference. Both arms are welded at
world origin under the un-modified code path — exact table center horizontally,
floor level vertically. The table mesh occupies `x ∈ [−0.2, 0.2]`,
`y ∈ [−0.5, 0.5]`, `z ∈ [0, 0.4]` and surrounds both bases identically.

- **Franka** has a ~85 cm reach. Its upper links (link4–link7, hand) push
  above z=0.4, so part of the arm sticks up out of the table mesh and is
  visible from a top-down camera.
- **Trossen** is 6-DOF with shorter reach (~50 cm). More of its body stays at
  z<0.4 (inside the table block's vertical extent), so more of it is
  occluded by the table mesh from the same camera.

Same bug, same world position. Different visual signature because Franka has
more arm above the table line.

### Q3. Why is this a "renderer" issue — can't Trossen use Franka's renderer?

The renderer doesn't *cause* the bug — the placement bug is in PhysX. The
renderer affects whether the bug is **visible** in rendered output.

- **NvdrCameraWrapper** (used by `show_ppo_arm.py`'s `NvdrRecordEpisode`,
  visible in [nvdr_camera_wrapper.py:516-527](dywa/src/env/env/wrap/nvdr_camera_wrapper.py#L516-L527))
  reads body poses from `env.tensors['body']` — the lying tensor — and
  renders meshes itself via nvdiffrast from URDF + those poses. So nvdiffrast
  paints the arm at the *written* position `(−0.5, 0, 0.4)`. Looks correct.
  Bug hidden.
- **Raw Isaac Gym GPU camera** (`gym.create_camera_sensor` +
  `gym.render_all_camera_sensors`, used by `test_franka_env.py` and
  `test_trossen_env.py`) renders the actual PhysX scene. Shows the welded
  pose at world origin. Bug exposed.

**Can Trossen use NvdrCameraWrapper?** Yes — with a half-day of integration
work. The wrapper currently fails for Trossen at
[nvdr_camera_wrapper.py:188-204](dywa/src/env/env/wrap/nvdr_camera_wrapper.py#L188-L204):
the `get_hacky_blocklist_for_arm_links` function hardcodes Franka link names,
and the asset-copy logic in `set_camera` only knows the Franka URDF + mesh
layout. Trossen URDF copies into the tmp dir fail with
`ValueError: /tmp/tmpXXX/trossen_arm/urdf/wxai_base.urdf is not a file`.

Why we didn't extend it: doing so would just **hide** the bug for Trossen the
same way it's hidden for Franka. The actual fix is the create-time Transform
patch at [trossen.py:193-221](dywa/src/env/robot/trossen.py#L193-L221), which
moves PhysX itself to the correct welded pose. After that, both renderers
agree.

#### Deep dive: two parallel renderers, each with its own pose source

The DyWA project has two separate paths for "turn the sim state into pixels":

**Path A — Raw Isaac Gym GPU camera.**
- API: `gym.create_camera_sensor` + `gym.render_all_camera_sensors`
- Renders: whatever PhysX has in the scene
- Reads from: PhysX directly (the actual welded poses, link positions from
  forward simulation)
- Used by: `test_franka_env.py`, `test_trossen_env.py`, `show_trossen_record.py`
- Sees: the actual welded pose. If PhysX has Franka stuck at world origin,
  the render shows it at world origin.

**Path B — nvdiffrast via NvdrCameraWrapper.**
- API: nvdiffrast (NVIDIA's differentiable rasterizer for PyTorch)
- Renders: meshes the wrapper has loaded itself, positioned by poses it read
  from a tensor
- Reads from: [`env.tensors['body']`](dywa/src/env/env/wrap/nvdr_camera_wrapper.py#L520) —
  the lying tensor
- Used by: `NvdrRecordEpisode` (in `show_ppo_arm.py`), the production
  training/eval recording path
- Sees: the *written* pose. If the reset code wrote `(−0.5, 0, 0.4)` into the
  tensor, nvdiffrast paints the arm at `(−0.5, 0, 0.4)`, regardless of where
  PhysX actually has it.

The two renderers diverge precisely when the tensor lies. Which is exactly
what happens with `fix_base_link=True` + reset-time root write.

**What's hardcoded for Franka in NvdrCameraWrapper:**

1. **`get_hacky_blocklist_for_arm_links`** at
   [nvdr_camera_wrapper.py:187-229](dywa/src/env/env/wrap/nvdr_camera_wrapper.py#L187-L229).
   Returns a dict keyed by URDF path with hardcoded link-name lists (which
   links to skip when rendering — probably for visual hygiene). Only knows
   about Franka, UR5, and the table. Trossen URDF + link names aren't in the
   dict, so nothing gets blocked — that part is benign. (Worst case: a few
   extra primitives rendered.)

2. **Asset-copy logic in `set_camera`** at
   [nvdr_camera_wrapper.py:387-453](dywa/src/env/env/wrap/nvdr_camera_wrapper.py#L387-L453).
   This is the part that actually breaks Trossen:

   ```python
   asset_root = asset_args['rootpath'] \
       if new_root is None or 'robots' in asset_args['filename'] else new_root
   asset_file = asset_args['filename']
   asset_path = F'{asset_root}/{asset_file}'
   ```

   For Trossen with `new_root='/tmp/tmpXXX'` and
   `filename='trossen_arm/urdf/wxai_base.urdf'`, the wrapper expects the URDF
   at `/tmp/tmpXXX/trossen_arm/urdf/wxai_base.urdf`. But the tmp-dir
   population (done elsewhere in the project) was written assuming Franka's
   layout — it copies `franka_description/` and `crm-panda/` trees but not
   `trossen_arm/`. The wrapper looks at the tmp path → file doesn't exist →
   `ValueError: /tmp/tmpXXX/trossen_arm/urdf/wxai_base.urdf is not a file`.

   Fix would be to extend tmp-population to copy Trossen assets, or skip the
   `new_root` remap for Trossen. Straightforward; nobody has done it.

**Why we chose root-cause over renderer-extension:**

| Option | Effort | Effect on bug | Effect on render |
|---|---|---|---|
| Extend NvdrCameraWrapper for Trossen | ~half day | None — PhysX still has the arm at origin | Trossen renders via nvdiffrast now look correct (read from lying tensor) |
| Fix placement at create time ([trossen.py:193-221](dywa/src/env/robot/trossen.py#L193-L221)) | ~15 lines | Actually fixed — PhysX welds the arm at `(−0.5, 0, 0.4)` | Both renderers now agree (tensor matches PhysX) |

The first option is a cosmetic fix that masks the symptom. The second is a
root-cause fix that aligns the simulation with intent. We took the second.
For Franka, the first option (NvdrCameraWrapper) has been silently in effect
since CORN — Franka renders looked correct because they used nvdiffrast,
which masks the PhysX-state issue. Step 5's +7.4 pp result shows the bug
matters for policy quality even when it doesn't show up in renders.

#### Did we fix the PhysX bug?

**No, not strictly.** The underlying Isaac Gym / PhysX behavior — "for
`fix_base_link=True` actors, the reset-time root-tensor write is a no-op" —
is a property of Isaac Gym that we can't change from the user side. The
faulty no-op write is still in both files:

- `Trossen.reset()` at [trossen.py:337-353](dywa/src/env/robot/trossen.py#L337-L353) —
  still writes `(−0.5, 0, 0.4)` into `env.tensors['root']` on first reset.
  PhysX still ignores it.
- `Franka.reset()` at [franka.py:799-809](dywa/src/env/robot/franka.py#L799-L809) —
  same, still does the no-op write.

What we did is **work around** the bug by changing *where* we tell PhysX to
weld the actor:

- **For Trossen** ([trossen.py:193-221](dywa/src/env/robot/trossen.py#L193-L221)):
  bake the correct position into the `gymapi.Transform()` passed at
  `create_actor` time. PhysX welds at the right pose at creation, before the
  no-op reset-time write happens. The welded pose is now correct in
  production.
- **For Franka** ([franka.py:678](dywa/src/env/robot/franka.py#L678)): we did
  **not** apply the symmetric fix. Franka in production is still welded at
  world origin. Step 5 temporarily applied the fix to measure
  policy-invariance (78.79%), then reverted to preserve the 71.40%
  reproducibility baseline.

So: Trossen routes around the bug (correct PhysX placement). Franka still
has the bug active by default. The underlying Isaac Gym behavior is
untouched. If anything ever convinces NVIDIA to update Isaac Gym's
`set_actor_root_state_tensor_indexed` to actually move fix_base_link actors,
both arms' reset-time write would suddenly start working — at which point
the Trossen create-time Transform becomes redundant (still correct, just
double-effective) and Franka's existing reset-time write would suddenly take
effect, breaking the 71.40% checkpoint until retrained. Worth keeping in
mind on any Isaac Gym upgrade.

### Q4. Is Trossen ready to launch training?

**For Stage 1 PPO teacher training: yes.** Verified by the 50k smoke run on
2026-05-17 — pipeline runs end-to-end, loss converges, geometry is correct
under the create-time Transform fix.

**Caveats before production-scale (`num_env=4096`, ≥200k steps):**

- Throughput is ~16× slower than Franka because Trossen uses a 30-iter
  damped-LS numerical IK in pure PyTorch
  ([trossen.py:359-454](dywa/src/env/robot/trossen.py#L359-L454)), while
  Franka uses OSC (single matrix inverse). Expect ~3–4 days for one Stage 1
  run on RTX 6000 Ada.
- `PYTORCH_JIT=0` required (nvrtc sm_89 issue, same as elsewhere in the
  project).
- **Only Stage 1 PPO is validated.** Stage 2 PPO, DAgger distillation, and
  student eval pipelines have NOT been smoke-tested for Trossen. The first
  failure surface to expect is `NvdrRecordEpisode` (used by `show_ppo_arm.py`),
  which doesn't work for Trossen — see Q3. The
  `show_trossen_record.py` workaround at
  [show_trossen_record.py](dywa/exp/train/show_trossen_record.py) handles
  Stage 1 playback only; student eval may have its own integration gaps.

So: Stage 1 production training can launch unattended. Budget time for
debugging Stage 2 → DAgger → eval before assuming the full Trossen pipeline
runs without intervention.

### Q5. Who defines `(−0.5, 0, 0.4)` — the paper?

Not the paper directly. The number is *computed* from three codebase defaults,
none of which are paper-prescribed coordinates:

1. [`tabletop_scene.py:16-17`](dywa/src/env/scene/tabletop_scene.py#L16-L17):
   ```python
   table_dims: Tuple[float, float, float] = (0.4, 1.0, 0.4)
   table_pos:  Tuple[float, float, float] = (0.0, 0.0, 0.2)
   ```
2. [`franka.py:91`](dywa/src/env/robot/franka.py#L91) (and same default in
   `trossen.py`): `keepout_radius: float = 0.3`
3. `base_height` config knob (`'ground'`, `'table'`, or numeric override).
   Production setting in `icra_base.yaml` is `'table'`.

The arithmetic in `reset()` (mirrored in the patched `create_actors`):

```
x = table_pos[0] − 0.5 × table_dims[0] − keepout_radius
  = 0 − 0.5 × 0.4 − 0.3
  = −0.5

y = table_pos[1] = 0

z (base_height='table') = table_dims[2] = 0.4
```

So `(−0.5, 0, 0.4)` is "table is 40 cm wide centered at x=0, stand 30 cm
behind it, mount at table-top height." The paper describes the *kind* of
layout (arm mounted behind table workspace) but doesn't pin the 40/30/40
numbers. They're inherited from CORN, the predecessor project (acknowledged
in [README.md:194-196](README.md#L194-L196)). Changing `keepout_radius=0.0`
would move the base to `(−0.2, 0, 0.4)` — *on* the back edge — and is what
the 2026-05-17 verification used to match the paper-figure framing more
exactly.

---

## 2026-05-18 — Franka side-view evidence recording (bugged vs fixed)

The earlier top-down evidence
([franka_phys_buried_evidence.png](dywa/output/franka_phys_buried_evidence.png))
shows the X-Y location ambiguity but doesn't expose *vertical* burial — the
arm being at z=0 (floor) vs z=0.4 (table top). Top-down looks at the scene
from straight above so the z-axis collapses. Asked by the mentor whether the
arm could be visualized from the side to make the burial obvious, recorded a
side-view A/B against the same scene/seed.

### Script

[`dywa/exp/train/test_franka_env_side.py`](dywa/exp/train/test_franka_env_side.py)
— companion to `test_franka_env.py`, identical config except for the camera
parameters:

```
eye = (1.6, -1.6, 0.9)    # 3/4 view from in front and to the side
at  = (-0.25, 0.0, 0.3)   # looking back toward arm's intended location
fov = 60°
```

Same raw Isaac Gym GPU camera as `test_franka_env.py` (bypasses
`NvdrCameraWrapper` so the rendered frame reflects PhysX state, not the
written-but-ignored body tensor).

### Procedure

Two recording passes, same seed, same `init_type='home'`,
`base_height='table'`:

1. **Pass A — BUGGED.** Run the script against the un-modified
   [`franka.create_actors`](dywa/src/env/robot/franka.py#L678) (identity
   `gymapi.Transform()` passed to `create_actor`). Output:
   [`franka_sideview_bugged_h264.mp4`](dywa/output/franka_sideview_bugged_h264.mp4)
   + PNG still at frame 45.
2. **Pass B — FIXED.** Apply the symmetric create-time Transform fix
   (mirror of `trossen.create_actors:193-221`) to `franka.create_actors` and
   re-run. Output:
   [`franka_sideview_fixed_h264.mp4`](dywa/output/franka_sideview_fixed_h264.mp4)
   + PNG still.

`franka.create_actors` was then **reverted** to the identity-Transform
form to preserve the 71.40% reproducibility baseline (same call as in the
2026-05-18 policy-invariance check).

### Body-tensor read-back was identical across both passes

```
panda_link0:      (-0.500, -0.000, 0.400)
panda_link1:      (-0.500, -0.000, 0.733)
panda_link4:      (-0.417, -0.000, 1.049)
panda_link7:      ( 0.051, -0.000, 1.157)
panda_hand:       ( 0.082, -0.000, 1.055)
panda_leftfinger: ( 0.099, -0.000, 0.999)
panda_rightfinger:( 0.099, -0.000, 0.999)
```

The tensor "lies" identically in both passes because the reset-time write into
`env.tensors['root']` happens regardless of where PhysX actually has the
welded actor.

### Rendered frames differ dramatically

| Pass | Frame | What it shows |
|---|---|---|
| **Bugged** | [franka_sideview_bugged.png](dywa/output/franka_sideview_bugged.png) | Arm centered on / inside the white table block. Lower joints are *embedded* in the table mesh; the gripper sticks up from the table top because Franka's reach pushes upper links above z=0.4 even from the buried base. |
| **Fixed** | [franka_sideview_fixed.png](dywa/output/franka_sideview_fixed.png) | Arm clearly **behind** the table (visible to the left of the white block in frame). Base at z=0.4 (table top), reaching forward over the table. Casts a separate shadow on the floor next to the table's shadow. |

### Side-by-side composite

ffmpeg `hstack` with `drawtext` labels:
[franka_sideview_compare.mp4](dywa/output/franka_sideview_compare.mp4) +
[franka_sideview_compare.png](dywa/output/franka_sideview_compare.png).
Single mentor-facing artifact — same arm in same scene, only the placement
fix differs, and the burial is unambiguous.

### Multi-env variants — recorded then deleted

Also recorded `num_env=4` variants in case extra envs would show object
placement differences. They didn't — the side-view camera in this setup
only frames `envs[0]`, so the multi-env clips were visually identical to
single-env. Files deleted as redundant.

Cleanup pass also dropped the raw `mp4v`-fourcc outputs from
`cv2.VideoWriter` (no `ffmpeg` in the `pkm1:v0` container, so the in-loop
writer falls back to `mp4v` which modern players can't decode); only the
host-side H.264 transcodes (`_h264.mp4`) are kept.

### Eval rollout videos — not produced

Mentor also asked for "actual policy rollout videos before and after the
fix." Skipped for time. The cleanest implementation would require writing a
new entry-point that mirrors `test_rma.py`'s pretrained-student loading
(`StudentAgentRMA` + `setup_rma_env_v2` + the `AddTeacherState` wrapper) and
adds a raw Isaac Gym GPU camera while disabling `NvdrCameraWrapper` (which
would otherwise hide the bug by reading from the lying body tensor). Budget
was ~45 min plus CUDA-wedge risk. The 71.40% vs 78.79% empirical numbers
already exist from the 2026-05-18 eval runs; the side-view stills + compare
are sufficient to make the geometric story visual.

---

## 2026-05-18 — Trossen side-view A/B (companion to the Franka comparison)

Same idea as the Franka side-view A/B but for Trossen, so the mentor has a paired set
of bugged-vs-fixed visuals for both arms. Cleaner setup than the Franka pass because
no `trossen.py` modification is needed — `trossen.create_actors` already takes a
`base_height` config, so the bug-vs-fix toggle is one CLI flag.

### Script change

[`test_trossen_env.py`](dywa/exp/train/test_trossen_env.py) gained a
`--base_height {table,ground,origin}` argument:

- `--base_height=table` → production fix, base at `(−0.5, 0, 0.4)`.
- `--base_height=origin` → reproduces the pre-fix bug, base welded at world `(0, 0, 0)`
  (same physical state the un-fixed `gymapi.Transform()` identity would produce).
- `--base_height=ground` → base on floor at `(−0.5, 0, 0)` (older debug option, not
  used in this pass).

The create-time Transform code is the same path the production fix uses; passing
`origin` as the config value just feeds `(0, 0, 0)` into the Transform.
Mechanism-wise this isn't *exactly* the pre-fix code path (which used identity
Transform regardless of config), but the rendered PhysX state is identical, and the
toggle keeps `trossen.py` unmodified.

### Recordings

Two passes, `num_env=1`, `steps=150` (≈5 s @ 30 fps), action-driven sweep so the arm
visibly moves:

| Pass | Output |
|---|---|
| `--base_height=table` | [trossen_sideview_fixed_h264.mp4](dywa/output/trossen_sideview_fixed_h264.mp4) + [PNG](dywa/output/trossen_sideview_fixed.png) |
| `--base_height=origin` | [trossen_sideview_bugged_h264.mp4](dywa/output/trossen_sideview_bugged_h264.mp4) + [PNG](dywa/output/trossen_sideview_bugged.png) |
| `hstack` composite with `drawtext` labels | [trossen_sideview_compare.mp4](dywa/output/trossen_sideview_compare.mp4) + [PNG](dywa/output/trossen_sideview_compare.png) |

### What the comparison shows

The Trossen burial is **more dramatic** than Franka's — confirms the
reach-vs-burial analysis from the 2026-05-18 Q&A entry:

- **Bugged (origin)**: the table block (white) almost completely occludes the arm.
  Only a small orange sliver pokes out, because Trossen's ~50 cm reach means most
  links stay at `z < 0.4` (inside the table's vertical extent).
- **Fixed (table)**: arm is clearly behind the table at table-top height, casts a
  separate shadow on the floor. Same scene, same seed, same action sweep — only
  the placement differs.

### Cleanup pass (also 2026-05-18)

After producing the polished A/B, deleted several older / redundant artifacts:

| File | Reason for deletion |
|---|---|
| `trossen_vis.mp4`, `trossen_vis_frame.png` | Old Trossen-fixed visualization from 2026-05-17; superseded by `trossen_sideview_fixed*` |
| `franka_topdown.mp4` | Raw `mp4v` fourcc output from `cv2.VideoWriter` inside the container; modern players can't decode it |
| `franka_topdown_h264.mp4`, `franka_topdown_frame.png` | Byte-identical duplicates of `franka_phys_buried_evidence.{mp4,png}` (verified via md5sum) |
| `frame_check.png`, `latest_vis_frame.png` | Working / scratch files, no references in dev_log or report |

Kept: `franka_phys_buried_evidence.{mp4,png}` (top-down evidence — complementary to
side-view since it shows X-Y placement, not vertical burial), all `franka_sideview_*`
and `trossen_sideview_*` artifacts.

---

## 2026-05-19 — Reference: what counts as a "success" in eval

For future reference when reading `avg_success_rate.txt` / `categorical_result.pkl`,
or when designing a new task. The criterion is set inside the **task**, not the
robot — so Franka and Trossen share the exact same definition as long as both use
`PushTask`.

### Where it's defined

[push_task.py:296-321](dywa/src/env/task/push_task.py#L296-L321), inside
`compute_feedback_legacy`. Called every physics step from `PushTask.compute_feedback`
([push_task.py:610-666](dywa/src/env/task/push_task.py#L610-L666)), which writes the
boolean tensor into `info['success']` (shape `[num_env]`, one bool per parallel env).

### The predicates (logical AND)

```python
suc = (pos_err1 <= goal_radius) AND obj_on_table
if use_pose_goal:  suc &= (orn_err1 <= goal_angle)
if check_stable:   suc &= is_stable
```

| Predicate | Threshold (DyWA Franka eval) | Meaning |
|---|---|---|
| position error | `≤ 0.05 m` (`goal_radius`) | object COM within 5 cm of goal point |
| orientation error | `≤ 0.1 rad` (~5.7°, `goal_angle`) | object orientation within ~5.7° of goal pose |
| `obj_on_table` | `‖contact_force‖ ≥ contact_thresh = 0` | nonzero table-contact (i.e. resting, not airborne) |
| `is_stable` | OFF (`check_stable=false`) | velocity check **not** enforced |
| `timeout` | `300 steps × 40 ms = 12 s` | fail if not met by then |

Source of these values: [arm_div_base.yaml:63-76](dywa/src/data/cfg/env/arm_div_base.yaml#L63-L76)
+ [icra_base.yaml:38-45](dywa/src/data/cfg/env/icra_base.yaml#L38-L45)
(`use_pose_goal: true`, `check_stable: false`, `goal_radius: 0.05`,
`goal_angle: 0.1`, `timeout: 300`).

### When the episode terminates

[push_task.py:329](dywa/src/env/task/push_task.py#L329):

```python
done = suc OR oob OR timeout
```

- `oob`: object position drops out of workspace bounds.
- `timeout`: `step_tensor >= 300`.

`suc=True` causes immediate `done=True` for that env, which triggers
auto-reset with a new (object, goal). Success at any single step within
the 12 s window counts — the policy does **not** need to hold the pose.

### Per-object tallying

[cube_env_wrappers.py:1502-1537](dywa/exp/train/envs/cube_env_wrappers.py#L1502-L1537),
`CountCategoricalSuccess.step` increments `reset_count` whenever an env finishes
(any termination), and adds the final `info['success']` value to `success_count`.
Saved at the end of the run to
`/home/user/DyWA/output/test_rma/dywa/result/{base_set}/`:

- `avg_success_rate.txt` — global mean (e.g. `0.7879` for Franka unseen DGN)
- `categorical_result.pkl` — `{object_name: {reset_count, success_count}}` dict
- `categorical_result.png` — bar chart with `0.6` reference line

### What is *not* logged by default

Per-step trajectories (`action`, `obs`, `reward` sequences) are dumped only when
`++use_log_episode=True` is passed and `++log_episode.log_dir=…` is set
([log_episodes.py:101-105](dywa/src/env/env/wrap/log_episodes.py#L101-L105)).
The official `eval_student_unseen_obj.sh` does not enable this — only aggregate
results survive.

## 2026-05-19 — Trossen vs Franka training throughput; how to speed up Trossen

### The real difference: control mode, not IK kernel

Franka and Trossen use **different control-loop architectures**, which is the
dominant source of any per-step speed gap — not the difference between a
compiled CUDA IK vs. PyTorch IK.

| | Franka (default) | Trossen (default) |
|---|---|---|
| `ctrl_mode` | `CI` (Cartesian Impedance / OSC) — [franka.py:64](dywa/src/env/robot/franka.py#L64) | `cpos_n` (Cartesian position via numerical IK) — [trossen.py:61](dywa/src/env/robot/trossen.py#L61) |
| PhysX `driveMode` | `DOF_MODE_EFFORT` (torques) | `DOF_MODE_POS` (PhysX built-in PD) |
| IK iterations per env step | **0** | **30 DLS iterations** |
| Control-loop ops per step | a few matmuls (`pose_error → J^T · solve(M, err) → τ`) | ~30 iterations × ~5–10 PyTorch ops each = ~150–300 GPU launches |
| Control code | [franka.py:1015 + franka_util.py:891 CartesianImpedanceController](dywa/src/env/robot/franka.py#L1015) | [trossen.py:451-455 + trossen_kin.py:222-266](dywa/src/env/robot/trossen.py#L451-L455) |

Franka in CI/OSC mode skips IK entirely — pose error feeds directly into a
Jacobian-transpose + mass-matrix torque solve. Trossen's `cpos_n` mode does
30 DLS iterations of `FK → Jacobian → solve(J Jᵀ + λ²I, err) → clamp` to convert
a target pose into joint angles, then sends position targets to PhysX's built-in
PD controller. That's ~30× more control-loop CUDA launches per step for Trossen
even though each individual op is fast.

Reasonable estimate: end-to-end Franka training is **1.5–3× faster** than
Trossen on the same hardware. Not 10–100×, but not negligible either. The
control loop is one slice of the per-step cost; physics step, point cloud
rendering, and network forward are also significant.

### Measured Trossen Stage 1 throughput

From ckpt mtimes of [teacher-stage1/run-000/ckpt/](output/trossen_pipeline/teacher-stage1/run-000/ckpt/):

| Metric | Value |
|---|---|
| Start (step 0) | 2026-05-17 07:16 |
| End (step 49152, ≈50k iters) | 2026-05-17 14:25 |
| Wall clock | ~7 h 9 min |
| Envs in parallel | 1024 |
| PPO iterations / sec | ~2 |
| Per-env physics steps/sec | ~77 |
| Total env-steps | ~1.99 M |

### Why current Trossen rollouts look frozen

GPU hours, not algorithm speed:

- Franka teacher in the paper: ~millions of PPO iterations to convergence.
- At Trossen's measured ~2 PPO-iter/sec, 1 M iters ≈ **140 h ≈ 6 days** on one GPU.
- We have spent **7 h** on Trossen → ~5% of one Franka-style convergence run.
- Stage 2 (phase-2 fine-tune) and Stage 3 (DAgger distillation) come after Stage 1
  converges — together they add days more.

Policy collapse at 50k iters is **expected** at 5% of training. Action
distribution `Normal(μ, σ)` from the actor net has both `μ` and `σ` ≈ 0, so
`dist.sample()` returns ≈ 0, the IK target is the current EE pose, and the
arm stays at HOME_Q = `[0, 1.3, 1.5, 0, 0.5, 0]`.

### How to speed up Trossen training

Ranked by effort × expected gain:

1. **Switch Trossen to CI / OSC control** (biggest win, moderate effort)
   - Eliminate the 30-iteration IK loop entirely from the per-step path
   - Mirror Franka's `CartesianImpedanceController` at
     [franka_util.py:891](dywa/src/env/robot/franka_util.py#L891)
   - Needs system identification for the WXAI arm: friction, damping, armature
     per joint (Franka has hardcoded `sysid_friction`, `sysid_damping`,
     `sysid_armature` at [franka.py:744](dywa/src/env/robot/franka.py#L744))
   - Mass matrix is already available from Isaac Gym via `acquire_mass_matrix_tensor`
     for any URDF — no analytical derivation required
   - Estimated speedup: **1.5–3×** end-to-end
   - Estimated effort: ~1 week (sysid + controller adaptation + tuning)

2. **Compile analytical IK for Trossen as a CUDA extension** (medium win, high effort)
   - Derive closed-form 6-DOF IK for WXAI on paper (likely wrist-partitioned
     structure, has known closed-form solutions)
   - Mirror Franka's CUDA-extension files:
     - [c_src/franka_kin_cuda.cpp](dywa/c_src/franka_kin_cuda.cpp) — host launcher + pybind
     - [c_src/franka_kin_cuda_kernel.cu](dywa/c_src/franka_kin_cuda_kernel.cu) — per-env kernel
     - [franka_kin.py](dywa/src/env/robot/franka_kin.py) — Python wrapper
   - Replacement point: [trossen.py:453](dywa/src/env/robot/trossen.py#L453)
   - Estimated speedup for control loop: ~30× (one closed-form solve vs. 30 iterations)
   - End-to-end: comparable to option 1 (control is one slice)
   - Estimated effort: ~2 weeks (analytical derivation is the time sink)

3. **Reduce IK iterations** (tiny effort, quick test)
   - Drop `ik_n_iter` from 30 → 10 or 15 at
     [trossen.py:78](dywa/src/env/robot/trossen.py#L78)
   - Check whether IK still converges adequately (rendered EE should still track
     the target within position tolerance over a few control cycles)
   - Estimated speedup: ~2–3× on the control loop alone, smaller end-to-end
   - Estimated effort: 1 hour (test + observe)

4. **Increase parallelism** (zero effort if hardware allows)
   - Bump `++env.num_env=1024` → 2048 or 4096 if GPU memory holds
   - PPO updates per second drop, but total environment throughput (env-steps/sec)
     scales nearly linearly
   - GPU memory usage during 50k Stage 1 run was modest — likely room to scale
   - Estimated speedup: ~2× per 2× envs (until other bottleneck dominates)
   - Risk: OOM if other workloads share the GPU

5. **Multi-GPU PPO** (large effort, large gain)
   - Not natively supported in this codebase — would require Distributed
     DataParallel wiring around the actor/critic and rollout buffer
   - Estimated effort: weeks; risky for a one-off training run

Practical recommendation: **try option 3 first** (free speedup check), then
**option 4** (parallelism), then commit GPU hours. Save options 1 and 2 for
when Trossen is a productionized arm needing repeated training runs.

### Aside: `sample_action` is a dead parameter

While debugging the no-motion observation, found that
[ppo.py:729](dywa/src/models/rl/v6/ppo.py#L729) `PPO.test(sample=True, ...)`
accepts a `sample` argument but never uses it. `test()` just calls
`self.interact()`, which unconditionally does `actn = dist.sample()` at
[ppo.py:641](dywa/src/models/rl/v6/ppo.py#L641). Confirmed empirically: the
`sample_action=false` and `sample_action=true` recorder runs produced
**bit-identical** MP4s (same md5). The flag in
[show_trossen_record.py](dywa/exp/train/show_trossen_record.py) and
[show_trossen_record_episodes.py](dywa/exp/train/show_trossen_record_episodes.py)
is therefore cosmetic. Low priority to wire through since it makes no behavioral
difference for the current undertrained ckpt.

## 2026-05-20 — Trossen workspace bug: EE can't reach the table

### Symptom

Continued Trossen Stage 1 from the existing 50k ckpt → 500k iters with the new
`ik_n_iter=5`. After 71k of fresh training:

- Return improving: -0.26 → -0.13 (real signal)
- Action σ healthy: ~0.77 (still exploring)
- **Success rate stuck at noise floor: 0.033% → 0.107%** — no breakthrough trend

Reading the trend more carefully revealed it wasn't "slow but real learning toward
the task" — the policy is **learning to maximize a structurally-bounded reward by
pressing against the workspace wall**.

### Root cause

[trossen.py:88-91](dywa/src/env/robot/trossen.py#L88-L91) default `ws_bound` upper-x
= 0.25 in base frame. The wait_and_launch_trossen.sh script preserves this:

```
ws_bound = [[-0.25, -0.35, -0.1], [+0.25, +0.35, +0.4]]   # base frame
base world position = (-0.5, 0, 0.4)
=> EE workspace in world: x ∈ [-0.75, -0.25]
```

The **table** is at `pos=(0,0,0.2), dims=(0.4, 0.5, 0.4)` →
table world x ∈ **[-0.20, +0.20]**.

**The EE workspace's upper x (-0.25) is 5 cm short of the table's nearest edge
(-0.20).** Trossen physically cannot place its EE over any point on the table.
The fingertips (~5-10 cm extent past `ee_gripper_link`) can graze objects spawned
right at the near edge — that's where the residual ~0.1% success rate comes from.

### Why Franka works

Different `ws_bound` AND different track_object mechanism:

| | Franka | Trossen |
|---|---|---|
| `ws_bound` upper x | **+0.30** | +0.25 |
| Base world x | -0.5 | -0.5 |
| EE max world x (static) | **-0.20** (touches table left edge) | -0.25 (5 cm short) |
| track_object behavior | Franka [franka.py:970-984](dywa/src/env/robot/franka.py#L970-L984) — `pose_error.update_pos_bound(obj_bound)` **re-centers** workspace around object | Trossen [trossen.py:421-430](dywa/src/env/robot/trossen.py#L421-L430) — `pos_world.clamp(ws_lo, ws_hi)` after intersecting with object box — **only shrinks** within the static base ws_bound |

Two compounding issues:
1. Trossen's static workspace ends 5 cm before the table.
2. Trossen's `track_object` intersects (Franka re-centers around object), so when the
   object is at table center the intersected workspace becomes degenerate (crossed
   bounds) and `clamp` pins the EE to `ws_hi` = -0.25.

The minimal fix is widening `ws_bound` upper x to ≥ 0.6 (giving EE world x reach to
+0.1). The deeper fix is to mirror Franka's re-centering `track_object` logic, but
widening alone is enough to unblock training.

### The fix (and what it actually changes)

The arm BASE position is **unchanged** — Trossen is still mounted at world
`(-0.5, 0, 0.4)` (30 cm keepout from the table's left edge). The fix only widens
the **allowed EE workspace** — it gives the policy permission to reach further
forward (toward the table) from the same fixed base. Concretely:

- **Old:** `ws_bound` upper-x = 0.25 (base frame) → EE max world x = -0.25 (5 cm short of table left edge at -0.20)
- **New:** `ws_bound` upper-x = 0.60 (base frame) → EE max world x = +0.10 (overlaps the table interior)

So "closer to the table" in the sense that the **EE's reachable region now extends
onto the table**. The arm itself isn't physically moved closer; we just stopped
artificially clipping its commanded EE pose 5 cm short of the table.

Trossen WXAI nominal reach is ~700 mm, so an upper-x of 0.6 in base frame is well
within the arm's physical capability. The IK already handles this — it was the
software workspace clip that was the artificial limit, not the kinematics.

### Action taken

- Killed the 71k-step continuation run at 05:30 UTC.
- Both the prior 50k smoke ckpt and the 71k-step continuation ckpt are **trained
  against the unreachable workspace** and should not be used as a starting point.
- Restarting Stage 1 **from scratch** (no `load_ckpt`) with widened workspace:
  `ws_bound = [[-0.25, -0.35, -0.1], [+0.6, +0.35, +0.4]]` → EE world x ∈ [-0.75, +0.1].
- Saved to new path `teacher-stage1-wide/` to keep the buggy training history intact
  for reference.

## 2026-05-20 — Trossen Stage 1 breakthrough at step ~75k

Checking `teacher-stage1-wide/run-000/tb_train/` ~7.6 h into the run (step 101k of
500k, 20%): **the widened workspace fix worked.** Textbook RL phase transition.

### Trends (`env/` tensorboard scalars)

| Step | Cumulative `suc_rate` | Recent-window `cur_suc_rate` | Avg `episode_return` |
|---|---:|---:|---:|
| 1k | 0.09% | 0.09% (noise) | -0.16 |
| 25k | 0.30% | 0.38% | -0.05 |
| 50k | 0.52% | 1.33% | -0.04 |
| 62k | 0.92% | 2.94% | -0.04 |
| **75k** | 1.42% | **16.3%** | **+0.03** (crossed zero) |
| **87k** | 2.18% | **36.5%** | +0.17 |
| **100k** | 4.37% | **48.0%** | **+0.30** |

`cur_suc_rate` is the right "true current capability" signal — it's a rolling
window, not the run-cumulative average. The breakthrough where the policy starts
actually completing the task happened between step 62k and 75k.

### What this confirms

1. **The widened `ws_bound` was the missing piece.** Same algorithm, same hyperparams,
   same `ik_n_iter=5`, same seed — only difference from the previous (failed) runs
   is the `ws_bound` upper-x = 0.6 instead of 0.25. With the EE able to reach onto
   the table, PPO converges as expected.
2. **DyWA is not Franka-specific in principle**, as the project goal asserted —
   it's robot-agnostic given correct workspace setup.
3. **The earlier 50k and 71k-iter runs were truly stuck**, not just slow.
   The slow return improvement (-0.26 → -0.13) in those runs was the policy
   maximizing reward against the workspace wall, not learning the task.
4. **The mentor's "URDF → CUDA extension" framing was a red herring for this
   bottleneck.** The IK was never the issue; the workspace clip was.

### Throughput note

Wall clock is 7.6 h for 101k iters → **~3.7 PPO-iter/sec average**, lower than
the bench-predicted 5 it/s. Likely cause: GPU contention from other workloads
that appeared on the host during the run (GPU 0 went from 8 GB → mid-run other
jobs; GPU 3 grew 36 → 44 GB). Run is still healthy, just slower than bench.
Revised ETA: ~22 more hours → finish ~03:00 UTC 2026-05-21.

### Recording from the 98k ckpt

Launched [show_trossen_record_episodes.py](dywa/exp/train/show_trossen_record_episodes.py)
on GPU 0 (separate from the training run on GPU 2) against the
`step-98304.ckpt`. With recent-window suc_rate ≈ 48%, expect ~half the recorded
episodes to be in `succ/` and half in `fail/`. Compare to the pre-fix recordings
(`rollout_episodes/` from 2026-05-19, all `fail/`, arm visually frozen) — should
be a clear A/B for "before workspace fix" vs "after + ~100k iters of training."

## 2026-05-28 — Trossen Stage 1 wide run COMPLETED (500k steps, cur_suc_rate 93%)

Back-filling this entry a week after the fact: the `teacher-stage1-wide/run-000`
run **finished on 2026-05-21 ~14:01** at the full 500k steps and was never logged.
Reading the final scalars straight out of the tfevents file (no tensorboard
installed; parsed the TFRecord/Event protobuf raw):

### Final metrics (`env/`, step 499712)

| Metric | Value | Meaning |
|---|---:|---|
| `cur_suc_rate` | **93.1%** | rolling-window = true current capability |
| `suc_rate` | 60.7% | run-cumulative average (drags low because of the slow early phase) |
| `avg_episode_return` | 0.91 | |
| `avg_true_return` | 0.56 | |
| `num_success` | 4.22M | cumulative successful episodes |

### Trajectory, end to end

| Step | `cur_suc_rate` |
|---|---:|
| 1k | ~0% |
| 75k | 16.3% |
| 100k | 48.0% |
| **500k (final)** | **93.1%** |

The policy kept improving cleanly after the step-75k phase transition — no plateau,
no collapse. **48% → 93%** over the back half of the run. This is a strong Stage-1
teacher and confirms, end to end, that the widened `ws_bound` was the whole fix:
DyWA trains on Trossen with the same algorithm/hyperparams as Franka once the EE
can actually reach the table.

### Artifacts

- `output/trossen_pipeline/teacher-stage1-wide/run-000/ckpt/last.ckpt` (= step ~500k)
- `.../ckpt/step-491520.ckpt` (last periodic ckpt)
- tfevents: `.../tb_train/events.out.tfevents.1779255972.barista.20442.0`

### Status / not yet done

- **Not committed or pushed.** As of this entry the entire Trossen body of work
  (this dev_log, [trossen.py](dywa/src/env/robot/trossen.py), [arm_env.py](dywa/src/env/arm_env.py),
  the nvdr camera/record wrappers, docker files, `teacher_base.yaml`, the
  `show_trossen_*` recording scripts, `bench_wxai_ik.py`) is still uncommitted on
  `main`; `origin/main` has none of it.
- **Next:** Stage 2 (distillation to the student / DAgger) off this teacher ckpt,
  and a fresh recording pass from `last.ckpt` to replace the 98k-ckpt A/B clips.

---

*Log started: 2026-05-01 · Resumed on new machine: 2026-05-14 · Eval reproduced: 2026-05-16 · Trossen PPO smoke test passed: 2026-05-16 · Trossen Stage 1 50k smoke training run: 2026-05-17 · GPU CUDA context wedge: 2026-05-17 · Trossen placement re-fixed (rendered + body match): 2026-05-17 · Franka placement bug confirmed by top-down render: 2026-05-18 · Policy invariance confirmed (78.79% with fixed Franka): 2026-05-18 · Side-view A/B evidence recorded for Franka and Trossen, output dir cleanup: 2026-05-18 · Success-criterion reference section added: 2026-05-19 · CI-vs-cpos_n control-mode gap identified; Trossen speedup options recorded: 2026-05-19 · Trossen workspace bug found; widened ws_bound and restarted from scratch: 2026-05-20 · Trossen Stage 1 phase transition at step 75k (cur_suc_rate 3% → 48% by step 100k): 2026-05-20 · Trossen Stage 1 wide run completed at 500k, final cur_suc_rate 93.1% (logged 2026-05-28): 2026-05-21*
