# DyWA — Learning Notes

Personal study notes on the DyWA paper and codebase. Not for development — for understanding
the architecture, the ideas it borrows, and why each piece exists.

**Paper:** *DyWA: Dynamics-adaptive World Action Model for Generalizable Non-prehensile Manipulation*
Lyu et al., arXiv [2503.16806](https://arxiv.org/abs/2503.16806) (2025)
**Project page:** <https://pku-epic.github.io/DyWA/>
**Authors:** PKU EPIC + Galbot (the same group that did CORN)

---

## 1. The problem

**Non-prehensile manipulation** = manipulation without grasping. A robot pushing, poking,
nudging an object — the object isn't held, so contact is intermittent and the dynamics
depend heavily on physics (mass, friction, inertia, shape). Compare to grasping where once
you've closed the gripper, the object moves rigidly with the hand.

**Generalization target:** unseen object shapes, unknown physical parameters, and (in the
hardest setting) only a **single-view** depth/point-cloud observation.

The headline result on the DexGraspNet benchmark (Table 1 of paper, success rate %):

| Observation | Seen objects | Unseen objects |
|---|---|---|
| Known state, 3-view | 87.9 | 85.0 |
| Unknown state, 3-view | 85.8 | 82.3 |
| Unknown state, **1-view** | 82.2 | **75.0** |

The 1-view-unknown-state-unseen-objects regime (last cell) is what most real robots
face — you have one depth camera, you don't know the object's mass, and the object isn't
something you trained on. **75% success on that** is the contribution. Baselines on the
same row: HACMan 2.9%, CORN 29.8%, CORN-PN++ 49.4%.

---

## 2. Background concepts the paper uses

You need a few pieces of background to understand the architecture choices: teacher-student
distillation, RMA, FiLM, DAgger, and PPO.

### 2.1 Teacher-student distillation (sim-to-real / privileged learning)

A standard trick: it's hard to train a policy from sparse, noisy observations. So you:

1. **Teacher:** train a policy in simulation with privileged info you can't get in the real
   world — e.g. exact object pose, friction coefficients, mass.
2. **Student:** train a second policy that takes only realistic observations (camera, joint
   encoders) and tries to mimic the teacher's actions on the same states.

The student inherits the teacher's competence but learns to operate without privileged info.

### 2.2 RMA — Rapid Motor Adaptation (Kumar et al. 2021)

A specific form of teacher-student distillation designed for **dynamics adaptation**. The
key insight: a policy that knows the dynamics (mass, friction, terrain) can act optimally;
a policy without that knowledge can sometimes *infer* the dynamics from a short history of
observations.

RMA structure:
- Teacher trained with explicit dynamics parameters as input.
- Student replaces the dynamics input with an **adapter module** that estimates dynamics
  from the last ~50 observation-action pairs.
- Student-adapter is trained via supervised regression against the teacher's dynamics
  latent.

DyWA inherits this idea but for manipulation rather than locomotion.

### 2.3 FiLM — Feature-wise Linear Modulation

FiLM is a way to inject conditioning information into a network. Given input features `x`
and a conditioning vector `z`, instead of concatenating `[x, z]`, you do:

```
γ, β = MLP(z)
x' = γ * x + β    # per-feature scale and shift
```

The conditioning `z` modulates `x` multiplicatively. This is more parameter-efficient than
concatenation and often empirically stronger for "context shapes behavior" use cases —
which is exactly what we want for "dynamics shape policy."

### 2.4 DAgger (Dataset Aggregation)

The distillation algorithm. Naive imitation learning suffers from distribution shift —
once the student makes a small mistake, it lands in states the teacher never demonstrated,
and errors compound.

DAgger fix:
1. Let the **student** drive the env (collect on-policy states).
2. At each visited state, ask the **teacher** what action it would take.
3. Add `(state, teacher_action)` to the training buffer.
4. Train the student on the aggregated buffer.

This keeps the student supervised on states it actually visits. Critical for distillation
because the student's observation distribution drifts away from the teacher's as training
progresses.

### 2.5 PPO — Proximal Policy Optimization

How the teacher is trained. Standard model-free RL. Not novel to this paper but mentioned
because it's the optimizer for Stage 1.

---

## 3. Problem framing in DyWA

The agent controls a Franka arm (7-DOF) in IsaacGym. The task: **6D object rearrangement via
non-prehensile manipulation** — pushing, flipping, etc., to move an object on the table to a
target 6D pose without grasping it. Each episode:

- An object is placed on a table with randomized starting pose and randomized physics
  (object mass, scale, friction; object/table/gripper restitution). The paper doesn't
  publish exact ranges; the code config has them — assume 5–10 cm scale and 0.1–0.5 kg mass
  if reading the code, but treat these as code-specific, not paper-quoted.
- Objects: **323 meshes from DexGraspNet** for training; 10 unseen objects × 5 scales = 50
  evaluation objects.
- The robot gets observations and outputs Cartesian-impedance commands (20-D action: 6D
  EE pose delta + 7D per-joint positional gains + 7D per-joint damping factors). No gripper
  dim — non-prehensile means the gripper isn't actively commanded.
- **Success** = object's final pose within **0.05 m and 0.1 rad** of the target pose.
- Episode ends on success or failure (OOB / timeout).

**Privileged state (teacher only):** ground-truth object geometry (full point cloud) and
physics parameters. The teacher's encoder produces both a geometry embedding `f^Geo` and a
physics embedding `f^Phy` that the student must regress (see §5.5).

**Realistic observations (student) — paper version:**
- Single-view partial point cloud (512 × 3)
- Joint state (Franka 7-DOF position + velocity)
- End-effector pose (SE(3))
- Goal: target point cloud (initial obs transformed to the target 6D pose)
- 5-step history of (observation, action) for dynamics inference

**Code-side observation breakdown** (slightly different keys in the YAML — useful when
reading the repo): `hand_state` (9-D), `robot_state` (14-D), `previous_action` (20-D),
`abs_goal` (9-D absolute) or `rel_goal` (9-D relative). These are the code's representation
of the same underlying observation set.

---

## 4. Training pipeline

The paper describes **two stages**: teacher PPO, then student DAgger distillation. The
repo's `phase-2` fine-tuning is an extra code-side refinement not described in the paper.

### Stage 1 — Teacher PPO training (paper)

| | |
|---|---|
| Code | `dywa/exp/train/train_ppo_arm.py` |
| Config | `+env=icra_base +run=icra_ours_abs_rel` |
| Envs | 4,096 in parallel |
| Steps | **200K iterations** (paper); the code is configured for more |
| Learning rate | 3e-4, PPO clip 0.3 |

The teacher policy sees the full privileged state (geometry + physics).

**Paper's teacher architecture** (Table 8): a simplified PointNet++ with one grouping layer
(16 keypoints, K=32 neighbors, [128] grouped features), followed by an MLP policy head.
The teacher produces `f^Geo` and `f^Phy` features that supervise the student's adaptation
latent `zₜ`.

**Code-side teacher architecture** (what the repo actually loads): a different design that
uses CORN's pre-trained ICP encoder with a cross-attention aggregator. Probably an
implementation choice for compatibility with the CORN codebase rather than a paper-quoted
design.

```
   point cloud  ───►  CORN's ICP encoder (frozen, pre-trained — code only)
                          │
                          ▼
                     [512-D point tokens]
                          │
   state vector  ───►  cross-attention   (16 query tokens query the point tokens,
   (hand_state,           │               conditioned on state including phys_params)
    rel_goal,             ▼
    previous_action,  [pooled feature]
    phys_params,          │
    robot_state)          ▼
                     fuser MLP [512]
                          │
                          ▼
                     PPO policy head ──► 20-D action
```

[icra_ours_abs_rel.yaml](dywa/src/data/cfg/run/icra_ours_abs_rel.yaml):

```yaml
icp_obs:
  icp:
    ckpt: 'corn/col-pre:512-32-balanced-SAM-wd-5e-05-920'  # CORN's encoder

net.state.feature.icp_emb:
  query_keys: ['rel_goal', 'previous_action', 'robot_state', 'phys_params']
  num_query: 16
  ctx_dim: 48  # 9 + 20 + 14 + 5  (rel_goal + prev_action + robot + phys)
```

Key point: **`phys_params` is a teacher-only feature**. The student never sees it directly.

### Stage 1.5 — Phase-2 teacher fine-tuning (code only, not in paper)

| | |
|---|---|
| Same config + | `++is_phase2=true ++agent.train.lr=2e-6` |
| Envs | 2,048 |
| Steps | ~50–100k |

Lower learning rate, refined target. Stage 1 makes a competent teacher; phase-2 makes a
**smoother, more deterministic** teacher — easier to distill. **Not described in the paper.**

### Stage 2 — Student DAgger distillation (paper)

| | |
|---|---|
| Code | `dywa/exp/train/train_rma.py` |
| Config | `+env=abs_goal_1view +run=teacher_base +student=dywa/base` |
| Envs | **1,024** (paper) — the code config uses 256 |
| Steps | **500K iterations** (paper) |
| Learning rate | 6e-4, Adam |

The student sees only point clouds + proprio. Trained to match teacher actions on
on-policy states (DAgger), with two auxiliary losses (see §5.5).

This is where the **novel architecture** lives.

---

## 5. The student architecture (the novel bit)

Source: [dywa/base.yaml](dywa/src/data/cfg/student/dywa/base.yaml)

```
   point cloud  ──►  simplified PointNet++ (paper Table 8)
   (single-view       Layer 1: 64 keypoints (FPS), K=32 KNN, group MLP → 32-D
    512×3)            Layer 2: 16 keypoints (FPS), K=32 KNN, group MLP → 128-D
                          │
                          ▼
                     16 patch tokens × 128-D
                          │
                          ▼
                     MLP encoder (residual)
                          │
                          ▼
                     [encoded vision feature]   ◄── auxiliary heads attach here:
                          │                          • world model head → Sₜ₊₁ (next obj pose)
                          │                          • zₜ regression → teacher (f^Geo, f^Phy)
                          │
   proprio       ───►  history buffer (last 5 timesteps)
   (joint state,           │
    EE pose,               ▼
    goal,             Conv1d + MaxPool over time (128 channels)
    previous_action)       │
                          ▼
                     MLP aggregator
                          │
                          ▼
                     [dynamics embedding zₜ]
                          │
                          ▼
                     FiLM-conditioned MLP decoder (3 FiLM blocks, early layers)
                     (zₜ modulates action features via γ, β)
                          │
                          ▼
                     action (20-D)
```

The five ingredients that make this work:

### 5.1 Point-cloud encoder

**Paper version** (Table 8): a simplified PointNet++ trained from scratch as part of the
student. Two grouping layers — 64 keypoints then 16 keypoints, K=32 neighbors each, grouped
feature dims [32, 128]. This is the student's actual visual encoder per the paper.

**Code version**: the repo also loads CORN's pre-trained ICP encoder
(`corn-public:512-32-balanced-SAM-wd-5e-05-920` from HuggingFace) for the teacher and
possibly as an auxiliary input on the student side. The downloaded weights are not the
"student visual backbone" the paper describes — they're CORN's contact-pretrained features
re-used by the teacher pipeline. Worth verifying in code which encoder actually feeds the
student's policy head.

### 5.2 PointNet-style patch tokenization

Rather than feeding the whole point cloud through a single network, the paper's encoder uses
**two PointNet++ grouping layers**:
- Layer 1: 64 keypoints chosen by farthest-point sampling, each grouping K=32 KNN neighbors,
  per-patch MLP produces a 32-D feature.
- Layer 2: 16 keypoints chosen from layer 1's output, K=32 neighbors, per-patch MLP produces
  a 128-D feature.

Final output: 16 patch tokens × 128-D. Hierarchical PointNet++-style — good inductive bias
for local geometry.

### 5.3 5-step temporal history with 1-D conv

The student maintains a buffer of the last 5 (proprioception, action) tuples. A 1-D conv
aggregates them along the time axis. This is **how the student implicitly infers dynamics**:
the recent history of state-action pairs constrains what mass / friction / inertia the
object plausibly has, even though those parameters are never observed directly.

This is the RMA idea in compressed form: instead of a separate "adapter network" predicting
dynamics latents, DyWA bakes the temporal aggregation into the main encoder.

### 5.4 FiLM-conditioned decoder

The dynamics embedding `zₜ` (from §5.3's history conv) modulates the action MLP's hidden
features via FiLM. Per Table 8, **3 FiLM blocks** are placed densely in the **early layers**
of the world action model; final layers are left unconditioned. Each block is two shallow
MLPs that take `zₜ` and emit γ and β:

```yaml
decoder:
  decoder_type: film
  decoder_mlp: [2368]
  film_pred_scale: True
```

Why FiLM and not concatenation? When the model needs to act differently in different
dynamic regimes (heavy object vs light, sticky vs slippery), multiplicative gating is a
much sharper inductive bias for "context changes behavior" than concatenation.

This is the "dynamics-adaptive" part of the name made concrete.

### 5.5 Two auxiliary losses on the encoder

The paper frames these as a **one-step world model** plus an **adaptation-latent regression**
— together they're what makes the model "world-aware" and "dynamics-adaptive".

- **World model head — next task state prediction.** An auxiliary head off the encoded
  feature predicts the *next* task state Sₜ₊₁ = (Tₜ₊₁ ∈ ℝ³, Rₜ₊₁ ∈ SO(3)). Rotation is
  represented as a 9-D matrix flattened. Losses: L2 on translation, L1 on rotation.
  This is the head wired up in the config as `vision_pose_predictor`:
  ```yaml
  vision_pose_predictor:
    xyz_mlp_states: [128, 64]
    rot_mlp_states: [128, 64]
  ```
  ⚠️ The config name suggests "predicts current pose," but per the paper it predicts the
  next step. Worth verifying in code which target it's actually trained against.
- **Adaptation-latent regression (`zₜ`).** The student's history-aggregated representation
  is supervised to match the teacher's concatenated geometry + physics features:
  `‖zₜ − concat(f^Geo, f^Phy)‖²`. This is the RMA-style adapter loss, implemented in the
  config as the `constraint` block:
  ```yaml
  constraint:
    loss_coef: 1
    margin: 1
    loss_type: contrastive
    dim: 128
  ```
  ⚠️ The config calls it `contrastive`, but the paper's equation is an L2 regression
  against teacher features, not a margin-based contrastive loss. Either the code uses a
  contrastive variant of the same idea, or the config name is legacy. Verify in code.

Together these are the "World" (one-step prediction of Sₜ₊₁) and the "dynamics-adaptive"
(zₜ regression) parts of *Dynamics-adaptive World Action Model*.

---

## 6. Why the name parses

**Dynamics-adaptive** → 5-step history conv produces `zₜ`, supervised to regress the
teacher's geometry+physics features. FiLM modulation then lets the policy condition its
behavior on `zₜ`. Together: online dynamics inference + dynamics-conditioned action head.

**World** → one-step prediction of next task state Sₜ₊₁. The encoder must retain enough
world information to forecast where the object ends up after this action.

**Action Model** → the FiLM-modulated MLP that outputs the 20-D Cartesian-impedance action.

So the full name parses as: an action model that jointly forecasts the next world state and
adapts its behavior to online-inferred dynamics. The contribution over plain CORN is the
**FiLM modulation + history conv + world-model head + adaptation latent**; the contribution
over plain RMA is replacing the standalone adapter with a temporal-conv + FiLM mechanism and
adding the one-step world model head.

---

## 7. Inference-time details worth knowing

### 7.1 The `add_teacher_state` thing

At eval time, the option `++add_teacher_state=1` appends the teacher's GRU hidden state to
the student's observation. This is from the RMA paper's eval protocol — it lets you compare
"student with privileged adapter signal" vs "student inferring dynamics on its own."

**Gotcha:** if you pass `++load_ckpt=<some teacher ckpt>` whose hidden-state distribution
differs from what the student saw at distillation time, the student is in OOD territory
and outputs high-variance actions → instant OOB → 0% success. The official eval omits
`++load_ckpt` for this reason. (Confirmed empirically — see [dev_log.md](dev_log.md).)

### 7.2 What action does the policy actually output

20 dimensions, per the paper (Table 10, action = `ΔTₑₑ ∈ SE(3), P ∈ ℝ⁷, ρ ∈ ℝ⁷`):

| Dims | Meaning |
|---|---|
| 0–2 | Δposition (xyz) of the EE |
| 3–5 | Δrotation (3-D, axis-angle or equivalent SE(3) parameterization) |
| 6–12 | P — per-joint positional impedance gains (7-D) |
| 13–19 | ρ — per-joint damping factors (7-D) |

No gripper command: non-prehensile = the gripper is closed/fixed and the policy never
modulates it. The variable impedance is the per-timestep compliance, applied per joint
(7-DOF Franka). This matters for pushing — heavy/sticky objects want stiffer joints,
light/slippery objects want compliant joints.

### 7.3 Hyper-scale of training

Paper numbers:

| Stage | num_env | Steps |
|---|---|---|
| Teacher PPO | 4,096 | 200K |
| Student DAgger | 1,024 | 500K |

Code-side extras not in the paper:

| Stage | num_env | Steps |
|---|---|---|
| Phase-2 teacher fine-tune | 2,048 | ~50–100k |
| Repo student config | 256 | configurable |

Rough wall-clock on a single RTX 4090: order of 1–4 GPU-days end-to-end (paper doesn't
publish wall-clock numbers; these are extrapolations from the repo). With 3× RTX 6000 Ada,
Stage 1 can be sharded and total wall-clock should drop substantially.

---

## 8. Related work to skim if you want broader context

- **CORN** (Lyu et al. 2024): the direct predecessor by the same group. Same teacher
  architecture, no FiLM, no history conv, no auxiliary heads. <https://sites.google.com/view/contact-non-prehensile>
- **RMA** (Kumar et al. 2021): the teacher-student dynamics-adaptation framework. Originally
  for quadruped locomotion. <https://arxiv.org/abs/2107.04034>
- **FiLM** (Perez et al. 2018): the feature modulation method. <https://arxiv.org/abs/1709.07871>
- **DAgger** (Ross et al. 2011): the imitation learning algorithm. <https://arxiv.org/abs/1011.0686>
- **PointNet / PointNet++** (Qi et al. 2017): the point cloud encoder family the patch
  tokenizer is built on. <https://arxiv.org/abs/1612.00593>

---

## 9. Code reading order if you want to dig in

1. **[README.md](README.md)** — high-level pipeline.
2. **[dywa/exp/train/train_ppo_arm.py](dywa/exp/train/train_ppo_arm.py)** — teacher training entrypoint. PPO loop.
3. **[dywa/exp/train/train_rma.py](dywa/exp/train/train_rma.py)** — student DAgger training entrypoint. Look for `cfg.train_step` outer loop.
4. **[dywa/exp/train/test_rma.py](dywa/exp/train/test_rma.py)** — student evaluation. Look for `AddTeacherState` wrapper.
5. **[dywa/src/data/cfg/student/dywa/base.yaml](dywa/src/data/cfg/student/dywa/base.yaml)** — the student architecture config (concrete numbers).
6. **[dywa/src/data/cfg/run/icra_ours_abs_rel.yaml](dywa/src/data/cfg/run/icra_ours_abs_rel.yaml)** — teacher / shared architecture.
7. **[dywa/src/models/rl/net/cross.py](dywa/src/models/rl/net/cross.py)** — the cross-attention aggregator.
8. **[dywa/src/models/rl/net/icp.py](dywa/src/models/rl/net/icp.py)** + **icp_v2.py** — how the CORN encoder is loaded and queried.
9. **[dywa/src/env/push_env.py](dywa/src/env/push_env.py)** — the simulation environment definition.

---

## 10. Open questions

### Resolved after reading the paper
- **Auxiliary loss structure** — not contrastive. The paper specifies `L = L_imitation +
  L_world + L_adapt`, with `L_world = L2(translation) + L1(rotation 9-D)` and
  `L_adapt = L2(zₜ, concat(f^Geo, f^Phy))`. Equal weighting, no contrastive component in the
  paper's equations.
- **FiLM placement** — 3 FiLM blocks, densely in early layers, final layers unconditioned
  (Table 8). Each block uses two shallow MLPs that emit per-feature γ, β from `zₜ`.
- **History length** — 5 timesteps of (obs, action), aggregated by Conv1d + MaxPool at 128
  channels. The paper doesn't publish a sweep over this length.

### Still open
- What's the ablation telling you the FiLM modulation contributes vs a concat baseline?
- How much of the 3-view-known-state ceiling (87.9%) vs 1-view-unknown (75.0%) gap is "view"
  vs "physics inference"? Worth checking the paper's ablation tables.
- The two ⚠️ flags in §5.5 — does the code actually compute L1/L2 regression as the paper
  says, or contrastive as the YAML key suggests?

Worth opening the code to answer the last one.

---

*Notes drafted: 2026-05-14. Personal learning notes — not part of the project's
development log ([dev_log.md](dev_log.md)).*
