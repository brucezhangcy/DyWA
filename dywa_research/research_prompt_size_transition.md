# Research prompt — object size and single-arm versus bimanual manipulation

Act as a robotics researcher and research engineer. Advance the strongest defensible
result, even when it contradicts the initial thesis. Read [README.md](README.md) and
follow [AGENTS.md](AGENTS.md); use [PROGRESS.md](PROGRESS.md) to avoid repeated attempts.

## Execution brief

Continue from the current evidence; do not restart the project or optimize for a
preferred story. Choose the smallest useful task that resolves the highest-priority
uncertainty. Proceed autonomously with scoped CPU analysis, reversible fixes and
necessary checks. Reuse passed checks and existing data when nothing material changed.

For any proposed configuration change, explain the causal reason, baseline/control,
expected observation, failure mode and rollback before spending compute. Keep the
validated recipe intact and isolate the intervention; do not bundle unrelated changes.
A changed parameter without supporting evidence is a hypothesis, not an improvement.

Keep the research folder lean: update README's current state and PROGRESS.md rather
than adding more status/audit/cleanup documents. Preserve raw evidence and checkpoint
provenance. Ask only for missing information that blocks the next decision, broader
edits outside scope, or explicit approval before GPU/remote experiments. Never infer
compute approval from permission to publish code.

Finish each session with the concrete result, supporting artifact/check, remaining
uncertainty and next decision. Report a clean negative finding when that is what the
evidence supports; do not manufacture a new experiment merely to show activity.

## Research question

For non-prehensile manipulation, how does object size affect a given controller's
success, what explains its failures, and when does adding a second arm improve the
reliability–cost trade-off?

Separate three possible contributions:

1. An empirical generalization frontier for a specified policy, task and environment.
2. A causal account separating geometry, observation, training distribution, dynamics
   and control limitations.
3. An interpretable routing system using comparable single-arm/bimanual outcomes.

Do not assume a universal physical scaling law, a monotonic collapse, bimanual
necessity, or a publishable novelty claim in advance. A failed finite-budget policy
does not establish that no single-arm controller could solve the task.

## Current evidence and known traps

The fixed-table frontier and its uncertainty, the exploratory competence predictor,
and the cross-simulator scripted baseline are summarized in `README.md` with exact
artifact paths. They are historical measurements, not permission to repeat experiments.

- Object scale is an enclosing radius in metres. Keep radius and diameter distinct.
- Training scale DR was approximately [0.05, 0.10] m. Larger eval objects probe
  distribution shift as well as physical difficulty; label that boundary explicitly.
- Valid-mode spawn placement does not apply train-mode radius contraction. It can
  replay a single pose per environment. Repeated episodes are not independent poses.
- Table dimensions affect spawn/goal regions and may interact with base tensors and
  software clipping. The pending v2 table fails a conservative large-radius support
  certificate. Do not treat a table-only override as a validated causal control.
- Reset-height scaling and delayed reset timing need correct measurement before
  interpreting distance, reach or penetration. Use the repaired recorder; its CPU
  tests do not certify Isaac Gym behavior.
- Mass is sampled independently of geometric scale in the audited pipeline. State
  that model explicitly; constant-density scaling is a separate intervention.
- Resolved success settings matter: `force_rad` overrides `task.goal_radius` in the
  evaluation entry. Threshold robustness alone does not eliminate other confounds.
- “Success rate ≥0.5” is a binary competence label. Its classifier probability does
  not mean the probability that the next episode succeeds.
- Goal type can depend on object geometry (yaw-only for thin objects). Account for
  that rule before attributing shape effects to mechanics.

## Choose the next experiment by its decision value

Do not blindly restart all phases. Select the earliest unresolved dependency from
`README.md` and reuse existing evidence. For each proposal state:

- Falsifiable hypothesis and a plausible alternative explanation.
- Intervention, matched control and quantities held fixed.
- Objects/poses/seeds, primary metric and denominator, uncertainty method.
- Exact resolved settings, checkpoint/normalizer, code revision and output location.
- Required compute, upper time budget, stop conditions and the next decision for
  each possible outcome. A pilot can estimate cost; do not invent a precise ETA.

Obtain explicit owner approval before GPU use, including small evaluations. After
a pilot, inspect results before requesting a larger budget; no automatic long queue.

## Evidence gates

### 1. Validate the measurement path

Propose the existing four-cell diagnostic before sweeps or retraining. Compare with
historical reference cells at the same seed/settings, with justified acceptance
criteria. Verify start/goal association, reset heights, actual support, robot base
and effective workspace, termination causes and incomplete episodes. Use raw camera
observations plus tensors for physical geometry checks. Stop if measurements disagree.

### 2. Measure a controlled size frontier

After the measurement gate and a reviewed geometry proposal, hold spawn/goal
regions, task-distance distribution, controller, observation, timeout and success
criteria fixed while varying size. Report per-object/per-pose outcomes with fixed
trial budgets, multiple seeds and grouped uncertainty. Include absolute and
size-relative tolerance controls where useful; identify unsupported/infeasible
starts using physical evidence rather than a ≤3-step heuristic alone.

Treat support, reach-around, mass/inertia, toppling and limited time as competing
mechanisms. A timeout is an outcome, not proof of underactuation. Use paired
interventions to distinguish them. Do not grow the table and silently change task
length at the same time.

### 3. Separate policy generalization from broader capability

Compare the fixed teacher with controlled training/fine-tuning over larger sizes,
including a small-range control with comparable budget and environment. Evaluate
on held-out objects/poses. If retraining helps, update the thesis. If it fails,
report the controller class and optimization budget; do not infer impossibility.

### 4. Obtain a comparable bimanual result

Prefer the same simulator, object/pose manifests, task, observations, timeout,
success criteria and accounting for both arm counts. Audit actor names, welded
base poses, DOF indexing, target buffers and observations before training a dual-arm
policy. The current scaffold is not a working baseline. Cross-simulator scripted
primitives may inform design but cannot establish a fair crossover or necessity.

### 5. Build and evaluate the explicit trigger

Predict episode success or use a count-aware capability model with reported
uncertainty. Group all data for an object into the same split; fit preprocessing
on training groups only. Use separate groups for calibration/threshold selection
and final testing. Keep the final test untouched by feature and threshold search.

Compare always-single, always-bimanual, size-threshold and learned routing on
matched outcomes. Report reliability, coverage, errors, calibration and compute or
arm-use cost. Include a reject/fallback option when neither controller meets the
target. A single-arm failure score is not evidence that two arms will succeed.

## Completion and communication

At each session end update `README.md` and prepend a concise entry to `PROGRESS.md`.
Link commands actually executed and result artifacts; separate checks that passed
from runtime behavior that remains unverified. Preserve raw evidence and log
negative results. Avoid new duplicate reports when a short state/log update suffices.

Before publication, verify related-work claims using primary sources. The current
literature spot-check and historical source notes are in `PROGRESS.md`; it does not establish “first” or “provably
better.” The deliverable may be a corrected benchmark or failure explanation if
that is what the evidence supports.
