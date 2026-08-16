# Auto-Research Log — Object-Size Transition & Single-vs-Bimanual Triggering

**Date:** 2026-06-20
**Method:** Web search + primary-source abstract fetches (arXiv / proceedings), run from the main
thread. Every arXiv ID below was returned by search and/or confirmed by fetching the abstract.
Companion to [research_prompt_size_transition.md](research_prompt_size_transition.md) (see its §8 for
the condensed positioning).

> **Scope.** Five threads tied to the proposal: (1) non-prehensile manipulation SOTA/baselines,
> (2) object-size/scale scaling, (3) single-vs-bimanual arm-count decision (the core novelty),
> (4) bimanual non-prehensile / large-object manipulation, (5) feasibility/affordance prediction.

---

## TL;DR

- **Both core claims are open.** No prior work (a) measures the *single-arm non-prehensile size
  frontier / crossover*, or (b) provides an *explicit, interpretable one-vs-two-arm trigger* driven
  by object physics.
- **One paper sits close and must be addressed: BiNoMaP (2509.21256).** Dual-arm non-prehensile,
  size-parameterized — but *always bimanual*, demo-based, and it never decides whether two arms are
  needed. It is also already present in your repos (`Aloha_real/binomap/`, `portfolio/BiNoMaP/`).
- **Biggest practical takeaway:** reuse BiNoMaP's bimanual primitives as the Phase-D bimanual
  controller instead of training dual-arm RL from scratch — spend novelty budget on the trigger and
  the frontier, which is where the contribution actually is.

---

## Thread 1 — Non-prehensile manipulation (base system + baselines)

| Paper | ID / venue | One-line | Role |
|---|---|---|---|
| **DyWA** | 2503.16806 · ICCV 2025 | Dynamics-adaptive world+action model; 1-view unseen success 75% sim / 68% real | **Base system being extended** |
| **CORN** | 2403.10760 · ICLR 2024 | Contact-based patch-transformer object representation, pretrained encoder | Predecessor; supplies the reused ICP/point encoder; baseline |
| **HACMan** | 2305.03942 · CoRL 2023 | Hybrid actor-critic maps; pick contact point + motion params; 89% sim-unseen, 50% real | Non-prehensile RL baseline |
| **HACMan++** | 2407.08585 | Spatially-grounded motion primitives, follow-up to HACMan | Baseline / primitive design reference |
| **Hierarchical & Modular Non-prehensile** | 2502.20843 · 2025 | Non-prehensile in general environments via modular nets | Recent follow-up |

**Note:** DyWA, CORN, HACMan are all **single-arm**. The published evaluation protocol the field
expects is *1-view, unknown-state, unseen objects* — report your single-arm frontier numbers on it
for direct comparability.

## Thread 2 — Object-size / scale scaling

- **Data Scaling Laws in Imitation Learning** (2410.18647, ICLR 2025): success follows a power law in
  *number of training objects/environments*; diversity matters more than demo count; gains saturate
  past ~200 objects. **This is data-quantity scaling, not object-physical-size scaling** — a useful
  contrast, not a precedent.
- General practice (ADR, "test at 0.5×–3× scale") randomizes/perturbs object size for robustness, but
  **no source characterizes a capability-vs-size frontier with a crossover** for non-prehensile
  manipulation. → **Phase A is novel.**

## Thread 3 — Single-vs-bimanual arm-count decision (CORE NOVELTY)

This is the scoop-check thread. Findings:
- Dual-hand *selection* work decides **which** hand executes a **grasp** (left/right), not **how
  many** arms. Different problem (prehensile, laterality).
- "Mixed coordination" / base-residual multi-arm work decides **when arms must coordinate** during a
  task, not whether the task needs one or two effectors.
- Bimanual taxonomies (Krebs & Asfour 2022) catalogue *types* of two-handed tasks but prescribe no
  learned trigger.
- **MonoDuo** (2605.29298) and **You-Only-Teach-Once** (2501.14208) transfer/learn bimanual policies
  but assume bimanual from the start.

→ **No work provides an explicit, interpretable trigger that decides one-vs-two arms from object
size/mass for non-prehensile manipulation.** The novelty holds.

## Thread 4 — Bimanual non-prehensile / large-object manipulation

| Paper | ID | Key facts | Threat |
|---|---|---|---|
| **BiNoMaP** | 2509.21256 (Zhou & Jia; under review, rev. Mar 2026) | Dual-arm non-prehensile primitives (push/poke/pivot/wrap); RL-free, learns from video demos; **parameterized primarily by object size**; cross-embodiment to two real dual-arm platforms | **CLOSEST.** But *always bimanual*, no arm-count decision, no single-arm frontier |
| **DART** | 2604.17833 (Apr 2026) | Dual-arm non-prehensile tray transport; NMPC + impedance + (analytic/regression/RL) dynamics; varies object mass/geometry/friction | Always-dual, no decision, different task |
| **PerAct2** | 2407.00278 | Bimanual manipulation benchmark + multi-task agent | Benchmark reference |
| **DA-VIL** | 2410.19712 | Adaptive dual-arm RL + variable impedance | Dual-arm control reference |

**On BiNoMaP being in your repos:** `Aloha_real/binomap/` and
`portfolio/BiNoMaP/BiNoMaP_03_aloha_trossen_box_flip_SUCCESS.mp4` indicate you already have a working
bimanual non-prehensile primitive stack on Trossen hardware. This is the natural Phase-D bimanual
controller.

## Thread 5 — Feasibility / affordance prediction (mechanism for the trigger)

- **RT-Affordance** (2411.02704): affordances as intermediate representations for manipulation
  reasoning.
- Grasp/push **feasibility models** predict P(success) of an action before executing it (used for
  action selection). This is precisely the abstraction the **Phase-C capability predictor**
  instantiates — but applied to *arm-count* choice rather than grasp-pose choice. Cite this line as
  the methodological precedent for "predict success before acting."

---

## Defending novelty — the three threats and the rebuttals

1. **BiNoMaP (2509.21256).** *Overlap:* bimanual non-prehensile + object-size parameterization.
   *Defense:* always-bimanual and demo-based; no trigger, no frontier, no necessity result. Your
   contribution is the decision + the measured crossover + the "two arms are *necessary* above `s*`"
   evidence. Cite prominently; reuse its primitives.
2. **DART (2604.17833).** *Overlap:* dual-arm non-prehensile, varies object physics. *Defense:*
   always-dual MPC for tray balancing; no arm-count decision; different task and objective.
3. **Data Scaling Laws (2410.18647).** *Overlap:* the phrase "scaling law." *Defense:* their axis is
   data quantity; yours is object physical scale with a mechanistic (controllability/stability)
   explanation and a crossover. Orthogonal.

## Recommended edits already applied to the proposal (§8)

- Added a verified Related-Work / positioning section (must-cites + threats + rebuttals).
- Phase D now recommends **reusing BiNoMaP primitives** as the bimanual controller (de-risk).
- Added field-expected experiments: report on the standard 1-view-unseen protocol; include a
  sim-to-real argument; add a **"necessity" experiment** (retrain single-arm above `s*`, show it
  still fails); add **BiNoMaP-style size-only parameterization as a Phase-C baseline** to beat.

## Caveats

- BiNoMaP is under review (not yet a finalized venue); track its status before submission.
- A few IDs are early-2026 preprints (DART 2604.17833, MonoDuo 2605.29298) — re-verify versions near
  submission.
- This pass used targeted search, not an exhaustive sweep; a second pass focused narrowly on
  "effector-count / morphology selection" outside manipulation (e.g., task-and-motion planning,
  multi-robot task allocation) may surface additional adjacent prior art worth pre-empting.

---

### Verified source links
DyWA https://arxiv.org/abs/2503.16806 · CORN https://arxiv.org/abs/2403.10760 ·
HACMan https://arxiv.org/abs/2305.03942 · HACMan++ https://arxiv.org/html/2407.08585 ·
Hierarchical/Modular https://arxiv.org/abs/2502.20843 · BiNoMaP https://arxiv.org/abs/2509.21256 ·
DART https://arxiv.org/abs/2604.17833 · PerAct2 https://arxiv.org/pdf/2407.00278 ·
DA-VIL https://arxiv.org/pdf/2410.19712 · Data Scaling Laws https://arxiv.org/html/2410.18647v1 ·
RT-Affordance https://arxiv.org/html/2411.02704v1


---

## 2026-08-16 — Fresh literature refresh + HOW WE DIFFER (targeted web search, NOT exhaustive)
Re-searched single-arm non-prehensile size frontier / scaling. Object-size limits ARE studied, but
under DIFFERENT framings than ours:

| Prior framing | Representative work | How OURS differs |
|---|---|---|
| Object too big to GRASP (aperture) -> non-prehensile / extrinsic dexterity | Ungraspable via Extrinsic Dexterity (2503.23120) | We study non-prehensile CONTROL vs size, not graspability; the arm CAN contact it, it just can't steer its pose |
| Non-prehensile EXPANDS a simple arm's operational space | Multi-Joint Failure (2410.01102) | We measure the LIMIT of that expansion vs object size, with a crossover s* |
| Geometry/shape-aware non-prehensile | DexNoMa | We quantify shape-beyond-size in a competence predictor (complementary) |
| Robustness to size variation (DR) | many | We treat size as the independent AXIS and measure success-vs-size, not just robustness |
| Data-quantity scaling laws | Data Scaling Laws (2410.18647) | Different axis: object physical size, not #demos/objects |
| Dual-arm non-prehensile | DART (2604.17833), BiNoMaP (2509.21256) | Always-bimanual; never measure where SINGLE-arm fails nor decide arm-count |

NOT FOUND (consistent with the 2026-06-20 pass): a measured single-arm non-prehensile success-vs-size
FRONTIER with a crossover s* AND a mechanistic proof the cause is controllability (not reach/table/
threshold/graspability). => that specific measurement + mechanism is our differentiator.
CAVEAT: US-only web search, hard-to-query niche, NOT exhaustive; a full related-work pass is still
required before any novelty claim. Differentiate explicitly from extrinsic-dexterity + geometry-aware.


---

## 2026-08-16 — Fresh literature refresh + HOW WE DIFFER (targeted web search, NOT exhaustive)
Object-size limits ARE studied, but under DIFFERENT framings than ours:

| Prior framing | Representative work | How OURS differs |
|---|---|---|
| Object too big to GRASP (aperture) -> non-prehensile / extrinsic dexterity | Ungraspable via Extrinsic Dexterity (2503.23120) | We study non-prehensile CONTROL vs size, not graspability; the arm CAN contact it, it just can't steer its pose |
| Non-prehensile EXPANDS a simple arm's operational space | Multi-Joint Failure (2410.01102) | We measure the LIMIT of that expansion vs size, with a crossover s* |
| Geometry/shape-aware non-prehensile | DexNoMa | We quantify shape-beyond-size in a competence predictor (complementary) |
| Robustness to size variation (DR) | many | We treat size as the independent AXIS (success-vs-size), not just robustness |
| Data-quantity scaling laws | Data Scaling Laws (2410.18647) | Different axis: object physical size, not #demos/objects |
| Dual-arm non-prehensile | DART (2604.17833), BiNoMaP (2509.21256) | Always-bimanual; never measure where SINGLE-arm fails nor decide arm-count |

NOT FOUND (consistent with 2026-06-20): a measured single-arm non-prehensile success-vs-size FRONTIER
with a crossover s* AND a mechanistic proof the cause is controllability (not reach/table/threshold/
graspability). => that measurement + mechanism is our differentiator.
CAVEAT: US-only web search, hard-to-query niche, NOT exhaustive; a full related-work pass is still
required before any novelty claim. Differentiate explicitly from extrinsic-dexterity + geometry-aware.
