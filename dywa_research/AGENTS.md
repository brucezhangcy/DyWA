# Research workflow instructions

These rules apply to `dywa_research/`. Explicit current user instructions take precedence.

## Resume efficiently

1. Read `STATUS.md`, then `research_prompt_size_transition.md`.
2. Inspect Git status and the newest `PROGRESS.md` entry. Read older history,
   `REVIEW.md`, `CRITIQUES.md` and result files only for the question being resolved.
3. Check whether the proposed task has already been attempted. Reuse recorded
   results and passed checks unless new changes or unresolved evidence justify a rerun.
4. Pick one uncertainty whose resolution changes the next decision. State the
   hypothesis, evidence needed, expected cost and stop rule before expensive work.

## Scope and permission

- Edit and clean only this folder by default. Preserve raw data, checkpoints,
  experiment settings and meaningful filenames. Explain any necessary exception.
- CPU analysis, reversible tooling fixes and documentation may proceed within the
  user's task. Preserve historical results; write re-analysis to a distinct result directory.
- Never launch GPU evaluation/training, remote compute, a watcher that launches
  jobs, or detached experiment without explicit owner approval for that run and budget.
  A launch flag is an interlock, not authorization. Never bypass a failed scientific gate.
- Propose scientific parameter changes with rationale and a control before applying
  them. Do not silently optimize hyperparameters based on test-set performance.
- Push only within the owner's authorized publication scope. Inspect current remote
  state; use normal fast-forward pushes and preserve concurrent changes.

## Evidence and engineering

- Distinguish observations, CPU derivations, implementation hypotheses and proposed
  experiments. No invented runs or numbers. Negative results are useful evidence.
- Preserve object/pose/seed grouping; prevent train/test and preprocessing leakage.
  Record metric denominators, immediate outcomes and right-censoring.
- For each real experiment record an ID, code commit, resolved configuration or
  immutable config artifact, exact command, checkpoint identity/normalizer, data
  split, seeds, environment, budget, outputs and completion status.
- Test changes at the layer they affect. Use CPU mocks for bookkeeping and failure
  propagation; label simulator behavior unverified until an approved runtime check.
- Clean generated caches and superseded duplicates only after checking dependencies;
  keep raw evidence and a brief deletion rationale. Do not accumulate duplicate summaries.

## Leave a useful handoff

Update `STATUS.md` in place and prepend one concise session entry to `PROGRESS.md`:

- Date, goal and experiment/task ID.
- Files/decisions changed and the reason.
- Commands actually run, result paths and validation outcome.
- What was learned, including disconfirming evidence and uncertainty.
- Compute used and approval scope; explicitly say when no simulation ran.
- Next decision/action, its stop condition, and any required owner input.

Keep the status page compact; keep historical results in the log and artifacts.
Do not mark proposed or queued work as completed.
