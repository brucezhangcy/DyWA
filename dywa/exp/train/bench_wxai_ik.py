#!/usr/bin/env python3
"""
Benchmark wxai_ik at different n_iter values.

Measures:
  - Wall-clock per call (warmed up, on GPU)
  - IK residual (position + orientation error)

Uses realistic target poses: sample random joint configs, FK them to get
reachable targets, then start IK from HOME_Q and see how well it converges
within N iterations. This mimics the per-step control-loop usage where the
target is ~6cm/0.1rad away from current EE pose.
"""
import time
import torch as th
from env.robot.trossen_kin import (
    wxai_fk, wxai_ik, JOINT_LOWER, JOINT_UPPER, HOME_Q,
)


def _rotation_error(R_tgt, R_cur):
    """axis-angle magnitude of R_tgt * R_cur^T"""
    R = R_tgt @ R_cur.transpose(-1, -2)
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1) * 0.5).clamp(-1.0, 1.0)
    return th.acos(cos)


def main():
    device = th.device('cuda:0')
    N = 4096            # batch size similar to training
    n_trials = 20       # repeat for timing stability
    n_iters = [5, 10, 15, 20, 25, 30]

    # Generate realistic targets: random joint configs within limits, FK to pose.
    th.manual_seed(0)
    q_lo = JOINT_LOWER.to(device)
    q_hi = JOINT_UPPER.to(device)
    q_rand = q_lo + (q_hi - q_lo) * th.rand(N, 6, device=device)
    with th.no_grad():
        T_target = wxai_fk(q_rand)

    q_init = HOME_Q.to(device).expand(N, 6).clone()

    # Warm-up CUDA kernels at each n_iter so timings exclude compile/launch overhead.
    for n_iter in n_iters:
        with th.no_grad():
            _ = wxai_ik(T_target, q_init=q_init, n_iter=n_iter, damping=0.05)
    th.cuda.synchronize()

    print(f"Benchmark: wxai_ik, batch={N}, device={device}")
    print(f"Targets sampled uniformly from joint limits, IK starts from HOME_Q.")
    print(f"(Note: real control-loop targets are near current EE; convergence will be faster there.)\n")

    print(f"{'n_iter':>7s} | {'avg ms':>8s} | {'pos_err mm (mean / 99th%)':>30s} | {'ori_err deg (mean / 99th%)':>30s}")
    print("-" * 90)

    baseline_ms = None
    for n_iter in n_iters:
        # Time
        th.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            with th.no_grad():
                q_sol = wxai_ik(T_target, q_init=q_init, n_iter=n_iter, damping=0.05)
        th.cuda.synchronize()
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) * 1000.0 / n_trials

        # Residual
        with th.no_grad():
            T_cur = wxai_fk(q_sol)
            pos_err = (T_target[..., :3, 3] - T_cur[..., :3, 3]).norm(dim=-1)
            ori_err = _rotation_error(T_target[..., :3, :3], T_cur[..., :3, :3])
        pos_mean = pos_err.mean().item() * 1000   # mm
        pos_p99 = th.quantile(pos_err, 0.99).item() * 1000
        ori_mean = ori_err.mean().item() * 57.2958  # deg
        ori_p99 = th.quantile(ori_err, 0.99).item() * 57.2958

        if baseline_ms is None and n_iter == 30:
            baseline_ms = avg_ms

        speedup = ""
        if n_iter != 30 and baseline_ms is None:
            pass  # compute later
        print(f"{n_iter:7d} | {avg_ms:8.2f} | {pos_mean:13.2f} / {pos_p99:13.2f}  | {ori_mean:13.2f} / {ori_p99:13.2f}")

    # Also test with realistic small deltas (this is what control loop actually sees)
    print()
    print("=" * 90)
    print("Realistic control-loop scenario: target = HOME_Q + small joint delta")
    print("(target EE pose ~ 6cm / 0.1rad from current EE — what one step actually requires)\n")

    small_delta = 0.05 * (q_hi - q_lo) * (2 * th.rand(N, 6, device=device) - 1)
    q_near_home = (HOME_Q.to(device) + small_delta).clamp(q_lo, q_hi)
    with th.no_grad():
        T_target_close = wxai_fk(q_near_home)
    q_init_close = HOME_Q.to(device).expand(N, 6).clone()

    print(f"{'n_iter':>7s} | {'avg ms':>8s} | {'pos_err mm (mean / 99th%)':>30s} | {'ori_err deg (mean / 99th%)':>30s}")
    print("-" * 90)
    for n_iter in n_iters:
        th.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            with th.no_grad():
                q_sol = wxai_ik(T_target_close, q_init=q_init_close, n_iter=n_iter, damping=0.05)
        th.cuda.synchronize()
        t1 = time.perf_counter()
        avg_ms = (t1 - t0) * 1000.0 / n_trials

        with th.no_grad():
            T_cur = wxai_fk(q_sol)
            pos_err = (T_target_close[..., :3, 3] - T_cur[..., :3, 3]).norm(dim=-1)
            ori_err = _rotation_error(T_target_close[..., :3, :3], T_cur[..., :3, :3])
        pos_mean = pos_err.mean().item() * 1000
        pos_p99 = th.quantile(pos_err, 0.99).item() * 1000
        ori_mean = ori_err.mean().item() * 57.2958
        ori_p99 = th.quantile(ori_err, 0.99).item() * 57.2958
        print(f"{n_iter:7d} | {avg_ms:8.2f} | {pos_mean:13.2f} / {pos_p99:13.2f}  | {ori_mean:13.2f} / {ori_p99:13.2f}")


if __name__ == '__main__':
    main()
