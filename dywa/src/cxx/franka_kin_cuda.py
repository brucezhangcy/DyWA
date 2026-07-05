# Stub for the Franka CUDA kinematics extension (not built here; no nvcc).
# The Trossen render path never instantiates Franka, so `forward` is never
# called. It raises clearly if it ever is, rather than failing silently.

def forward(*args, **kwargs):
    raise RuntimeError(
        "cxx.franka_kin_cuda.forward called, but the Franka CUDA kinematics "
        "extension was not built on this machine (no nvcc). This stub exists "
        "only so non-Franka (Trossen) code paths can import successfully."
    )
