# Stub for the UR5 CUDA kinematics extension (not built here; no nvcc).
# The Trossen render path never instantiates UR5, so `forward` is never called.

def forward(*args, **kwargs):
    raise RuntimeError(
        "cxx.ur5_kin_cuda.forward called, but the UR5 CUDA kinematics extension "
        "was not built on this machine (no nvcc). This stub exists only so "
        "non-UR5 (Trossen) code paths can import successfully."
    )
