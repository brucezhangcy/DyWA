"""Stub of the nvtx profiling package (see progress log). Only nvtx.annotate is
used in the repo, as both a decorator and a context manager; this no-op supports
both. Zero effect on computed results."""
class _Annotate:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __call__(self, fn): return fn
def annotate(*a, **k):
    return _Annotate(*a, **k)
def mark(*a, **k): pass
def push_range(*a, **k): pass
def pop_range(*a, **k): pass
