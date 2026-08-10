"""Stub of nvdiffrast.torch. Rendering is disabled in Phase-A eval
(use_nvdr_record_episode=false), so these are import-satisfying placeholders.
Any actual call raises loudly so a wrong assumption surfaces immediately."""
def _stub(*a, **k):
    raise RuntimeError('nvdiffrast stub invoked: rendering path is disabled; '
                       'do not enable nvdr recording without real nvdiffrast.')
class RasterizeGLContext:  # noqa
    def __init__(self, *a, **k): _stub()
class RasterizeCudaContext:  # noqa
    def __init__(self, *a, **k): _stub()
rasterize = _stub
interpolate = _stub
antialias = _stub
texture = _stub
