import numpy as np

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_raycast_3d import closest_point_on_triangle_fast as _cy_closest_tri
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False

def closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Find the closest point on triangle ABC to point P.
    """
    if _USE_CYTHON:
        p64 = getattr(p, '_c64', None)  # if someone cached
        if p64 is None:
            p64 = np.ascontiguousarray(p, dtype=np.float64)
        a64 = np.ascontiguousarray(a, dtype=np.float64)
        b64 = np.ascontiguousarray(b, dtype=np.float64)
        c64 = np.ascontiguousarray(c, dtype=np.float64)
        return _cy_closest_tri(p64, a64, b64, c64)
    # Check if P in vertex region outside A
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a

    # Check if P in vertex region outside B
    bp = p - b
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b

    # Check if P in edge region of AB
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab

    # Check if P in vertex region outside C
    cp = p - c
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c

    # Check if P in edge region of AC
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac

    # Check if P in edge region of BC
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b)

    # P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + ab * v + ac * w
