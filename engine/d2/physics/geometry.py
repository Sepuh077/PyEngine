import numpy as np


def closest_point_on_segment(p, a, b):
    """Find closest point on line segment AB to point P (all 2D numpy arrays)."""
    ab = b - a
    dot_ab = np.dot(ab, ab)
    if dot_ab < 1e-10:
        return a.copy()
    t = np.dot(p - a, ab) / dot_ab
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab


def project_polygon_onto_axis(vertices, axis):
    """Project all polygon vertices onto an axis, return (min, max) interval."""
    dots = np.dot(vertices, axis)
    return float(np.min(dots)), float(np.max(dots))


def polygon_axes(vertices):
    """Return the edge-normal axes for a convex polygon (2D numpy array of shape (N,2))."""
    n = len(vertices)
    axes = []
    for i in range(n):
        edge = vertices[(i + 1) % n] - vertices[i]
        # Perpendicular (outward normal assuming CCW winding)
        normal = np.array([-edge[1], edge[0]], dtype=np.float64)
        length = np.linalg.norm(normal)
        if length > 1e-10:
            axes.append(normal / length)
    return axes
