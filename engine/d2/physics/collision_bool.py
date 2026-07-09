import math
import numpy as np
from typing import Optional

from engine.d2.physics.types import ColliderType2D
from engine.d2.physics.geometry import closest_point_on_segment, project_polygon_onto_axis, polygon_axes

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_collision_2d import (
        aabb_overlap_2d_fast as _cy_aabb_2d,
        circle_vs_circle_fast as _cy_cc,
        obb_vs_obb_2d_fast as _cy_obb2d,
        circle_vs_obb_2d_fast as _cy_co2d,
        closest_point_on_segment_fast as _cy_seg,
        segment_segment_dist_sq_fast as _cy_seg_seg,
    )
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False


# =========================================================================
# AABB broadphase
# =========================================================================

def aabb_overlap_2d(a_aabb, b_aabb) -> bool:
    """2-axis AABB overlap test.  Each aabb is (min_np_2d, max_np_2d)."""
    if a_aabb is None or b_aabb is None:
        return False
    a_min, a_max = a_aabb
    b_min, b_max = b_aabb
    if _USE_CYTHON:
        return _cy_aabb_2d(
            np.ascontiguousarray(a_min, dtype=np.float64),
            np.ascontiguousarray(a_max, dtype=np.float64),
            np.ascontiguousarray(b_min, dtype=np.float64),
            np.ascontiguousarray(b_max, dtype=np.float64),
        )
    return not (
        a_max[0] < b_min[0] or a_max[1] < b_min[1] or
        a_min[0] > b_max[0] or a_min[1] > b_max[1]
    )


# =========================================================================
# Circle vs Circle
# =========================================================================

def circle_vs_circle(a, b) -> bool:
    """a, b each = (center_np_2d, radius)."""
    ca, ra = a
    cb, rb = b
    if _USE_CYTHON:
        return _cy_cc(float(ca[0]), float(ca[1]), ra, float(cb[0]), float(cb[1]), rb)
    diff = ca - cb
    dist_sq = float(np.dot(diff, diff))
    radius_sum = ra + rb
    return dist_sq <= radius_sum * radius_sum


# =========================================================================
# OBB vs OBB (2D SAT — 4 axes)
# =========================================================================

def _obb_corners(center, angle, half_ext):
    """Return the 4 corners of a 2D OBB."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    ux = np.array([cos_a, sin_a], dtype=np.float64)
    uy = np.array([-sin_a, cos_a], dtype=np.float64)
    dx = ux * half_ext[0]
    dy = uy * half_ext[1]
    return np.array([
        center - dx - dy,
        center + dx - dy,
        center + dx + dy,
        center - dx + dy,
    ])


def _obb_axes(angle):
    """Return the two unit axes of an OBB given its rotation angle."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return [
        np.array([cos_a, sin_a], dtype=np.float64),
        np.array([-sin_a, cos_a], dtype=np.float64),
    ]


def _project_obb(center, angle, half_ext, axis):
    """Project OBB onto an axis, return (min, max) interval."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    ux = np.array([cos_a, sin_a], dtype=np.float64)
    uy = np.array([-sin_a, cos_a], dtype=np.float64)
    c_proj = float(np.dot(center, axis))
    r = abs(float(np.dot(ux, axis))) * half_ext[0] + abs(float(np.dot(uy, axis))) * half_ext[1]
    return c_proj - r, c_proj + r


def obb_vs_obb_2d(a, b) -> bool:
    """SAT test for two 2D OBBs.  Each = (center_np, angle, half_ext_np)."""
    ca, aa, ea = a
    cb, ab, eb = b

    if _USE_CYTHON:
        return _cy_obb2d(
            float(ca[0]), float(ca[1]), float(aa), float(ea[0]), float(ea[1]),
            float(cb[0]), float(cb[1]), float(ab), float(eb[0]), float(eb[1]),
        )

    axes = _obb_axes(aa) + _obb_axes(ab)
    for axis in axes:
        a_min, a_max = _project_obb(ca, aa, ea, axis)
        b_min, b_max = _project_obb(cb, ab, eb, axis)
        if a_max < b_min or b_max < a_min:
            return False
    return True


# =========================================================================
# Circle vs OBB
# =========================================================================

def circle_vs_obb_2d(circle, obb) -> bool:
    """circle = (center_np, radius), obb = (center_np, angle, half_ext_np)."""
    cs, rs = circle
    cb, angle, eb = obb

    if _USE_CYTHON:
        return _cy_co2d(
            float(cs[0]), float(cs[1]), float(rs),
            float(cb[0]), float(cb[1]), float(angle), float(eb[0]), float(eb[1]),
        )

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    d = cs - cb
    local_x = d[0] * cos_a + d[1] * sin_a
    local_y = -d[0] * sin_a + d[1] * cos_a

    cx = np.clip(local_x, -eb[0], eb[0])
    cy = np.clip(local_y, -eb[1], eb[1])

    dx = local_x - cx
    dy = local_y - cy
    dist_sq = dx * dx + dy * dy
    return dist_sq <= rs * rs


# =========================================================================
# Capsule helpers
# =========================================================================

def _capsule_segment(capsule):
    """Return the two endpoint centers of the capsule's internal segment."""
    center, radius, half_h, direction = capsule
    if direction == 0:
        # Vertical
        a = center + np.array([0.0, -half_h], dtype=np.float64)
        b = center + np.array([0.0,  half_h], dtype=np.float64)
    else:
        # Horizontal
        a = center + np.array([-half_h, 0.0], dtype=np.float64)
        b = center + np.array([ half_h, 0.0], dtype=np.float64)
    return a, b


def capsule_vs_circle(capsule, circle) -> bool:
    """capsule = (center_np, radius, half_height, direction), circle = (center_np, radius)."""
    seg_a, seg_b = _capsule_segment(capsule)
    cc, rc = circle
    cp = closest_point_on_segment(cc, seg_a, seg_b)
    diff = cc - cp
    dist_sq = float(np.dot(diff, diff))
    r_sum = capsule[1] + rc
    return dist_sq <= r_sum * r_sum


def capsule_vs_obb(capsule, obb) -> bool:
    """Capsule vs OBB: test the capsule's internal segment against the Minkowski-expanded OBB."""
    cap_center, cap_r, cap_hh, cap_dir = capsule
    ob_center, ob_angle, ob_ext = obb

    # Strategy: find closest point on capsule segment to OBB, then check circle vs OBB
    seg_a, seg_b = _capsule_segment(capsule)

    cos_a = math.cos(ob_angle)
    sin_a = math.sin(ob_angle)

    # Transform segment into OBB local space
    def to_local(p):
        d = p - ob_center
        return np.array([d[0] * cos_a + d[1] * sin_a,
                         -d[0] * sin_a + d[1] * cos_a], dtype=np.float64)

    la = to_local(seg_a)
    lb = to_local(seg_b)

    # Find closest point on segment to the OBB center (origin in local space)
    # Then clamp that to the expanded box
    # Actually, the proper test: closest point on segment to the AABB
    # We do it by clamping each segment point to box and checking distance

    # Minkowski expansion: expand box by capsule radius and test segment vs expanded box
    expanded_ext = ob_ext + cap_r

    # Slab method: check if the segment intersects the expanded AABB
    # Quick check: closest point on segment to the box center
    seg_closest = closest_point_on_segment(np.zeros(2, dtype=np.float64), la, lb)
    cx = np.clip(seg_closest[0], -expanded_ext[0], expanded_ext[0])
    cy = np.clip(seg_closest[1], -expanded_ext[1], expanded_ext[1])

    # But we actually need: closest point on AABB to the segment
    # Simplified robust approach: sample both endpoints and the closest-on-segment point
    best_dist_sq = float('inf')
    for pt in [la, lb, seg_closest]:
        clamped = np.array([np.clip(pt[0], -ob_ext[0], ob_ext[0]),
                            np.clip(pt[1], -ob_ext[1], ob_ext[1])], dtype=np.float64)
        diff = pt - clamped
        d2 = float(np.dot(diff, diff))
        if d2 < best_dist_sq:
            best_dist_sq = d2

    # Also: closest point on segment to each of the 4 box corners
    corners_local = np.array([
        [-ob_ext[0], -ob_ext[1]],
        [ ob_ext[0], -ob_ext[1]],
        [ ob_ext[0],  ob_ext[1]],
        [-ob_ext[0],  ob_ext[1]],
    ], dtype=np.float64)
    for corner in corners_local:
        cp = closest_point_on_segment(corner, la, lb)
        diff = corner - cp
        d2 = float(np.dot(diff, diff))
        if d2 < best_dist_sq:
            best_dist_sq = d2

    # Also: closest point on each box edge to the segment
    for i in range(4):
        edge_a = corners_local[i]
        edge_b = corners_local[(i + 1) % 4]
        # Closest pair between two segments
        d2 = _segment_segment_dist_sq(la, lb, edge_a, edge_b)
        if d2 < best_dist_sq:
            best_dist_sq = d2

    return best_dist_sq <= cap_r * cap_r


def _segment_segment_dist_sq(a1, a2, b1, b2):
    """Squared distance between two line segments in 2D."""
    d1 = a2 - a1
    d2 = b2 - b1
    r = a1 - b1
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a < 1e-10 and e < 1e-10:
        diff = a1 - b1
        return float(np.dot(diff, diff))

    if a < 1e-10:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = float(np.dot(d1, r))
        if e < 1e-10:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            b_val = float(np.dot(d1, d2))
            denom = a * e - b_val * b_val
            if abs(denom) > 1e-10:
                s = np.clip((b_val * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b_val * s + f) / e
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b_val - c) / a, 0.0, 1.0)

    closest_a = a1 + d1 * s
    closest_b = b1 + d2 * t
    diff = closest_a - closest_b
    return float(np.dot(diff, diff))


def capsule_vs_capsule(cap_a, cap_b) -> bool:
    """Two capsules: closest distance between their internal segments <= sum of radii."""
    a1, a2 = _capsule_segment(cap_a)
    b1, b2 = _capsule_segment(cap_b)
    dist_sq = _segment_segment_dist_sq(a1, a2, b1, b2)
    r_sum = cap_a[1] + cap_b[1]
    return dist_sq <= r_sum * r_sum


# =========================================================================
# Polygon helpers
# =========================================================================

def polygon_vs_circle(poly_collider, circle_collider) -> bool:
    """poly_collider has .world_points (N,2 np), circle_collider has .circle (center, r)."""
    verts = poly_collider.world_points
    cc, rc = circle_collider.circle
    if verts is None or len(verts) < 3:
        return False

    # SAT: polygon edge normals + axis from circle center to closest vertex
    axes = polygon_axes(verts)

    # Additional axis: direction from closest vertex to circle center
    dists = np.linalg.norm(verts - cc, axis=1)
    closest_vert = verts[np.argmin(dists)]
    diff = cc - closest_vert
    diff_len = np.linalg.norm(diff)
    if diff_len > 1e-10:
        axes.append(diff / diff_len)

    for axis in axes:
        p_min, p_max = project_polygon_onto_axis(verts, axis)
        c_proj = float(np.dot(cc, axis))
        c_min = c_proj - rc
        c_max = c_proj + rc
        if p_max < c_min or c_max < p_min:
            return False
    return True


def polygon_vs_obb(poly_collider, obb_collider) -> bool:
    """SAT between polygon world_points and an OBB."""
    verts = poly_collider.world_points
    ob_center, ob_angle, ob_ext = obb_collider.obb
    if verts is None or len(verts) < 3:
        return False

    obb_corners = _obb_corners(ob_center, ob_angle, ob_ext)

    # Axes: polygon edges + OBB edges
    axes = polygon_axes(verts) + _obb_axes(ob_angle)

    for axis in axes:
        p_min, p_max = project_polygon_onto_axis(verts, axis)
        o_min, o_max = project_polygon_onto_axis(obb_corners, axis)
        if p_max < o_min or o_max < p_min:
            return False
    return True


def polygon_vs_polygon(poly_a, poly_b) -> bool:
    """SAT between two polygons (both have .world_points)."""
    va = poly_a.world_points
    vb = poly_b.world_points
    if va is None or vb is None or len(va) < 3 or len(vb) < 3:
        return False

    axes = polygon_axes(va) + polygon_axes(vb)
    for axis in axes:
        a_min, a_max = project_polygon_onto_axis(va, axis)
        b_min, b_max = project_polygon_onto_axis(vb, axis)
        if a_max < b_min or b_max < a_min:
            return False
    return True


def polygon_vs_capsule(poly_collider, capsule_collider) -> bool:
    """Polygon vs capsule: SAT with additional capsule-segment axis."""
    verts = poly_collider.world_points
    cap = capsule_collider.capsule
    if verts is None or len(verts) < 3 or cap is None:
        return False
    seg_a, seg_b = _capsule_segment(cap)
    cap_r = cap[1]

    # Find closest point on polygon edges to the capsule segment
    best_dist_sq = float('inf')
    n_verts = len(verts)
    for i in range(n_verts):
        edge_a = verts[i]
        edge_b = verts[(i + 1) % n_verts]
        d2 = _segment_segment_dist_sq(seg_a, seg_b, edge_a, edge_b)
        if d2 < best_dist_sq:
            best_dist_sq = d2

    # Also check if capsule segment endpoints are inside polygon
    if best_dist_sq > cap_r * cap_r:
        # Quick containment check for segment endpoints
        if _point_in_polygon(seg_a, verts) or _point_in_polygon(seg_b, verts):
            return True
        return False
    return True


def _point_in_polygon(p, verts) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(verts)
    inside = False
    j = n - 1
    for i in range(n):
        vi = verts[i]
        vj = verts[j]
        if ((vi[1] > p[1]) != (vj[1] > p[1])) and \
           (p[0] < (vj[0] - vi[0]) * (p[1] - vi[1]) / (vj[1] - vi[1] + 1e-30) + vi[0]):
            inside = not inside
        j = i
    return inside


# =========================================================================
# Top-level dispatch
# =========================================================================

def objects_collide_2d(a, b) -> bool:
    """
    Full 2D collision check: broadphase AABB + narrowphase dispatch by type.
    a, b are Collider2D instances.
    """
    # Broadphase
    if not aabb_overlap_2d(a.get_world_aabb(), b.get_world_aabb()):
        return False

    ta = a.type
    tb = b.type

    # ---- Circle vs Circle ----
    if ta == ColliderType2D.CIRCLE and tb == ColliderType2D.CIRCLE:
        return circle_vs_circle(a.circle, b.circle)

    # ---- Box vs Box ----
    if ta == ColliderType2D.BOX and tb == ColliderType2D.BOX:
        return obb_vs_obb_2d(a.obb, b.obb)

    # ---- Circle vs Box ----
    if ta == ColliderType2D.CIRCLE and tb == ColliderType2D.BOX:
        return circle_vs_obb_2d(a.circle, b.obb)
    if ta == ColliderType2D.BOX and tb == ColliderType2D.CIRCLE:
        return circle_vs_obb_2d(b.circle, a.obb)

    # ---- Capsule vs Circle ----
    if ta == ColliderType2D.CAPSULE and tb == ColliderType2D.CIRCLE:
        return capsule_vs_circle(a.capsule, b.circle)
    if ta == ColliderType2D.CIRCLE and tb == ColliderType2D.CAPSULE:
        return capsule_vs_circle(b.capsule, a.circle)

    # ---- Capsule vs Box ----
    if ta == ColliderType2D.CAPSULE and tb == ColliderType2D.BOX:
        return capsule_vs_obb(a.capsule, b.obb)
    if ta == ColliderType2D.BOX and tb == ColliderType2D.CAPSULE:
        return capsule_vs_obb(b.capsule, a.obb)

    # ---- Capsule vs Capsule ----
    if ta == ColliderType2D.CAPSULE and tb == ColliderType2D.CAPSULE:
        return capsule_vs_capsule(a.capsule, b.capsule)

    # ---- Polygon vs Circle ----
    if ta == ColliderType2D.POLYGON and tb == ColliderType2D.CIRCLE:
        return polygon_vs_circle(a, b)
    if ta == ColliderType2D.CIRCLE and tb == ColliderType2D.POLYGON:
        return polygon_vs_circle(b, a)

    # ---- Polygon vs Box ----
    if ta == ColliderType2D.POLYGON and tb == ColliderType2D.BOX:
        return polygon_vs_obb(a, b)
    if ta == ColliderType2D.BOX and tb == ColliderType2D.POLYGON:
        return polygon_vs_obb(b, a)

    # ---- Polygon vs Capsule ----
    if ta == ColliderType2D.POLYGON and tb == ColliderType2D.CAPSULE:
        return polygon_vs_capsule(a, b)
    if ta == ColliderType2D.CAPSULE and tb == ColliderType2D.POLYGON:
        return polygon_vs_capsule(b, a)

    # ---- Polygon vs Polygon ----
    if ta == ColliderType2D.POLYGON and tb == ColliderType2D.POLYGON:
        return polygon_vs_polygon(a, b)

    # Fallback — treat both as OBBs if they have one
    if hasattr(a, 'obb') and a.obb and hasattr(b, 'obb') and b.obb:
        return obb_vs_obb_2d(a.obb, b.obb)

    return False
