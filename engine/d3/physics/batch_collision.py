"""Batch collision processing for 3D — broadphase *and* narrowphase.

Instead of calling Cython once per pair (lots of Python ↔ C overhead), this
module collects all collider geometry into contiguous NumPy arrays, groups
pairs by collision-type combination, and dispatches each group to a single
Cython batch call.

Public API
----------
batch_broadphase_3d(colliders)
    Sweep-and-prune broadphase.  Returns (M, 2) int32 pair-index array.

batch_narrowphase_manifold_3d(colliders, pairs)
    Batch narrowphase.  Returns a list of
    ``(idx_a, idx_b, CollisionManifold | None)`` for every input pair.

batch_narrowphase_bool_3d(colliders, pairs)
    Batch narrowphase bool-only.  Returns a list of ``(idx_a, idx_b, bool)``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from engine.d3.physics.collider import Collider3D
from engine.d3.physics.collision_manifold import (
    CollisionManifold,
    _make_manifold,
    _obb_contact_point,
    _obb_multi_contact_points,
)
from engine.d3.physics.types import ColliderType

# ---------------------------------------------------------------------------
# Try to load Cython batch module
# ---------------------------------------------------------------------------
try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled")
    from engine.cython.cy_batch_collision import (
        batch_broadphase_3d as _cy_bp3d,
        batch_sphere_sphere_bool_3d as _cy_ss_bool,
        batch_sphere_sphere_manifold_3d as _cy_ss_man,
        batch_obb_obb_bool_3d as _cy_oo_bool,
        batch_obb_obb_manifold_3d as _cy_oo_man,
        batch_sphere_obb_bool_3d as _cy_so_bool,
        batch_collision_pack_3d as _cy_pack3d,
        batch_frustum_cull_3d as _cy_frustum_cull,
        batch_continuous_sweep_3d as _cy_cont_sweep,
    )
    _BATCH_CYTHON = True
except Exception:
    _BATCH_CYTHON = False


# ---------------------------------------------------------------------------
# Batch broadphase
# ---------------------------------------------------------------------------

def batch_broadphase_3d(
    colliders: Sequence[Collider3D],
) -> np.ndarray:
    """Sweep-and-prune AABB broadphase over *colliders*.

    Returns an (M, 2) int32 array of overlapping index pairs (i < j).
    Colliders with ``aabb is None`` are excluded.
    """
    n = len(colliders)
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)

    mins = np.empty((n, 3), dtype=np.float64)
    maxs = np.empty((n, 3), dtype=np.float64)
    valid = np.ones(n, dtype=bool)

    for i, c in enumerate(colliders):
        if c.aabb is None:
            valid[i] = False
            mins[i] = 0.0
            maxs[i] = 0.0
        else:
            mins[i] = c.aabb[0]
            maxs[i] = c.aabb[1]

    if _BATCH_CYTHON:
        # Filter to valid rows, run Cython, then remap indices
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 2:
            return np.empty((0, 2), dtype=np.int32)
        v_mins = np.ascontiguousarray(mins[valid_idx], dtype=np.float64)
        v_maxs = np.ascontiguousarray(maxs[valid_idx], dtype=np.float64)
        raw = _cy_bp3d(v_mins, v_maxs)
        if len(raw) == 0:
            return np.empty((0, 2), dtype=np.int32)
        # Remap local indices back to original
        return valid_idx[raw].astype(np.int32)

    # Pure-Python fallback: brute-force O(n²)
    pairs: list = []
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            if (maxs[i, 0] >= mins[j, 0] and mins[i, 0] <= maxs[j, 0] and
                maxs[i, 1] >= mins[j, 1] and mins[i, 1] <= maxs[j, 1] and
                maxs[i, 2] >= mins[j, 2] and mins[i, 2] <= maxs[j, 2]):
                pairs.append((i, j))
    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)


# ---------------------------------------------------------------------------
# Helper: classify collider type
# ---------------------------------------------------------------------------

def _ctype(c: Collider3D) -> int:
    return int(getattr(c, "type", ColliderType.CUBE))


# ---------------------------------------------------------------------------
# Batch narrowphase (bool only)
# ---------------------------------------------------------------------------

def batch_narrowphase_bool_3d(
    colliders: Sequence[Collider3D],
    pairs: np.ndarray,
) -> List[Tuple[int, int, bool]]:
    """Batch bool collision test for every pair.

    *pairs* is (M, 2) int32 indexing into *colliders*.
    Returns list of ``(idx_a, idx_b, colliding_bool)``.
    """
    if len(pairs) == 0:
        return []

    # Group pairs by type combination
    groups: dict = {}  # (type_a, type_b) -> [(pair_local_idx, idx_a, idx_b), ...]
    for k in range(len(pairs)):
        ia, ib = int(pairs[k, 0]), int(pairs[k, 1])
        ta, tb = _ctype(colliders[ia]), _ctype(colliders[ib])
        groups.setdefault((ta, tb), []).append((k, ia, ib))

    results: list = [None] * len(pairs)

    for (ta, tb), items in groups.items():
        _batch_bool_group_3d(colliders, ta, tb, items, results)

    return [(int(pairs[k, 0]), int(pairs[k, 1]), bool(results[k]))
            for k in range(len(pairs))]


def _batch_bool_group_3d(colliders, ta, tb, items, results):
    """Process a homogeneous group of pairs using Cython batch calls."""
    m = len(items)
    S, C, Y = int(ColliderType.SPHERE), int(ColliderType.CUBE), int(ColliderType.CYLINDER)

    # --- Sphere vs Sphere ---
    if ta == S and tb == S and _BATCH_CYTHON:
        ca_arr = np.empty((m, 3), dtype=np.float64)
        ra_arr = np.empty(m, dtype=np.float64)
        cb_arr = np.empty((m, 3), dtype=np.float64)
        rb_arr = np.empty(m, dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            s = colliders[ia].sphere; ca_arr[i] = s[0]; ra_arr[i] = s[1]
            s = colliders[ib].sphere; cb_arr[i] = s[0]; rb_arr[i] = s[1]
        hits = _cy_ss_bool(
            np.ascontiguousarray(ca_arr), np.ascontiguousarray(ra_arr),
            np.ascontiguousarray(cb_arr), np.ascontiguousarray(rb_arr),
        )
        for i, (k, _, _) in enumerate(items):
            results[k] = bool(hits[i])
        return

    # --- OBB vs OBB ---
    if ta == C and tb == C and _BATCH_CYTHON:
        ca_arr = np.empty((m, 3), dtype=np.float64)
        aa_arr = np.empty((m, 3, 3), dtype=np.float64)
        ea_arr = np.empty((m, 3), dtype=np.float64)
        cb_arr = np.empty((m, 3), dtype=np.float64)
        ab_arr = np.empty((m, 3, 3), dtype=np.float64)
        eb_arr = np.empty((m, 3), dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            o = colliders[ia].obb; ca_arr[i] = o[0]; aa_arr[i] = o[1]; ea_arr[i] = o[2]
            o = colliders[ib].obb; cb_arr[i] = o[0]; ab_arr[i] = o[1]; eb_arr[i] = o[2]
        hits = _cy_oo_bool(
            np.ascontiguousarray(ca_arr), np.ascontiguousarray(aa_arr),
            np.ascontiguousarray(ea_arr), np.ascontiguousarray(cb_arr),
            np.ascontiguousarray(ab_arr), np.ascontiguousarray(eb_arr),
        )
        for i, (k, _, _) in enumerate(items):
            results[k] = bool(hits[i])
        return

    # --- Sphere vs OBB ---
    if ta == S and tb == C and _BATCH_CYTHON:
        sc_arr = np.empty((m, 3), dtype=np.float64)
        sr_arr = np.empty(m, dtype=np.float64)
        oc_arr = np.empty((m, 3), dtype=np.float64)
        oa_arr = np.empty((m, 3, 3), dtype=np.float64)
        oe_arr = np.empty((m, 3), dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            s = colliders[ia].sphere; sc_arr[i] = s[0]; sr_arr[i] = s[1]
            o = colliders[ib].obb; oc_arr[i] = o[0]; oa_arr[i] = o[1]; oe_arr[i] = o[2]
        hits = _cy_so_bool(
            np.ascontiguousarray(sc_arr), np.ascontiguousarray(sr_arr),
            np.ascontiguousarray(oc_arr), np.ascontiguousarray(oa_arr),
            np.ascontiguousarray(oe_arr),
        )
        for i, (k, _, _) in enumerate(items):
            results[k] = bool(hits[i])
        return

    # --- Fallback: per-pair using existing collision_bool functions ---
    from engine.d3.physics.collision_bool import objects_collide
    for _, ia, ib in items:
        k = _
        results[k] = objects_collide(colliders[ia], colliders[ib])


# ---------------------------------------------------------------------------
# Batch narrowphase (manifold)
# ---------------------------------------------------------------------------

def batch_narrowphase_manifold_3d(
    colliders: Sequence[Collider3D],
    pairs: np.ndarray,
) -> List[Tuple[int, int, Optional[CollisionManifold]]]:
    """Batch manifold computation for every pair.

    *pairs* is (M, 2) int32 indexing into *colliders*.
    Returns list of ``(idx_a, idx_b, manifold_or_None)``.
    """
    if len(pairs) == 0:
        return []

    # Group pairs by type combination
    groups: dict = {}
    for k in range(len(pairs)):
        ia, ib = int(pairs[k, 0]), int(pairs[k, 1])
        ta, tb = _ctype(colliders[ia]), _ctype(colliders[ib])
        groups.setdefault((ta, tb), []).append((k, ia, ib))

    results: list = [None] * len(pairs)

    for (ta, tb), items in groups.items():
        _batch_manifold_group_3d(colliders, ta, tb, items, results)

    return [(int(pairs[k, 0]), int(pairs[k, 1]), results[k])
            for k in range(len(pairs))]


def _batch_manifold_group_3d(colliders, ta, tb, items, results):
    """Process a homogeneous group using Cython batch manifold calls."""
    m = len(items)
    S, C = int(ColliderType.SPHERE), int(ColliderType.CUBE)

    # --- Sphere vs Sphere ---
    if ta == S and tb == S and _BATCH_CYTHON:
        ca_arr = np.empty((m, 3), dtype=np.float64)
        ra_arr = np.empty(m, dtype=np.float64)
        cb_arr = np.empty((m, 3), dtype=np.float64)
        rb_arr = np.empty(m, dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            s = colliders[ia].sphere; ca_arr[i] = s[0]; ra_arr[i] = s[1]
            s = colliders[ib].sphere; cb_arr[i] = s[0]; rb_arr[i] = s[1]
        hit, norms, deps, cons = _cy_ss_man(
            np.ascontiguousarray(ca_arr), np.ascontiguousarray(ra_arr),
            np.ascontiguousarray(cb_arr), np.ascontiguousarray(rb_arr),
        )
        for i, (k, _, _) in enumerate(items):
            if hit[i]:
                results[k] = _make_manifold(norms[i], deps[i], cons[i])
        return

    # --- OBB vs OBB ---
    if ta == C and tb == C and _BATCH_CYTHON:
        ca_arr = np.empty((m, 3), dtype=np.float64)
        aa_arr = np.empty((m, 3, 3), dtype=np.float64)
        ea_arr = np.empty((m, 3), dtype=np.float64)
        cb_arr = np.empty((m, 3), dtype=np.float64)
        ab_arr = np.empty((m, 3, 3), dtype=np.float64)
        eb_arr = np.empty((m, 3), dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            o = colliders[ia].obb; ca_arr[i] = o[0]; aa_arr[i] = o[1]; ea_arr[i] = o[2]
            o = colliders[ib].obb; cb_arr[i] = o[0]; ab_arr[i] = o[1]; eb_arr[i] = o[2]
        hit, norms, deps = _cy_oo_man(
            np.ascontiguousarray(ca_arr), np.ascontiguousarray(aa_arr),
            np.ascontiguousarray(ea_arr), np.ascontiguousarray(cb_arr),
            np.ascontiguousarray(ab_arr), np.ascontiguousarray(eb_arr),
        )
        for i, (k, ia, ib) in enumerate(items):
            if hit[i]:
                normal = norms[i]
                depth = float(deps[i])
                ca_o = colliders[ia].obb
                cb_o = colliders[ib].obb
                contact = _obb_contact_point(
                    ca_o[0], ca_o[1], ca_o[2],
                    cb_o[0], cb_o[1], cb_o[2],
                    normal, depth,
                )
                multi = _obb_multi_contact_points(
                    ca_o[0], ca_o[1], ca_o[2],
                    cb_o[0], cb_o[1], cb_o[2],
                    normal, depth,
                )
                results[k] = _make_manifold(normal, depth, contact, multi)
        return

    # --- Fallback: per-pair using existing manifold functions ---
    from engine.d3.physics.collision_manifold import get_collision_manifold
    for k_idx, ia, ib in items:
        results[k_idx] = get_collision_manifold(colliders[ia], colliders[ib])


# ---------------------------------------------------------------------------
# End-to-end batch collision packing (AABB + broadphase + type grouping)
# ---------------------------------------------------------------------------

def batch_collision_pack_e2e(
    colliders: Sequence[Collider3D],
) -> tuple:
    """End-to-end collision packing: extract AABBs, broadphase, and group pair types.

    Moves the AABB extraction, numpy array construction, and pair-type
    grouping into a single Cython call when available, avoiding repeated
    Python ↔ C boundary crossings.

    Returns
    -------
    (pairs, pair_types) :
        pairs      : (M, 2) int32 – overlapping pair indices (i < j).
        pair_types : (M, 2) int32 – (type_a, type_b) for each pair.
    """
    n = len(colliders)
    if n < 2:
        return (np.empty((0, 2), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32))

    mins = np.empty((n, 3), dtype=np.float64)
    maxs = np.empty((n, 3), dtype=np.float64)
    types = np.empty(n, dtype=np.int32)
    valid = np.ones(n, dtype=np.uint8)

    for i, c in enumerate(colliders):
        if c.aabb is None:
            valid[i] = 0
            mins[i] = 0.0
            maxs[i] = 0.0
        else:
            mins[i] = c.aabb[0]
            maxs[i] = c.aabb[1]
        types[i] = _ctype(c)

    if _BATCH_CYTHON:
        return _cy_pack3d(
            np.ascontiguousarray(mins, dtype=np.float64),
            np.ascontiguousarray(maxs, dtype=np.float64),
            np.ascontiguousarray(types, dtype=np.int32),
            np.ascontiguousarray(valid, dtype=np.uint8),
        )

    # Pure-Python fallback: broadphase then manually build pair types
    pairs_arr = batch_broadphase_3d(colliders)
    if len(pairs_arr) == 0:
        return (np.empty((0, 2), dtype=np.int32),
                np.empty((0, 2), dtype=np.int32))

    pair_types = np.empty((len(pairs_arr), 2), dtype=np.int32)
    for k in range(len(pairs_arr)):
        ia, ib = int(pairs_arr[k, 0]), int(pairs_arr[k, 1])
        pair_types[k, 0] = types[ia]
        pair_types[k, 1] = types[ib]

    return (pairs_arr, pair_types)


# ---------------------------------------------------------------------------
# Batch frustum culling
# ---------------------------------------------------------------------------

def batch_frustum_cull_spheres(
    centers: np.ndarray,
    radii: np.ndarray,
    planes: np.ndarray,
) -> np.ndarray:
    """Batch frustum cull for packed sphere centers/radii.

    Parameters
    ----------
    centers : (N, 3) float – world-space sphere centres.
    radii   : (N,) float – sphere radii.
    planes  : (6, 4) float – frustum planes (inward normals).

    Returns
    -------
    ndarray bool (N,) – True if sphere is inside / intersects the frustum.
    """
    n = int(len(radii)) if radii is not None else 0
    if n == 0:
        return np.empty(0, dtype=bool)

    centers_c = np.ascontiguousarray(centers, dtype=np.float64)
    radii_c = np.ascontiguousarray(radii, dtype=np.float64)
    planes_c = np.ascontiguousarray(planes, dtype=np.float32)

    if _BATCH_CYTHON:
        return _cy_frustum_cull(centers_c, radii_c, planes_c)

    result = np.ones(n, dtype=bool)
    for i in range(n):
        cx, cy, cz = centers_c[i]
        r = float(radii_c[i])
        for p in range(6):
            dist = (
                planes_c[p, 0] * cx
                + planes_c[p, 1] * cy
                + planes_c[p, 2] * cz
                + planes_c[p, 3]
            )
            if dist < -r:
                result[i] = False
                break
    return result


def batch_frustum_cull(
    colliders: Sequence[Collider3D],
    planes: np.ndarray,
) -> np.ndarray:
    """Batch frustum cull: test all collider bounding spheres against frustum planes.

    Parameters
    ----------
    colliders : sequence of Collider3D (must have sphere or aabb set).
    planes    : (6, 4) float32 frustum planes (normals pointing inward).

    Returns
    -------
    ndarray bool (N,) – True if the collider is visible (inside frustum).
    """
    n = len(colliders)
    if n == 0:
        return np.empty(0, dtype=bool)

    centers = np.empty((n, 3), dtype=np.float64)
    radii = np.empty(n, dtype=np.float64)

    for i, c in enumerate(colliders):
        if c.sphere is not None:
            centers[i] = c.sphere[0]
            radii[i] = float(c.sphere[1])
        elif c.aabb is not None:
            amin, amax = c.aabb
            centers[i] = (np.asarray(amin) + np.asarray(amax)) * 0.5
            half = (np.asarray(amax) - np.asarray(amin)) * 0.5
            radii[i] = float(np.linalg.norm(half))
        else:
            # No bounds: always visible
            centers[i] = 0.0
            radii[i] = 1e10

    return batch_frustum_cull_spheres(centers, radii, planes)


# ---------------------------------------------------------------------------
# Batch continuous collision sweep
# ---------------------------------------------------------------------------

def batch_continuous_sweep(
    colliders: Sequence[Collider3D],
    prev_positions: np.ndarray,
    curr_positions: np.ndarray,
    step_size: float = 0.1,
) -> tuple:
    """Compute swept AABBs and find broadphase pairs for continuous movers.

    Parameters
    ----------
    colliders : sequence of Collider3D.
    prev_positions : (N, 3) previous frame world positions.
    curr_positions : (N, 3) current frame world positions.
    step_size : float – sweep substep size.

    Returns
    -------
    (swept_mins, swept_maxs, cont_pairs, step_counts) :
        swept_mins/maxs : (N, 3) – expanded AABBs.
        cont_pairs      : (M, 2) int32 – continuous candidate pairs.
        step_counts     : (N,) int32 – substeps per continuous body.
    """
    from engine.d3.physics.types import CollisionMode

    n = len(colliders)
    if n == 0:
        return (np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.int32),
                np.empty(0, dtype=np.int32))

    is_cont = np.zeros(n, dtype=np.uint8)
    half_ext = np.zeros((n, 3), dtype=np.float64)

    for i, c in enumerate(colliders):
        if c.collision_mode == CollisionMode.CONTINUOUS:
            is_cont[i] = 1
        if c.aabb is not None:
            amin, amax = c.aabb
            half_ext[i] = (np.asarray(amax, dtype=np.float64) -
                           np.asarray(amin, dtype=np.float64)) * 0.5

    if _BATCH_CYTHON:
        return _cy_cont_sweep(
            np.ascontiguousarray(prev_positions, dtype=np.float64),
            np.ascontiguousarray(curr_positions, dtype=np.float64),
            np.ascontiguousarray(half_ext, dtype=np.float64),
            np.ascontiguousarray(is_cont, dtype=np.uint8),
            float(step_size),
        )

    # Pure-Python fallback
    sw_mins = np.empty((n, 3), dtype=np.float64)
    sw_maxs = np.empty((n, 3), dtype=np.float64)
    step_counts = np.ones(n, dtype=np.int32)

    for i in range(n):
        if is_cont[i] == 0:
            sw_mins[i] = curr_positions[i] - half_ext[i]
            sw_maxs[i] = curr_positions[i] + half_ext[i]
        else:
            delta = curr_positions[i] - prev_positions[i]
            speed = float(np.linalg.norm(delta))
            if speed > 1e-6:
                step_counts[i] = max(1, int(speed / step_size))
            # Union of AABB at prev and curr
            for j in range(3):
                mn = min(prev_positions[i, j], curr_positions[i, j]) - half_ext[i, j]
                mx = max(prev_positions[i, j], curr_positions[i, j]) + half_ext[i, j]
                sw_mins[i, j] = mn
                sw_maxs[i, j] = mx

    # Find pairs with at least one continuous (set avoids O(n²) list scans)
    pair_set = set()
    for i in range(n):
        if is_cont[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if (sw_maxs[i, 0] >= sw_mins[j, 0] and sw_mins[i, 0] <= sw_maxs[j, 0] and
                sw_maxs[i, 1] >= sw_mins[j, 1] and sw_mins[i, 1] <= sw_maxs[j, 1] and
                sw_maxs[i, 2] >= sw_mins[j, 2] and sw_mins[i, 2] <= sw_maxs[j, 2]):
                pair_set.add((min(i, j), max(i, j)))

    if not pair_set:
        return (sw_mins, sw_maxs, np.empty((0, 2), dtype=np.int32), step_counts)

    return (sw_mins, sw_maxs, np.array(list(pair_set), dtype=np.int32), step_counts)
