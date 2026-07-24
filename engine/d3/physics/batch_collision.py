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
