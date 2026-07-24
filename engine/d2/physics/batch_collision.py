"""Batch collision processing for 2D — broadphase *and* narrowphase.

Mirrors ``engine.d3.physics.batch_collision`` but for 2D colliders (circles,
boxes, capsules, polygons).  Packs geometry into contiguous arrays and sends
them to Cython in a single call per type-group, eliminating per-pair FFI
overhead.

Public API
----------
batch_broadphase_2d(colliders)
    Sweep-and-prune broadphase.  Returns (M, 2) int32 pair-index array.

batch_narrowphase_manifold_2d(colliders, pairs)
    Returns list of ``(idx_a, idx_b, CollisionManifold2D | None)``.

batch_narrowphase_bool_2d(colliders, pairs)
    Returns list of ``(idx_a, idx_b, bool)``.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from engine.d2.physics.collider import Collider2D
from engine.d2.physics.collision_manifold import (
    CollisionManifold2D,
    _make_manifold,
    _obb_contact_point,
)
from engine.d2.physics.types import ColliderType2D

# ---------------------------------------------------------------------------
# Try to load Cython batch module
# ---------------------------------------------------------------------------
try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled")
    from engine.cython.cy_batch_collision import (
        batch_broadphase_2d as _cy_bp2d,
        batch_circle_circle_bool_2d as _cy_cc_bool,
        batch_circle_circle_manifold_2d as _cy_cc_man,
        batch_obb_obb_bool_2d as _cy_bb_bool,
        batch_obb_obb_manifold_2d as _cy_bb_man,
        batch_circle_obb_bool_2d as _cy_co_bool,
        batch_circle_obb_manifold_2d as _cy_co_man,
    )
    _BATCH_CYTHON = True
except Exception:
    _BATCH_CYTHON = False


# ---------------------------------------------------------------------------
# Batch broadphase
# ---------------------------------------------------------------------------

def batch_broadphase_2d(
    colliders: Sequence[Collider2D],
) -> np.ndarray:
    """Sweep-and-prune AABB broadphase for 2D colliders.

    Returns (M, 2) int32 array of overlapping pair indices (i < j).
    """
    n = len(colliders)
    if n < 2:
        return np.empty((0, 2), dtype=np.int32)

    mins = np.empty((n, 2), dtype=np.float64)
    maxs = np.empty((n, 2), dtype=np.float64)
    valid = np.ones(n, dtype=bool)

    for i, c in enumerate(colliders):
        aabb = c.get_world_aabb()
        if aabb is None:
            valid[i] = False
            mins[i] = 0.0
            maxs[i] = 0.0
        else:
            mins[i] = aabb[0]
            maxs[i] = aabb[1]

    if _BATCH_CYTHON:
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < 2:
            return np.empty((0, 2), dtype=np.int32)
        v_mins = np.ascontiguousarray(mins[valid_idx], dtype=np.float64)
        v_maxs = np.ascontiguousarray(maxs[valid_idx], dtype=np.float64)
        raw = _cy_bp2d(v_mins, v_maxs)
        if len(raw) == 0:
            return np.empty((0, 2), dtype=np.int32)
        return valid_idx[raw].astype(np.int32)

    # Pure-Python fallback
    pairs: list = []
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i + 1, n):
            if not valid[j]:
                continue
            if (maxs[i, 0] >= mins[j, 0] and mins[i, 0] <= maxs[j, 0] and
                maxs[i, 1] >= mins[j, 1] and mins[i, 1] <= maxs[j, 1]):
                pairs.append((i, j))
    if not pairs:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(pairs, dtype=np.int32)


# ---------------------------------------------------------------------------
# Batch narrowphase (bool)
# ---------------------------------------------------------------------------

def batch_narrowphase_bool_2d(
    colliders: Sequence[Collider2D],
    pairs: np.ndarray,
) -> List[Tuple[int, int, bool]]:
    """Batch bool collision test for 2D pairs.

    *pairs* is (M, 2) int32 indexing into *colliders*.
    Returns list of ``(idx_a, idx_b, colliding_bool)``.
    """
    if len(pairs) == 0:
        return []

    # Group by type combination
    groups: dict = {}
    for k in range(len(pairs)):
        ia, ib = int(pairs[k, 0]), int(pairs[k, 1])
        ta = int(colliders[ia].type)
        tb = int(colliders[ib].type)
        groups.setdefault((ta, tb), []).append((k, ia, ib))

    results: list = [None] * len(pairs)

    for (ta, tb), items in groups.items():
        _batch_bool_group_2d(colliders, ta, tb, items, results)

    return [(int(pairs[k, 0]), int(pairs[k, 1]), bool(results[k]))
            for k in range(len(pairs))]


def _batch_bool_group_2d(colliders, ta, tb, items, results):
    m = len(items)
    CIRCLE = int(ColliderType2D.CIRCLE)
    BOX = int(ColliderType2D.BOX)

    # --- Circle vs Circle ---
    if ta == CIRCLE and tb == CIRCLE and _BATCH_CYTHON:
        ca_arr = np.empty((m, 2), dtype=np.float64)
        ra_arr = np.empty(m, dtype=np.float64)
        cb_arr = np.empty((m, 2), dtype=np.float64)
        rb_arr = np.empty(m, dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            c = colliders[ia].circle; ca_arr[i] = c[0]; ra_arr[i] = c[1]
            c = colliders[ib].circle; cb_arr[i] = c[0]; rb_arr[i] = c[1]
        hits = _cy_cc_bool(
            np.ascontiguousarray(ca_arr), np.ascontiguousarray(ra_arr),
            np.ascontiguousarray(cb_arr), np.ascontiguousarray(rb_arr),
        )
        for i, (k, _, _) in enumerate(items):
            results[k] = bool(hits[i])
        return

    # --- Box vs Box ---
    if ta == BOX and tb == BOX and _BATCH_CYTHON:
        ca_arr = np.empty((m, 2), dtype=np.float64)
        aa_arr = np.empty(m, dtype=np.float64)
        ea_arr = np.empty((m, 2), dtype=np.float64)
        cb_arr = np.empty((m, 2), dtype=np.float64)
        ab_arr = np.empty(m, dtype=np.float64)
        eb_arr = np.empty((m, 2), dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            o = colliders[ia].obb; ca_arr[i] = o[0]; aa_arr[i] = o[1]; ea_arr[i] = o[2]
            o = colliders[ib].obb; cb_arr[i] = o[0]; ab_arr[i] = o[1]; eb_arr[i] = o[2]
        hits = _cy_bb_bool(
            np.ascontiguousarray(ca_arr), np.ascontiguousarray(aa_arr),
            np.ascontiguousarray(ea_arr), np.ascontiguousarray(cb_arr),
            np.ascontiguousarray(ab_arr), np.ascontiguousarray(eb_arr),
        )
        for i, (k, _, _) in enumerate(items):
            results[k] = bool(hits[i])
        return

    # --- Circle vs Box ---
    if ta == CIRCLE and tb == BOX and _BATCH_CYTHON:
        cc_arr = np.empty((m, 2), dtype=np.float64)
        cr_arr = np.empty(m, dtype=np.float64)
        oc_arr = np.empty((m, 2), dtype=np.float64)
        oa_arr = np.empty(m, dtype=np.float64)
        oe_arr = np.empty((m, 2), dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            c = colliders[ia].circle; cc_arr[i] = c[0]; cr_arr[i] = c[1]
            o = colliders[ib].obb; oc_arr[i] = o[0]; oa_arr[i] = o[1]; oe_arr[i] = o[2]
        hits = _cy_co_bool(
            np.ascontiguousarray(cc_arr), np.ascontiguousarray(cr_arr),
            np.ascontiguousarray(oc_arr), np.ascontiguousarray(oa_arr),
            np.ascontiguousarray(oe_arr),
        )
        for i, (k, _, _) in enumerate(items):
            results[k] = bool(hits[i])
        return

    # --- Fallback: per-pair ---
    from engine.d2.physics.collision_bool import objects_collide_2d
    for k_idx, ia, ib in items:
        results[k_idx] = objects_collide_2d(colliders[ia], colliders[ib])


# ---------------------------------------------------------------------------
# Batch narrowphase (manifold)
# ---------------------------------------------------------------------------

def batch_narrowphase_manifold_2d(
    colliders: Sequence[Collider2D],
    pairs: np.ndarray,
) -> List[Tuple[int, int, Optional[CollisionManifold2D]]]:
    """Batch manifold computation for 2D pairs.

    Returns list of ``(idx_a, idx_b, manifold_or_None)``.
    """
    if len(pairs) == 0:
        return []

    groups: dict = {}
    for k in range(len(pairs)):
        ia, ib = int(pairs[k, 0]), int(pairs[k, 1])
        ta = int(colliders[ia].type)
        tb = int(colliders[ib].type)
        groups.setdefault((ta, tb), []).append((k, ia, ib))

    results: list = [None] * len(pairs)

    for (ta, tb), items in groups.items():
        _batch_manifold_group_2d(colliders, ta, tb, items, results)

    return [(int(pairs[k, 0]), int(pairs[k, 1]), results[k])
            for k in range(len(pairs))]


def _batch_manifold_group_2d(colliders, ta, tb, items, results):
    m = len(items)
    CIRCLE = int(ColliderType2D.CIRCLE)
    BOX = int(ColliderType2D.BOX)

    # --- Circle vs Circle ---
    if ta == CIRCLE and tb == CIRCLE and _BATCH_CYTHON:
        ca_arr = np.empty((m, 2), dtype=np.float64)
        ra_arr = np.empty(m, dtype=np.float64)
        cb_arr = np.empty((m, 2), dtype=np.float64)
        rb_arr = np.empty(m, dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            c = colliders[ia].circle; ca_arr[i] = c[0]; ra_arr[i] = c[1]
            c = colliders[ib].circle; cb_arr[i] = c[0]; rb_arr[i] = c[1]
        hit, norms, deps, cons = _cy_cc_man(
            np.ascontiguousarray(ca_arr), np.ascontiguousarray(ra_arr),
            np.ascontiguousarray(cb_arr), np.ascontiguousarray(rb_arr),
        )
        for i, (k, _, _) in enumerate(items):
            if hit[i]:
                results[k] = _make_manifold(norms[i], deps[i], cons[i])
        return

    # --- Box vs Box ---
    if ta == BOX and tb == BOX and _BATCH_CYTHON:
        ca_arr = np.empty((m, 2), dtype=np.float64)
        aa_arr = np.empty(m, dtype=np.float64)
        ea_arr = np.empty((m, 2), dtype=np.float64)
        cb_arr = np.empty((m, 2), dtype=np.float64)
        ab_arr = np.empty(m, dtype=np.float64)
        eb_arr = np.empty((m, 2), dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            o = colliders[ia].obb; ca_arr[i] = o[0]; aa_arr[i] = o[1]; ea_arr[i] = o[2]
            o = colliders[ib].obb; cb_arr[i] = o[0]; ab_arr[i] = o[1]; eb_arr[i] = o[2]
        hit, norms, deps = _cy_bb_man(
            np.ascontiguousarray(ca_arr), np.ascontiguousarray(aa_arr),
            np.ascontiguousarray(ea_arr), np.ascontiguousarray(cb_arr),
            np.ascontiguousarray(ab_arr), np.ascontiguousarray(eb_arr),
        )
        for i, (k, ia, ib) in enumerate(items):
            if hit[i]:
                normal = norms[i]
                depth = float(deps[i])
                oa = colliders[ia].obb
                ob = colliders[ib].obb
                contact = _obb_contact_point(
                    oa[0], oa[1], oa[2],
                    ob[0], ob[1], ob[2],
                    normal, depth,
                )
                results[k] = _make_manifold(normal, depth, contact)
        return

    # --- Circle vs Box ---
    if ta == CIRCLE and tb == BOX and _BATCH_CYTHON:
        cc_arr = np.empty((m, 2), dtype=np.float64)
        cr_arr = np.empty(m, dtype=np.float64)
        oc_arr = np.empty((m, 2), dtype=np.float64)
        oa_arr = np.empty(m, dtype=np.float64)
        oe_arr = np.empty((m, 2), dtype=np.float64)
        for i, (_, ia, ib) in enumerate(items):
            c = colliders[ia].circle; cc_arr[i] = c[0]; cr_arr[i] = c[1]
            o = colliders[ib].obb; oc_arr[i] = o[0]; oa_arr[i] = o[1]; oe_arr[i] = o[2]
        hit, norms, deps, cons = _cy_co_man(
            np.ascontiguousarray(cc_arr), np.ascontiguousarray(cr_arr),
            np.ascontiguousarray(oc_arr), np.ascontiguousarray(oa_arr),
            np.ascontiguousarray(oe_arr),
        )
        for i, (k, _, _) in enumerate(items):
            if hit[i]:
                results[k] = _make_manifold(norms[i], deps[i], cons[i])
        return

    # --- Fallback: per-pair ---
    from engine.d2.physics.collision_manifold import get_collision_manifold_2d
    for k_idx, ia, ib in items:
        results[k_idx] = get_collision_manifold_2d(colliders[ia], colliders[ib])
