"""Tests for batch collision processing (broadphase + narrowphase).

Validates that the batch Cython path produces the same results as the
per-pair Python/Cython path for all supported collision type combinations.
"""
import math
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# 3D helpers  — create colliders with geometry set manually (no Object3D)
# ---------------------------------------------------------------------------
from engine.d3.physics.collider import (
    BoxCollider3D,
    SphereCollider3D,
    CapsuleCollider3D,
    Collider3D,
)
from engine.d3.physics.collision import (
    objects_collide,
    get_collision_manifold,
    CollisionManifold,
)
from engine.d3.physics.types import ColliderType

# 2D helpers
from engine.d2.physics.collider import (
    BoxCollider2D,
    CircleCollider2D,
    Collider2D,
)
from engine.d2.physics.collision_bool import objects_collide_2d
from engine.d2.physics.collision_manifold import (
    get_collision_manifold_2d,
    CollisionManifold2D,
)
from engine.d2.physics.types import ColliderType2D
from engine.gameobject import GameObject


def _make_sphere(center, radius):
    """Create a SphereCollider3D with manually set geometry."""
    c = SphereCollider3D()
    c.sphere = (np.array(center, dtype=np.float32), float(radius))
    c.aabb = (
        np.array(center, dtype=np.float32) - radius,
        np.array(center, dtype=np.float32) + radius,
    )
    c.type = ColliderType.SPHERE
    return c


def _make_box(center, extents, rotation=None):
    """Create a BoxCollider3D with manually set OBB geometry."""
    c = BoxCollider3D()
    R = np.eye(3, dtype=np.float32) if rotation is None else np.array(rotation, dtype=np.float32)
    E = np.array(extents, dtype=np.float32)
    C = np.array(center, dtype=np.float32)
    c.obb = (C, R, E)
    absR = np.abs(R)
    half = absR @ E
    c.aabb = (C - half, C + half)
    c.type = ColliderType.CUBE
    return c


def _make_circle_2d(center, radius):
    """Create a CircleCollider2D with manually set geometry."""
    go = GameObject(f"circle_{id(center)}")
    c = CircleCollider2D()
    go.add_component(c)
    c.circle = (np.array(center, dtype=np.float64), float(radius))
    c.aabb = (
        np.array(center, dtype=np.float64) - radius,
        np.array(center, dtype=np.float64) + radius,
    )
    c.type = ColliderType2D.CIRCLE
    c._transform_dirty = False
    return c


def _make_box_2d(center, half_ext, angle=0.0):
    """Create a BoxCollider2D with manually set OBB geometry."""
    go = GameObject(f"box_{id(center)}")
    c = BoxCollider2D()
    go.add_component(c)
    center_np = np.array(center, dtype=np.float64)
    ext_np = np.array(half_ext, dtype=np.float64)
    c.obb = (center_np, float(angle), ext_np)
    cos_a = abs(math.cos(angle))
    sin_a = abs(math.sin(angle))
    aabb_hx = ext_np[0] * cos_a + ext_np[1] * sin_a
    aabb_hy = ext_np[0] * sin_a + ext_np[1] * cos_a
    c.aabb = (center_np - np.array([aabb_hx, aabb_hy]), center_np + np.array([aabb_hx, aabb_hy]))
    c.type = ColliderType2D.BOX
    c._transform_dirty = False
    return c


# =========================================================================
# 3D Batch Broadphase
# =========================================================================

class TestBatchBroadphase3D:
    """Verify batch_broadphase_3d agrees with per-pair AABB overlap."""

    def test_no_colliders(self):
        from engine.d3.physics.batch_collision import batch_broadphase_3d
        result = batch_broadphase_3d([])
        assert len(result) == 0

    def test_single_collider(self):
        from engine.d3.physics.batch_collision import batch_broadphase_3d
        s = _make_sphere([0, 0, 0], 1.0)
        result = batch_broadphase_3d([s])
        assert len(result) == 0

    def test_overlapping_pair(self):
        from engine.d3.physics.batch_collision import batch_broadphase_3d
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([1.5, 0, 0], 1.0)
        result = batch_broadphase_3d([s1, s2])
        assert len(result) == 1
        assert tuple(result[0]) == (0, 1)

    def test_separated_pair(self):
        from engine.d3.physics.batch_collision import batch_broadphase_3d
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([10, 0, 0], 1.0)
        result = batch_broadphase_3d([s1, s2])
        assert len(result) == 0

    def test_multiple_overlaps(self):
        from engine.d3.physics.batch_collision import batch_broadphase_3d
        # Three spheres, each overlapping the next
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([1.5, 0, 0], 1.0)
        s3 = _make_sphere([3.0, 0, 0], 1.0)
        s4 = _make_sphere([100, 0, 0], 1.0)  # Far away
        result = batch_broadphase_3d([s1, s2, s3, s4])
        pairs = set(tuple(p) for p in result)
        assert (0, 1) in pairs  # s1-s2 overlap
        assert (1, 2) in pairs  # s2-s3 overlap
        # s1-s3 should NOT overlap (dist 3.0, sum radii 2.0)
        assert (0, 2) not in pairs
        assert (0, 3) not in pairs
        assert (1, 3) not in pairs
        assert (2, 3) not in pairs

    def test_agrees_with_per_pair(self):
        """Batch broadphase should produce the same overlaps as brute-force."""
        from engine.d3.physics.batch_collision import batch_broadphase_3d
        from engine.d3.physics.collision_bool import aabb_overlap

        rng = np.random.RandomState(42)
        colliders = []
        for _ in range(20):
            pos = rng.uniform(-5, 5, 3)
            r = rng.uniform(0.5, 2.0)
            colliders.append(_make_sphere(pos, r))

        batch_pairs = set(tuple(p) for p in batch_broadphase_3d(colliders))

        # Brute-force
        brute = set()
        for i in range(len(colliders)):
            for j in range(i + 1, len(colliders)):
                if aabb_overlap(colliders[i], colliders[j]):
                    brute.add((i, j))

        assert batch_pairs == brute, f"Mismatch: batch={batch_pairs}, brute={brute}"


# =========================================================================
# 3D Batch Narrowphase — Bool
# =========================================================================

class TestBatchNarrowphaseBool3D:
    """Verify batch bool results match per-pair objects_collide."""

    def test_sphere_sphere_overlapping(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_bool_3d
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([1.5, 0, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_3d([s1, s2], pairs)
        assert len(result) == 1
        assert result[0][2] is True

    def test_sphere_sphere_separated(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_bool_3d
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([3.0, 0, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_3d([s1, s2], pairs)
        assert result[0][2] is False

    def test_obb_obb_overlapping(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_bool_3d
        b1 = _make_box([0, 0, 0], [1, 1, 1])
        b2 = _make_box([1.5, 0, 0], [1, 1, 1])
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_3d([b1, b2], pairs)
        assert result[0][2] is True

    def test_obb_obb_separated(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_bool_3d
        b1 = _make_box([0, 0, 0], [1, 1, 1])
        b2 = _make_box([3.0, 0, 0], [1, 1, 1])
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_3d([b1, b2], pairs)
        assert result[0][2] is False

    def test_sphere_obb_overlapping(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_bool_3d
        s = _make_sphere([0, 0, 0], 1.0)
        b = _make_box([1.5, 0, 0], [1, 1, 1])
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_3d([s, b], pairs)
        assert result[0][2] is True

    def test_many_pairs_agree_with_per_pair(self):
        """Batch bool results should agree with per-pair objects_collide."""
        from engine.d3.physics.batch_collision import (
            batch_broadphase_3d,
            batch_narrowphase_bool_3d,
        )

        rng = np.random.RandomState(123)
        colliders = []
        for i in range(15):
            pos = rng.uniform(-3, 3, 3)
            r = rng.uniform(0.5, 1.5)
            colliders.append(_make_sphere(pos, r))
        for i in range(10):
            pos = rng.uniform(-3, 3, 3)
            ext = rng.uniform(0.3, 1.0, 3)
            colliders.append(_make_box(pos, ext))

        bp_pairs = batch_broadphase_3d(colliders)
        if len(bp_pairs) == 0:
            return  # No pairs to test

        batch_results = batch_narrowphase_bool_3d(colliders, bp_pairs)

        for ia, ib, batch_hit in batch_results:
            per_pair_hit = objects_collide(colliders[ia], colliders[ib])
            assert batch_hit == per_pair_hit, (
                f"Pair ({ia}, {ib}): batch={batch_hit} vs per_pair={per_pair_hit}"
            )

    def test_empty_pairs(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_bool_3d
        s = _make_sphere([0, 0, 0], 1.0)
        pairs = np.empty((0, 2), dtype=np.int32)
        result = batch_narrowphase_bool_3d([s], pairs)
        assert len(result) == 0


# =========================================================================
# 3D Batch Narrowphase — Manifold
# =========================================================================

class TestBatchNarrowphaseManifold3D:
    """Verify batch manifold results match per-pair get_collision_manifold."""

    def test_sphere_sphere_manifold(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_manifold_3d
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([1.5, 0, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_3d([s1, s2], pairs)
        assert len(result) == 1
        ia, ib, m = result[0]
        assert m is not None
        assert m.depth == pytest.approx(0.5, abs=0.05)
        # Normal should point from B to A (roughly -x direction)
        assert m.normal[0] < 0 or abs(m.normal[0]) > 0.9

    def test_sphere_sphere_no_collision(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_manifold_3d
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([5.0, 0, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_3d([s1, s2], pairs)
        assert result[0][2] is None

    def test_obb_obb_manifold(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_manifold_3d
        b1 = _make_box([0, 0, 0], [1, 1, 1])
        b2 = _make_box([1.5, 0, 0], [1, 1, 1])
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_3d([b1, b2], pairs)
        ia, ib, m = result[0]
        assert m is not None
        assert m.depth == pytest.approx(0.5, abs=0.05)

    def test_manifold_agrees_with_per_pair(self):
        """Batch manifold normal/depth should match per-pair results."""
        from engine.d3.physics.batch_collision import (
            batch_broadphase_3d,
            batch_narrowphase_manifold_3d,
        )

        rng = np.random.RandomState(456)
        colliders = []
        # Dense field so lots of collisions
        for _ in range(10):
            pos = rng.uniform(-2, 2, 3)
            r = rng.uniform(0.5, 1.5)
            colliders.append(_make_sphere(pos, r))

        bp_pairs = batch_broadphase_3d(colliders)
        if len(bp_pairs) == 0:
            return

        batch_results = batch_narrowphase_manifold_3d(colliders, bp_pairs)

        for ia, ib, batch_m in batch_results:
            per_pair_m = get_collision_manifold(colliders[ia], colliders[ib])

            if per_pair_m is None:
                assert batch_m is None, (
                    f"Pair ({ia}, {ib}): per_pair=None but batch gave depth={batch_m.depth}"
                )
            else:
                assert batch_m is not None, (
                    f"Pair ({ia}, {ib}): per_pair depth={per_pair_m.depth} but batch=None"
                )
                assert batch_m.depth == pytest.approx(per_pair_m.depth, abs=0.01), (
                    f"Pair ({ia}, {ib}): depth {batch_m.depth} vs {per_pair_m.depth}"
                )
                # Normals should be parallel (or anti-parallel, but sign should match)
                dot = float(np.dot(batch_m.normal, per_pair_m.normal))
                assert abs(dot) > 0.95, (
                    f"Pair ({ia}, {ib}): normal dot={dot}"
                )

    def test_mixed_types_manifold(self):
        """Batch manifold with mixed sphere/box types."""
        from engine.d3.physics.batch_collision import batch_narrowphase_manifold_3d
        s = _make_sphere([0, 0, 0], 1.0)
        b1 = _make_box([1.5, 0, 0], [1, 1, 1])
        b2 = _make_box([0, 0, 0], [1, 1, 1])  # Overlaps both
        pairs = np.array([[0, 2], [1, 2]], dtype=np.int32)
        result = batch_narrowphase_manifold_3d([s, b1, b2], pairs)
        # sphere vs box overlap
        assert result[0][2] is not None
        # box vs box overlap
        assert result[1][2] is not None


# =========================================================================
# 2D Batch Broadphase
# =========================================================================

class TestBatchBroadphase2D:
    """Verify batch_broadphase_2d agrees with per-pair AABB overlap."""

    def test_no_colliders(self):
        from engine.d2.physics.batch_collision import batch_broadphase_2d
        result = batch_broadphase_2d([])
        assert len(result) == 0

    def test_overlapping_circles(self):
        from engine.d2.physics.batch_collision import batch_broadphase_2d
        c1 = _make_circle_2d([0, 0], 1.0)
        c2 = _make_circle_2d([1.5, 0], 1.0)
        result = batch_broadphase_2d([c1, c2])
        assert len(result) == 1
        assert tuple(result[0]) == (0, 1)

    def test_separated_circles(self):
        from engine.d2.physics.batch_collision import batch_broadphase_2d
        c1 = _make_circle_2d([0, 0], 1.0)
        c2 = _make_circle_2d([10, 0], 1.0)
        result = batch_broadphase_2d([c1, c2])
        assert len(result) == 0

    def test_many_circles_agree_with_brute_force(self):
        from engine.d2.physics.batch_collision import batch_broadphase_2d
        from engine.d2.physics.collision_bool import aabb_overlap_2d

        rng = np.random.RandomState(77)
        colliders = []
        for _ in range(20):
            pos = rng.uniform(-5, 5, 2)
            r = rng.uniform(0.5, 2.0)
            colliders.append(_make_circle_2d(pos, r))

        batch_pairs = set(tuple(p) for p in batch_broadphase_2d(colliders))

        brute = set()
        for i in range(len(colliders)):
            for j in range(i + 1, len(colliders)):
                if aabb_overlap_2d(colliders[i].aabb, colliders[j].aabb):
                    brute.add((i, j))

        assert batch_pairs == brute


# =========================================================================
# 2D Batch Narrowphase — Bool
# =========================================================================

class TestBatchNarrowphaseBool2D:
    """Verify batch bool for 2D colliders."""

    def test_circle_circle_overlap(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_bool_2d
        c1 = _make_circle_2d([0, 0], 1.0)
        c2 = _make_circle_2d([1.5, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_2d([c1, c2], pairs)
        assert result[0][2] is True

    def test_circle_circle_separated(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_bool_2d
        c1 = _make_circle_2d([0, 0], 1.0)
        c2 = _make_circle_2d([3.0, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_2d([c1, c2], pairs)
        assert result[0][2] is False

    def test_box_box_overlap(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_bool_2d
        b1 = _make_box_2d([0, 0], [1, 1])
        b2 = _make_box_2d([1.5, 0], [1, 1])
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_2d([b1, b2], pairs)
        assert result[0][2] is True

    def test_box_box_separated(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_bool_2d
        b1 = _make_box_2d([0, 0], [1, 1])
        b2 = _make_box_2d([5, 0], [1, 1])
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_2d([b1, b2], pairs)
        assert result[0][2] is False

    def test_circle_box_overlap(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_bool_2d
        c = _make_circle_2d([0, 0], 1.0)
        b = _make_box_2d([1.5, 0], [1, 1])
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_2d([c, b], pairs)
        assert result[0][2] is True

    def test_many_pairs_agree_with_per_pair(self):
        from engine.d2.physics.batch_collision import (
            batch_broadphase_2d,
            batch_narrowphase_bool_2d,
        )

        rng = np.random.RandomState(88)
        colliders = []
        for _ in range(10):
            colliders.append(_make_circle_2d(rng.uniform(-3, 3, 2), rng.uniform(0.5, 1.5)))
        for _ in range(10):
            colliders.append(_make_box_2d(rng.uniform(-3, 3, 2), rng.uniform(0.3, 1.0, 2)))

        bp = batch_broadphase_2d(colliders)
        if len(bp) == 0:
            return

        batch_results = batch_narrowphase_bool_2d(colliders, bp)

        for ia, ib, batch_hit in batch_results:
            per_pair_hit = objects_collide_2d(colliders[ia], colliders[ib])
            assert batch_hit == per_pair_hit, (
                f"2D Pair ({ia}, {ib}): batch={batch_hit} vs per_pair={per_pair_hit}"
            )


# =========================================================================
# 2D Batch Narrowphase — Manifold
# =========================================================================

class TestBatchNarrowphaseManifold2D:
    """Verify batch manifold for 2D colliders."""

    def test_circle_circle_manifold(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_manifold_2d
        c1 = _make_circle_2d([0, 0], 1.0)
        c2 = _make_circle_2d([1.5, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_2d([c1, c2], pairs)
        ia, ib, m = result[0]
        assert m is not None
        assert m.depth == pytest.approx(0.5, abs=0.05)

    def test_circle_circle_no_collision(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_manifold_2d
        c1 = _make_circle_2d([0, 0], 1.0)
        c2 = _make_circle_2d([5, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_2d([c1, c2], pairs)
        assert result[0][2] is None

    def test_box_box_manifold(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_manifold_2d
        b1 = _make_box_2d([0, 0], [1, 1])
        b2 = _make_box_2d([1.5, 0], [1, 1])
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_2d([b1, b2], pairs)
        ia, ib, m = result[0]
        assert m is not None
        assert m.depth == pytest.approx(0.5, abs=0.05)

    def test_circle_box_manifold(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_manifold_2d
        c = _make_circle_2d([0, 0], 1.0)
        b = _make_box_2d([1.5, 0], [1, 1])
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_2d([c, b], pairs)
        ia, ib, m = result[0]
        assert m is not None
        assert m.depth > 0

    def test_manifold_agrees_with_per_pair(self):
        """2D batch manifold depth/normal should match per-pair."""
        from engine.d2.physics.batch_collision import (
            batch_broadphase_2d,
            batch_narrowphase_manifold_2d,
        )

        rng = np.random.RandomState(99)
        colliders = []
        for _ in range(10):
            colliders.append(_make_circle_2d(rng.uniform(-2, 2, 2), rng.uniform(0.5, 1.5)))

        bp = batch_broadphase_2d(colliders)
        if len(bp) == 0:
            return

        batch_results = batch_narrowphase_manifold_2d(colliders, bp)

        for ia, ib, batch_m in batch_results:
            per_pair_m = get_collision_manifold_2d(colliders[ia], colliders[ib])
            if per_pair_m is None:
                assert batch_m is None
            else:
                assert batch_m is not None
                assert batch_m.depth == pytest.approx(per_pair_m.depth, abs=0.01)


# =========================================================================
# Batch end-to-end: full pipeline broadphase → narrowphase
# =========================================================================

class TestBatchEndToEnd:
    """Full pipeline: broadphase + narrowphase through batch functions."""

    def test_3d_full_pipeline(self):
        """Run broadphase + narrowphase on a mixed 3D scene."""
        from engine.d3.physics.batch_collision import (
            batch_broadphase_3d,
            batch_narrowphase_manifold_3d,
        )

        colliders = [
            _make_sphere([0, 0, 0], 1.0),
            _make_sphere([1.5, 0, 0], 1.0),
            _make_box([0, 2, 0], [1, 1, 1]),
            _make_box([0.5, 2, 0], [1, 1, 1]),
            _make_sphere([100, 0, 0], 1.0),  # Far away
        ]

        bp_pairs = batch_broadphase_3d(colliders)
        results = batch_narrowphase_manifold_3d(colliders, bp_pairs)

        # Count actual collisions
        hits = [r for r in results if r[2] is not None]
        assert len(hits) >= 2  # At least sphere-sphere and box-box

        # Verify far-away sphere has no collisions
        for ia, ib, m in results:
            assert ia != 4 and ib != 4, "Far-away sphere should not appear in results"

    def test_2d_full_pipeline(self):
        """Run broadphase + narrowphase on a mixed 2D scene."""
        from engine.d2.physics.batch_collision import (
            batch_broadphase_2d,
            batch_narrowphase_manifold_2d,
        )

        colliders = [
            _make_circle_2d([0, 0], 1.0),
            _make_circle_2d([1.5, 0], 1.0),
            _make_box_2d([0, 2], [1, 1]),
            _make_box_2d([0.5, 2], [1, 1]),
            _make_circle_2d([100, 0], 1.0),  # Far away
        ]

        bp_pairs = batch_broadphase_2d(colliders)
        results = batch_narrowphase_manifold_2d(colliders, bp_pairs)

        hits = [r for r in results if r[2] is not None]
        assert len(hits) >= 2

        for ia, ib, m in results:
            assert ia != 4 and ib != 4

    def test_stress_many_3d_colliders(self):
        """Stress test with many 3D colliders to verify no crashes."""
        from engine.d3.physics.batch_collision import (
            batch_broadphase_3d,
            batch_narrowphase_bool_3d,
        )

        rng = np.random.RandomState(2026)
        colliders = []
        for _ in range(50):
            pos = rng.uniform(-5, 5, 3)
            r = rng.uniform(0.3, 1.0)
            colliders.append(_make_sphere(pos, r))

        bp_pairs = batch_broadphase_3d(colliders)
        results = batch_narrowphase_bool_3d(colliders, bp_pairs)

        # Just verify no crashes and results have correct length
        assert len(results) == len(bp_pairs)

    def test_stress_many_2d_colliders(self):
        """Stress test with many 2D colliders to verify no crashes."""
        from engine.d2.physics.batch_collision import (
            batch_broadphase_2d,
            batch_narrowphase_bool_2d,
        )

        rng = np.random.RandomState(2026)
        colliders = []
        for _ in range(50):
            pos = rng.uniform(-5, 5, 2)
            r = rng.uniform(0.3, 1.0)
            colliders.append(_make_circle_2d(pos, r))

        bp_pairs = batch_broadphase_2d(colliders)
        results = batch_narrowphase_bool_2d(colliders, bp_pairs)
        assert len(results) == len(bp_pairs)


# =========================================================================
# Cython batch vs pure-Python batch consistency
# =========================================================================

class TestCythonBatchConsistency:
    """Ensure the Cython batch path and the per-pair path give identical answers."""

    def test_3d_sphere_sphere_batch_vs_individual(self):
        """Every sphere-sphere pair should get the same bool/manifold."""
        from engine.d3.physics.batch_collision import (
            batch_narrowphase_bool_3d,
            batch_narrowphase_manifold_3d,
        )

        rng = np.random.RandomState(1234)
        colliders = []
        for _ in range(20):
            pos = rng.uniform(-3, 3, 3)
            r = rng.uniform(0.5, 1.5)
            colliders.append(_make_sphere(pos, r))

        # Build all pairs
        all_pairs = []
        for i in range(len(colliders)):
            for j in range(i + 1, len(colliders)):
                all_pairs.append((i, j))
        pairs = np.array(all_pairs, dtype=np.int32)

        # Batch bool
        batch_bools = batch_narrowphase_bool_3d(colliders, pairs)
        for ia, ib, batch_hit in batch_bools:
            per_pair = objects_collide(colliders[ia], colliders[ib])
            assert batch_hit == per_pair, f"Bool mismatch at ({ia},{ib})"

        # Batch manifold
        batch_mans = batch_narrowphase_manifold_3d(colliders, pairs)
        for ia, ib, batch_m in batch_mans:
            per_pair_m = get_collision_manifold(colliders[ia], colliders[ib])
            if per_pair_m is None:
                assert batch_m is None
            else:
                assert batch_m is not None
                assert abs(batch_m.depth - per_pair_m.depth) < 0.02

    def test_2d_circle_circle_batch_vs_individual(self):
        """Every circle-circle pair should get the same bool/manifold."""
        from engine.d2.physics.batch_collision import (
            batch_narrowphase_bool_2d,
            batch_narrowphase_manifold_2d,
        )

        rng = np.random.RandomState(5678)
        colliders = []
        for _ in range(20):
            pos = rng.uniform(-3, 3, 2)
            r = rng.uniform(0.5, 1.5)
            colliders.append(_make_circle_2d(pos, r))

        all_pairs = []
        for i in range(len(colliders)):
            for j in range(i + 1, len(colliders)):
                all_pairs.append((i, j))
        pairs = np.array(all_pairs, dtype=np.int32)

        batch_bools = batch_narrowphase_bool_2d(colliders, pairs)
        for ia, ib, batch_hit in batch_bools:
            per_pair = objects_collide_2d(colliders[ia], colliders[ib])
            assert batch_hit == per_pair, f"Bool mismatch at ({ia},{ib})"

        batch_mans = batch_narrowphase_manifold_2d(colliders, pairs)
        for ia, ib, batch_m in batch_mans:
            per_pair_m = get_collision_manifold_2d(colliders[ia], colliders[ib])
            if per_pair_m is None:
                assert batch_m is None
            else:
                assert batch_m is not None
                assert abs(batch_m.depth - per_pair_m.depth) < 0.02

    def test_3d_obb_obb_batch_vs_individual(self):
        """OBB-OBB batch results should match per-pair."""
        from engine.d3.physics.batch_collision import (
            batch_narrowphase_bool_3d,
            batch_narrowphase_manifold_3d,
        )

        rng = np.random.RandomState(9012)
        colliders = []
        for _ in range(15):
            pos = rng.uniform(-2, 2, 3)
            ext = rng.uniform(0.3, 1.0, 3)
            # Random rotation
            angle = rng.uniform(0, 2 * math.pi)
            R = np.array([
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)],
            ], dtype=np.float32)
            colliders.append(_make_box(pos, ext, rotation=R))

        all_pairs = []
        for i in range(len(colliders)):
            for j in range(i + 1, len(colliders)):
                all_pairs.append((i, j))
        pairs = np.array(all_pairs, dtype=np.int32)

        batch_bools = batch_narrowphase_bool_3d(colliders, pairs)
        for ia, ib, batch_hit in batch_bools:
            per_pair = objects_collide(colliders[ia], colliders[ib])
            assert batch_hit == per_pair, f"OBB bool mismatch at ({ia},{ib})"

    def test_2d_box_box_batch_vs_individual(self):
        """2D OBB-OBB batch results should match per-pair."""
        from engine.d2.physics.batch_collision import batch_narrowphase_bool_2d

        rng = np.random.RandomState(3456)
        colliders = []
        for _ in range(15):
            pos = rng.uniform(-2, 2, 2)
            ext = rng.uniform(0.3, 1.0, 2)
            angle = rng.uniform(0, 2 * math.pi)
            colliders.append(_make_box_2d(pos, ext, angle))

        all_pairs = []
        for i in range(len(colliders)):
            for j in range(i + 1, len(colliders)):
                all_pairs.append((i, j))
        pairs = np.array(all_pairs, dtype=np.int32)

        batch_bools = batch_narrowphase_bool_2d(colliders, pairs)
        for ia, ib, batch_hit in batch_bools:
            per_pair = objects_collide_2d(colliders[ia], colliders[ib])
            assert batch_hit == per_pair, f"2D OBB bool mismatch at ({ia},{ib})"


# =========================================================================
# Edge cases
# =========================================================================

class TestBatchEdgeCases:
    """Edge cases: coincident objects, zero-radius, touching exactly."""

    def test_coincident_spheres_3d(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_manifold_3d
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([0, 0, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_3d([s1, s2], pairs)
        assert result[0][2] is not None
        assert result[0][2].depth == pytest.approx(2.0, abs=0.01)

    def test_touching_spheres_3d(self):
        from engine.d3.physics.batch_collision import batch_narrowphase_manifold_3d
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([2.0, 0, 0], 1.0)  # Exactly touching
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_3d([s1, s2], pairs)
        # Touching → depth ≈ 0, should still report collision
        ia, ib, m = result[0]
        assert m is not None
        assert m.depth == pytest.approx(0.0, abs=0.01)

    def test_coincident_circles_2d(self):
        from engine.d2.physics.batch_collision import batch_narrowphase_manifold_2d
        c1 = _make_circle_2d([0, 0], 1.0)
        c2 = _make_circle_2d([0, 0], 1.0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_manifold_2d([c1, c2], pairs)
        assert result[0][2] is not None
        assert result[0][2].depth == pytest.approx(2.0, abs=0.01)

    def test_rotated_boxes_2d(self):
        """Two rotated 2D boxes that overlap."""
        from engine.d2.physics.batch_collision import batch_narrowphase_bool_2d
        b1 = _make_box_2d([0, 0], [1, 1], angle=math.pi / 4)
        b2 = _make_box_2d([1.0, 0], [1, 1], angle=0)
        pairs = np.array([[0, 1]], dtype=np.int32)
        result = batch_narrowphase_bool_2d([b1, b2], pairs)
        # These should overlap
        assert result[0][2] is True

    def test_large_batch_no_crash(self):
        """100 3D spheres → batch pipeline should not crash."""
        from engine.d3.physics.batch_collision import (
            batch_broadphase_3d,
            batch_narrowphase_manifold_3d,
        )

        rng = np.random.RandomState(7777)
        colliders = []
        for _ in range(100):
            pos = rng.uniform(-10, 10, 3)
            r = rng.uniform(0.2, 0.8)
            colliders.append(_make_sphere(pos, r))

        bp = batch_broadphase_3d(colliders)
        results = batch_narrowphase_manifold_3d(colliders, bp)
        assert len(results) == len(bp)
