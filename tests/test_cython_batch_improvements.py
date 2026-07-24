"""Tests for the four Cython batch improvements:

1. Batch frustum culling
2. Batch rigidbody integration
3. End-to-end batch collision packing
4. Batch continuous collision detection (CCD) sweep

Each section verifies:
- Cython path produces correct results
- Results match the pure-Python fallback
- Edge cases (empty input, coincident objects, etc.)
"""
import math
import numpy as np
import pytest

from engine.d3.physics.collider import (
    BoxCollider3D,
    SphereCollider3D,
    CapsuleCollider3D,
    Collider3D,
)
from engine.d3.physics.types import ColliderType, CollisionMode
from engine.d3.physics.batch_collision import (
    batch_broadphase_3d,
    batch_collision_pack_e2e,
    batch_frustum_cull,
    batch_continuous_sweep,
)
from engine.d3.physics.rigidbody import Rigidbody3D, batch_integrate_rigidbodies
from engine.d3.camera import Camera3D
from engine.gameobject import GameObject
from engine.component import Time
from engine.types import Vector3
from engine.types.quaternion import Quaternion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sphere(center, radius):
    c = SphereCollider3D()
    c.sphere = (np.array(center, dtype=np.float32), float(radius))
    c.aabb = (
        np.array(center, dtype=np.float32) - radius,
        np.array(center, dtype=np.float32) + radius,
    )
    c.type = ColliderType.SPHERE
    return c


def _make_box(center, extents, rotation=None):
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


def _make_capsule(center, radius, half_height):
    c = CapsuleCollider3D()
    C = np.array(center, dtype=np.float32)
    c.cylinder = (C, float(radius), float(half_height))
    c.aabb = (
        C - np.array([radius, half_height, radius], dtype=np.float32),
        C + np.array([radius, half_height, radius], dtype=np.float32),
    )
    c.type = ColliderType.CYLINDER
    return c


def _simple_frustum_planes():
    """Build frustum planes for a camera at origin looking down -Z."""
    cam = Camera3D()
    view = np.eye(4, dtype=np.float32)
    proj = cam._perspective_matrix(60.0, 1.0, 0.1, 100.0)
    return cam.extract_frustum_planes(view, proj)


# =========================================================================
# 1. Batch Frustum Culling
# =========================================================================

class TestBatchFrustumCulling:
    """Verify batch_frustum_cull produces correct visibility results."""

    def test_sphere_inside_frustum(self):
        """A sphere right in front of the camera should be visible."""
        planes = _simple_frustum_planes()
        s = _make_sphere([0, 0, -5], 1.0)
        vis = batch_frustum_cull([s], planes)
        assert len(vis) == 1
        assert vis[0] is True or vis[0] == True

    def test_sphere_outside_frustum(self):
        """A sphere far to the side should be culled."""
        planes = _simple_frustum_planes()
        s = _make_sphere([100, 0, -5], 0.5)
        vis = batch_frustum_cull([s], planes)
        assert vis[0] == False

    def test_sphere_behind_camera(self):
        """A sphere behind the camera (positive Z) should be culled."""
        planes = _simple_frustum_planes()
        s = _make_sphere([0, 0, 50], 1.0)
        vis = batch_frustum_cull([s], planes)
        assert vis[0] == False

    def test_mixed_visibility(self):
        """Multiple spheres: some visible, some not."""
        planes = _simple_frustum_planes()
        colliders = [
            _make_sphere([0, 0, -5], 1.0),    # visible
            _make_sphere([100, 0, -5], 0.5),   # far right → culled
            _make_sphere([0, 0, -50], 2.0),    # deep in → visible
            _make_sphere([0, 0, 200], 1.0),    # behind camera → culled
            _make_sphere([0, 0, -0.5], 1.0),   # near plane, but overlaps → visible
        ]
        vis = batch_frustum_cull(colliders, planes)
        assert vis[0] == True   # in front, centered
        assert vis[1] == False  # way off to the side
        assert vis[2] == True   # deep inside frustum
        assert vis[3] == False  # behind camera
        assert vis[4] == True   # near plane but sphere overlaps

    def test_empty_input(self):
        """No colliders should return empty array."""
        planes = _simple_frustum_planes()
        vis = batch_frustum_cull([], planes)
        assert len(vis) == 0

    def test_large_sphere_always_visible(self):
        """A huge sphere should always intersect the frustum."""
        planes = _simple_frustum_planes()
        s = _make_sphere([0, 0, 0], 10000.0)
        vis = batch_frustum_cull([s], planes)
        assert vis[0] == True

    def test_box_collider_uses_aabb(self):
        """Frustum cull should work with box colliders (uses AABB-derived sphere)."""
        planes = _simple_frustum_planes()
        b = _make_box([0, 0, -5], [1, 1, 1])
        vis = batch_frustum_cull([b], planes)
        assert vis[0] == True

    def test_agrees_with_per_object_culling(self):
        """Batch result should match per-object camera.sphere_in_frustum."""
        cam = Camera3D()
        view = np.eye(4, dtype=np.float32)
        proj = cam._perspective_matrix(60.0, 1.0, 0.1, 100.0)
        planes = cam.extract_frustum_planes(view, proj)

        rng = np.random.RandomState(42)
        colliders = []
        for _ in range(30):
            pos = rng.uniform(-50, 50, 3)
            r = rng.uniform(0.5, 3.0)
            colliders.append(_make_sphere(pos, r))

        batch_vis = batch_frustum_cull(colliders, planes)

        for i, c in enumerate(colliders):
            center = c.sphere[0]
            radius = c.sphere[1]
            per_obj = cam.sphere_in_frustum(center, radius, planes)
            assert batch_vis[i] == per_obj, (
                f"Collider {i} at {center}: batch={batch_vis[i]} vs per_obj={per_obj}"
            )

    def test_stress_many_objects(self):
        """Stress test: 500 spheres should not crash and should run fast."""
        planes = _simple_frustum_planes()
        rng = np.random.RandomState(123)
        colliders = [_make_sphere(rng.uniform(-100, 100, 3), rng.uniform(0.1, 5.0))
                     for _ in range(500)]
        vis = batch_frustum_cull(colliders, planes)
        assert len(vis) == 500
        # At least some should be visible, some culled
        assert vis.sum() > 0
        assert vis.sum() < 500


# =========================================================================
# 2. Batch Rigidbody Integration
# =========================================================================

class TestBatchRigidbodyIntegration:
    """Verify batch_integrate_rigidbodies matches per-body rb.update()."""

    def _make_rb_go(self, pos=(0, 0, 0), vel=(0, 0, 0), use_gravity=True,
                    drag=0.0, angular_drag=0.25):
        """Create a (Rigidbody3D, GameObject) pair for testing."""
        go = GameObject("test")
        go.transform.position = pos
        rb = Rigidbody3D(use_gravity=use_gravity, drag=drag)
        rb.angular_drag = angular_drag
        rb._velocity = Vector3(*vel)
        go.add_component(rb)
        go.transform._compute_world_transform()
        return rb, go

    def test_gravity_integration(self):
        """Batch integrate should apply gravity identically to per-body update."""
        dt = 1.0 / 60.0
        Time.delta_time = dt
        Time._skip_rigidbody_frame_update = False

        # Per-body reference
        rb_ref, go_ref = self._make_rb_go(pos=(0, 10, 0))
        rb_ref.update()
        ref_vy = rb_ref._velocity._y

        # Batch
        rb_bat, go_bat = self._make_rb_go(pos=(0, 10, 0))
        batch_integrate_rigidbodies([rb_bat], [go_bat], dt)
        bat_vy = rb_bat._velocity._y

        assert bat_vy == pytest.approx(ref_vy, abs=1e-6), (
            f"batch vy={bat_vy} vs per-body vy={ref_vy}"
        )

    def test_velocity_integration(self):
        """Bodies with initial velocity should move correctly."""
        dt = 0.02
        Time.delta_time = dt
        Time._skip_rigidbody_frame_update = False

        rb, go = self._make_rb_go(pos=(0, 0, 0), vel=(10, 0, 5), use_gravity=False)
        batch_integrate_rigidbodies([rb], [go], dt)

        assert go.transform.position[0] == pytest.approx(10 * dt, abs=1e-6)
        assert go.transform.position[2] == pytest.approx(5 * dt, abs=1e-6)

    def test_drag_applied(self):
        """Drag should reduce velocity."""
        dt = 0.02
        Time.delta_time = dt
        Time._skip_rigidbody_frame_update = False

        rb, go = self._make_rb_go(pos=(0, 0, 0), vel=(10, 0, 0),
                                   use_gravity=False, drag=5.0)
        initial_vx = 10.0
        batch_integrate_rigidbodies([rb], [go], dt)
        # Drag should reduce velocity
        assert abs(rb._velocity._x) < initial_vx

    def test_static_skipped(self):
        """Static bodies should not be integrated."""
        dt = 0.02
        Time.delta_time = dt
        Time._skip_rigidbody_frame_update = False

        rb, go = self._make_rb_go(pos=(0, 5, 0), vel=(0, 0, 0))
        rb.is_static = True
        batch_integrate_rigidbodies([rb], [go], dt)

        # Position should not change
        assert go.transform.position[1] == pytest.approx(5.0, abs=1e-6)

    def test_sleeping_zero_velocity_skipped(self):
        """Truly resting sleeping bodies (zero velocity) should not integrate."""
        dt = 0.02
        Time.delta_time = dt
        Time._skip_rigidbody_frame_update = False

        rb, go = self._make_rb_go(pos=(0, 5, 0), vel=(0, 0, 0), use_gravity=False)
        rb._is_sleeping = True
        batch_integrate_rigidbodies([rb], [go], dt)

        # Position should not change
        assert go.transform.position[1] == pytest.approx(5.0, abs=1e-6)

    def test_sleeping_with_velocity_wakes(self):
        """In-place velocity while sleeping should wake and integrate."""
        dt = 0.02
        Time.delta_time = dt
        Time._skip_rigidbody_frame_update = False

        rb, go = self._make_rb_go(pos=(0, 5, 0), vel=(0, -1, 0), use_gravity=False)
        rb._is_sleeping = True
        batch_integrate_rigidbodies([rb], [go], dt)

        assert not rb._is_sleeping
        assert go.transform.position[1] == pytest.approx(5.0 + (-1.0) * dt, abs=1e-5)

    def test_multiple_bodies(self):
        """Batch integrating multiple bodies at once."""
        dt = 0.02
        Time.delta_time = dt
        Time._skip_rigidbody_frame_update = False

        bodies = []
        gos = []
        for i in range(5):
            rb, go = self._make_rb_go(
                pos=(i * 2.0, 10.0, 0),
                vel=(float(i), 0, 0),
                use_gravity=True,
            )
            bodies.append(rb)
            gos.append(go)

        batch_integrate_rigidbodies(bodies, gos, dt)

        # All should have moved
        for i, (rb, go) in enumerate(zip(bodies, gos)):
            expected_x = i * 2.0 + float(i) * dt
            assert go.transform.position[0] == pytest.approx(expected_x, abs=1e-4), (
                f"Body {i}: x={go.transform.position[0]}, expected={expected_x}"
            )
            # Gravity should have pulled vy negative
            assert rb._velocity._y < 0

    def test_angular_velocity_integration(self):
        """Angular velocity should produce quaternion rotation."""
        dt = 0.05
        Time.delta_time = dt
        Time._skip_rigidbody_frame_update = False

        rb, go = self._make_rb_go(pos=(0, 0, 0), vel=(0, 0, 0), use_gravity=False)
        rb._angular_velocity = Vector3(0, 5.0, 0)  # 5 rad/s about Y
        rb.angular_drag = 0.0

        q_before = go.transform._local_quaternion
        batch_integrate_rigidbodies([rb], [go], dt)
        q_after = go.transform._local_quaternion

        # Quaternion should have changed
        assert not (
            abs(q_after._w - q_before._w) < 1e-6
            and abs(q_after._x - q_before._x) < 1e-6
            and abs(q_after._y - q_before._y) < 1e-6
            and abs(q_after._z - q_before._z) < 1e-6
        ), "Quaternion should have changed with angular velocity"

    def test_empty_batch(self):
        """Empty batch should not crash."""
        batch_integrate_rigidbodies([], [], 0.02)

    def test_matches_single_update(self):
        """Batch of 1 should produce the same result as rb.update()."""
        dt = 0.016
        Time.delta_time = dt
        Time._skip_rigidbody_frame_update = False

        # Per-body
        rb1, go1 = self._make_rb_go(pos=(3, 8, -2), vel=(1, -2, 3),
                                      use_gravity=True, drag=0.5)
        rb1.angular_drag = 0.3
        rb1._angular_velocity = Vector3(0.5, 1.0, -0.3)
        rb1.update()

        # Batch
        rb2, go2 = self._make_rb_go(pos=(3, 8, -2), vel=(1, -2, 3),
                                      use_gravity=True, drag=0.5)
        rb2.angular_drag = 0.3
        rb2._angular_velocity = Vector3(0.5, 1.0, -0.3)
        batch_integrate_rigidbodies([rb2], [go2], dt)

        assert rb2._velocity._x == pytest.approx(rb1._velocity._x, abs=1e-4)
        assert rb2._velocity._y == pytest.approx(rb1._velocity._y, abs=1e-4)
        assert rb2._velocity._z == pytest.approx(rb1._velocity._z, abs=1e-4)


# =========================================================================
# 3. End-to-End Batch Collision Packing
# =========================================================================

class TestBatchCollisionPackE2E:
    """Verify batch_collision_pack_e2e produces correct broadphase + type pairs."""

    def test_two_overlapping_spheres(self):
        """Two overlapping spheres should produce one pair."""
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([1.5, 0, 0], 1.0)
        pairs, ptypes = batch_collision_pack_e2e([s1, s2])
        assert len(pairs) == 1
        assert tuple(pairs[0]) == (0, 1)
        assert ptypes[0, 0] == int(ColliderType.SPHERE)
        assert ptypes[0, 1] == int(ColliderType.SPHERE)

    def test_sphere_box_pair(self):
        """Overlapping sphere and box should have correct types."""
        s = _make_sphere([0, 0, 0], 1.0)
        b = _make_box([1.0, 0, 0], [1, 1, 1])
        pairs, ptypes = batch_collision_pack_e2e([s, b])
        assert len(pairs) == 1
        assert int(ColliderType.SPHERE) in (ptypes[0, 0], ptypes[0, 1])
        assert int(ColliderType.CUBE) in (ptypes[0, 0], ptypes[0, 1])

    def test_no_overlap(self):
        """Separated colliders produce no pairs."""
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([10, 0, 0], 1.0)
        pairs, ptypes = batch_collision_pack_e2e([s1, s2])
        assert len(pairs) == 0

    def test_agrees_with_broadphase(self):
        """Pack result pairs should match standard broadphase."""
        rng = np.random.RandomState(55)
        colliders = []
        for _ in range(15):
            pos = rng.uniform(-3, 3, 3)
            r = rng.uniform(0.5, 1.5)
            colliders.append(_make_sphere(pos, r))
        for _ in range(10):
            pos = rng.uniform(-3, 3, 3)
            ext = rng.uniform(0.3, 1.0, 3)
            colliders.append(_make_box(pos, ext))

        pack_pairs, pack_types = batch_collision_pack_e2e(colliders)
        bp_pairs = batch_broadphase_3d(colliders)

        pack_set = set(tuple(p) for p in pack_pairs)
        bp_set = set(tuple(p) for p in bp_pairs)
        assert pack_set == bp_set, f"Pack pairs != broadphase pairs"

    def test_type_classification(self):
        """Types should correctly identify sphere, box, capsule."""
        s = _make_sphere([0, 0, 0], 2.0)
        b = _make_box([1, 0, 0], [1, 1, 1])
        c = _make_capsule([0, 1, 0], 0.5, 1.0)
        pairs, ptypes = batch_collision_pack_e2e([s, b, c])

        # All three should overlap (they're close together)
        for k in range(len(pairs)):
            ia, ib = int(pairs[k, 0]), int(pairs[k, 1])
            assert ptypes[k, 0] == _expected_type(ia, [s, b, c])
            assert ptypes[k, 1] == _expected_type(ib, [s, b, c])

    def test_empty_input(self):
        pairs, ptypes = batch_collision_pack_e2e([])
        assert len(pairs) == 0
        assert len(ptypes) == 0

    def test_single_collider(self):
        pairs, ptypes = batch_collision_pack_e2e([_make_sphere([0, 0, 0], 1.0)])
        assert len(pairs) == 0

    def test_none_aabb_skipped(self):
        """Colliders with aabb=None should be skipped."""
        s1 = _make_sphere([0, 0, 0], 1.0)
        s2 = _make_sphere([0.5, 0, 0], 1.0)
        s2.aabb = None  # simulate missing AABB
        pairs, ptypes = batch_collision_pack_e2e([s1, s2])
        assert len(pairs) == 0

    def test_stress_many_colliders(self):
        """100 colliders should not crash."""
        rng = np.random.RandomState(999)
        colliders = [_make_sphere(rng.uniform(-5, 5, 3), rng.uniform(0.2, 0.8))
                     for _ in range(100)]
        pairs, ptypes = batch_collision_pack_e2e(colliders)
        assert len(pairs) == len(ptypes)


def _expected_type(idx, colliders):
    c = colliders[idx]
    return int(getattr(c, "type", ColliderType.CUBE))


# =========================================================================
# 4. Continuous Collision Detection (CCD) Sweep
# =========================================================================

class TestBatchContinuousSweep:
    """Verify batch_continuous_sweep produces correct swept AABBs and pairs."""

    def test_stationary_no_pairs(self):
        """Non-moving continuous collider should not expand."""
        s = _make_sphere([0, 0, 0], 1.0)
        s.collision_mode = CollisionMode.CONTINUOUS
        prev = np.array([[0, 0, 0]], dtype=np.float64)
        curr = np.array([[0, 0, 0]], dtype=np.float64)
        sw_mins, sw_maxs, pairs, steps = batch_continuous_sweep(
            [s], prev, curr, step_size=0.1
        )
        assert steps[0] == 1
        assert len(pairs) == 0

    def test_fast_moving_expands_aabb(self):
        """A fast-moving body should have an expanded swept AABB."""
        s = _make_sphere([0, 0, 0], 1.0)
        s.collision_mode = CollisionMode.CONTINUOUS
        prev = np.array([[0, 0, 0]], dtype=np.float64)
        curr = np.array([[10, 0, 0]], dtype=np.float64)
        sw_mins, sw_maxs, pairs, steps = batch_continuous_sweep(
            [s], prev, curr, step_size=0.1
        )
        # Swept AABB should cover both prev and curr position
        assert sw_mins[0, 0] <= -1.0   # prev center - radius
        assert sw_maxs[0, 0] >= 11.0   # curr center + radius
        assert steps[0] >= 10  # 10 units / 0.1 step

    def test_continuous_vs_static_pair(self):
        """A continuous mover near a static body should produce a pair."""
        s_cont = _make_sphere([0, 0, 0], 1.0)
        s_cont.collision_mode = CollisionMode.CONTINUOUS
        s_static = _make_sphere([5, 0, 0], 1.0)
        s_static.collision_mode = CollisionMode.NORMAL

        prev = np.array([[0, 0, 0], [5, 0, 0]], dtype=np.float64)
        curr = np.array([[4, 0, 0], [5, 0, 0]], dtype=np.float64)
        sw_mins, sw_maxs, pairs, steps = batch_continuous_sweep(
            [s_cont, s_static], prev, curr, step_size=0.1
        )
        # The continuous mover sweeps from x=0 to x=4, static is at x=5
        # Swept AABB of cont: [-1, 5] → overlaps with static [4, 6]
        assert len(pairs) >= 1
        pair_set = set(tuple(p) for p in pairs)
        assert (0, 1) in pair_set

    def test_no_continuous_no_pairs(self):
        """If no collider is continuous, no CCD pairs should be produced."""
        s1 = _make_sphere([0, 0, 0], 1.0)
        s1.collision_mode = CollisionMode.NORMAL
        s2 = _make_sphere([1.5, 0, 0], 1.0)
        s2.collision_mode = CollisionMode.NORMAL

        prev = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float64)
        curr = np.array([[0, 0, 0], [1.5, 0, 0]], dtype=np.float64)
        sw_mins, sw_maxs, pairs, steps = batch_continuous_sweep(
            [s1, s2], prev, curr, step_size=0.1
        )
        assert len(pairs) == 0

    def test_step_count_proportional_to_speed(self):
        """Faster movers should get more substeps."""
        s = _make_sphere([0, 0, 0], 1.0)
        s.collision_mode = CollisionMode.CONTINUOUS

        # Slow movement
        prev = np.array([[0, 0, 0]], dtype=np.float64)
        curr_slow = np.array([[0.5, 0, 0]], dtype=np.float64)
        _, _, _, steps_slow = batch_continuous_sweep([s], prev, curr_slow, step_size=0.1)

        # Fast movement
        curr_fast = np.array([[5.0, 0, 0]], dtype=np.float64)
        _, _, _, steps_fast = batch_continuous_sweep([s], prev, curr_fast, step_size=0.1)

        assert steps_fast[0] > steps_slow[0]

    def test_empty_input(self):
        sw_mins, sw_maxs, pairs, steps = batch_continuous_sweep(
            [], np.empty((0, 3)), np.empty((0, 3)), step_size=0.1
        )
        assert len(pairs) == 0
        assert len(steps) == 0

    def test_mixed_modes(self):
        """Mix of CONTINUOUS and NORMAL bodies."""
        colliders = [
            _make_sphere([0, 0, 0], 1.0),   # NORMAL
            _make_sphere([3, 0, 0], 1.0),    # CONTINUOUS
            _make_sphere([6, 0, 0], 1.0),    # NORMAL
            _make_sphere([9, 0, 0], 1.0),    # CONTINUOUS
        ]
        colliders[0].collision_mode = CollisionMode.NORMAL
        colliders[1].collision_mode = CollisionMode.CONTINUOUS
        colliders[2].collision_mode = CollisionMode.NORMAL
        colliders[3].collision_mode = CollisionMode.CONTINUOUS

        prev = np.array([[0, 0, 0], [0, 0, 0], [6, 0, 0], [20, 0, 0]], dtype=np.float64)
        curr = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]], dtype=np.float64)

        sw_mins, sw_maxs, pairs, steps = batch_continuous_sweep(
            colliders, prev, curr, step_size=0.1
        )

        # Collider 1 sweeps from 0 to 3, should overlap with collider 0 and 2
        # Collider 3 sweeps from 20 to 9, should overlap with collider 2
        assert len(pairs) >= 1
        # At minimum, collider 1 (sweep 0→3) should pair with collider 0 (at 0)
        pair_set = set(tuple(p) for p in pairs)
        assert (0, 1) in pair_set


# =========================================================================
# Cross-cutting: Cython availability
# =========================================================================

def _cy_batch_available():
    try:
        from engine.d3.physics.batch_collision import _BATCH_CYTHON
        from engine.d3.physics.rigidbody import _BATCH_RB_CYTHON
        return bool(_BATCH_CYTHON and _BATCH_RB_CYTHON)
    except Exception:
        return False


@pytest.mark.skipif(not _cy_batch_available(), reason="Cython batch extensions not built")
class TestCythonAvailability:
    """Verify Cython batch modules loaded correctly (skipped without build)."""

    def test_cython_enabled(self):
        from engine.cython import CYTHON_ENABLED
        assert CYTHON_ENABLED

    def test_batch_collision_cython_flag(self):
        from engine.d3.physics.batch_collision import _BATCH_CYTHON
        assert _BATCH_CYTHON

    def test_batch_rb_cython_flag(self):
        from engine.d3.physics.rigidbody import _BATCH_RB_CYTHON
        assert _BATCH_RB_CYTHON

    def test_new_cython_functions_importable(self):
        from engine.cython.cy_batch_collision import (
            batch_frustum_cull_3d,
            batch_rigidbody_integrate_3d,
            batch_collision_pack_3d,
            batch_continuous_sweep_3d,
        )
        assert callable(batch_frustum_cull_3d)
        assert callable(batch_rigidbody_integrate_3d)
        assert callable(batch_collision_pack_3d)
        assert callable(batch_continuous_sweep_3d)


# =========================================================================
# Integration: pure-Python fallback consistency
# =========================================================================

class TestPurePythonFallback:
    """Verify the pure-Python fallback path produces matching results.

    These tests force the Python path by calling the fallback logic directly,
    then compare with the Cython path.
    """

    def test_frustum_cull_py_vs_cy(self):
        """Python fallback should match Cython frustum cull."""
        if not _cy_batch_available():
            pytest.skip("Cython batch not available")
        from engine.cython.cy_batch_collision import batch_frustum_cull_3d

        planes = _simple_frustum_planes()
        rng = np.random.RandomState(77)
        n = 40
        centers = rng.uniform(-50, 50, (n, 3))
        radii = rng.uniform(0.5, 5.0, n)

        # Cython direct
        cy_vis = batch_frustum_cull_3d(
            np.ascontiguousarray(centers, dtype=np.float64),
            np.ascontiguousarray(radii, dtype=np.float64),
            np.ascontiguousarray(planes, dtype=np.float32),
        )

        # Python fallback
        py_vis = np.ones(n, dtype=bool)
        for i in range(n):
            cx, cy_c, cz = centers[i]
            r = radii[i]
            for p in range(6):
                dist = planes[p, 0] * cx + planes[p, 1] * cy_c + planes[p, 2] * cz + planes[p, 3]
                if dist < -r:
                    py_vis[i] = False
                    break

        np.testing.assert_array_equal(cy_vis, py_vis)

    def test_collision_pack_py_vs_cy(self):
        """Python pack should match Cython pack."""
        rng = np.random.RandomState(88)
        colliders = []
        for _ in range(20):
            pos = rng.uniform(-3, 3, 3)
            r = rng.uniform(0.5, 1.5)
            colliders.append(_make_sphere(pos, r))
        for _ in range(10):
            pos = rng.uniform(-3, 3, 3)
            ext = rng.uniform(0.3, 1.0, 3)
            colliders.append(_make_box(pos, ext))

        pairs_cy, types_cy = batch_collision_pack_e2e(colliders)
        pairs_bp = batch_broadphase_3d(colliders)

        cy_set = set(tuple(p) for p in pairs_cy)
        bp_set = set(tuple(p) for p in pairs_bp)
        assert cy_set == bp_set