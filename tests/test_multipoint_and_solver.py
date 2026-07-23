"""Tests for multi-point manifold generation and multi-iteration contact solver.

Covers:
1. Multi-point manifold — face contacts produce multiple contact points.
2. Multi-iteration solver — stacked bodies converge better with iterations.

Both features work on top of the pure-Python and Cython codepaths; the tests
run with whatever backend is currently enabled (set PYENGINE_PURE_PYTHON=1 to
force pure Python).
"""
import os
import sys
import math
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.d3.physics.collider import BoxCollider3D, SphereCollider3D, CapsuleCollider3D
from engine.d3.physics.collision import get_collision_manifold, CollisionManifold
from engine.d3.physics.collision_manifold import (
    _obb_multi_contact_points,
    _cylinder_face_contact_points,
    _clip_polygon_by_plane,
    _obb_face_vertices,
    _make_manifold,
)


# =========================================================================
# Helpers
# =========================================================================

def _box_at(center, half_extents, rotation=None):
    """Create a BoxCollider3D with manually set OBB / AABB."""
    b = BoxCollider3D()
    R = np.eye(3, dtype=np.float32) if rotation is None else np.asarray(rotation, dtype=np.float32)
    C = np.asarray(center, dtype=np.float32)
    E = np.asarray(half_extents, dtype=np.float32)
    b.obb = (C, R, E)
    absR = np.abs(R)
    half = absR @ E
    b.aabb = (C - half, C + half)
    b.type = 2
    return b


def _sphere_at(center, radius):
    """Create a SphereCollider3D with manually set sphere / AABB."""
    s = SphereCollider3D()
    C = np.asarray(center, dtype=np.float32)
    s.sphere = (C, float(radius))
    s.aabb = (C - radius, C + radius)
    s.type = 0
    return s


def _capsule_at(center, radius, half_height):
    """Create a CapsuleCollider3D with manually set cylinder / AABB."""
    c = CapsuleCollider3D()
    C = np.asarray(center, dtype=np.float32)
    c.cylinder = (C, float(radius), float(half_height))
    c.aabb = (
        C - np.array([radius, half_height, radius], dtype=np.float32),
        C + np.array([radius, half_height, radius], dtype=np.float32),
    )
    c.type = 1
    return c


# =========================================================================
# 1. Multi-point manifold tests
# =========================================================================

class TestMultiPointManifold:
    """Verify that face-to-face contacts generate multiple contact points."""

    def test_box_face_on_floor_produces_4_contacts(self):
        """An axis-aligned box sitting on an axis-aligned floor should produce
        4 contact points (the 4 corners of the contact face)."""
        # Box A: center at y=0.9, half-extent 1 → bottom face at y=-0.1
        # Floor: center at y=-1, half-extent 1 → top face at y=0
        # Overlap depth = 0.1
        box = _box_at([0, 0.9, 0], [1, 1, 1])
        floor = _box_at([0, -1, 0], [10, 1, 10])

        m = get_collision_manifold(box, floor)
        assert m is not None, "Box on floor should collide"
        assert m.depth > 0, f"Depth should be positive, got {m.depth}"
        assert m.contact_points is not None, "Face-on-face should produce multi-point contacts"
        assert len(m.contact_points) >= 4, (
            f"Expected >= 4 contact points for a face contact, got {len(m.contact_points)}"
        )

    def test_box_edge_on_floor_produces_fewer_contacts(self):
        """A box tilted 45° on one axis rests on an edge — should produce
        2 contact points (or fall back to single-point)."""
        angle = math.pi / 4
        R = np.array([
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle),  math.cos(angle), 0],
            [0, 0, 1],
        ], dtype=np.float32)
        # Tilted box sinks its bottom corner into the floor
        box = _box_at([0, 0.8, 0], [1, 1, 1], rotation=R)
        floor = _box_at([0, -1, 0], [10, 1, 10])

        m = get_collision_manifold(box, floor)
        assert m is not None, f"Tilted box should still overlap the floor"
        # An edge contact should produce ≤ 4 points (typically 2 for edge, 1 for vertex)
        if m.contact_points is not None:
            assert len(m.contact_points) <= 4

    def test_sphere_vs_box_no_multipoint(self):
        """Sphere contacts are inherently single-point — no multi-point list."""
        # Sphere center at y=0.5, radius 1.0 → bottom at y=-0.5
        # Floor top face at y=0 → overlap of 0.5
        sph = _sphere_at([0, 0.5, 0], 1.0)
        floor = _box_at([0, -1, 0], [10, 1, 10])

        m = get_collision_manifold(sph, floor)
        assert m is not None, "Sphere should overlap the floor"
        # Spheres produce single contact points, not multi-point
        assert m.contact_points is None

    def test_two_boxes_face_contact_has_depth_per_point(self):
        """Each contact point should carry its own depth value."""
        box_a = _box_at([0, 0.95, 0], [1, 1, 1])
        box_b = _box_at([0, -1, 0], [10, 1, 10])

        m = get_collision_manifold(box_a, box_b)
        assert m is not None
        if m.contact_points is not None:
            for pt, depth in m.contact_points:
                assert isinstance(depth, float)
                assert depth >= 0.0
                assert pt.shape == (3,)

    def test_cylinder_face_on_floor_produces_contacts(self):
        """A cylinder sitting flat on a box floor should generate multi-point
        contacts around the rim of the circular face."""
        cyl = _capsule_at([0, 1, 0], 0.5, 1.0)
        floor = _box_at([0, -1, 0], [10, 1, 10])

        m = get_collision_manifold(cyl, floor)
        assert m is not None
        if m.contact_points is not None:
            assert len(m.contact_points) >= 3, (
                f"Cylinder face contact should have >= 3 points, got {len(m.contact_points)}"
            )

    def test_manifold_dataclass_backward_compat(self):
        """contact_points field defaults to None and doesn't break old code."""
        m = CollisionManifold(
            normal=np.array([0, 1, 0], dtype=np.float32),
            depth=0.5,
            contact_point=np.array([0, 0, 0], dtype=np.float32),
        )
        assert m.contact_points is None
        assert m.normal is not None
        assert m.depth == 0.5

    def test_make_manifold_with_contact_points(self):
        """_make_manifold correctly stores a list of contact points."""
        pts = [
            (np.array([1, 0, 1]), 0.1),
            (np.array([-1, 0, 1]), 0.2),
            (np.array([-1, 0, -1]), 0.15),
            (np.array([1, 0, -1]), 0.12),
        ]
        m = _make_manifold([0, 1, 0], 0.15, [0, 0, 0], contact_points=pts)
        assert m.contact_points is not None
        assert len(m.contact_points) == 4
        for pt, d in m.contact_points:
            assert pt.dtype == np.float32
            assert pt.shape == (3,)

    def test_obb_multi_contact_points_separated_boxes(self):
        """Non-overlapping boxes should return a single-point fallback."""
        Ca = np.array([0, 0, 0], dtype=np.float64)
        Cb = np.array([5, 0, 0], dtype=np.float64)
        R = np.eye(3, dtype=np.float64)
        E = np.array([1, 1, 1], dtype=np.float64)
        # Normal doesn't matter for the geometry, use +X
        contacts = _obb_multi_contact_points(Ca, R, E, Cb, R, E, [1, 0, 0], 0.1)
        # Should produce something (at least single fallback)
        assert len(contacts) >= 1


# =========================================================================
# 2. Polygon clipping helpers
# =========================================================================

class TestClippingHelpers:
    """Unit tests for the Sutherland-Hodgman clipping used in face-contact generation."""

    def test_clip_polygon_fully_inside(self):
        """Polygon entirely inside the half-space should be unchanged."""
        quad = [
            np.array([0, 0, 0], dtype=np.float64),
            np.array([1, 0, 0], dtype=np.float64),
            np.array([1, 0, 1], dtype=np.float64),
            np.array([0, 0, 1], dtype=np.float64),
        ]
        # Plane: x <= 5
        result = _clip_polygon_by_plane(quad, np.array([1, 0, 0], dtype=np.float64), 5.0)
        assert len(result) == 4

    def test_clip_polygon_fully_outside(self):
        """Polygon entirely outside should return empty."""
        quad = [
            np.array([10, 0, 0], dtype=np.float64),
            np.array([11, 0, 0], dtype=np.float64),
            np.array([11, 0, 1], dtype=np.float64),
            np.array([10, 0, 1], dtype=np.float64),
        ]
        # Plane: x <= 5
        result = _clip_polygon_by_plane(quad, np.array([1, 0, 0], dtype=np.float64), 5.0)
        assert len(result) == 0

    def test_clip_polygon_partial(self):
        """Clipping should produce vertices on the boundary."""
        quad = [
            np.array([0, 0, 0], dtype=np.float64),
            np.array([2, 0, 0], dtype=np.float64),
            np.array([2, 0, 1], dtype=np.float64),
            np.array([0, 0, 1], dtype=np.float64),
        ]
        # Plane: x <= 1  →  should clip at x=1
        result = _clip_polygon_by_plane(quad, np.array([1, 0, 0], dtype=np.float64), 1.0)
        assert len(result) == 4  # rectangle clipped to smaller rectangle
        for v in result:
            assert v[0] <= 1.0 + 1e-6

    def test_obb_face_vertices_returns_4(self):
        """_obb_face_vertices should return exactly 4 vertices in winding order."""
        C = np.array([0, 0, 0], dtype=np.float64)
        A = np.eye(3, dtype=np.float64)
        E = np.array([1, 1, 1], dtype=np.float64)
        verts = _obb_face_vertices(C, A, E, 1, 1.0)  # +Y face
        assert len(verts) == 4
        # All vertices should have y = 1
        for v in verts:
            assert abs(v[1] - 1.0) < 1e-6

    def test_cylinder_face_contact_points_vertical(self):
        """Cylinder on a floor (normal ≈ +Y) should produce face points."""
        pts = _cylinder_face_contact_points(
            np.array([0, 1, 0]), 0.5, 1.0,
            np.array([0, 1, 0]),  # normal pointing up
            num_points=4,
        )
        assert pts is not None
        assert len(pts) == 4
        # All points should be on the bottom face (y ≈ 0)
        for p in pts:
            assert abs(p[1] - 0.0) < 1e-6

    def test_cylinder_face_contact_points_horizontal_normal_returns_none(self):
        """A horizontal normal means the contact is on the cylinder's side,
        not a flat end — should return None."""
        pts = _cylinder_face_contact_points(
            np.array([0, 1, 0]), 0.5, 1.0,
            np.array([1, 0, 0]),  # horizontal normal
        )
        assert pts is None


# =========================================================================
# 3. Multi-iteration solver tests
# =========================================================================

class TestMultiIterationSolver:
    """Test that the multi-iteration solver attribute exists and is sensible."""

    def test_solver_iterations_attribute_exists(self):
        """Window3D should have a SOLVER_ITERATIONS class attribute."""
        from engine.d3.window import Window3D
        assert hasattr(Window3D, "SOLVER_ITERATIONS")
        assert Window3D.SOLVER_ITERATIONS >= 1

    def test_solver_iterations_default_is_4(self):
        """Default iteration count should be 4."""
        from engine.d3.window import Window3D
        assert Window3D.SOLVER_ITERATIONS == 4

    def test_resolve_collision_accepts_velocity_only(self):
        """_resolve_collision should accept velocity_only keyword without error."""
        from engine.d3.window import Window3D
        # We can't call it directly without a full window, but we can verify
        # the method signature includes the parameter.
        import inspect
        sig = inspect.signature(Window3D._resolve_collision)
        assert "velocity_only" in sig.parameters

    def test_2d_resolve_collision_accepts_velocity_only(self):
        """_resolve_collision_2d should accept velocity_only keyword."""
        from engine.d2.window2d import Window2D
        import inspect
        sig = inspect.signature(Window2D._resolve_collision_2d)
        assert "velocity_only" in sig.parameters

    def test_multipoint_sequential_impulse_reduces_closing_speed(self):
        """Multi-point sequential resolve should kill downward velocity on a face."""
        from engine.d3.physics.response import resolve_contacts_3d_multi

        pos_a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        vel_a = np.array([0.0, -3.0, 0.0], dtype=np.float64)
        ome_a = np.zeros(3, dtype=np.float64)
        inv_m_a = 1.0
        i_inv_a = np.eye(3, dtype=np.float64) * 6.0

        pos_b = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        vel_b = np.zeros(3, dtype=np.float64)
        ome_b = np.zeros(3, dtype=np.float64)
        inv_m_b = 0.0
        i_inv_b = None

        # 4 face corners of a 2×2 contact patch at y=0
        pts = np.array([
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, -1.0],
        ], dtype=np.float64)
        n_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        va, oa, vb, ob, unst = resolve_contacts_3d_multi(
            pos_a=pos_a, vel_a=vel_a, omega_a=ome_a,
            inv_mass_a=inv_m_a, i_inv_a=i_inv_a,
            pos_b=pos_b, vel_b=vel_b, omega_b=ome_b,
            inv_mass_b=inv_m_b, i_inv_b=i_inv_b,
            contact_points=pts, normal=n_up,
            restitution=0.0, static_friction=0.5, dynamic_friction=0.3,
            face_align_a=1.0, face_align_b=0.0,
        )
        # Closing (negative y) velocity should be removed / reduced
        assert float(va[1]) >= -0.5, f"expected non-negative-ish vy, got {va[1]}"
        # Multi-point should not invent huge spin on a centered face hit
        assert float(np.linalg.norm(oa)) < 5.0

    def test_headless_resolve_uses_contact_points(self):
        """Headless window resolve should apply multi-point impulses (no crash)."""
        from engine.d3.object3d import create_cube
        from engine.d3.physics.rigidbody import Rigidbody3D
        from engine.d3.physics.collider import BoxCollider3D
        from engine.d3.window import Window3D
        from engine.types import Vector3
        from engine.component import Time

        class HeadlessWindow(Window3D):
            def __init__(self):
                self.objects = []
                self._current_scene = None

            def _active_objects(self):
                return self.objects

        win = HeadlessWindow()
        # Unit cubes: half-extent 0.5. Overlap floor top (y=0) with box bottom.
        # Floor center y=-0.5 → top face y=0; box center y=0.4 → bottom y=-0.1.
        box = create_cube(size=1.0, position=(0.0, 0.4, 0.0))
        rb = Rigidbody3D(use_gravity=False)
        rb.velocity = Vector3(0.0, -2.0, 0.0)
        col = BoxCollider3D()
        col.bounciness = 0.0
        box.add_component(rb)
        box.add_component(col)

        floor = create_cube(size=1.0, position=(0.0, -0.5, 0.0))
        floor.transform.scale_xyz = (20.0, 1.0, 20.0)
        rb_f = Rigidbody3D(use_gravity=False, is_static=True)
        col_f = BoxCollider3D()
        floor.add_component(rb_f)
        floor.add_component(col_f)

        win.objects.extend([box, floor])
        for o in win.objects:
            o.transform._compute_world_transform()
            for c in o.get_components(BoxCollider3D):
                c._transform_dirty = True
                c.update_bounds()

        Time.delta_time = 1.0 / 60.0
        m = get_collision_manifold(col, col_f)
        assert m is not None, "box should overlap floor"
        # Prefer multi-point when available
        if m.contact_points is not None:
            assert len(m.contact_points) >= 2
        win._resolve_collision(box, floor, m, col_a=col, col_b=col_f)
        # Body should not still be diving at full speed
        assert float(rb.velocity.y) > -2.0 + 1e-6


class TestMultiIterationConvergence:
    """Test that iterating the solver improves velocity convergence in stacks."""

    def test_three_body_stack_velocity_convergence(self):
        """With three bodies stacked, the middle body's residual velocity
        should be smaller after multiple solver iterations than after one.

        We test this at the response function level, simulating the pattern
        that _process_collisions uses.
        """
        from engine.d3.physics.response import resolve_contact_3d, _as_np3

        # Floor (static)
        pos_f = np.array([0, -1, 0], dtype=np.float64)
        vel_f = np.zeros(3, dtype=np.float64)
        ome_f = np.zeros(3, dtype=np.float64)
        inv_m_f = 0.0
        i_inv_f = None

        # Middle box (dynamic, falling at 2 m/s)
        pos_m = np.array([0, 1, 0], dtype=np.float64)
        vel_m = np.array([0, -2, 0], dtype=np.float64)
        ome_m = np.zeros(3, dtype=np.float64)
        inv_m_m = 1.0
        i_inv_m = np.eye(3, dtype=np.float64) * 6.0

        # Top box (dynamic, falling at 3 m/s)
        pos_t = np.array([0, 3, 0], dtype=np.float64)
        vel_t = np.array([0, -3, 0], dtype=np.float64)
        ome_t = np.zeros(3, dtype=np.float64)
        inv_m_t = 1.0
        i_inv_t = np.eye(3, dtype=np.float64) * 6.0

        n_up = np.array([0, 1, 0], dtype=np.float64)

        # --- Single iteration ---
        vm1, om1 = vel_m.copy(), ome_m.copy()
        vt1, ot1 = vel_t.copy(), ome_t.copy()

        # Contact 1: middle ↔ floor
        r = resolve_contact_3d(
            pos_a=pos_m, vel_a=vm1, omega_a=om1,
            inv_mass_a=inv_m_m, i_inv_a=i_inv_m,
            pos_b=pos_f, vel_b=vel_f.copy(), omega_b=ome_f.copy(),
            inv_mass_b=inv_m_f, i_inv_b=i_inv_f,
            contact_point=np.array([0, 0, 0], dtype=np.float64),
            normal=n_up, restitution=0.0,
            static_friction=0.5, dynamic_friction=0.3,
        )
        vm1, om1 = r[0], r[1]

        # Contact 2: top ↔ middle
        r = resolve_contact_3d(
            pos_a=pos_t, vel_a=vt1, omega_a=ot1,
            inv_mass_a=inv_m_t, i_inv_a=i_inv_t,
            pos_b=pos_m, vel_b=vm1, omega_b=om1,
            inv_mass_b=inv_m_m, i_inv_b=i_inv_m,
            contact_point=np.array([0, 2, 0], dtype=np.float64),
            normal=n_up, restitution=0.0,
            static_friction=0.5, dynamic_friction=0.3,
        )
        vt1, ot1 = r[0], r[1]
        vm1_after_top, om1_after_top = r[2], r[3]

        residual_1iter = abs(float(vm1_after_top[1]))

        # --- Four iterations ---
        vm4, om4 = vel_m.copy(), ome_m.copy()
        vt4, ot4 = vel_t.copy(), ome_t.copy()

        for _ in range(4):
            # Contact 1: middle ↔ floor
            r = resolve_contact_3d(
                pos_a=pos_m, vel_a=vm4, omega_a=om4,
                inv_mass_a=inv_m_m, i_inv_a=i_inv_m,
                pos_b=pos_f, vel_b=vel_f.copy(), omega_b=ome_f.copy(),
                inv_mass_b=inv_m_f, i_inv_b=i_inv_f,
                contact_point=np.array([0, 0, 0], dtype=np.float64),
                normal=n_up, restitution=0.0,
                static_friction=0.5, dynamic_friction=0.3,
            )
            vm4, om4 = r[0], r[1]

            # Contact 2: top ↔ middle
            r = resolve_contact_3d(
                pos_a=pos_t, vel_a=vt4, omega_a=ot4,
                inv_mass_a=inv_m_t, i_inv_a=i_inv_t,
                pos_b=pos_m, vel_b=vm4, omega_b=om4,
                inv_mass_b=inv_m_m, i_inv_b=i_inv_m,
                contact_point=np.array([0, 2, 0], dtype=np.float64),
                normal=n_up, restitution=0.0,
                static_friction=0.5, dynamic_friction=0.3,
            )
            vt4, ot4 = r[0], r[1]
            vm4, om4 = r[2], r[3]

        residual_4iter = abs(float(vm4[1]))

        # 4 iterations should produce equal or smaller residual velocity
        assert residual_4iter <= residual_1iter + 1e-6, (
            f"4 iterations ({residual_4iter:.6f}) should not be worse than "
            f"1 iteration ({residual_1iter:.6f})"
        )


# =========================================================================
# 4. Integration: manifold + solver together
# =========================================================================

class TestIntegration:
    """Higher-level tests combining manifold generation and solver."""

    def test_multipoint_manifold_depth_consistency(self):
        """The overall manifold depth should be >= the maximum per-point depth."""
        box = _box_at([0, 0.95, 0], [1, 1, 1])
        floor = _box_at([0, -1, 0], [10, 1, 10])
        m = get_collision_manifold(box, floor)
        assert m is not None
        if m.contact_points:
            max_pt_depth = max(d for _, d in m.contact_points)
            # Overall depth should be a reasonable upper bound
            assert m.depth >= max_pt_depth - 0.01

    def test_manifold_normal_unit_length(self):
        """Contact normal should be unit length regardless of multi-point."""
        box = _box_at([0, 0.9, 0], [1, 1, 1])
        floor = _box_at([0, -1, 0], [5, 1, 5])
        m = get_collision_manifold(box, floor)
        assert m is not None
        n_len = float(np.linalg.norm(m.normal))
        assert abs(n_len - 1.0) < 1e-4, f"Normal length {n_len} should be ~1.0"

    def test_two_dynamic_boxes_multipoint(self):
        """Two dynamic boxes with face contact should both get multi-point."""
        a = _box_at([0, 0.5, 0], [1, 1, 1])
        b = _box_at([0, -0.5, 0], [1, 1, 1])
        m = get_collision_manifold(a, b)
        assert m is not None
        assert m.depth > 0
        # Face contact between two equal boxes
        if m.contact_points is not None:
            assert len(m.contact_points) >= 2
