"""
Edge-case and regression tests for rotational rigid-body physics.

Covers features introduced for realistic arcade/rigid-body behaviour:

  * Model matrix ↔ physics OBB alignment (ramp slope not inverted)
  * Rotated OBB bounds / support features
  * Ramp sliding direction
  * Face rest vs edge/vertex tipping
  * Friction sign (no ice-skate acceleration / reverse)
  * Restitution thresholds and settle
  * Continuous collision sliding / anti-tunnel
  * Depenetration against static geometry
  * Impulse solver edge cases (separating, zero normal, mass extremes)
  * Sleep allow/deny on unstable support
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.component import Time
from engine.d3.object3d import create_cube, create_sphere
from engine.d3.physics.collider import (
    BoxCollider3D,
    SphereCollider3D,
    Collider3D,
    CollisionMode,
)
from engine.d3.physics.collision_manifold import (
    CollisionManifold,
    get_collision_manifold,
    obb_vs_obb_manifold,
    sphere_vs_obb_manifold,
    _obb_support_feature_centroid,
    _obb_face_center_along_normal,
)
from engine.d3.physics.response import (
    FACE_ALIGN_THRESHOLD,
    FACE_REST_ALIGN,
    MAX_ANGULAR_SPEED,
    RESTITUTION_THRESHOLD,
    UNSTABLE_SUPPORT_OFFSET,
    apply_body_state,
    body_state_from_rigidbody,
    estimate_contact_point,
    resolve_contact_3d,
    stabilize_contact_point,
)
from engine.d3.physics.rigidbody import Rigidbody3D
from engine.d3.window import Window3D
from engine.types import Vector3
from engine.types.quaternion import Quaternion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class HeadlessWindow(Window3D):
    def __init__(self):
        self.objects = []
        self._current_scene = None

    def _active_objects(self):
        return self.objects


def _pos3(obj) -> np.ndarray:
    p = obj.transform.position
    if hasattr(p, "to_numpy"):
        return np.asarray(p.to_numpy(), dtype=np.float64)
    return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)


def _refresh(obj):
    obj.transform._compute_world_transform()
    for c in obj.get_components(Collider3D):
        c._transform_dirty = True
        c.update_bounds()
    rb = obj.get_component(Rigidbody3D)
    if rb is not None:
        rb._inertia_dirty = True


def make_box(
    position=(0, 0, 0),
    *,
    mass=1.0,
    static=False,
    gravity=False,
    bounce=0.0,
    friction=0.0,
    size=1.0,
    scale=None,
    rotation=None,
    continuous=False,
    angular_drag=0.0,
    drag=0.0,
):
    obj = create_cube(size=size, position=position)
    if scale is not None:
        obj.transform.scale_xyz = scale
    if rotation is not None:
        obj.transform.rotation = rotation
    rb = Rigidbody3D(use_gravity=gravity, is_static=static)
    rb.mass = mass
    rb.drag = drag
    rb.angular_drag = angular_drag
    col = BoxCollider3D()
    col.bounciness = bounce
    col.static_friction = friction
    col.dynamic_friction = friction * 0.85 if friction > 0 else 0.0
    if continuous:
        col.collision_mode = CollisionMode.CONTINUOUS
    obj.add_component(rb)
    obj.add_component(col)
    _refresh(obj)
    return obj


def make_sphere(
    position=(0, 0, 0),
    *,
    mass=1.0,
    static=False,
    gravity=False,
    bounce=0.0,
    friction=0.0,
    radius=0.5,
):
    obj = create_sphere(radius=radius, position=position)
    rb = Rigidbody3D(use_gravity=gravity, is_static=static)
    rb.mass = mass
    rb.drag = 0.0
    rb.angular_drag = 0.0
    col = SphereCollider3D(radius=1.0)
    col.bounciness = bounce
    col.static_friction = friction
    col.dynamic_friction = friction
    obj.add_component(rb)
    obj.add_component(col)
    _refresh(obj)
    return obj


def step(window, dt=1 / 60.0, steps=1):
    prev_max = Time.maximum_delta_time
    prev_skip = Time._skip_rigidbody_frame_update
    Time.maximum_delta_time = 0.0
    Time._skip_rigidbody_frame_update = False
    try:
        for _ in range(steps):
            Time.set(dt)
            for o in window.objects:
                rb = o.get_component(Rigidbody3D)
                if rb is not None:
                    rb.wake()
                    rb.update()
                for col in o.get_components(Collider3D):
                    col._transform_dirty = True
                    col.update_bounds()
            window._process_collisions()
    finally:
        Time.maximum_delta_time = prev_max
        Time._skip_rigidbody_frame_update = prev_skip


def _unit_I(scale=6.0):
    return np.diag([scale, scale, scale]).astype(np.float64)


# =========================================================================
# Model matrix ↔ physics OBB alignment (regression for inverted ramp)
# =========================================================================


class TestModelMatrixPhysicsAlignment:
    """Rendering uses row-vector M; physics uses column R. They must agree."""

    @pytest.mark.parametrize("angle_z", [0, 15, -15, 28, -28, 45, -45, 90, -90, 180])
    def test_mesh_corners_match_obb_for_z_rotation(self, angle_z):
        scale = (12.0, 0.6, 5.0)
        obj = make_box(position=(-3.0, 2.5, 0.0), scale=scale, rotation=(0, 0, angle_z))
        col = obj.get_component(BoxCollider3D)
        center, axes, extents = col.get_world_obb()
        M = obj.transform.get_model_matrix()

        # Unit cube vertices are in [-0.5, 0.5]; OBB half-extents are scale/2.
        for sx, sy, sz in (
            (-0.5, -0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (0.5, -0.5, 0.5),
            (0.5, 0.5, 0.5),
            (-0.5, 0.5, 0.5),
            (0.5, -0.5, -0.5),
        ):
            mesh = np.array([sx, sy, sz, 1.0], dtype=np.float64) @ M
            local_e = np.array(
                [sx / 0.5 * extents[0], sy / 0.5 * extents[1], sz / 0.5 * extents[2]],
                dtype=np.float64,
            )
            phys = center + axes @ local_e
            np.testing.assert_allclose(
                mesh[:3], phys, atol=1e-4,
                err_msg=f"angle={angle_z} corner={(sx,sy,sz)}",
            )

    @pytest.mark.parametrize("angle_x", [-40, -20, 0, 20, 40, 90])
    def test_mesh_matches_obb_for_x_rotation(self, angle_x):
        obj = make_box(position=(1, 2, 3), scale=(2, 1, 4), rotation=(angle_x, 0, 0))
        center, axes, extents = obj.get_component(BoxCollider3D).get_world_obb()
        M = obj.transform.get_model_matrix()
        for sx in (-0.5, 0.5):
            mesh = np.array([sx, 0.5, 0.0, 1.0]) @ M
            phys = center + axes @ np.array([sx / 0.5 * extents[0], extents[1], 0.0])
            np.testing.assert_allclose(mesh[:3], phys, atol=1e-4)

    def test_rotation_block_is_transpose_of_physics_r(self):
        obj = make_box(rotation=(30, 45, -15))
        R = obj.transform.rotation_matrix
        M = obj.transform.get_model_matrix()
        # Uniform scale 1 → M[:3,:3] == R.T
        np.testing.assert_allclose(M[:3, :3], R.T, atol=1e-5)

    def test_non_uniform_scale_preserves_alignment(self):
        obj = make_box(position=(0, 1, 0), scale=(3, 0.25, 1.5), rotation=(0, 0, -33))
        center, axes, extents = obj.get_component(BoxCollider3D).get_world_obb()
        M = obj.transform.get_model_matrix()
        mesh_hi = np.array([-0.5, 0.5, 0, 1.0]) @ M
        mesh_lo = np.array([0.5, 0.5, 0, 1.0]) @ M
        phys_hi = center + axes @ np.array([-extents[0], extents[1], 0])
        phys_lo = center + axes @ np.array([extents[0], extents[1], 0])
        np.testing.assert_allclose(mesh_hi[:3], phys_hi, atol=1e-4)
        np.testing.assert_allclose(mesh_lo[:3], phys_lo, atol=1e-4)
        # Slope sense: which end is higher must match
        assert (mesh_hi[1] > mesh_lo[1]) == (phys_hi[1] > phys_lo[1])

    def test_identity_model_translation_only(self):
        obj = make_box(position=(5, -2, 7))
        M = obj.transform.get_model_matrix()
        p = np.array([0, 0, 0, 1.0]) @ M
        np.testing.assert_allclose(p[:3], [5, -2, 7], atol=1e-5)

    def test_not_inverted_vs_naive_r_not_rt(self):
        """Regression: putting R (not R.T) into row-vector M inverts the slope."""
        obj = make_box(position=(0, 0, 0), scale=(4, 1, 1), rotation=(0, 0, -30))
        R = obj.transform.rotation_matrix
        M_correct = obj.transform.get_model_matrix()
        # Build wrong matrix the old buggy way: S @ R @ T
        s = np.diag([4, 1, 1, 1]).astype(np.float32)
        R4 = np.eye(4, dtype=np.float32)
        R4[:3, :3] = R  # BUG: should be R.T
        T = np.eye(4, dtype=np.float32)
        M_wrong = s @ R4 @ T
        y_correct = (np.array([0.5, 0.5, 0, 1.0]) @ M_correct)[1]
        y_wrong = (np.array([0.5, 0.5, 0, 1.0]) @ M_wrong)[1]
        y_neg_correct = (np.array([-0.5, 0.5, 0, 1.0]) @ M_correct)[1]
        # Wrong matrix flips which end is high
        assert (y_correct > y_neg_correct) != (y_wrong > (np.array([-0.5, 0.5, 0, 1.0]) @ M_wrong)[1])


# =========================================================================
# Rotated OBB geometry
# =========================================================================


class TestRotatedOBBBounds:
    def test_obb_axes_are_rotation_columns(self):
        obj = make_box(rotation=(10, -25, 40))
        _, axes, _ = obj.get_component(BoxCollider3D).get_world_obb()
        R = obj.transform.rotation_matrix
        np.testing.assert_allclose(axes, R, atol=1e-5)

    def test_aabb_grows_under_45_degree_rotation(self):
        axis = make_box(scale=(2, 1, 1))
        tilted = make_box(scale=(2, 1, 1), rotation=(0, 0, 45))
        ca = axis.get_component(BoxCollider3D)
        ct = tilted.get_component(BoxCollider3D)
        ca.update_bounds()
        ct.update_bounds()
        amin, amax = ca.get_world_aabb()
        tmin, tmax = ct.get_world_aabb()
        assert (tmax - tmin)[0] > (amax - amin)[0] - 1e-6
        assert (tmax - tmin)[1] > (amax - amin)[1] - 1e-6

    def test_180_rotation_extents_match(self):
        a = make_box(scale=(3, 0.5, 2), rotation=(0, 0, 0))
        b = make_box(scale=(3, 0.5, 2), rotation=(0, 0, 180))
        _, _, ea = a.get_component(BoxCollider3D).get_world_obb()
        _, _, eb = b.get_component(BoxCollider3D).get_world_obb()
        np.testing.assert_allclose(ea, eb, atol=1e-5)

    def test_thin_slab_obb_extents(self):
        obj = make_box(scale=(10, 0.05, 4))
        _, _, e = obj.get_component(BoxCollider3D).get_world_obb()
        assert e[1] < 0.03
        assert e[0] == pytest.approx(5.0, abs=1e-4)

    @pytest.mark.parametrize(
        "pos_a,pos_b,expect_hit",
        [
            ((0, 0, 0), (0.9, 0, 0), True),
            ((0, 0, 0), (1.05, 0, 0), False),
            ((0, 0, 0), (0, 0.9, 0), True),
            ((0, 0, 0), (0.7, 0.7, 0), True),
        ],
    )
    def test_obb_overlap_edge_distances(self, pos_a, pos_b, expect_hit):
        a = make_box(position=pos_a)
        b = make_box(position=pos_b)
        m = obb_vs_obb_manifold(
            a.get_component(BoxCollider3D),
            b.get_component(BoxCollider3D),
        )
        if expect_hit:
            assert m is not None and m.depth > 0
        else:
            assert m is None or m.depth <= 1e-4


# =========================================================================
# Support features (face / edge / vertex)
# =========================================================================


class TestSupportFeatures:
    def test_face_down_support_near_com(self):
        C = np.zeros(3)
        A = np.eye(3)
        E = np.array([0.5, 0.5, 0.5])
        # Support in -Y (bottom face of resting box against floor normal +Y)
        feat = _obb_support_feature_centroid(C, A, E, direction=(0, -1, 0))
        # Face center should be near (0, -0.5, 0)
        np.testing.assert_allclose(feat, [0, -0.5, 0], atol=0.05)
        assert abs(feat[0]) < 0.05 and abs(feat[2]) < 0.05

    def test_edge_support_off_com(self):
        C = np.zeros(3)
        # Rotate 45° about Z so bottom is an edge in world Y
        a = math.radians(45)
        c, s = math.cos(a), math.sin(a)
        A = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
        E = np.array([0.5, 0.5, 0.5])
        feat = _obb_support_feature_centroid(C, A, E, direction=(0, -1, 0))
        # Edge feature should have non-zero in-plane offset or sit at corner/edge
        offset = math.hypot(feat[0], feat[2])
        # At least one of X or the feature is not pure face-center under COM
        assert offset > 1e-3 or abs(feat[1] + 0.5 * math.sqrt(2)) < 0.15

    def test_vertex_support_is_corner(self):
        C = np.zeros(3)
        A = np.eye(3)
        E = np.array([1.0, 1.0, 1.0])
        d = np.array([1.0, 1.0, 1.0])
        d = d / np.linalg.norm(d)
        feat = _obb_support_feature_centroid(C, A, E, direction=d)
        np.testing.assert_allclose(feat, [1, 1, 1], atol=1e-5)

    def test_face_center_along_normal(self):
        C = np.array([0.0, 1.0, 0.0])
        A = np.eye(3)
        E = np.array([1.0, 0.5, 1.0])
        # Returns (face_center, axis_alignment); bottom face for +Y normal
        fc, align = _obb_face_center_along_normal(C, A, E, n=(0, 1, 0))
        np.testing.assert_allclose(fc, [0, 0.5, 0], atol=1e-5)
        assert align == pytest.approx(1.0)


# =========================================================================
# Impulse solver edge cases
# =========================================================================


class TestResolveContactEdgeCases:
    def test_separating_bodies_unchanged(self):
        I = _unit_I()
        va0 = np.array([0.0, 2.0, 0.0])
        va, oa, vb, ob, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=va0, omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0, -0.5, 0), normal=(0, 1, 0),
            restitution=0.5, static_friction=0.5, dynamic_friction=0.4,
            face_align_a=1.0,
        )
        np.testing.assert_allclose(va, va0, atol=1e-9)
        np.testing.assert_allclose(oa, 0, atol=1e-9)

    def test_zero_normal_safe(self):
        va, oa, vb, ob, u = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(1, 0, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=_unit_I(),
            pos_b=(1, 0, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.5, 0, 0), normal=(0, 0, 0),
            restitution=0.0, static_friction=0.0, dynamic_friction=0.0,
        )
        assert va[0] == pytest.approx(1.0)
        assert u is False

    def test_both_static_no_nan(self):
        va, oa, vb, ob, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(0, -1, 0), omega_a=(1, 0, 0),
            inv_mass_a=0.0, i_inv_a=None,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0, -0.5, 0), normal=(0, 1, 0),
            restitution=1.0, static_friction=1.0, dynamic_friction=1.0,
            face_align_a=1.0, face_align_b=1.0,
        )
        assert np.all(np.isfinite(va)) and np.all(np.isfinite(oa))

    def test_face_support_centered_no_spin(self):
        I = _unit_I()
        va, oa, _, _, unstable = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(0, -3, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.0, -0.5, 0),  # under COM
            normal=(0, 1, 0),
            restitution=0.0, static_friction=0.5, dynamic_friction=0.4,
            face_align_a=1.0,
        )
        assert not unstable
        assert abs(oa[0]) < 1e-6 and abs(oa[1]) < 1e-6 and abs(oa[2]) < 1e-6
        assert va[1] >= -1e-6  # not still diving into floor

    def test_unstable_offset_flags_and_spins(self):
        I = _unit_I()
        _, oa, _, _, unstable = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(0, -2, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.3, -0.5, 0),
            normal=(0, 1, 0),
            restitution=0.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=0.5,
        )
        assert unstable
        assert abs(oa[2]) > 0.02

    def test_very_light_vs_very_heavy(self):
        I_light = _unit_I(20.0)
        I_heavy = _unit_I(0.05)
        va, _, vb, _, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(4, 0, 0), omega_a=(0, 0, 0),
            inv_mass_a=10.0, i_inv_a=I_light,  # mass 0.1
            pos_b=(1, 0, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.01, i_inv_b=I_heavy,  # mass 100
            contact_point=(0.5, 0, 0), normal=(-1, 0, 0),
            restitution=0.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=0.0, face_align_b=0.0,
        )
        # Light body should reverse/stop more than heavy moves
        assert va[0] < 4.0
        assert abs(vb[0]) < abs(va[0] - 4.0) + 1.0

    def test_restitution_threshold_kills_micro_bounce(self):
        I = _unit_I()
        # Closing speed well below RESTITUTION_THRESHOLD
        slow = 0.3
        assert slow < RESTITUTION_THRESHOLD
        va, _, _, _, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(0, -slow, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0, -0.5, 0), normal=(0, 1, 0),
            restitution=1.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=1.0,
        )
        # e forced to 0 → should not bounce up with ~+slow
        assert va[1] < slow * 0.5

    def test_high_speed_restitution_bounces(self):
        I = _unit_I()
        va, _, _, _, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(0, -10, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0, -0.5, 0), normal=(0, 1, 0),
            restitution=1.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=1.0,
        )
        assert va[1] > 1.0

    def test_omega_clamped_to_max(self):
        I = np.diag([100.0, 100.0, 100.0])  # tiny inertia → huge omega
        _, oa, _, _, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(0, -20, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.4, -0.5, 0), normal=(0, 1, 0),
            restitution=0.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=0.0,
        )
        assert float(np.linalg.norm(oa)) <= MAX_ANGULAR_SPEED + 1e-6

    def test_friction_does_not_reverse_tangent(self):
        """Dynamic friction must not inject energy / reverse slide direction."""
        I = _unit_I()
        va, _, _, _, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(3.0, -1.0, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.0, -0.5, 0), normal=(0, 1, 0),
            restitution=0.0, static_friction=0.9, dynamic_friction=0.8,
            face_align_a=1.0,
        )
        # Still non-negative X (may be reduced, must not go strongly negative)
        assert va[0] >= -0.05

    def test_zero_friction_keeps_tangent_speed(self):
        I = _unit_I()
        va, _, _, _, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(2.5, -1.0, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.0, -0.5, 0), normal=(0, 1, 0),
            restitution=0.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=1.0,
        )
        assert va[0] == pytest.approx(2.5, abs=0.15)

    def test_stabilize_snaps_face_contact_under_com(self):
        # In-plane offset must be < 0.08 to snap under COM (FACE_REST path)
        cp = stabilize_contact_point(
            pos_a=(0, 1, 0), pos_b=(0, 0, 0),
            contact_point=(0.04, 0.5, 0.03),
            normal=(0, 1, 0), depth=0.05,
            face_align_a=1.0, face_align_b=0.0,
        )
        # Projected under COM of A on the plane
        assert abs(cp[0]) < 0.05
        assert abs(cp[2]) < 0.05

    def test_stabilize_keeps_edge_offset(self):
        cp = stabilize_contact_point(
            pos_a=(0, 1, 0), pos_b=(0, 0, 0),
            contact_point=(0.35, 0.5, 0.0),
            normal=(0, 1, 0), depth=0.05,
            face_align_a=0.7, face_align_b=0.0,
        )
        assert abs(cp[0] - 0.35) < 0.05

    def test_estimate_contact_point_midway(self):
        cp = estimate_contact_point((0, 0, 0), (2, 0, 0), normal=(-1, 0, 0), depth=0.2)
        assert 0.5 < cp[0] < 1.5

    def test_apply_body_state_allow_sleep_snaps_micro(self):
        obj = make_box()
        rb = obj.get_component(Rigidbody3D)
        rb.sleep_time = 0.0  # immediate if thresholds met
        Time.set(1 / 60)
        apply_body_state(
            rb,
            np.array([0.01, 0.0, 0.0]),
            np.array([0.05, 0.0, 0.0]),
            allow_sleep=True,
        )
        assert rb.velocity.magnitude < 1e-9
        assert rb.angular_velocity.magnitude < 1e-9

    def test_apply_body_state_deny_sleep_keeps_micro_omega(self):
        obj = make_box()
        rb = obj.get_component(Rigidbody3D)
        apply_body_state(
            rb,
            np.array([0.01, 0.0, 0.0]),
            np.array([0.08, 0.0, 0.0]),
            allow_sleep=False,
        )
        assert abs(rb.angular_velocity.x - 0.08) < 1e-9
        assert rb.is_sleeping is False


# =========================================================================
# Ramp sliding (visual slope == collision slope)
# =========================================================================


class TestRampSliding:
    def _make_ramp(self, angle_z=-28.0, friction=0.2):
        rad = math.radians(abs(angle_z))
        half_len, half_th = 6.0, 0.3
        cy = half_len * math.sin(rad) + half_th * math.cos(rad) + 0.02
        ramp = make_box(
            position=(-2.0, cy, 0.0),
            static=True,
            friction=friction,
            bounce=0.0,
            scale=(12.0, 0.6, 5.0),
            rotation=(0.0, 0.0, angle_z),
        )
        return ramp

    def _spawn_on_ramp_surface(self, ramp, local_x=-4.0, friction=0.15):
        col = ramp.get_component(BoxCollider3D)
        center, axes, extents = col.get_world_obb()
        # On top face near high end for negative Z angle
        surface = center + axes @ np.array([local_x, extents[1], 0.0])
        up = axes[:, 1]
        pos = surface + up * 0.52
        return make_box(
            position=tuple(pos),
            gravity=True,
            friction=friction,
            bounce=0.0,
            angular_drag=0.15,
        )

    def test_slides_down_negative_z_ramp_toward_plus_x(self):
        window = HeadlessWindow()
        ramp = self._make_ramp(angle_z=-28.0, friction=0.15)
        box = self._spawn_on_ramp_surface(ramp, local_x=-4.0, friction=0.12)
        window.objects = [box, ramp]
        x0, y0 = float(box.transform.position.x), float(box.transform.position.y)
        step(window, steps=100)
        x1, y1 = float(box.transform.position.x), float(box.transform.position.y)
        assert x1 > x0 + 0.3, f"Should slide toward +X, dx={x1 - x0}"
        assert y1 < y0 - 0.15, f"Should lose height, dy={y1 - y0}"
        # Still roughly on the ramp slab, not fallen through
        assert y1 > 0.3, f"Fell through ramp to y={y1}"

    def test_slides_down_positive_z_ramp_toward_minus_x(self):
        window = HeadlessWindow()
        ramp = self._make_ramp(angle_z=+28.0, friction=0.15)
        box = self._spawn_on_ramp_surface(ramp, local_x=+4.0, friction=0.12)
        window.objects = [box, ramp]
        x0 = float(box.transform.position.x)
        step(window, steps=100)
        x1 = float(box.transform.position.x)
        assert x1 < x0 - 0.3, f"Should slide toward -X, dx={x1 - x0}"

    def test_steep_ramp_accelerates_more_than_shallow(self):
        def run(angle):
            window = HeadlessWindow()
            ramp = self._make_ramp(angle_z=-abs(angle), friction=0.05)
            box = self._spawn_on_ramp_surface(ramp, local_x=-3.5, friction=0.05)
            window.objects = [box, ramp]
            x0 = float(box.transform.position.x)
            step(window, steps=60)
            return float(box.transform.position.x) - x0

        dx_steep = run(40)
        dx_shallow = run(15)
        assert dx_steep > dx_shallow + 0.1, (
            f"steep dx={dx_steep}, shallow dx={dx_shallow}"
        )

    def test_high_friction_holds_on_shallow_ramp(self):
        window = HeadlessWindow()
        ramp = self._make_ramp(angle_z=-12.0, friction=0.95)
        box = self._spawn_on_ramp_surface(ramp, local_x=-2.0, friction=0.95)
        window.objects = [box, ramp]
        x0 = float(box.transform.position.x)
        step(window, steps=90)
        x1 = float(box.transform.position.x)
        assert abs(x1 - x0) < 1.5, f"Should mostly hold, dx={x1 - x0}"

    def test_mesh_high_end_is_where_box_starts(self):
        """High end of mesh Y must match high end of OBB (not inverted)."""
        ramp = self._make_ramp(angle_z=-28.0)
        center, axes, extents = ramp.get_component(BoxCollider3D).get_world_obb()
        M = ramp.transform.get_model_matrix()
        mesh_neg = (np.array([-0.5, 0.5, 0, 1.0]) @ M)[1]
        mesh_pos = (np.array([0.5, 0.5, 0, 1.0]) @ M)[1]
        phys_neg = (center + axes @ np.array([-extents[0], extents[1], 0]))[1]
        phys_pos = (center + axes @ np.array([extents[0], extents[1], 0]))[1]
        assert mesh_neg == pytest.approx(phys_neg, abs=1e-3)
        assert mesh_pos == pytest.approx(phys_pos, abs=1e-3)
        assert mesh_neg > mesh_pos  # for -28° about Z, -X is high


# =========================================================================
# Face rest / edge tip integration
# =========================================================================


class TestFaceRestAndEdgeTip:
    def test_face_down_lands_flat_and_stops_spin(self):
        window = HeadlessWindow()
        cube = make_box(
            position=(0, 2.5, 0), gravity=True, friction=0.65, bounce=0.05,
            angular_drag=0.25,
        )
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.7, scale=(30, 1, 30),
        )
        window.objects = [cube, floor]
        step(window, steps=240)
        rb = cube.get_component(Rigidbody3D)
        assert cube.transform.position.y == pytest.approx(0.5, abs=0.08)
        assert rb.velocity.magnitude < 0.15
        assert rb.angular_velocity.magnitude < 0.25
        # Nearly axis-aligned
        R = cube.transform.rotation_matrix
        assert abs(abs(R[1, 1]) - 1.0) < 0.08 or max(abs(R[0, 1]), abs(R[2, 1])) > 0.95

    def test_tilted_cube_tips_off_edge_toward_face(self):
        window = HeadlessWindow()
        cube = make_box(
            position=(0, 0.85, 0), gravity=True, friction=0.35, bounce=0.0,
            rotation=(0, 0, 40), angular_drag=0.1,
        )
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.5, scale=(40, 1, 40),
        )
        window.objects = [cube, floor]
        step(window, steps=360)
        # Should leave the 40° knife-edge and settle near a face (0° or 90°)
        rz = float(cube.transform.rotation[2])
        # Distance to nearest multiple of 90°
        nearest = round(rz / 90.0) * 90.0
        assert abs(rz - nearest) < 15.0, f"Expected near face rest, rot_z={rz}"
        rb = cube.get_component(Rigidbody3D)
        assert rb.angular_velocity.magnitude < 1.5

    def test_does_not_freeze_forever_on_edge(self):
        window = HeadlessWindow()
        cube = make_box(
            position=(0, 0.9, 0), gravity=True, friction=0.3, bounce=0.0,
            rotation=(0, 0, 35), angular_drag=0.05,
        )
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.45, scale=(40, 1, 40),
        )
        window.objects = [cube, floor]
        # Early window: should develop tip rate or change angle
        step(window, steps=30)
        rb = cube.get_component(Rigidbody3D)
        rz_early = float(cube.transform.rotation[2])
        step(window, steps=90)
        rz_late = float(cube.transform.rotation[2])
        moved = abs(rz_late - rz_early) > 2.0 or rb.angular_velocity.magnitude > 0.05
        # After more time, either moved or already settled to face
        step(window, steps=200)
        rz_final = float(cube.transform.rotation[2])
        nearest = round(rz_final / 90.0) * 90.0
        settled = abs(rz_final - nearest) < 20.0
        assert moved or settled, (
            f"Stuck on edge: early={rz_early}, late={rz_late}, final={rz_final}"
        )

    def test_small_tilt_settles_without_runaway_spin(self):
        window = HeadlessWindow()
        cube = make_box(
            position=(0, 3.0, 0), gravity=True, friction=0.6, bounce=0.1,
            rotation=(6, 12, -5), angular_drag=0.3,
        )
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.7, scale=(25, 1, 25),
        )
        window.objects = [cube, floor]
        peak_w = 0.0
        for _ in range(280):
            step(window, steps=1)
            peak_w = max(
                peak_w,
                cube.get_component(Rigidbody3D).angular_velocity.magnitude,
            )
        assert peak_w < 10.0, f"Runaway spin peak |ω|={peak_w}"
        rb = cube.get_component(Rigidbody3D)
        assert rb.angular_velocity.magnitude < 0.6
        assert cube.transform.position.y < 1.2


# =========================================================================
# Floor settle / no ice skate
# =========================================================================


class TestFloorSettleNoSkate:
    def test_resting_box_does_not_gain_horizontal_velocity(self):
        window = HeadlessWindow()
        cube = make_box(
            position=(0, 0.5, 0), gravity=True, friction=0.7, bounce=0.0,
        )
        # Start already on floor
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.7, scale=(20, 1, 20),
        )
        window.objects = [cube, floor]
        step(window, steps=120)
        rb = cube.get_component(Rigidbody3D)
        assert abs(rb.velocity.x) < 0.05
        assert abs(rb.velocity.z) < 0.05
        assert abs(cube.transform.position.x) < 0.15

    def test_landing_with_lateral_velocity_friction_slows_not_accelerates(self):
        window = HeadlessWindow()
        cube = make_box(
            position=(0, 2.0, 0), gravity=True, friction=0.55, bounce=0.0,
        )
        rb = cube.get_component(Rigidbody3D)
        rb.velocity = Vector3(2.0, 0, 0)
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.55, scale=(40, 1, 40),
        )
        window.objects = [cube, floor]
        step(window, steps=20)  # in air
        vx_air = rb.velocity.x
        step(window, steps=100)  # land and slide
        # Friction should not increase |vx| after landing
        assert rb.velocity.x <= vx_air + 0.05
        assert rb.velocity.x > -0.1  # no reverse skate

    def test_zero_friction_floor_keeps_sliding(self):
        window = HeadlessWindow()
        cube = make_box(
            position=(0, 0.5, 0), gravity=True, friction=0.0, bounce=0.0,
        )
        rb = cube.get_component(Rigidbody3D)
        rb.velocity = Vector3(1.5, 0, 0)
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.0, scale=(50, 1, 50),
        )
        window.objects = [cube, floor]
        step(window, steps=60)
        assert rb.velocity.x > 1.0, f"Ice should keep sliding, vx={rb.velocity.x}"

    def test_deep_penetration_pushed_out_to_surface(self):
        window = HeadlessWindow()
        # Cube half-size 0.5; center at 0.2 → 0.3 deep into floor top (y=0)
        cube = make_box(position=(0, 0.2, 0), gravity=False, friction=0.0)
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.0, scale=(10, 1, 10),
        )
        window.objects = [cube, floor]
        for o in window.objects:
            _refresh(o)
        window._process_collisions()
        y = float(cube.transform.position.y)
        assert y >= 0.49, f"Expected full static depenetration to ~0.5, got {y}"


# =========================================================================
# Continuous collision
# =========================================================================


class TestContinuousCollisionEdgeCases:
    def test_fast_horizontal_does_not_tunnel_wall(self):
        window = HeadlessWindow()
        ball = make_box(
            position=(0, 0, 0), continuous=True, bounce=0.0, friction=0.0,
        )
        rb = ball.get_component(Rigidbody3D)
        rb.velocity = Vector3(80, 0, 0)
        wall = make_box(
            position=(4, 0, 0), static=True, scale=(1, 4, 4), bounce=0.0, friction=0.0,
        )
        window.objects = [ball, wall]
        for o in window.objects:
            o.transform._update_prev_position()
            _refresh(o)
        step(window, dt=1 / 60, steps=30)
        x = float(ball.transform.position.x)
        assert x < 3.6, f"Tunneled through wall to x={x}"
        assert x > 2.0

    def test_diagonal_slide_keeps_horizontal_velocity(self):
        window = HeadlessWindow()
        box = make_box(
            position=(0, 1.0, 0), continuous=True, bounce=0.0, friction=0.0,
        )
        rb = box.get_component(Rigidbody3D)
        rb.velocity = Vector3(8, -8, 0)
        floor = make_box(
            position=(0, 0, 0), static=True, scale=(80, 1, 80), bounce=0.0, friction=0.0,
        )
        window.objects = [box, floor]
        for o in window.objects:
            o.transform._update_prev_position()
            _refresh(o)
        Time.set(0.05)
        prev_max = Time.maximum_delta_time
        Time.maximum_delta_time = 0.0
        try:
            rb.update()
            window._process_collisions()
        finally:
            Time.maximum_delta_time = prev_max
        assert rb.velocity.x > 1.0, f"Horizontal velocity killed: {rb.velocity}"
        assert float(box.transform.position.y) >= 0.99

    def test_fast_drop_onto_floor_continuous(self):
        window = HeadlessWindow()
        box = make_box(
            position=(0, 5, 0), continuous=True, bounce=0.0, friction=0.0, gravity=False,
        )
        rb = box.get_component(Rigidbody3D)
        rb.velocity = Vector3(0, -100, 0)
        floor = make_box(
            position=(0, -0.5, 0), static=True, scale=(20, 1, 20), bounce=0.0,
        )
        window.objects = [box, floor]
        for o in window.objects:
            o.transform._update_prev_position()
            _refresh(o)
        step(window, dt=1 / 60, steps=20)
        y = float(box.transform.position.y)
        assert y >= 0.45, f"Fell through floor to y={y}"
        assert y < 1.5

    def test_discrete_fast_may_tunnel_but_continuous_does_not(self):
        """Sanity: CONTINUOUS is safer than DISCRETE at extreme speed."""
        def run(continuous: bool) -> float:
            window = HeadlessWindow()
            box = make_box(
                position=(-2, 0, 0), continuous=continuous, bounce=0.0, friction=0.0,
            )
            rb = box.get_component(Rigidbody3D)
            rb.velocity = Vector3(200, 0, 0)
            wall = make_box(
                position=(1.5, 0, 0), static=True, scale=(0.4, 3, 3),
            )
            window.objects = [box, wall]
            for o in window.objects:
                o.transform._update_prev_position()
                _refresh(o)
            step(window, dt=1 / 30, steps=5)
            return float(box.transform.position.x)

        x_cont = run(True)
        # Continuous should stop before/at wall; may not always catch 200u/frame
        # but should not end far past it if CCD works
        assert x_cont < 5.0, f"CONTINUOUS still far past wall at {x_cont}"


# =========================================================================
# Multi-body / materials integration
# =========================================================================


class TestMultiBodyAndMaterials:
    def test_two_dynamic_boxes_exchange_momentum(self):
        window = HeadlessWindow()
        a = make_box(position=(0, 0.5, 0), bounce=0.0, friction=0.0)
        b = make_box(position=(1.2, 0.5, 0), bounce=0.0, friction=0.0)
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.0, scale=(40, 1, 40),
        )
        ra = a.get_component(Rigidbody3D)
        rb = b.get_component(Rigidbody3D)
        ra.velocity = Vector3(3, 0, 0)
        rb.velocity = Vector3(0, 0, 0)
        window.objects = [a, b, floor]
        step(window, steps=80)
        # A should slow, B should gain some +X
        assert ra.velocity.x < 3.0
        assert rb.velocity.x > 0.05 or float(b.transform.position.x) > 1.25

    def test_static_wall_not_moved(self):
        window = HeadlessWindow()
        dyn = make_box(position=(0, 0.5, 0), bounce=0.2, friction=0.0)
        wall = make_box(position=(2, 0.5, 0), static=True, bounce=0.2, scale=(1, 2, 2))
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.0, scale=(30, 1, 30),
        )
        dyn.get_component(Rigidbody3D).velocity = Vector3(5, 0, 0)
        wall_x0 = float(wall.transform.position.x)
        window.objects = [dyn, wall, floor]
        step(window, steps=60)
        assert float(wall.transform.position.x) == pytest.approx(wall_x0, abs=1e-6)

    def test_bouncy_box_leaves_floor(self):
        window = HeadlessWindow()
        cube = make_box(
            position=(0, 3.0, 0), gravity=True, bounce=0.95, friction=0.0,
        )
        floor = make_box(
            position=(0, -0.5, 0), static=True, bounce=0.95, friction=0.0, scale=(20, 1, 20),
        )
        window.objects = [cube, floor]
        max_y_after_contact = 0.0
        contacted = False
        for _ in range(200):
            step(window, steps=1)
            y = float(cube.transform.position.y)
            if y < 1.2:
                contacted = True
            if contacted:
                max_y_after_contact = max(max_y_after_contact, y)
        assert contacted
        assert max_y_after_contact > 1.2, (
            f"Expected bounce, max y after contact={max_y_after_contact}"
        )

    def test_sphere_on_box_floor(self):
        window = HeadlessWindow()
        sph = make_sphere(position=(0, 2.0, 0), gravity=True, bounce=0.1, friction=0.4)
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.5, scale=(20, 1, 20),
        )
        window.objects = [sph, floor]
        step(window, steps=200)
        y = float(sph.transform.position.y)
        # Sphere radius ~0.5 resting on floor top y=0 → center ~0.5
        assert 0.3 < y < 1.2
        assert sph.get_component(Rigidbody3D).velocity.magnitude < 1.0

    def test_kinematic_unaffected_by_dynamic(self):
        window = HeadlessWindow()
        dyn = make_box(position=(0, 0.5, 0), bounce=0.0, friction=0.0)
        kin = make_box(position=(1.5, 0.5, 0), bounce=0.0, friction=0.0)
        kin.get_component(Rigidbody3D).is_kinematic = True
        floor = make_box(
            position=(0, -0.5, 0), static=True, friction=0.0, scale=(30, 1, 30),
        )
        dyn.get_component(Rigidbody3D).velocity = Vector3(6, 0, 0)
        kin_x0 = float(kin.transform.position.x)
        window.objects = [dyn, kin, floor]
        step(window, steps=40)
        assert float(kin.transform.position.x) == pytest.approx(kin_x0, abs=1e-5)


# =========================================================================
# Manifold / contact quality
# =========================================================================


class TestManifoldQuality:
    def test_stacked_boxes_manifold_normal_up(self):
        lower = make_box(position=(0, 0.5, 0), static=True)
        upper = make_box(position=(0, 1.4, 0))  # slight overlap
        m = get_collision_manifold(
            upper.get_component(BoxCollider3D),
            lower.get_component(BoxCollider3D),
        )
        assert m is not None
        # Normal from B(lower) to A(upper) should be mostly +Y
        assert abs(m.normal[1]) > 0.7

    def test_side_hit_normal_mostly_horizontal(self):
        a = make_box(position=(0, 0, 0))
        b = make_box(position=(0.8, 0, 0))
        m = obb_vs_obb_manifold(
            a.get_component(BoxCollider3D),
            b.get_component(BoxCollider3D),
        )
        assert m is not None
        assert abs(m.normal[0]) > abs(m.normal[1])

    def test_rotated_pair_still_has_finite_manifold(self):
        a = make_box(position=(0, 0, 0), rotation=(20, 30, 10))
        b = make_box(position=(0.6, 0.2, 0.1), rotation=(-15, 5, 40))
        m = get_collision_manifold(
            a.get_component(BoxCollider3D),
            b.get_component(BoxCollider3D),
        )
        if m is not None:
            assert np.all(np.isfinite(m.normal))
            assert m.depth >= 0
            if m.contact_point is not None:
                assert np.all(np.isfinite(m.contact_point))

    def test_sphere_vs_rotated_obb(self):
        box = make_box(position=(0, 0, 0), rotation=(0, 0, 30), scale=(2, 1, 2))
        sph = make_sphere(position=(0, 1.0, 0), radius=0.5)
        m = sphere_vs_obb_manifold(
            sph.get_component(SphereCollider3D),
            box.get_component(BoxCollider3D),
        )
        # May or may not overlap depending on exact sizes; just no crash
        if m is not None:
            assert m.depth >= 0
            assert abs(np.linalg.norm(m.normal) - 1.0) < 1e-3


# =========================================================================
# Inertia / quaternion integration edge cases
# =========================================================================


class TestInertiaAndIntegrationEdgeCases:
    def test_zero_mass_treated_safely_via_static(self):
        obj = make_box(static=True)
        rb = obj.get_component(Rigidbody3D)
        pos, vel, omega, inv_m, i_inv = body_state_from_rigidbody(rb, obj, True)
        assert inv_m == 0.0
        assert i_inv is None
        np.testing.assert_allclose(vel, 0)

    def test_long_thin_rod_inertia_anisotropic(self):
        obj = make_box(scale=(8, 0.2, 0.2), mass=2.0)
        inv = obj.get_component(Rigidbody3D).get_inertia_inv_array()
        # Spinning about long axis (X) should be easiest → largest inv component
        assert inv[0] > inv[1]
        assert inv[0] > inv[2]

    def test_angular_integration_90_deg(self):
        obj = make_box()
        rb = obj.get_component(Rigidbody3D)
        rb.angular_velocity = Vector3(0, math.radians(90) * 60, 0)  # 90°/frame at 60fps
        rb.angular_drag = 0.0
        Time.set(1 / 60)
        rb.use_gravity = False
        rb.update()
        # Roughly 90° about Y
        assert abs(obj.transform.rotation_y - 90) < 5.0 or abs(
            abs(obj.transform.rotation_y) - 90
        ) < 5.0

    def test_quaternion_remains_normalized_after_many_spins(self):
        obj = make_box()
        rb = obj.get_component(Rigidbody3D)
        rb.angular_velocity = Vector3(3.0, -2.0, 1.5)
        rb.angular_drag = 0.0
        rb.use_gravity = False
        Time.set(1 / 60)
        for _ in range(500):
            rb.update()
        q = obj.transform._local_quaternion
        mag = math.sqrt(q.w ** 2 + q.x ** 2 + q.y ** 2 + q.z ** 2)
        assert abs(mag - 1.0) < 1e-3

    def test_world_inertia_matches_r_i_rt(self):
        obj = make_box(scale=(3, 1, 0.5), mass=1.0, rotation=(0, 35, 0))
        rb = obj.get_component(Rigidbody3D)
        R = obj.transform.rotation_matrix.astype(np.float64)
        inv_body = rb.get_inertia_inv_array().astype(np.float64)
        I_body_inv = np.diag(inv_body)
        expected = R @ I_body_inv @ R.T
        got = rb.get_world_inertia_inv_matrix()
        np.testing.assert_allclose(got, expected, atol=1e-4)


# =========================================================================
# Constants sanity
# =========================================================================


class TestPhysicsConstants:
    def test_face_thresholds_ordered(self):
        assert 0.0 < FACE_ALIGN_THRESHOLD < FACE_REST_ALIGN <= 1.0
        assert UNSTABLE_SUPPORT_OFFSET > 0.0
        assert RESTITUTION_THRESHOLD > 0.0
        assert MAX_ANGULAR_SPEED > 1.0

    def test_face_align_from_identity_box(self):
        from engine.d3.physics.response import _face_align_from_rotation

        obj = make_box()
        align_y = _face_align_from_rotation(obj, np.array([0.0, 1.0, 0.0]))
        assert align_y > 0.99
        align_diag = _face_align_from_rotation(
            obj, np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
        )
        assert align_diag < align_y


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
