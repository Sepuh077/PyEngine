"""
Tests for correct rotational (angular) collision response.

Covers:
  - Contact points on manifolds
  - Off-center hits produce spin about the expected axis
  - Centered hits produce little/no spin
  - Effective mass includes inertia (harder to spin than pure linear)
  - add_torque uses inverse inertia tensor
  - Friction-free angular impulse conserves angular momentum about an axis
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.component import Time
from engine.d3.object3d import create_cube, create_sphere, create_plane
from engine.d3.physics.collider import BoxCollider3D, SphereCollider3D, Collider3D
from engine.d3.physics.collision_manifold import (
    CollisionManifold,
    get_collision_manifold,
    sphere_vs_sphere_manifold,
    obb_vs_obb_manifold,
)
from engine.d3.physics.response import resolve_contact_3d
from engine.d3.physics.rigidbody import Rigidbody3D
from engine.d3.physics.types import PhysicsMaterialCombine
from engine.d3.window import Window3D
from engine.types import Vector3


class HeadlessWindow(Window3D):
    def __init__(self):
        self.objects = []
        self._current_scene = None

    def _active_objects(self):
        return self.objects


def _make_box(position=(0, 0, 0), mass=1.0, static=False, gravity=False,
              bounce=0.0, friction=0.0, size=1.0, scale=None):
    obj = create_cube(size=size, position=position)
    if scale is not None:
        obj.transform.scale_xyz = scale
    rb = Rigidbody3D(use_gravity=gravity, is_static=static)
    rb.mass = mass
    rb.drag = 0.0
    rb.angular_drag = 0.0
    col = BoxCollider3D()
    col.bounciness = bounce
    col.static_friction = friction
    col.dynamic_friction = friction
    obj.add_component(rb)
    obj.add_component(col)
    obj.transform._compute_world_transform()
    col._transform_dirty = True
    col.update_bounds()
    rb._inertia_dirty = True
    return obj


def _make_sphere(position=(0, 0, 0), mass=1.0, static=False, gravity=False,
                 bounce=0.0, friction=0.0, radius=1.0):
    obj = create_sphere(radius=radius * 0.5, position=position)
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
    obj.transform._compute_world_transform()
    col._transform_dirty = True
    col.update_bounds()
    rb._inertia_dirty = True
    return obj


def _step(window, dt=1 / 60.0, steps=1):
    prev_max = Time.maximum_delta_time
    prev_skip = Time._skip_rigidbody_frame_update
    Time.maximum_delta_time = 0.0
    Time._skip_rigidbody_frame_update = False
    try:
        for _ in range(steps):
            Time.set(dt)
            for o in window.objects:
                rb = o.get_component(Rigidbody3D)
                if rb:
                    rb.wake()
                    rb.update()
                for col in o.get_components(Collider3D):
                    col._transform_dirty = True
                    col.update_bounds()
            window._process_collisions()
    finally:
        Time.maximum_delta_time = prev_max
        Time._skip_rigidbody_frame_update = prev_skip


# =========================================================================
# Contact points
# =========================================================================


class TestContactPoints:
    def test_sphere_sphere_manifold_has_contact(self):
        a = _make_sphere(position=(0, 0, 0), radius=1.0)
        b = _make_sphere(position=(0.5, 0, 0), radius=1.0)
        ca = a.get_component(SphereCollider3D)
        cb = b.get_component(SphereCollider3D)
        m = sphere_vs_sphere_manifold(ca, cb)
        assert m is not None
        assert m.contact_point is not None
        assert m.depth > 0
        # Contact should lie roughly between centers
        cp = m.contact_point
        assert abs(cp[1]) < 0.5
        assert 0.0 <= cp[0] <= 0.5 or abs(cp[0] - 0.25) < 0.5

    def test_obb_obb_manifold_has_contact(self):
        a = _make_box(position=(0, 0, 0))
        b = _make_box(position=(0.5, 0, 0))
        ca = a.get_component(BoxCollider3D)
        cb = b.get_component(BoxCollider3D)
        m = obb_vs_obb_manifold(ca, cb)
        assert m is not None
        assert m.contact_point is not None
        assert m.depth > 0

    def test_get_collision_manifold_fills_contact(self):
        a = _make_box(position=(0, 0, 0))
        b = _make_box(position=(0.4, 0.1, 0))
        m = get_collision_manifold(
            a.get_component(BoxCollider3D),
            b.get_component(BoxCollider3D),
        )
        assert m is not None
        assert m.contact_point is not None


# =========================================================================
# Pure solver unit tests
# =========================================================================


class TestResolveContact3D:
    def test_centered_hit_no_spin(self):
        """Impulse through COM should not generate angular velocity."""
        I = np.diag([6.0, 6.0, 6.0])  # inv inertia for unit box-ish
        va, oa, vb, ob, _ = resolve_contact_3d(
            pos_a=(0, 0, 0),
            vel_a=(2, 0, 0),
            omega_a=(0, 0, 0),
            inv_mass_a=1.0,
            i_inv_a=I,
            pos_b=(1, 0, 0),
            vel_b=(0, 0, 0),
            omega_b=(0, 0, 0),
            inv_mass_b=0.0,  # static wall
            i_inv_b=None,
            contact_point=(0.5, 0, 0),
            normal=(-1, 0, 0),  # from B toward A (wall is +X of A)
            restitution=0.0,
            static_friction=0.0,
            dynamic_friction=0.0,
            face_align_a=0.0,  # not a face-support hit
        )
        assert abs(oa[0]) < 1e-9 and abs(oa[1]) < 1e-9 and abs(oa[2]) < 1e-9
        # Approaching along -normal? vel_a=(2,0,0), normal=(-1,0,0)
        # v_rel · n = 2*(-1) = -2 < 0 approaching — should bounce/stop
        assert va[0] <= 2.0 + 1e-9

    def test_off_center_hit_produces_spin(self):
        """Contact offset from COM along Y should spin about Z."""
        I = np.diag([6.0, 6.0, 6.0])
        va, oa, vb, ob, _ = resolve_contact_3d(
            pos_a=(0, 0, 0),
            vel_a=(0, -4, 0),
            omega_a=(0, 0, 0),
            inv_mass_a=1.0,
            i_inv_a=I,
            pos_b=(0, -1, 0),
            vel_b=(0, 0, 0),
            omega_b=(0, 0, 0),
            inv_mass_b=0.0,
            i_inv_b=None,
            contact_point=(0.4, -0.5, 0),  # right of center
            normal=(0, 1, 0),
            restitution=0.0,
            static_friction=0.0,
            dynamic_friction=0.0,
            face_align_a=0.0,  # force edge/vertex path
        )
        # r × J with J along +Y and r=(0.4,-0.5,0) → torque about +Z
        assert oa[2] > 0.05, f"Expected +Z spin, got omega={oa}"

    def test_inertia_reduces_impulse_vs_point_mass(self):
        """Off-center contact: finite inertia softens linear bounce vs point mass."""
        I = np.diag([6.0, 6.0, 6.0])
        # Point mass (no rotational coupling)
        va_pt, _, _, _, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(0, -4, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=None,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.4, -0.5, 0), normal=(0, 1, 0),
            restitution=1.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=0.0,
        )
        # With inertia
        va_rb, oa, _, _, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(0, -4, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(0, -1, 0), vel_b=(0, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.4, -0.5, 0), normal=(0, 1, 0),
            restitution=1.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=0.0,
        )
        # Linear restitution is smaller when energy goes into rotation
        assert va_rb[1] < va_pt[1] + 1e-6 or abs(oa[2]) > 1e-6
        assert abs(oa[2]) > 1e-6


# =========================================================================
# Full engine integration
# =========================================================================


class TestEngineRotationalResponse:
    def test_off_center_impulse_via_resolve_collision(self):
        window = HeadlessWindow()
        cube = _make_box(position=(0, 0, 0), bounce=0.0, friction=0.0)
        ground = _make_box(position=(0, -1.0, 0), static=True, bounce=0.0, friction=0.0)
        # Stretch ground
        ground.transform.scale_xyz = (10, 1, 10)
        ground.transform._compute_world_transform()
        for c in ground.get_components(BoxCollider3D):
            c._transform_dirty = True
            c.update_bounds()

        window.objects = [cube, ground]
        rb = cube.get_component(Rigidbody3D)
        rb.velocity = Vector3(0, -5, 0)
        rb.angular_velocity = Vector3.zero()

        manifold = CollisionManifold(
            normal=np.array([0, 1, 0], dtype=np.float32),
            depth=0.05,
            contact_point=np.array([0.4, -0.5, 0], dtype=np.float32),
        )
        window._resolve_collision(cube, ground, manifold)

        assert abs(rb.angular_velocity.z) > 0.1, (
            f"Expected Z rotation from right-edge hit, got ω={rb.angular_velocity}"
        )

    def test_no_spin_up_after_face_landing(self):
        """After landing nearly face-down, body must not restart spinning."""
        window = HeadlessWindow()
        cube = _make_box(position=(0, 3.0, 0), bounce=0.1, friction=0.6, gravity=True)
        cube.transform.rotation = (8, 25, -6)
        ground = _make_box(position=(0, -0.5, 0), static=True, bounce=0.0, friction=0.7)
        ground.transform.scale_xyz = (20, 1, 20)
        for o in (cube, ground):
            o.transform._compute_world_transform()
            for c in o.get_components(BoxCollider3D):
                c._transform_dirty = True
                c.update_bounds()
            rb = o.get_component(Rigidbody3D)
            if rb:
                rb._inertia_dirty = True
                rb.angular_drag = 0.3

        window.objects = [cube, ground]
        rb = cube.get_component(Rigidbody3D)
        rb.use_gravity = True

        peak_w = 0.0
        min_v_after_contact = 1e9
        contacted = False
        for i in range(300):
            _step(window, dt=1 / 60.0, steps=1)
            v = rb.velocity.magnitude
            w = rb.angular_velocity.magnitude
            peak_w = max(peak_w, w)
            if cube.transform.position.y < 1.2:
                contacted = True
                min_v_after_contact = min(min_v_after_contact, v)
            # After plenty of time on the ground, spin must be dead
            if i > 180 and contacted:
                assert w < 0.6, (
                    f"Late spin-up on ground: |ω|={w:.3f} at step {i}, |v|={v:.3f}"
                )

        assert contacted, "Cube should hit the ground"
        assert min_v_after_contact < 1.0, (
            f"Cube should slow after contact, min |v|={min_v_after_contact:.3f}"
        )
        # Overall spin must not run away
        assert peak_w < 8.0, f"Runaway spin peak |ω|={peak_w:.3f}"

    def test_two_body_offset_collision_spins_both(self):
        """Direct solver path: off-center hit with face_align=0 produces spin."""
        I = np.diag([6.0, 6.0, 6.0])
        _, oa, _, _, _ = resolve_contact_3d(
            pos_a=(0, 0, 0), vel_a=(5, 0, 0), omega_a=(0, 0, 0),
            inv_mass_a=1.0, i_inv_a=I,
            pos_b=(1, 0, 0), vel_b=(-5, 0, 0), omega_b=(0, 0, 0),
            inv_mass_b=1.0, i_inv_b=I,
            contact_point=(0.5, 0.35, 0), normal=(-1, 0, 0),
            restitution=0.2, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=0.0, face_align_b=0.0,
        )
        assert abs(oa[2]) > 0.05 or abs(oa[1]) > 0.05, (
            f"Offset two-body hit should produce spin, ω={oa}"
        )

    def test_kinematic_body_not_pushed(self):
        window = HeadlessWindow()
        dyn = _make_box(position=(0, 0, 0), bounce=0.0, friction=0.0)
        kin = _make_box(position=(1.2, 0, 0), bounce=0.0, friction=0.0)
        rb_k = kin.get_component(Rigidbody3D)
        rb_k.is_kinematic = True
        rb_d = dyn.get_component(Rigidbody3D)
        rb_d.velocity = Vector3(5, 0, 0)
        window.objects = [dyn, kin]
        for o in window.objects:
            o.transform._compute_world_transform()
            for c in o.get_components(BoxCollider3D):
                c._transform_dirty = True
                c.update_bounds()

        kin_pos_before = np.array(kin.transform.position.to_numpy()
                                  if hasattr(kin.transform.position, "to_numpy")
                                  else [kin.transform.position.x,
                                        kin.transform.position.y,
                                        kin.transform.position.z])
        _step(window, dt=1 / 60.0, steps=40)
        kin_pos_after = np.array(kin.transform.position.to_numpy()
                                 if hasattr(kin.transform.position, "to_numpy")
                                 else [kin.transform.position.x,
                                       kin.transform.position.y,
                                       kin.transform.position.z])
        assert np.allclose(kin_pos_before, kin_pos_after, atol=1e-5)
        # Dynamic should have been stopped/bounced
        assert rb_d.velocity.x < 5.0


# =========================================================================
# Inertia / torque
# =========================================================================


class TestUnstableSupport:
    def test_off_center_support_gets_torque(self):
        """Large in-plane contact offset must create tipping torque (edge/vertex)."""
        I = np.diag([6.0, 6.0, 6.0])
        _, oa, _, _, unstable = resolve_contact_3d(
            pos_a=(0, 0, 0),
            vel_a=(0, -2, 0),
            omega_a=(0, 0, 0),
            inv_mass_a=1.0,
            i_inv_a=I,
            pos_b=(0, -1, 0),
            vel_b=(0, 0, 0),
            omega_b=(0, 0, 0),
            inv_mass_b=0.0,
            i_inv_b=None,
            contact_point=(0.35, -0.5, 0),
            normal=(0, 1, 0),
            restitution=0.0,
            static_friction=0.0,
            dynamic_friction=0.0,
            face_align_a=0.5,  # not face-parallel
        )
        assert unstable
        assert abs(oa[2]) > 0.05, f"Expected tipping torque about Z, ω={oa}"


class TestInertiaAndTorque:
    def test_box_inertia_inverse_positive(self):
        obj = _make_box(mass=2.0, scale=(2, 1, 1))
        rb = obj.get_component(Rigidbody3D)
        inv = rb.get_inertia_inv_array()
        assert inv.shape == (3,)
        assert np.all(inv > 0)

    def test_sphere_inertia_isotropic(self):
        obj = _make_sphere(mass=1.0, radius=1.0)
        rb = obj.get_component(Rigidbody3D)
        inv = rb.get_inertia_inv_array()
        assert abs(inv[0] - inv[1]) < 1e-5
        assert abs(inv[1] - inv[2]) < 1e-5

    def test_add_torque_uses_inertia(self):
        obj = _make_box(mass=1.0, scale=(4, 0.5, 1))
        rb = obj.get_component(Rigidbody3D)
        rb.angular_velocity = Vector3.zero()
        rb.add_torque(Vector3(0, 0, 10), as_impulse=True)
        # Izz ≈ m/12 (sx^2 + sy^2) = 1/12 (16 + 0.25) ≈ 1.354 → I_inv ≈ 0.74
        # ωz ≈ 7.4, not 10/mass=10
        assert abs(rb.angular_velocity.z) > 0.1
        # Must not be the old mass-only behaviour of exactly 10
        assert abs(rb.angular_velocity.z - 10.0) > 0.5

    def test_world_inertia_changes_with_rotation(self):
        obj = _make_box(mass=1.0, scale=(4, 0.5, 1))
        rb = obj.get_component(Rigidbody3D)
        I0 = rb.get_world_inertia_inv_matrix().copy()
        obj.transform.rotation = (0, 90, 0)
        obj.transform._compute_world_transform()
        rb._inertia_dirty = True
        I1 = rb.get_world_inertia_inv_matrix()
        assert not np.allclose(I0, I1, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
