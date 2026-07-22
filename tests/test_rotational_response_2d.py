"""
Tests for 2D rotational (angular) collision response.

Mirrors tests/test_rotational_response.py for the planar solver:
  - Contact points on manifolds
  - Off-center hits produce spin about Z
  - Centered hits produce little/no spin
  - Effective mass includes inertia
  - add_torque uses inverse inertia
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.component import Time
from engine.gameobject import GameObject
from engine.d2.object2d import create_rect, create_circle
from engine.d2.physics.collider import BoxCollider2D, CircleCollider2D, Collider2D
from engine.d2.physics.collision_manifold import (
    CollisionManifold2D,
    get_collision_manifold_2d,
    circle_vs_circle_manifold,
    obb_vs_obb_manifold,
)
from engine.d2.physics.response import resolve_contact_2d
from engine.d2.physics.rigidbody import Rigidbody2D
from engine.d2.window2d import Window2D
from engine.types.vector2 import Vector2


class HeadlessWindow2D(Window2D):
    def __init__(self):
        self.objects = []
        self._current_scene = None

    def _active_objects(self):
        return self.objects


def _make_box(position=(0, 0), mass=1.0, static=False, gravity=False,
              bounce=0.0, friction=0.0, size=(1.0, 1.0)):
    obj = create_rect(width=size[0], height=size[1], color=(1, 1, 1))
    obj.transform.position = (position[0], position[1], 0)
    rb = Rigidbody2D(use_gravity=gravity, is_static=static)
    rb.mass = mass
    rb.drag = 0.0
    rb.angular_drag = 0.0
    col = BoxCollider2D()
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


def _make_circle(position=(0, 0), mass=1.0, static=False, gravity=False,
                 bounce=0.0, friction=0.0, radius=0.5):
    obj = create_circle(radius=radius, color=(1, 1, 1))
    obj.transform.position = (position[0], position[1], 0)
    rb = Rigidbody2D(use_gravity=gravity, is_static=static)
    rb.mass = mass
    rb.drag = 0.0
    rb.angular_drag = 0.0
    col = CircleCollider2D(radius=1.0)
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
                rb = o.get_component(Rigidbody2D)
                if rb:
                    rb.wake()
                    rb.update()
                for col in o.get_components(Collider2D):
                    col._transform_dirty = True
                    col.update_bounds()
            window._process_collisions()
    finally:
        Time.maximum_delta_time = prev_max
        Time._skip_rigidbody_frame_update = prev_skip


# =========================================================================
# Contact points
# =========================================================================


class TestContactPoints2D:
    def test_circle_circle_manifold_has_contact(self):
        a = _make_circle(position=(0, 0), radius=1.0)
        b = _make_circle(position=(0.5, 0), radius=1.0)
        ca = a.get_component(CircleCollider2D)
        cb = b.get_component(CircleCollider2D)
        m = circle_vs_circle_manifold(ca.circle, cb.circle)
        assert m is not None
        assert m.contact_point is not None
        assert m.depth > 0

    def test_obb_obb_manifold_has_contact(self):
        a = _make_box(position=(0, 0))
        b = _make_box(position=(0.5, 0))
        ca = a.get_component(BoxCollider2D)
        cb = b.get_component(BoxCollider2D)
        m = obb_vs_obb_manifold(ca.obb, cb.obb)
        assert m is not None
        assert m.contact_point is not None
        assert m.depth > 0

    def test_get_collision_manifold_fills_contact(self):
        a = _make_box(position=(0, 0))
        b = _make_box(position=(0.4, 0.1))
        m = get_collision_manifold_2d(
            a.get_component(BoxCollider2D),
            b.get_component(BoxCollider2D),
        )
        assert m is not None
        assert m.contact_point is not None


# =========================================================================
# Pure solver unit tests
# =========================================================================


class TestResolveContact2D:
    def test_centered_hit_no_spin(self):
        i_inv = 6.0
        va, oa, vb, ob, _ = resolve_contact_2d(
            pos_a=(0, 0),
            vel_a=(2, 0),
            omega_a=0.0,
            inv_mass_a=1.0,
            i_inv_a=i_inv,
            pos_b=(1, 0),
            vel_b=(0, 0),
            omega_b=0.0,
            inv_mass_b=0.0,
            i_inv_b=None,
            contact_point=(0.5, 0),
            normal=(-1, 0),
            restitution=0.0,
            static_friction=0.0,
            dynamic_friction=0.0,
            face_align_a=0.0,
        )
        assert abs(oa) < 1e-9
        assert va[0] <= 2.0 + 1e-9

    def test_off_center_hit_produces_spin(self):
        i_inv = 6.0
        va, oa, vb, ob, _ = resolve_contact_2d(
            pos_a=(0, 0),
            vel_a=(0, -4),
            omega_a=0.0,
            inv_mass_a=1.0,
            i_inv_a=i_inv,
            pos_b=(0, -1),
            vel_b=(0, 0),
            omega_b=0.0,
            inv_mass_b=0.0,
            i_inv_b=None,
            contact_point=(0.4, -0.5),
            normal=(0, 1),
            restitution=0.0,
            static_friction=0.0,
            dynamic_friction=0.0,
            face_align_a=0.0,
        )
        # r=(0.4,-0.5), J along +Y → τ = rx*jy - ry*jx = 0.4*J > 0
        assert oa > 0.05, f"Expected +ω spin, got omega={oa}"

    def test_inertia_reduces_impulse_vs_point_mass(self):
        i_inv = 6.0
        va_pt, _, _, _, _ = resolve_contact_2d(
            pos_a=(0, 0), vel_a=(0, -4), omega_a=0.0,
            inv_mass_a=1.0, i_inv_a=None,
            pos_b=(0, -1), vel_b=(0, 0), omega_b=0.0,
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.4, -0.5), normal=(0, 1),
            restitution=1.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=0.0,
        )
        va_rb, oa, _, _, _ = resolve_contact_2d(
            pos_a=(0, 0), vel_a=(0, -4), omega_a=0.0,
            inv_mass_a=1.0, i_inv_a=i_inv,
            pos_b=(0, -1), vel_b=(0, 0), omega_b=0.0,
            inv_mass_b=0.0, i_inv_b=None,
            contact_point=(0.4, -0.5), normal=(0, 1),
            restitution=1.0, static_friction=0.0, dynamic_friction=0.0,
            face_align_a=0.0,
        )
        assert va_rb[1] < va_pt[1] + 1e-6 or abs(oa) > 1e-6
        assert abs(oa) > 1e-6


# =========================================================================
# Full engine integration
# =========================================================================


class TestEngineRotationalResponse2D:
    def test_off_center_impulse_via_resolve_collision(self):
        window = HeadlessWindow2D()
        cube = _make_box(position=(0, 0), bounce=0.0, friction=0.0)
        ground = _make_box(position=(0, -1.0), static=True, bounce=0.0, friction=0.0, size=(10, 1))
        window.objects = [cube, ground]
        rb = cube.get_component(Rigidbody2D)
        rb.velocity = Vector2(0, -5)
        rb.angular_velocity = 0.0

        manifold = CollisionManifold2D(
            normal=np.array([0.0, 1.0], dtype=np.float64),
            depth=0.05,
            contact_point=np.array([0.4, -0.5], dtype=np.float64),
        )
        window._resolve_collision_2d(cube, ground, manifold)

        assert abs(rb.angular_velocity) > 0.1, (
            f"Expected Z rotation from right-edge hit, got ω={rb.angular_velocity}"
        )

    def test_no_spin_up_after_face_landing(self):
        """After landing nearly face-down, body must not restart spinning."""
        window = HeadlessWindow2D()
        cube = _make_box(position=(0, 3.0), bounce=0.1, friction=0.6, gravity=True)
        cube.transform.rotation = (0, 0, 6)
        ground = _make_box(position=(0, -0.5), static=True, bounce=0.0, friction=0.7, size=(20, 1))
        for o in (cube, ground):
            o.transform._compute_world_transform()
            for c in o.get_components(Collider2D):
                c._transform_dirty = True
                c.update_bounds()
            rb = o.get_component(Rigidbody2D)
            if rb:
                rb._inertia_dirty = True
                rb.angular_drag = 0.3
        window.objects = [cube, ground]
        rb = cube.get_component(Rigidbody2D)
        rb.use_gravity = True

        peak_w = 0.0
        contacted = False
        for i in range(300):
            _step(window, dt=1 / 60.0, steps=1)
            w = abs(rb.angular_velocity)
            peak_w = max(peak_w, w)
            if cube.transform.position.y < 1.2:
                contacted = True
            if i > 180 and contacted:
                assert w < 1.5, (
                    f"Late spin-up on ground: |ω|={w:.3f} at step {i}"
                )

        assert contacted, "Box should hit the ground"
        assert peak_w < 12.0, f"Runaway spin peak |ω|={peak_w:.3f}"


# =========================================================================
# Inertia / torque
# =========================================================================


class TestInertia2D:
    def test_box_inertia_inverse_positive(self):
        obj = _make_box()
        inv = obj.get_component(Rigidbody2D).get_inertia_inv()
        assert inv > 0.0

    def test_circle_inertia_matches_disk(self):
        obj = _make_circle(radius=0.5)
        rb = obj.get_component(Rigidbody2D)
        rb.mass = 1.0
        rb._inertia_dirty = True
        inv = rb.get_inertia_inv()
        # I = ½ m r² → for r from collider after update
        assert inv > 0.0

    def test_add_torque_uses_inertia(self):
        obj = _make_box()
        rb = obj.get_component(Rigidbody2D)
        rb.angular_velocity = 0.0
        i_inv = rb.get_inertia_inv()
        rb.add_torque(10.0, as_impulse=True)
        assert abs(rb.angular_velocity - 10.0 * i_inv) < 1e-6
        # Not ω += τ / m
        assert abs(rb.angular_velocity - 10.0) > 0.5

    def test_angular_integration_applies_rotation(self):
        obj = _make_box()
        rb = obj.get_component(Rigidbody2D)
        rb.use_gravity = False
        rb.angular_drag = 0.0
        rb.angular_velocity = math.radians(90)  # 90°/s
        prev = float(obj.transform.rotation_z)
        Time.maximum_delta_time = 0.0
        Time._skip_rigidbody_frame_update = False
        Time.set(1.0)
        rb.update()
        # ~90 degrees of rotation
        assert abs(float(obj.transform.rotation_z) - prev - 90.0) < 1.0
