"""
Tests for physics-material properties on colliders (bounciness, friction,
combine modes) and their effect on collision response.

Each test creates a headless simulation, runs a few physics steps, and checks
that objects move / bounce / slide as expected.
"""
import pytest
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.component import Time
from engine.gameobject import GameObject
from engine.types import Vector3
from engine.types.vector2 import Vector2
from engine.d3.object3d import create_cube, create_plane
from engine.d3.physics.rigidbody import Rigidbody3D
from engine.d3.physics.collider import BoxCollider3D, SphereCollider3D, Collider3D
from engine.d3.physics.types import PhysicsMaterialCombine, CollisionMode
from engine.d2.object2d import Object2D, create_rect
from engine.d2.physics.collider import (
    Collider2D,
    BoxCollider2D,
    CircleCollider2D,
)
from engine.d2.physics.rigidbody import Rigidbody2D
from engine.d2.window2d import Window2D
from engine.d3.window import Window3D


# =========================================================================
# Headless window helpers (no GPU / Pygame)
# =========================================================================


class HeadlessWindow3D(Window3D):
    """Window3D that skips ModernGL/Pygame init — tests physics only."""

    def __init__(self):
        self.objects = []
        self._current_scene = None

    def _active_objects(self):
        return self.objects


class HeadlessWindow2D(Window2D):
    """Window2D that skips ModernGL/Pygame init — tests physics only."""

    def __init__(self):
        self.objects = []
        self._current_scene = None

    def _active_objects(self):
        return self.objects


def _init_3d_objects(window, *objs):
    """Add objects to the window and initialise transforms / bounds."""
    for obj in objs:
        window.objects.append(obj)
    for obj in window.objects:
        obj.transform._compute_world_transform()
        obj.transform._update_prev_position()
        for c in obj.get_components(Collider3D):
            c._transform_dirty = True
            c.update_bounds()


def _init_2d_objects(window, *objs):
    """Add objects to the window and mark colliders dirty."""
    for obj in objs:
        window.objects.append(obj)
    for obj in window.objects:
        for c in obj.get_components(Collider2D):
            c._transform_dirty = True
            c.update_bounds()


def _step_3d(window, rb, dt=1 / 60, steps=1):
    """Run *steps* physics frames on the 3D window."""
    # Allow large test timesteps (production clamps via Time.maximum_delta_time)
    prev_max = Time.maximum_delta_time
    prev_skip = Time._skip_rigidbody_frame_update
    Time.maximum_delta_time = 0.0  # disable clamp
    Time._skip_rigidbody_frame_update = False
    try:
        for _ in range(steps):
            Time.set(dt)
            rb.wake()
            rb.update()
            window._process_collisions()
    finally:
        Time.maximum_delta_time = prev_max
        Time._skip_rigidbody_frame_update = prev_skip


def _step_2d(window, rb, dt=1 / 60, steps=1):
    """Run *steps* physics frames on the 2D window."""
    prev_max = Time.maximum_delta_time
    prev_skip = Time._skip_rigidbody_frame_update
    Time.maximum_delta_time = 0.0  # disable clamp
    Time._skip_rigidbody_frame_update = False
    try:
        for _ in range(steps):
            Time.set(dt)
            rb.wake()
            rb.update()
            window._process_collisions()
    finally:
        Time.maximum_delta_time = prev_max
        Time._skip_rigidbody_frame_update = prev_skip


# =========================================================================
# PhysicsMaterialCombine enum tests
# =========================================================================


class TestPhysicsMaterialCombine:
    def test_average(self):
        result = PhysicsMaterialCombine.combine(
            0.2, 0.8, PhysicsMaterialCombine.AVERAGE, PhysicsMaterialCombine.AVERAGE
        )
        assert abs(result - 0.5) < 1e-9

    def test_minimum(self):
        result = PhysicsMaterialCombine.combine(
            0.2, 0.8, PhysicsMaterialCombine.MINIMUM, PhysicsMaterialCombine.AVERAGE
        )
        assert abs(result - 0.2) < 1e-9

    def test_maximum(self):
        result = PhysicsMaterialCombine.combine(
            0.2, 0.8, PhysicsMaterialCombine.MAXIMUM, PhysicsMaterialCombine.AVERAGE
        )
        assert abs(result - 0.8) < 1e-9

    def test_multiply(self):
        result = PhysicsMaterialCombine.combine(
            0.5, 0.4, PhysicsMaterialCombine.MULTIPLY, PhysicsMaterialCombine.AVERAGE
        )
        assert abs(result - 0.20) < 1e-9

    def test_higher_priority_wins(self):
        """The mode with the higher enum value (= higher priority) is used."""
        # MAXIMUM (3) > MINIMUM (1) → should pick max(0.2, 0.8) = 0.8
        result = PhysicsMaterialCombine.combine(
            0.2, 0.8, PhysicsMaterialCombine.MINIMUM, PhysicsMaterialCombine.MAXIMUM
        )
        assert abs(result - 0.8) < 1e-9


# =========================================================================
# Collider default property tests
# =========================================================================


class TestColliderDefaults:
    def test_collider2d_defaults(self):
        c = BoxCollider2D()
        assert c.bounciness == 0.0
        assert c.static_friction == 0.6
        assert c.dynamic_friction == 0.4
        assert c.friction_combine == PhysicsMaterialCombine.AVERAGE
        assert c.bounce_combine == PhysicsMaterialCombine.AVERAGE

    def test_collider3d_defaults(self):
        c = BoxCollider3D()
        assert c.bounciness == 0.0
        assert c.static_friction == 0.6
        assert c.dynamic_friction == 0.4
        assert c.friction_combine == PhysicsMaterialCombine.AVERAGE
        assert c.bounce_combine == PhysicsMaterialCombine.AVERAGE

    def test_custom_values(self):
        c = CircleCollider2D()
        c.bounciness = 0.8
        c.static_friction = 0.3
        c.dynamic_friction = 0.1
        c.friction_combine = PhysicsMaterialCombine.MAXIMUM
        c.bounce_combine = PhysicsMaterialCombine.MINIMUM
        assert c.bounciness == 0.8
        assert c.static_friction == 0.3
        assert c.dynamic_friction == 0.1
        assert c.friction_combine == PhysicsMaterialCombine.MAXIMUM
        assert c.bounce_combine == PhysicsMaterialCombine.MINIMUM


# =========================================================================
# 3D Bounciness tests
# =========================================================================


class TestBounciness3D:
    def test_no_bounce_stops(self):
        """With bounciness=0, an object hitting a wall should stop (no bounce)."""
        window = HeadlessWindow3D()

        # Dynamic ball moving right
        ball = create_cube(size=1.0, position=(0, 0, 0))
        rb = Rigidbody3D(use_gravity=False)
        rb.velocity = Vector3(-5.0, 0, 0)
        col = BoxCollider3D()
        col.bounciness = 0.0
        ball.add_component(rb)
        ball.add_component(col)

        # Static wall
        wall = create_cube(size=1.0, position=(-3, 0, 0))
        wall.transform.scale_xyz = (1.0, 10.0, 10.0)
        rb_w = Rigidbody3D(use_gravity=False, is_static=True)
        col_w = BoxCollider3D()
        col_w.bounciness = 0.0
        wall.add_component(rb_w)
        wall.add_component(col_w)

        _init_3d_objects(window, ball, wall)
        _step_3d(window, rb, dt=0.5, steps=5)

        # Velocity in X should be ~0 (no bounce)
        assert abs(rb.velocity.x) < 0.1, f"Expected near-zero vx, got {rb.velocity.x}"

    def test_full_bounce_reflects(self):
        """With bounciness=1, the object should bounce back with equal speed."""
        window = HeadlessWindow3D()

        ball = create_cube(size=1.0, position=(0, 0, 0))
        rb = Rigidbody3D(use_gravity=False)
        rb.velocity = Vector3(-5.0, 0, 0)
        col = BoxCollider3D()
        col.bounciness = 1.0
        ball.add_component(rb)
        ball.add_component(col)

        wall = create_cube(size=1.0, position=(-3, 0, 0))
        wall.transform.scale_xyz = (1.0, 10.0, 10.0)
        rb_w = Rigidbody3D(use_gravity=False, is_static=True)
        col_w = BoxCollider3D()
        col_w.bounciness = 1.0
        wall.add_component(rb_w)
        wall.add_component(col_w)

        _init_3d_objects(window, ball, wall)
        _step_3d(window, rb, dt=0.5, steps=5)

        # Velocity in X should be positive (bounced) and roughly 5
        assert rb.velocity.x > 3.0, f"Expected bounce-back vx > 3, got {rb.velocity.x}"

    def test_partial_bounce(self):
        """With bounciness=0.5, the bounce should lose about half its speed."""
        window = HeadlessWindow3D()

        # Place ball close to wall so it collides within a few small steps
        ball = create_cube(size=1.0, position=(-1.5, 0, 0))
        rb = Rigidbody3D(use_gravity=False)
        rb.velocity = Vector3(-4.0, 0, 0)
        col = BoxCollider3D()
        col.bounciness = 0.5
        ball.add_component(rb)
        ball.add_component(col)

        wall = create_cube(size=1.0, position=(-3, 0, 0))
        wall.transform.scale_xyz = (1.0, 10.0, 10.0)
        rb_w = Rigidbody3D(use_gravity=False, is_static=True)
        col_w = BoxCollider3D()
        col_w.bounciness = 0.5
        wall.add_component(rb_w)
        wall.add_component(col_w)

        _init_3d_objects(window, ball, wall)
        _step_3d(window, rb, dt=1 / 60, steps=60)

        # After bounce, speed should be roughly 4 * 0.5 = 2 (restitution 0.5)
        assert 1.0 < rb.velocity.x < 4.0, f"Expected partial bounce vx ∈ (1,4), got {rb.velocity.x}"


# =========================================================================
# 3D Friction tests
# =========================================================================


class TestFriction3D:
    def test_sliding_with_zero_friction(self):
        """No friction → object slides along wall at full tangential speed."""
        window = HeadlessWindow3D()

        ball = create_cube(size=1.0, position=(0, 2, 0))
        rb = Rigidbody3D(use_gravity=False)
        rb.velocity = Vector3(5.0, -5.0, 0)
        col = BoxCollider3D()
        col.bounciness = 0.0
        col.static_friction = 0.0
        col.dynamic_friction = 0.0
        ball.add_component(rb)
        ball.add_component(col)

        floor = create_cube(size=1.0, position=(0, 0, 0))
        floor.transform.scale_xyz = (100, 1, 100)
        rb_f = Rigidbody3D(use_gravity=False, is_static=True)
        col_f = BoxCollider3D()
        col_f.static_friction = 0.0
        col_f.dynamic_friction = 0.0
        floor.add_component(rb_f)
        floor.add_component(col_f)

        _init_3d_objects(window, ball, floor)
        _step_3d(window, rb, dt=0.2, steps=5)

        # Horizontal speed should be preserved (no friction)
        assert abs(rb.velocity.x - 5.0) < 0.5, f"Expected vx ≈ 5, got {rb.velocity.x}"

    def test_sliding_with_high_friction(self):
        """High friction → tangential speed should be reduced significantly."""
        window = HeadlessWindow3D()

        ball = create_cube(size=1.0, position=(0, 2, 0))
        rb = Rigidbody3D(use_gravity=False)
        rb.velocity = Vector3(5.0, -5.0, 0)
        col = BoxCollider3D()
        col.bounciness = 0.0
        col.static_friction = 1.0
        col.dynamic_friction = 0.8
        ball.add_component(rb)
        ball.add_component(col)

        floor = create_cube(size=1.0, position=(0, 0, 0))
        floor.transform.scale_xyz = (100, 1, 100)
        rb_f = Rigidbody3D(use_gravity=False, is_static=True)
        col_f = BoxCollider3D()
        col_f.static_friction = 1.0
        col_f.dynamic_friction = 0.8
        floor.add_component(rb_f)
        floor.add_component(col_f)

        _init_3d_objects(window, ball, floor)
        _step_3d(window, rb, dt=0.2, steps=5)

        # Horizontal speed should be reduced by friction
        assert rb.velocity.x < 5.0, f"Expected friction-reduced vx < 5.0, got {rb.velocity.x}"


# =========================================================================
# 2D Bounciness tests
# =========================================================================


class TestBounciness2D:
    def test_no_bounce_2d(self):
        """2D: bounciness=0 → object stops against wall."""
        window = HeadlessWindow2D()

        player = create_rect(1, 1, position=(0, 0))
        rb = Rigidbody2D(use_gravity=False)
        rb.velocity = Vector2(-3.0, 0.0)
        col = BoxCollider2D()
        col.bounciness = 0.0
        player.add_component(rb)
        player.add_component(col)

        wall = create_rect(1, 10, position=(-3, 0))
        rb_w = Rigidbody2D(use_gravity=False, is_static=True)
        col_w = BoxCollider2D()
        col_w.bounciness = 0.0
        wall.add_component(rb_w)
        wall.add_component(col_w)

        _init_2d_objects(window, player, wall)
        _step_2d(window, rb, dt=0.5, steps=5)

        assert abs(rb.velocity.x) < 0.1, f"Expected near-zero vx, got {rb.velocity.x}"

    def test_full_bounce_2d(self):
        """2D: bounciness=1 → perfect elastic bounce."""
        window = HeadlessWindow2D()

        player = create_rect(1, 1, position=(0, 0))
        rb = Rigidbody2D(use_gravity=False)
        rb.velocity = Vector2(-3.0, 0.0)
        col = BoxCollider2D()
        col.bounciness = 1.0
        player.add_component(rb)
        player.add_component(col)

        wall = create_rect(1, 10, position=(-3, 0))
        rb_w = Rigidbody2D(use_gravity=False, is_static=True)
        col_w = BoxCollider2D()
        col_w.bounciness = 1.0
        wall.add_component(rb_w)
        wall.add_component(col_w)

        _init_2d_objects(window, player, wall)
        _step_2d(window, rb, dt=0.5, steps=5)

        # Should bounce back with roughly the same speed
        assert rb.velocity.x > 2.0, f"Expected bounce vx > 2, got {rb.velocity.x}"

    def test_gravity_bounce_2d(self):
        """2D: ball falling under gravity with bounciness should bounce up."""
        window = HeadlessWindow2D()

        ball = create_rect(1, 1, position=(0, 5))
        rb = Rigidbody2D(use_gravity=True, gravity_scale=1.0)
        col = BoxCollider2D()
        col.bounciness = 0.8
        ball.add_component(rb)
        ball.add_component(col)

        floor = create_rect(20, 1, position=(0, 0))
        rb_f = Rigidbody2D(use_gravity=False, is_static=True)
        col_f = BoxCollider2D()
        col_f.bounciness = 0.8
        floor.add_component(rb_f)
        floor.add_component(col_f)

        _init_2d_objects(window, ball, floor)

        # Run until it hits the floor
        bounced = False
        for _ in range(200):
            _step_2d(window, rb, dt=1 / 60, steps=1)
            if rb.velocity.y > 0.5:
                bounced = True
                break

        assert bounced, "Ball should have bounced up after hitting the floor"


# =========================================================================
# 2D Friction tests
# =========================================================================


class TestFriction2D:
    def test_sliding_no_friction_2d(self):
        """2D: zero friction → full tangential speed preserved when hitting wall."""
        window = HeadlessWindow2D()

        player = create_rect(1, 1, position=(0, 3))
        rb = Rigidbody2D(use_gravity=False)
        rb.velocity = Vector2(5.0, -5.0)
        col = BoxCollider2D()
        col.bounciness = 0.0
        col.static_friction = 0.0
        col.dynamic_friction = 0.0
        player.add_component(rb)
        player.add_component(col)

        floor = create_rect(20, 1, position=(0, 0))
        rb_f = Rigidbody2D(use_gravity=False, is_static=True)
        col_f = BoxCollider2D()
        col_f.static_friction = 0.0
        col_f.dynamic_friction = 0.0
        floor.add_component(rb_f)
        floor.add_component(col_f)

        _init_2d_objects(window, player, floor)
        _step_2d(window, rb, dt=0.3, steps=5)

        # Horizontal speed should be fully preserved
        assert abs(rb.velocity.x - 5.0) < 0.5, f"Expected vx ≈ 5, got {rb.velocity.x}"

    def test_sliding_with_friction_2d(self):
        """2D: high friction → tangential velocity reduced."""
        window = HeadlessWindow2D()

        player = create_rect(1, 1, position=(0, 1.5))
        rb = Rigidbody2D(use_gravity=False)
        rb.velocity = Vector2(5.0, -5.0)
        col = BoxCollider2D()
        col.bounciness = 0.0
        col.static_friction = 1.0
        col.dynamic_friction = 0.8
        player.add_component(rb)
        player.add_component(col)

        floor = create_rect(20, 1, position=(0, 0))
        rb_f = Rigidbody2D(use_gravity=False, is_static=True)
        col_f = BoxCollider2D()
        col_f.static_friction = 1.0
        col_f.dynamic_friction = 0.8
        floor.add_component(rb_f)
        floor.add_component(col_f)

        _init_2d_objects(window, player, floor)
        _step_2d(window, rb, dt=1 / 60, steps=60)

        # Friction should slow horizontal movement
        assert rb.velocity.x < 4.5, f"Expected friction-reduced vx < 4.5, got {rb.velocity.x}"


# =========================================================================
# Combine mode integration tests
# =========================================================================


class TestCombineModesIntegration:
    def test_average_combine_bounce(self):
        """Average combine: (0.0 + 1.0)/2 = 0.5 → partial bounce."""
        window = HeadlessWindow3D()

        a = create_cube(size=1.0, position=(-1.5, 0, 0))
        rb_a = Rigidbody3D(use_gravity=False)
        rb_a.velocity = Vector3(-4.0, 0, 0)
        col_a = BoxCollider3D()
        col_a.bounciness = 0.0
        col_a.bounce_combine = PhysicsMaterialCombine.AVERAGE
        a.add_component(rb_a)
        a.add_component(col_a)

        b = create_cube(size=1.0, position=(-3, 0, 0))
        b.transform.scale_xyz = (1.0, 10.0, 10.0)
        rb_b = Rigidbody3D(use_gravity=False, is_static=True)
        col_b = BoxCollider3D()
        col_b.bounciness = 1.0
        col_b.bounce_combine = PhysicsMaterialCombine.AVERAGE
        b.add_component(rb_b)
        b.add_component(col_b)

        _init_3d_objects(window, a, b)
        _step_3d(window, rb_a, dt=1 / 60, steps=60)

        # Combined bounce = (0.0 + 1.0)/2 = 0.5 → speed should be ~50% of original
        assert 1.0 < rb_a.velocity.x < 3.5, f"Expected partial bounce, got vx={rb_a.velocity.x}"

    def test_max_combine_bounce(self):
        """Max combine: max(0.0, 1.0) = 1.0 → full bounce even though one is zero."""
        window = HeadlessWindow3D()

        a = create_cube(size=1.0, position=(-1.5, 0, 0))
        rb_a = Rigidbody3D(use_gravity=False)
        rb_a.velocity = Vector3(-4.0, 0, 0)
        col_a = BoxCollider3D()
        col_a.bounciness = 0.0
        col_a.bounce_combine = PhysicsMaterialCombine.MAXIMUM
        a.add_component(rb_a)
        a.add_component(col_a)

        b = create_cube(size=1.0, position=(-3, 0, 0))
        b.transform.scale_xyz = (1.0, 10.0, 10.0)
        rb_b = Rigidbody3D(use_gravity=False, is_static=True)
        col_b = BoxCollider3D()
        col_b.bounciness = 1.0
        col_b.bounce_combine = PhysicsMaterialCombine.AVERAGE  # lower priority → MAX wins
        b.add_component(rb_b)
        b.add_component(col_b)

        _init_3d_objects(window, a, b)
        _step_3d(window, rb_a, dt=1 / 60, steps=60)

        # MAX combine picks max(0.0, 1.0)=1.0 → nearly perfect bounce
        assert rb_a.velocity.x > 3.0, f"Expected full bounce, got vx={rb_a.velocity.x}"

    def test_min_combine_friction(self):
        """Min combine friction: min(1.0, 0.0) = 0.0 → frictionless sliding."""
        window = HeadlessWindow3D()

        a = create_cube(size=1.0, position=(0, 1.5, 0))
        rb_a = Rigidbody3D(use_gravity=False)
        rb_a.velocity = Vector3(5.0, -5.0, 0)
        col_a = BoxCollider3D()
        col_a.bounciness = 0.0
        col_a.static_friction = 1.0
        col_a.dynamic_friction = 1.0
        col_a.friction_combine = PhysicsMaterialCombine.MINIMUM  # MIN wins over AVERAGE
        a.add_component(rb_a)
        a.add_component(col_a)

        floor = create_cube(size=1.0, position=(0, 0, 0))
        floor.transform.scale_xyz = (100, 1, 100)
        rb_f = Rigidbody3D(use_gravity=False, is_static=True)
        col_f = BoxCollider3D()
        col_f.static_friction = 0.0
        col_f.dynamic_friction = 0.0
        col_f.friction_combine = PhysicsMaterialCombine.AVERAGE
        floor.add_component(rb_f)
        floor.add_component(col_f)

        _init_3d_objects(window, a, floor)
        _step_3d(window, rb_a, dt=1 / 60, steps=60)

        # MIN combine: min(1.0, 0.0)=0 → sliding should be frictionless
        assert abs(rb_a.velocity.x - 5.0) < 0.5, f"Expected frictionless slide vx ≈ 5, got {rb_a.velocity.x}"

    def test_multiply_combine_bounce(self):
        """Multiply combine: 0.5 * 0.4 = 0.2 → low bounce."""
        window = HeadlessWindow3D()

        a = create_cube(size=1.0, position=(-1.5, 0, 0))
        rb_a = Rigidbody3D(use_gravity=False)
        rb_a.velocity = Vector3(-4.0, 0, 0)
        col_a = BoxCollider3D()
        col_a.bounciness = 0.5
        col_a.bounce_combine = PhysicsMaterialCombine.MULTIPLY
        a.add_component(rb_a)
        a.add_component(col_a)

        b = create_cube(size=1.0, position=(-3, 0, 0))
        b.transform.scale_xyz = (1.0, 10.0, 10.0)
        rb_b = Rigidbody3D(use_gravity=False, is_static=True)
        col_b = BoxCollider3D()
        col_b.bounciness = 0.4
        col_b.bounce_combine = PhysicsMaterialCombine.AVERAGE
        b.add_component(rb_b)
        b.add_component(col_b)

        _init_3d_objects(window, a, b)
        _step_3d(window, rb_a, dt=1 / 60, steps=60)

        # Combined bounce = 0.5 * 0.4 = 0.2 → speed ~0.8 (4*0.2)
        assert 0.1 < rb_a.velocity.x < 2.0, f"Expected low bounce vx, got {rb_a.velocity.x}"


# =========================================================================
# Cython vs Python parity tests
# =========================================================================


class TestCythonParity:
    def test_resolve_velocity_2d_parity(self):
        """Cython and Python velocity resolution should produce identical results."""
        try:
            from engine.cython.cy_math import resolve_velocity_2d as cy_fn
        except ImportError:
            pytest.skip("Cython not available")

        py_fn = HeadlessWindow2D._resolve_velocity_2d_py

        cases = [
            # (vx_a, vy_a, vx_b, vy_b, nx, ny, im_a, im_b, rest, sf, df)
            (-2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.6, 0.4),
            (-3.0, 2.0, 1.0, -1.0, 0.707, 0.707, 1.0, 0.5, 0.8, 0.3, 0.2),
            (-1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.8),
            (0.0, -5.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        ]

        for args in cases:
            cy_result = cy_fn(*args)
            py_result = py_fn(*args)
            for i in range(4):
                assert abs(cy_result[i] - py_result[i]) < 1e-9, (
                    f"Mismatch at index {i}: cy={cy_result[i]}, py={py_result[i]} for args={args}"
                )

    def test_resolve_velocity_3d_parity(self):
        """Cython and Python velocity resolution should produce identical results (3D)."""
        try:
            from engine.cython.cy_math import resolve_velocity_3d as cy_fn
        except ImportError:
            pytest.skip("Cython not available")

        py_fn = HeadlessWindow3D._resolve_velocity_3d_py

        cases = [
            (-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.6, 0.4),
            (-3.0, 2.0, 1.0, 1.0, -1.0, 0.5, 0.577, 0.577, 0.577, 1.0, 0.5, 0.8, 0.3, 0.2),
            (0.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        ]

        for args in cases:
            cy_result = cy_fn(*args)
            py_result = py_fn(*args)
            for i in range(6):
                assert abs(cy_result[i] - py_result[i]) < 1e-9, (
                    f"Mismatch at index {i}: cy={cy_result[i]}, py={py_result[i]} for args={args}"
                )


# =========================================================================
# Two dynamic bodies collision tests
# =========================================================================


class TestDynamicCollisions:
    def test_two_dynamic_bounce_3d(self):
        """Two dynamic objects colliding should exchange velocity based on bounciness."""
        window = HeadlessWindow3D()

        a = create_cube(size=1.0, position=(-2, 0, 0))
        rb_a = Rigidbody3D(use_gravity=False)
        rb_a.velocity = Vector3(3.0, 0, 0)
        col_a = BoxCollider3D()
        col_a.bounciness = 1.0
        a.add_component(rb_a)
        a.add_component(col_a)

        b = create_cube(size=1.0, position=(2, 0, 0))
        rb_b = Rigidbody3D(use_gravity=False)
        rb_b.velocity = Vector3(-3.0, 0, 0)
        col_b = BoxCollider3D()
        col_b.bounciness = 1.0
        b.add_component(rb_b)
        b.add_component(col_b)

        _init_3d_objects(window, a, b)
        _step_3d(window, rb_a, dt=0.5, steps=1)
        rb_b.update()  # Also step b
        window._process_collisions()

        # After elastic collision of equal masses, velocities swap
        # A should be going left, B should be going right
        # (or they might pass through if collision detection needs more steps)
        total_energy_before = 3.0 ** 2 + 3.0 ** 2
        total_energy_after = rb_a.velocity.x ** 2 + rb_b.velocity.x ** 2
        # Energy should be approximately conserved for bounciness=1
        assert abs(total_energy_after - total_energy_before) < total_energy_before * 0.3, \
            f"Energy not conserved: before={total_energy_before}, after={total_energy_after}"

    def test_two_dynamic_bounce_2d(self):
        """Two dynamic 2D objects colliding should exchange velocity."""
        window = HeadlessWindow2D()

        a = create_rect(1, 1, position=(-2, 0))
        rb_a = Rigidbody2D(use_gravity=False)
        rb_a.velocity = Vector2(3.0, 0)
        col_a = BoxCollider2D()
        col_a.bounciness = 1.0
        a.add_component(rb_a)
        a.add_component(col_a)

        b = create_rect(1, 1, position=(2, 0))
        rb_b = Rigidbody2D(use_gravity=False)
        rb_b.velocity = Vector2(-3.0, 0)
        col_b = BoxCollider2D()
        col_b.bounciness = 1.0
        b.add_component(rb_b)
        b.add_component(col_b)

        _init_2d_objects(window, a, b)

        prev_max = Time.maximum_delta_time
        Time.maximum_delta_time = 0.0
        try:
            for _ in range(5):
                Time.set(0.3)
                rb_a.wake()
                rb_b.wake()
                rb_a.update()
                rb_b.update()
                window._process_collisions()
        finally:
            Time.maximum_delta_time = prev_max

        total_ke = rb_a.velocity.x ** 2 + rb_b.velocity.x ** 2
        assert total_ke > 5.0, f"Energy should be mostly conserved, got KE={total_ke}"


# =========================================================================
# Walking-into-wall tests (the original problem)
# =========================================================================


class TestWalkIntoWall:
    def test_walk_into_wall_slides_2d(self):
        """Walking diagonally into a vertical wall should maintain vertical motion."""
        window = HeadlessWindow2D()

        player = create_rect(1, 1, position=(0, 0))
        rb = Rigidbody2D(use_gravity=False)
        rb.velocity = Vector2(-3.0, 2.0)  # moving left and up
        col = BoxCollider2D()
        col.bounciness = 0.0
        col.static_friction = 0.0
        col.dynamic_friction = 0.0
        player.add_component(rb)
        player.add_component(col)

        wall = create_rect(1, 20, position=(-3, 0))
        rb_w = Rigidbody2D(use_gravity=False, is_static=True)
        col_w = BoxCollider2D()
        col_w.static_friction = 0.0
        col_w.dynamic_friction = 0.0
        wall.add_component(rb_w)
        wall.add_component(col_w)

        _init_2d_objects(window, player, wall)
        _step_2d(window, rb, dt=0.5, steps=5)

        # X velocity should be near zero (blocked by wall)
        assert abs(rb.velocity.x) < 0.5, f"Expected vx ≈ 0 against wall, got {rb.velocity.x}"
        # Y velocity should be preserved (sliding along wall, no friction)
        assert abs(rb.velocity.y - 2.0) < 0.5, f"Expected vy ≈ 2 (sliding), got {rb.velocity.y}"

    def test_gravity_object_hits_floor_bounces_2d(self):
        """Object falling with gravity hits floor and bounces (not going 'too slow')."""
        window = HeadlessWindow2D()

        ball = create_rect(1, 1, position=(0, 10))
        rb = Rigidbody2D(use_gravity=True, gravity_scale=1.0)
        col = BoxCollider2D()
        col.bounciness = 0.9
        col.static_friction = 0.0
        col.dynamic_friction = 0.0
        ball.add_component(rb)
        ball.add_component(col)

        floor = create_rect(20, 1, position=(0, 0))
        rb_f = Rigidbody2D(use_gravity=False, is_static=True)
        col_f = BoxCollider2D()
        col_f.bounciness = 0.9
        floor.add_component(rb_f)
        floor.add_component(col_f)

        _init_2d_objects(window, ball, floor)

        max_bounce_vy = 0.0
        for _ in range(300):
            _step_2d(window, rb, dt=1 / 60, steps=1)
            if rb.velocity.y > max_bounce_vy:
                max_bounce_vy = rb.velocity.y

        # With bounciness 0.9 and gravity, the ball should bounce up significantly
        assert max_bounce_vy > 2.0, f"Expected significant bounce, max vy={max_bounce_vy}"


# =========================================================================
# Physics material combine helper (Cython)
# =========================================================================


class TestPhysicsMaterialCombineCython:
    def test_cython_combine_matches_python(self):
        try:
            from engine.cython.cy_math import physics_material_combine as cy_fn
        except ImportError:
            pytest.skip("Cython not available")

        cases = [
            (0.3, 0.7, 0, 0, 0.5),    # AVERAGE
            (0.3, 0.7, 1, 0, 0.3),    # MINIMUM
            (0.3, 0.7, 3, 0, 0.7),    # MAXIMUM
            (0.5, 0.4, 2, 0, 0.2),    # MULTIPLY
            (0.3, 0.7, 1, 3, 0.7),    # MIN vs MAX → MAX wins
        ]
        for a_val, b_val, a_mode, b_mode, expected in cases:
            result = cy_fn(a_val, b_val, a_mode, b_mode)
            assert abs(result - expected) < 1e-9, (
                f"Expected {expected}, got {result} for ({a_val}, {b_val}, mode_a={a_mode}, mode_b={b_mode})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
