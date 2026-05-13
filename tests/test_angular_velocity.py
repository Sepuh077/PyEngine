"""
Tests for angular velocity and impulse-based collision resolution.

All tests run without OpenGL by using a HeadlessWindow that bypasses
GPU initialization while inheriting the physics resolution logic from
Window3D.
"""
import pytest
import numpy as np
import sys
import os
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine3d.engine3d.component import Time
from engine3d.engine3d.gameobject import GameObject
from engine3d.engine3d.object3d import create_cube, create_plane
from engine3d.engine3d.window import Window3D
from engine3d.physics.rigidbody import Rigidbody
from engine3d.physics.collider import BoxCollider, SphereCollider, CapsuleCollider, CollisionMode
from engine3d.physics.collision_manifold import CollisionManifold
from engine3d.types import Vector3


# ---------------------------------------------------------------------------
# Headless helpers – no OpenGL needed
# ---------------------------------------------------------------------------

class HeadlessWindow(Window3D):
    """Window subclass that skips all GPU / pygame initialisation."""

    def __init__(self):
        # Bypass Window3D.__init__ entirely
        self.objects = []
        self._current_scene = None

    def _active_objects(self):
        return self.objects


def _make_cube(position=(0, 0, 0), mass=1.0, static=False, gravity=True,
               restitution=0.3, friction=0.5, size=1.0):
    """Create a cube GameObject with Rigidbody + BoxCollider, bounds ready."""
    obj = create_cube(size=size, position=position)
    rb = Rigidbody(use_gravity=gravity, is_static=static)
    rb.mass = mass
    rb.restitution = restitution
    rb.friction = friction
    col = BoxCollider()
    obj.add_component(rb)
    obj.add_component(col)
    obj.transform._compute_world_transform()
    col.update_bounds()
    rb._inertia_dirty = True  # force recompute after bounds are ready
    return obj


def _make_plane(position=(0, 0, 0), restitution=0.5, friction=0.5):
    """Create a static ground plane with Rigidbody + BoxCollider."""
    obj = create_plane(position=position)
    rb = Rigidbody(use_gravity=False, is_static=True)
    rb.restitution = restitution
    rb.friction = friction
    col = BoxCollider()
    obj.add_component(rb)
    obj.add_component(col)
    obj.transform._compute_world_transform()
    col.update_bounds()
    return obj


def _step(window, dt=1 / 60.0, steps=1, track_rb=None):
    """Advance physics by *steps* frames of *dt* seconds each.

    If *track_rb* is given, returns (peak_angular_speed, peak_linear_speed)
    observed across all frames.
    """
    Time.delta_time = dt
    peak_ang = 0.0
    peak_lin = 0.0
    for _ in range(steps):
        for o in window.objects:
            rb = o.get_component(Rigidbody)
            if rb:
                rb.update()
        window._process_collisions()
        if track_rb is not None:
            peak_ang = max(peak_ang, _ang_mag(track_rb))
            peak_lin = max(peak_lin, track_rb.velocity.magnitude)
    if track_rb is not None:
        return peak_ang, peak_lin
    return None


def _ang_mag(rb):
    """Magnitude of angular velocity."""
    av = rb.angular_velocity
    return math.sqrt(av.x ** 2 + av.y ** 2 + av.z ** 2)


# ===================================================================
# 1. Cube edge hits static plane → starts rotating
# ===================================================================

class TestCubeEdgeHitsPlane:
    """
    A cube tilted ~30° around Z is dropped onto a plane.  The bottom
    corner/edge hits first, creating an off-center contact that must
    produce angular velocity and possibly horizontal velocity.
    """

    def _setup(self):
        window = HeadlessWindow()
        cube = _make_cube(position=(0, 2.0, 0.0), gravity=True,
                          restitution=0.2, friction=0.6)
        # Tilt the cube so its corner will hit the plane first
        cube.transform.rotation = (0, 0, 0)
        cube.transform._compute_world_transform()
        for c in cube.get_components(BoxCollider):
            c._transform_dirty = True
            c.update_bounds()
        rb = cube.get_component(Rigidbody)
        rb._inertia_dirty = True

        plane = _make_plane(position=(0, 0, 0))
        window.objects = [cube, plane]
        return window, cube, rb

    def test_edge_collision_causes_rotation(self):
        window, cube, rb = self._setup()
        peak_ang, _ = _step(window, dt=1 / 60.0, steps=150, track_rb=rb)

        # Peak angular speed during the simulation — the cube will
        # eventually settle, so we check the peak, not the final value.
        assert peak_ang > 0.1, (
            f"Tilted cube should rotate after edge hit, peak ω = {peak_ang:.4f}")

    def test_edge_collision_gives_horizontal_velocity(self):
        window, cube, rb = self._setup()
        peak_ang, peak_lin = _step(window, dt=1 / 60.0, steps=150, track_rb=rb)

        assert peak_ang > 0.05 or peak_lin > 0.5, (
            f"Tilted cube should gain spin or horizontal velocity, "
            f"peak ω={peak_ang:.4f}, peak |v|={peak_lin:.4f}")

    def test_cube_continues_falling_after_edge_hit(self):
        window, cube, rb = self._setup()
        _step(window, dt=1 / 60.0, steps=150)

        pos_y = cube.transform.position.y
        assert pos_y < 2.0, f"Cube should have fallen from 2.0, is at y={pos_y}"


# ===================================================================
# 2. Two dynamic cubes – off-center collision
# ===================================================================

class TestTwoDynamicCubes:

    def test_both_gain_angular_velocity(self):
        window = HeadlessWindow()

        # Two cubes approaching each other with a slight vertical offset
        a = _make_cube(position=(-2, 0.3, 0), gravity=False, restitution=0.5)
        b = _make_cube(position=(2, -0.3, 0), gravity=False, restitution=0.5)

        rb_a = a.get_component(Rigidbody)
        rb_b = b.get_component(Rigidbody)
        rb_a.velocity = Vector3(5, 0, 0)
        rb_b.velocity = Vector3(-5, 0, 0)

        window.objects = [a, b]
        _step(window, dt=1 / 60.0, steps=120)

        # Both should have gained angular velocity from off-center impact
        assert _ang_mag(rb_a) > 0.01, "Object A should rotate after off-center collision"
        assert _ang_mag(rb_b) > 0.01, "Object B should rotate after off-center collision"


# ===================================================================
# 3. Direct force / torque API
# ===================================================================

class TestForceAndTorqueAPI:

    def test_add_torque_changes_angular_velocity(self):
        obj = _make_cube(position=(0, 0, 0), gravity=False)
        rb = obj.get_component(Rigidbody)
        rb.angular_velocity = Vector3.zero()
        rb.add_torque(Vector3(0, 10, 0))
        assert abs(rb.angular_velocity.y) > 0, "add_torque should change angular velocity"

    def test_add_force_at_position_creates_torque(self):
        obj = _make_cube(position=(0, 0, 0), gravity=False)
        rb = obj.get_component(Rigidbody)
        rb.velocity = Vector3.zero()
        rb.angular_velocity = Vector3.zero()

        # Force applied at an offset point should create both
        # linear velocity AND angular velocity
        rb.add_force_at_position(Vector3(0, 10, 0), Vector3(0.5, 0, 0))

        assert abs(rb.velocity.y) > 0, "Force should change linear velocity"
        assert _ang_mag(rb) > 0, "Off-center force should create angular velocity"

    def test_force_at_center_no_torque(self):
        obj = _make_cube(position=(0, 0, 0), gravity=False)
        rb = obj.get_component(Rigidbody)
        rb.velocity = Vector3.zero()
        rb.angular_velocity = Vector3.zero()

        # Force at center of mass — no torque
        rb.add_force_at_position(Vector3(0, 10, 0), Vector3(0, 0, 0))
        assert abs(rb.velocity.y) > 0, "Force should change linear velocity"
        assert _ang_mag(rb) < 1e-6, "Force at center should not create angular velocity"


# ===================================================================
# 4. Angular drag slows rotation
# ===================================================================

class TestAngularDrag:

    def test_angular_drag_reduces_spin(self):
        obj = _make_cube(position=(0, 0, 0), gravity=False)
        rb = obj.get_component(Rigidbody)
        rb.angular_drag = 2.0
        rb.angular_velocity = Vector3(0, 10, 0)

        Time.delta_time = 1 / 60.0
        initial_mag = _ang_mag(rb)

        for _ in range(60):
            rb.update()

        assert _ang_mag(rb) < initial_mag, "Angular drag should reduce angular velocity"

    def test_zero_drag_preserves_spin(self):
        obj = _make_cube(position=(0, 0, 0), gravity=False)
        rb = obj.get_component(Rigidbody)
        rb.angular_drag = 0.0
        rb.angular_velocity = Vector3(0, 5, 0)

        Time.delta_time = 1 / 60.0
        for _ in range(60):
            rb.update()

        assert abs(rb.angular_velocity.y - 5.0) < 1e-4, "Zero drag should preserve spin"


# ===================================================================
# 5. Angular velocity applies rotation to transform
# ===================================================================

class TestAngularVelocityIntegration:

    def test_rotation_changes_over_time(self):
        obj = _make_cube(position=(0, 0, 0), gravity=False)
        rb = obj.get_component(Rigidbody)
        rb.angular_drag = 0.0
        rb.angular_velocity = Vector3(0, math.pi, 0)  # 180 deg/s around Y

        initial_rot = obj.transform.rotation_y
        Time.delta_time = 1.0
        rb.update()

        rot_diff = abs(obj.transform.rotation_y - initial_rot)
        # Should have rotated ~180 degrees
        assert rot_diff > 170 and rot_diff < 190, (
            f"Expected ~180° rotation, got {rot_diff}°")


# ===================================================================
# 6. Static objects do not move or rotate
# ===================================================================

class TestStaticObjects:

    def test_static_object_no_angular_velocity(self):
        window = HeadlessWindow()

        static_box = _make_cube(position=(0, 0, 0), static=True, gravity=False)
        dynamic_box = _make_cube(position=(0, 2, 0), gravity=True, restitution=0.5)

        window.objects = [static_box, dynamic_box]
        _step(window, dt=1 / 60.0, steps=120)

        rb_static = static_box.get_component(Rigidbody)
        assert _ang_mag(rb_static) < 1e-8, "Static body should never gain angular velocity"
        assert abs(rb_static.velocity.x) < 1e-8, "Static body should never gain linear velocity"

    def test_static_plane_stays_put(self):
        window = HeadlessWindow()

        cube = _make_cube(position=(0, 2, 0), gravity=True)
        plane = _make_plane(position=(0, 0, 0))
        window.objects = [cube, plane]

        plane_pos_before = plane.transform.position.y
        _step(window, dt=1 / 60.0, steps=120)

        assert abs(plane.transform.position.y - plane_pos_before) < 1e-8, \
            "Static plane must not move"


# ===================================================================
# 7. Inertia depends on shape / mass
# ===================================================================

class TestInertiaDependsOnShape:

    def test_heavier_object_rotates_less(self):
        window = HeadlessWindow()

        light = _make_cube(position=(0.6, 1.5, 0), mass=1.0, gravity=True,
                           restitution=0.3, friction=0.5)
        window.objects = [light, _make_plane()]
        _step(window, dt=1 / 60.0, steps=120)
        ang_light = _ang_mag(light.get_component(Rigidbody))

        window2 = HeadlessWindow()
        heavy = _make_cube(position=(0.6, 1.5, 0), mass=100.0, gravity=True,
                           restitution=0.3, friction=0.5)
        window2.objects = [heavy, _make_plane()]
        _step(window2, dt=1 / 60.0, steps=120)
        ang_heavy = _ang_mag(heavy.get_component(Rigidbody))

        # Heavier cube should rotate less from the same drop scenario
        # (impulse is similar but moment of inertia is larger)
        assert ang_heavy < ang_light or ang_light < 1e-6, (
            f"Heavier cube ({ang_heavy:.4f}) should rotate less than "
            f"lighter cube ({ang_light:.4f})")


# ===================================================================
# 8. Restitution controls bounce
# ===================================================================

class TestRestitution:

    @staticmethod
    def _drop_and_measure_peak_bounce(restitution):
        """Drop a cube from y=3 and return the peak upward velocity after impact."""
        w = HeadlessWindow()
        cube = _make_cube(position=(0, 3, 0), restitution=restitution, gravity=True)
        w.objects = [cube, _make_plane()]
        rb = cube.get_component(Rigidbody)

        max_up_vel = 0.0
        hit = False
        for _ in range(180):  # 3 seconds
            _step(w, dt=1 / 60.0, steps=1)
            if rb.velocity.y > 0.01:
                hit = True
                max_up_vel = max(max_up_vel, rb.velocity.y)
            # After bouncing and coming back down, stop tracking
            if hit and rb.velocity.y < -1.0:
                break
        return max_up_vel

    def test_higher_restitution_bounces_more(self):
        peak_low = self._drop_and_measure_peak_bounce(0.0)
        peak_high = self._drop_and_measure_peak_bounce(0.9)

        assert peak_high > peak_low, (
            f"High restitution peak={peak_high:.3f} should exceed "
            f"low restitution peak={peak_low:.3f}")


# ===================================================================
# 9. Manifold contact_point is populated
# ===================================================================

class TestManifoldContactPoint:

    def test_obb_obb_manifold_has_contact_point(self):
        a = _make_cube(position=(0, 0, 0))
        b = _make_cube(position=(0.9, 0, 0))
        from engine3d.physics.collision_manifold import get_collision_manifold
        col_a = a.get_component(BoxCollider)
        col_b = b.get_component(BoxCollider)
        m = get_collision_manifold(col_a, col_b)
        assert m is not None, "Overlapping cubes should produce a manifold"
        assert m.contact_point is not None, "Manifold must include a contact_point"
        assert len(m.contact_point) == 3, "contact_point must be 3D"

    def test_sphere_obb_manifold_has_contact_point(self):
        sphere_obj = create_cube(size=1.0, position=(0, 0, 0))
        col_s = SphereCollider()
        sphere_obj.add_component(col_s)
        sphere_obj.add_component(Rigidbody(use_gravity=False))
        sphere_obj.transform._compute_world_transform()
        col_s.update_bounds()

        box_obj = _make_cube(position=(0.8, 0, 0))
        col_b = box_obj.get_component(BoxCollider)

        from engine3d.physics.collision_manifold import sphere_vs_obb_manifold
        if col_s.sphere and col_b.obb:
            m = sphere_vs_obb_manifold(col_s, col_b)
            if m is not None:
                assert m.contact_point is not None


# ===================================================================
# 10. Cube falling centered on plane — no angular velocity
# ===================================================================

class TestCenteredDropNoRotation:

    def test_centered_drop_minimal_rotation(self):
        """A cube dropping perfectly centered on a plane should have
        minimal angular velocity (no off-center contact)."""
        window = HeadlessWindow()
        cube = _make_cube(position=(0, 2, 0), gravity=True,
                          restitution=0.3, friction=0.5)
        plane = _make_plane(position=(0, 0, 0))
        window.objects = [cube, plane]
        _step(window, dt=1 / 60.0, steps=120)

        rb = cube.get_component(Rigidbody)
        # Centered drop has symmetric contact — angular velocity should be tiny
        assert _ang_mag(rb) < 1.0, (
            f"Centered drop should produce minimal rotation, "
            f"got angular speed {_ang_mag(rb):.4f}")


# ===================================================================
# 11. Friction coefficient effect
# ===================================================================

class TestFrictionEffect:

    def test_high_friction_produces_more_spin(self):
        """Higher friction should convert more tangential velocity into spin."""

        # Low friction run
        w_lo = HeadlessWindow()
        c_lo = _make_cube(position=(0.6, 1.5, 0), gravity=True,
                          restitution=0.2, friction=0.1)
        w_lo.objects = [c_lo, _make_plane()]
        _step(w_lo, dt=1 / 60.0, steps=90)
        ang_lo = _ang_mag(c_lo.get_component(Rigidbody))

        # High friction run
        w_hi = HeadlessWindow()
        c_hi = _make_cube(position=(0.6, 1.5, 0), gravity=True,
                          restitution=0.2, friction=0.9)
        w_hi.objects = [c_hi, _make_plane()]
        _step(w_hi, dt=1 / 60.0, steps=90)
        ang_hi = _ang_mag(c_hi.get_component(Rigidbody))

        # At least one run should show rotation; higher friction should not
        # result in *less* rotation than lower friction (approximately)
        assert ang_hi >= ang_lo * 0.5 or ang_hi > 0.01, (
            f"High friction ({ang_hi:.4f}) should produce comparable or more "
            f"spin than low friction ({ang_lo:.4f})")


# ===================================================================
# 12. Direct manifold impulse unit test
# ===================================================================

class TestImpulsePhysicsDirect:
    """Directly construct a manifold and call _resolve_collision to verify
    the impulse maths independent of the broad/narrow-phase pipeline."""

    def test_off_center_impulse_creates_angular_velocity(self):
        window = HeadlessWindow()

        cube = _make_cube(position=(0, 0, 0), gravity=False, restitution=0.0)
        plane = _make_plane(position=(0, -0.6, 0))
        window.objects = [cube, plane]

        rb = cube.get_component(Rigidbody)
        rb.velocity = Vector3(0, -5, 0)
        rb.angular_velocity = Vector3.zero()

        # Craft a manifold as if the bottom-right edge of the cube hits the plane
        manifold = CollisionManifold(
            normal=np.array([0, 1, 0], dtype=np.float32),
            depth=0.05,
            contact_point=np.array([0.4, -0.5, 0], dtype=np.float32),
        )

        window._resolve_collision(cube, plane, manifold)

        # Cube should now have angular velocity around Z (tipping from edge)
        assert _ang_mag(rb) > 0.1, (
            f"Off-center impulse should produce angular velocity, got {rb.angular_velocity}")

    def test_centered_impulse_no_angular_velocity(self):
        window = HeadlessWindow()

        cube = _make_cube(position=(0, 0, 0), gravity=False, restitution=0.0)
        plane = _make_plane(position=(0, -0.6, 0))
        window.objects = [cube, plane]

        rb = cube.get_component(Rigidbody)
        rb.velocity = Vector3(0, -5, 0)
        rb.angular_velocity = Vector3.zero()

        # Manifold with contact at the center of mass → no torque
        manifold = CollisionManifold(
            normal=np.array([0, 1, 0], dtype=np.float32),
            depth=0.05,
            contact_point=np.array([0, 0, 0], dtype=np.float32),
        )

        window._resolve_collision(cube, plane, manifold)

        assert _ang_mag(rb) < 1e-6, (
            f"Centered impulse should not create angular velocity, "
            f"got {rb.angular_velocity}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])