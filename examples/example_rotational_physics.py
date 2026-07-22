"""
Rotational physics playground
=============================

Interactive demo for verifying linear + **angular** collision response.

Controls
--------
  WASD / Arrows : move camera focus on XZ
  Q / E         : raise / lower camera
  Mouse drag    : orbit camera (hold right mouse)
  Space         : spawn a tumbling cube above the scene
  1             : spawn a sphere
  2             : fire a fast cube from the camera toward the aim point
  3             : drop a stack of 4 cubes
  C             : toggle collider wireframes
  R             : reset scene
  F             : toggle freefall gravity on/off for new spawns
  Esc           : quit

Watch for: boxes tipping on edges, spinning after off-center hits,
and spheres rolling after glancing collisions (friction).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import Window3D, Scene3D, Time
from engine.d3.object3d import create_cube, create_sphere, create_plane
from engine.d3.physics import BoxCollider3D, SphereCollider3D, Rigidbody3D, CollisionMode
from engine.input import Keys
from engine.types import Color, Vector3


class RotationalPhysicsScene(Scene3D):
    def setup(self):
        super().setup()

        self._dynamic = []
        self._spawn_gravity = True
        self._orbit_yaw = 0.6
        self._orbit_pitch = 0.45
        self._orbit_dist = 18.0
        self._focus = Vector3(0, 1, 0)
        self._rmb_down = False
        self._last_mouse = None
        self.show_colliders = True

        self._build_static_world()
        self._seed_demo_objects()
        self._update_camera()

        self.light.direction = (0.4, -1.0, -0.3)
        self.light.ambient = 0.35

        print(__doc__)

    # ------------------------------------------------------------------
    # World
    # ------------------------------------------------------------------

    def _build_static_world(self):
        # Thick floor (plane mesh has zero thickness — bad for box colliders)
        floor = self.add_object(create_cube(size=1.0, color=Color.DARK_GRAY))
        floor.transform.position = (0, -0.25, 0)
        floor.transform.scale_xyz = (40, 0.5, 40)
        floor.add_component(BoxCollider3D())
        floor.add_component(Rigidbody3D(is_static=True, use_gravity=False))
        col = floor.get_component(BoxCollider3D)
        col.bounciness = 0.05
        col.static_friction = 0.7
        col.dynamic_friction = 0.55

        # Ramp: thick tilted slab, slope along +X (down toward origin).
        # Rotation about Z tilts the top face; collider OBB uses the same rotation.
        ramp_angle = -28.0  # degrees about Z
        ramp = self.add_object(create_cube(size=1.0, color=(0.35, 0.5, 0.35)))
        ramp.transform.scale_xyz = (12.0, 0.6, 5.0)  # long, reasonably thick, wide
        ramp.transform.rotation = (0.0, 0.0, ramp_angle)
        # Sit low end near the floor: place so the lowest corners rest on y=0
        # After scale, half-height = 0.3; half-length = 6.
        import math as _m
        rad = _m.radians(abs(ramp_angle))
        half_len, half_th = 6.0, 0.3
        # Center height so bottom edge is just above the floor top (y=0)
        # Lowest point of OBB ≈ cy - half_len*sin(θ) - half_th*cos(θ)
        cy = half_len * _m.sin(rad) + half_th * _m.cos(rad) + 0.02
        ramp.transform.position = (-5.0, cy, 0.0)
        ramp.add_component(BoxCollider3D())
        ramp.add_component(Rigidbody3D(is_static=True, use_gravity=False))
        rcol = ramp.get_component(BoxCollider3D)
        rcol.static_friction = 0.35
        rcol.dynamic_friction = 0.28
        rcol.bounciness = 0.05

        # Pedestal / step for uneven landings
        step = self.add_object(create_cube(size=1.0, color=(0.4, 0.35, 0.3)))
        step.transform.position = (4, 0.5, 2)
        step.transform.scale_xyz = (2.0, 1.0, 2.0)
        step.add_component(BoxCollider3D())
        step.add_component(Rigidbody3D(is_static=True, use_gravity=False))

        # Wall
        wall = self.add_object(create_cube(size=1.0, color=(0.3, 0.3, 0.45)))
        wall.transform.position = (0, 2, -8)
        wall.transform.scale_xyz = (12, 4, 0.4)
        wall.add_component(BoxCollider3D())
        wall.add_component(Rigidbody3D(is_static=True, use_gravity=False))
        wcol = wall.get_component(BoxCollider3D)
        wcol.bounciness = 0.6

        self._statics = [floor, ramp, step, wall]
        # Ensure physics bounds match final transforms (incl. ramp rotation)
        for obj in self._statics:
            obj.transform._compute_world_transform()
            for c in obj.get_components(BoxCollider3D):
                c._transform_dirty = True
                c.update_bounds()

    def _seed_demo_objects(self):
        # Resting cube on the step (should tip if nudged)
        self._spawn_box(position=(4.2, 1.8, 2), color=Color.ORANGE, bounce=0.1)
        # Drop onto the high end of the ramp so they slide down it
        self._spawn_box(position=(-9.5, 5.5, 0.5), color=Color.RED, bounce=0.1)
        self._spawn_box(position=(-9.0, 6.0, -0.8), color=Color.YELLOW, bounce=0.12)
        # Sphere on the floor with a sideways kick so friction can roll it
        s = self._spawn_sphere(position=(2, 1.5, 4), color=Color.CYAN, bounce=0.35)
        s.get_component(Rigidbody3D).velocity = Vector3(4, 0, 0)

    def _clear_dynamic(self):
        for obj in list(self._dynamic):
            try:
                self.remove_object(obj)
            except Exception:
                pass
        self._dynamic.clear()

    def _reset(self):
        self._clear_dynamic()
        self._seed_demo_objects()

    # ------------------------------------------------------------------
    # Spawning helpers
    # ------------------------------------------------------------------

    def _spawn_box(self, position, color=None, bounce=0.15, mass=1.0, scale=None):
        # Color is a namespace of RGB tuples (0-1), not a constructor
        color = color or Color.random_bright()
        obj = self.add_object(create_cube(size=1.0, color=color))

        obj.transform.position = position
        if scale is not None:
            obj.transform.scale_xyz = scale
        # Random initial tilt so landings are rarely flat
        obj.transform.rotation = (
            random.uniform(-30, 30),
            random.uniform(0, 360),
            random.uniform(-30, 30),
        )
        col = BoxCollider3D()
        col.bounciness = min(bounce, 0.25)
        col.static_friction = 0.7
        col.dynamic_friction = 0.5
        # Continuous collision helps with fast drops onto the thin-ish ramp
        col.collision_mode = CollisionMode.CONTINUOUS
        rb = Rigidbody3D(use_gravity=self._spawn_gravity, is_static=False)
        rb.mass = max(mass, 1.0)
        rb.angular_drag = 0.35
        rb.drag = 0.05
        obj.add_component(col)
        obj.add_component(rb)
        # Mild random spin (impacts should create most of the tumbling)
        rb.angular_velocity = Vector3(
            random.uniform(-0.5, 0.5),
            random.uniform(-0.3, 0.3),
            random.uniform(-0.5, 0.5),
        )
        self._dynamic.append(obj)
        return obj

    def _spawn_sphere(self, position, color=None, bounce=0.35, mass=1.0):
        color = color or Color.CYAN
        obj = self.add_object(create_sphere(radius=0.5, color=color))
        obj.transform.position = position
        col = SphereCollider3D(radius=1.0)
        col.bounciness = min(bounce, 0.35)
        col.static_friction = 0.5
        col.dynamic_friction = 0.35
        col.collision_mode = CollisionMode.CONTINUOUS
        rb = Rigidbody3D(use_gravity=self._spawn_gravity, is_static=False)
        rb.mass = max(mass, 1.0)
        rb.angular_drag = 0.4
        obj.add_component(col)
        obj.add_component(rb)
        self._dynamic.append(obj)
        return obj

    def _spawn_stack(self, base=(6, 0.6, -2), n=4):
        for i in range(n):
            self._spawn_box(
                position=(base[0], base[1] + i * 1.05, base[2]),
                color=(0.8, 0.5 + i * 0.1, 0.2),
                bounce=0.05,
                scale=(1.0, 1.0, 1.0),
            )
            # No random tilt for stack — zero rotation
            self._dynamic[-1].transform.rotation = (0, 0, 0)
            self._dynamic[-1].get_component(Rigidbody3D).angular_velocity = Vector3.zero()

    def _fire_projectile(self):
        # Fire from camera toward focus
        cam = self.camera.position
        if hasattr(cam, "to_numpy"):
            c = cam.to_numpy()
        else:
            c = (cam[0], cam[1], cam[2]) if not hasattr(cam, "x") else (cam.x, cam.y, cam.z)
        f = self._focus
        fx, fy, fz = float(f.x), float(f.y), float(f.z)
        direction = Vector3(fx - c[0], fy - c[1], fz - c[2])
        if direction.magnitude < 1e-6:
            direction = Vector3(0, 0, -1)
        direction = direction.normalized
        spawn = Vector3(c[0], c[1], c[2]) + direction * 1.5
        box = self._spawn_box(position=(spawn.x, spawn.y, spawn.z), color=Color.WHITE, bounce=0.4)
        rb = box.get_component(Rigidbody3D)
        rb.velocity = direction * 12.0
        rb.angular_velocity = Vector3(
            random.uniform(-1.5, 1.5),
            random.uniform(-1.0, 1.0),
            random.uniform(-1.5, 1.5),
        )

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _update_camera(self):
        pitch = max(-1.2, min(1.2, self._orbit_pitch))
        self._orbit_pitch = pitch
        x = self._focus.x + self._orbit_dist * math.cos(pitch) * math.sin(self._orbit_yaw)
        y = self._focus.y + self._orbit_dist * math.sin(pitch)
        z = self._focus.z + self._orbit_dist * math.cos(pitch) * math.cos(self._orbit_yaw)
        self.camera.position = (x, y, z)
        self.camera.look_at((self._focus.x, self._focus.y, self._focus.z))

    # ------------------------------------------------------------------
    # Input / frame
    # ------------------------------------------------------------------

    def on_update(self):
        dt = Time.delta_time
        speed = 8.0 * dt

        # Pan focus on XZ relative to camera yaw
        forward = Vector3(math.sin(self._orbit_yaw), 0, math.cos(self._orbit_yaw))
        right = Vector3(math.cos(self._orbit_yaw), 0, -math.sin(self._orbit_yaw))

        # Keyboard held keys via window if available
        import pygame
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self._focus = self._focus + forward * speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self._focus = self._focus - forward * speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self._focus = self._focus - right * speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self._focus = self._focus + right * speed
        if keys[pygame.K_q]:
            self._focus = Vector3(self._focus.x, self._focus.y + speed, self._focus.z)
        if keys[pygame.K_e]:
            self._focus = Vector3(self._focus.x, self._focus.y - speed, self._focus.z)

        # RMB orbit
        buttons = pygame.mouse.get_pressed()
        mx, my = pygame.mouse.get_pos()
        if buttons[2]:
            if self._last_mouse is not None:
                dx = mx - self._last_mouse[0]
                dy = my - self._last_mouse[1]
                self._orbit_yaw += dx * 0.005
                self._orbit_pitch += dy * 0.005
            self._last_mouse = (mx, my)
        else:
            self._last_mouse = None

        # Scroll zoom
        for event in pygame.event.get(pygame.MOUSEWHEEL):
            self._orbit_dist = max(4.0, min(40.0, self._orbit_dist - event.y * 1.2))

        self._update_camera()

        # Caption with collider toggle state
        try:
            self.window.set_caption(
                f"Rotational Physics — colliders {'ON' if self.show_colliders else 'OFF'} "
                f"(C) | gravity spawn {'ON' if self._spawn_gravity else 'OFF'} (F) | "
                f"{getattr(self.window, 'fps', 0):.0f} FPS"
            )
        except Exception:
            pass

    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()
        elif key == Keys.SPACE:
            self._spawn_box(
                position=(
                    self._focus.x + random.uniform(-1, 1),
                    self._focus.y + 4.0,
                    self._focus.z + random.uniform(-1, 1),
                )
            )
        elif key == Keys.KEY_1:
            self._spawn_sphere(
                position=(
                    self._focus.x + random.uniform(-0.5, 0.5),
                    self._focus.y + 4.0,
                    self._focus.z + random.uniform(-0.5, 0.5),
                )
            )
        elif key == Keys.KEY_2:
            self._fire_projectile()
        elif key == Keys.KEY_3:
            self._spawn_stack()
        elif key == Keys.C:
            self.show_colliders = not self.show_colliders
            print(f"[colliders] {'ON' if self.show_colliders else 'OFF'}")
        elif key == Keys.R:
            self._reset()
        elif key == Keys.F:
            self._spawn_gravity = not self._spawn_gravity
            print(f"[spawn gravity] {'ON' if self._spawn_gravity else 'OFF'}")

    def on_draw(self):
        super().on_draw()
        if not self.show_colliders:
            return
        # Wireframe colliders: cyan dynamics, yellow statics
        for obj in self._dynamic:
            self.window.draw_collider(obj, Color.CYAN)
        for obj in getattr(self, "_statics", ()):
            self.window.draw_collider(obj, Color.YELLOW)


def main():
    window = Window3D(1100, 700, "PyEngine — Rotational Physics Playground")
    window.show_scene(RotationalPhysicsScene())
    window.run()


if __name__ == "__main__":
    main()
