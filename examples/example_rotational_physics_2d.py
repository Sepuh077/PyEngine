"""
2D rotational physics playground
================================

Interactive demo for linear + **angular** collision response in 2D.

Controls
--------
  WASD / Arrows : pan camera
  Space         : spawn a tumbling box above the scene
  1             : spawn a circle
  2             : fire a fast box from the left
  3             : drop a stack of 4 boxes
  R             : reset scene
  F             : toggle gravity on new spawns
  Esc           : quit

Watch for: boxes tipping on corners, spinning after off-center hits,
and circles rolling after glancing collisions (friction).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d2 import Window2D, Scene2D, create_rect, create_circle
from engine.d2.physics import BoxCollider2D, CircleCollider2D, Rigidbody2D, CollisionMode
from engine.input import Keys
from engine.types import Color
from engine.types.vector2 import Vector2


class RotationalPhysics2DScene(Scene2D):
    def setup(self):
        super().setup()
        self._dynamic = []
        self._spawn_gravity = True
        self.main_camera.position = (0, 2)
        self.main_camera.orthographic_size = 8.0

        self._build_static_world()
        self._seed_demo_objects()
        print(__doc__)

    def _build_static_world(self):
        floor = self.add_object(create_rect(width=40, height=1, color=Color.DARK_GRAY))
        floor.transform.position = (0, -0.5, 0)
        floor.add_component(BoxCollider2D())
        floor.add_component(Rigidbody2D(is_static=True, use_gravity=False))
        fcol = floor.get_component(BoxCollider2D)
        fcol.bounciness = 0.05
        fcol.static_friction = 0.7
        fcol.dynamic_friction = 0.55

        ramp_angle = -28.0
        ramp = self.add_object(create_rect(width=12, height=0.6, color=(0.35, 0.5, 0.35)))
        ramp.transform.rotation = (0, 0, ramp_angle)
        rad = math.radians(abs(ramp_angle))
        half_len, half_th = 6.0, 0.3
        cy = half_len * math.sin(rad) + half_th * math.cos(rad) + 0.02
        ramp.transform.position = (-5.0, cy, 0)
        ramp.add_component(BoxCollider2D())
        ramp.add_component(Rigidbody2D(is_static=True, use_gravity=False))
        rcol = ramp.get_component(BoxCollider2D)
        rcol.static_friction = 0.35
        rcol.dynamic_friction = 0.28
        rcol.bounciness = 0.05

        step = self.add_object(create_rect(width=2, height=1, color=(0.4, 0.35, 0.3)))
        step.transform.position = (4, 0.5, 0)
        step.add_component(BoxCollider2D())
        step.add_component(Rigidbody2D(is_static=True, use_gravity=False))

        wall = self.add_object(create_rect(width=0.4, height=4, color=(0.3, 0.3, 0.45)))
        wall.transform.position = (8, 2, 0)
        wall.add_component(BoxCollider2D())
        wall.add_component(Rigidbody2D(is_static=True, use_gravity=False))
        wcol = wall.get_component(BoxCollider2D)
        wcol.bounciness = 0.6

        self._statics = [floor, ramp, step, wall]
        for obj in self._statics:
            obj.transform._compute_world_transform()
            for c in obj.get_components(BoxCollider2D):
                c._transform_dirty = True
                c.update_bounds()

    def _seed_demo_objects(self):
        self._spawn_box(position=(4.2, 1.8), color=Color.ORANGE, bounce=0.1)
        self._spawn_box(position=(-9.5, 5.5), color=Color.RED, bounce=0.1)
        self._spawn_box(position=(-9.0, 6.0), color=Color.YELLOW, bounce=0.12)
        s = self._spawn_circle(position=(2, 1.5), color=Color.CYAN, bounce=0.35)
        s.get_component(Rigidbody2D).velocity = Vector2(4, 0)

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

    def _spawn_box(self, position, color=None, bounce=0.15, mass=1.0, size=(1.0, 1.0)):
        color = color or Color.random_bright()
        obj = self.add_object(create_rect(width=size[0], height=size[1], color=color))
        obj.transform.position = (position[0], position[1], 0)
        # Mild tilt so landings tip realistically without wild tumbling
        obj.transform.rotation = (0, 0, random.uniform(-18, 18))
        col = BoxCollider2D()
        col.bounciness = min(bounce, 0.25)
        col.static_friction = 0.7
        col.dynamic_friction = 0.5
        col.collision_mode = CollisionMode.CONTINUOUS
        rb = Rigidbody2D(use_gravity=self._spawn_gravity, is_static=False)
        rb.mass = max(mass, 1.0)
        rb.angular_drag = 0.4
        rb.drag = 0.05
        obj.add_component(col)
        obj.add_component(rb)
        # Small initial spin (rad/s); collisions should create most of the rotation
        rb.angular_velocity = random.uniform(-0.35, 0.35)
        self._dynamic.append(obj)
        return obj

    def _spawn_circle(self, position, color=None, bounce=0.35, mass=1.0, radius=0.5):
        color = color or Color.CYAN
        obj = self.add_object(create_circle(radius=radius, color=color))
        obj.transform.position = (position[0], position[1], 0)
        col = CircleCollider2D(radius=1.0)
        col.bounciness = min(bounce, 0.35)
        col.static_friction = 0.5
        col.dynamic_friction = 0.35
        col.collision_mode = CollisionMode.CONTINUOUS
        rb = Rigidbody2D(use_gravity=self._spawn_gravity, is_static=False)
        rb.mass = max(mass, 1.0)
        rb.angular_drag = 0.4
        obj.add_component(col)
        obj.add_component(rb)
        self._dynamic.append(obj)
        return obj

    def _spawn_stack(self, base=(6, 0.6), n=4):
        for i in range(n):
            self._spawn_box(
                position=(base[0], base[1] + i * 1.05),
                color=(0.8, 0.5 + i * 0.1, 0.2),
                bounce=0.05,
                size=(1.0, 1.0),
            )
            self._dynamic[-1].transform.rotation = (0, 0, 0)
            self._dynamic[-1].get_component(Rigidbody2D).angular_velocity = 0.0

    def _fire_projectile(self):
        box = self._spawn_box(position=(-10, 3), color=Color.WHITE, bounce=0.4)
        rb = box.get_component(Rigidbody2D)
        rb.velocity = Vector2(14, 2)
        rb.angular_velocity = random.uniform(-2.0, 2.0)

    def on_update(self):
        from engine.component import Time
        import pygame

        dt = Time.delta_time
        speed = 8.0 * dt
        cam = self.main_camera
        pos = cam.position
        dx = dy = 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dy += speed
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dy -= speed
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= speed
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += speed
        if dx or dy:
            cam.position = (pos.x + dx, pos.y + dy)

        try:
            self.window.set_caption(
                f"2D Rotational Physics | gravity spawn "
                f"{'ON' if self._spawn_gravity else 'OFF'} (F) | "
                f"{getattr(self.window, 'fps', 0):.0f} FPS"
            )
        except Exception:
            pass

    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()
        elif key == Keys.SPACE:
            self._spawn_box(position=(random.uniform(-2, 2), 7))
        elif key == Keys.KEY_1:
            self._spawn_circle(position=(random.uniform(-2, 2), 7))
        elif key == Keys.KEY_2:
            self._fire_projectile()
        elif key == Keys.KEY_3:
            self._spawn_stack()
        elif key == Keys.R:
            self._reset()
        elif key == Keys.F:
            self._spawn_gravity = not self._spawn_gravity
            print(f"[spawn gravity] {'ON' if self._spawn_gravity else 'OFF'}")


def main():
    window = Window2D(1100, 700, "PyEngine — 2D Rotational Physics")
    window.show_scene(RotationalPhysics2DScene())
    window.run()


if __name__ == "__main__":
    main()
