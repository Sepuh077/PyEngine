"""
Example 2D Particles

Demonstrates the ParticleSystem2D – a lightweight particle emitter that
draws circles or rectangles without creating any GameObjects per particle.

Controls:
  1-3       – Switch emission shape (Circle / Cone / Rect)
  G         – Toggle gravity
  SPACE     – Toggle play/stop
  ESC       – Quit
"""
import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d2 import (
    Window2D,
    Scene2D,
    create_rect,
    ParticleSystem2D,
    ParticleBurst2D,
    CircleShape2D,
    ConeShape2D,
    RectShape2D,
    linear_size_over_lifetime,
    linear_color_over_lifetime,
    linear_velocity_over_lifetime,
)
from engine.gameobject import GameObject
from engine.component import Script, Time
from engine.input import Keys
from engine.types import Color


class ParticleDemo(Scene2D):
    """Scene with multiple 2D particle systems to showcase different configs."""

    def setup(self):
        super().setup()
        self.main_camera.orthographic_size = 6.0

        # ---------- Emitter 1: fire-like (circle, gravity off) ----------
        self.fire_ps = ParticleSystem2D(
            position=(0.0, 0.0),
            play_on_awake=True,
            particle_life=1.5,
            speed=3.0,
            size=0.25,
            color=Color.ORANGE,
            size_over_lifetime=linear_size_over_lifetime(0.3, 0.02),
            color_over_lifetime=linear_color_over_lifetime(Color.YELLOW, Color.RED),
            velocity_over_lifetime=linear_velocity_over_lifetime(3.0, 0.5),
            max_particles=300,
            burst=ParticleBurst2D(interval=0.05, count=4),
            gravity_scale=0.0,
            shape=ConeShape2D(angle_degrees=40.0, direction=(0.0, 1.0)),
            particle_shape_type="circle",
        )
        fire_go = GameObject("Fire Emitter")
        fire_go.transform.position = (-4.0, -2.0, 0.0)
        fire_go.add_component(self.fire_ps)
        self.add_object(fire_go)

        # ---------- Emitter 2: sparkle burst (circle shape, gravity on) ----------
        self.spark_ps = ParticleSystem2D(
            position=(0.0, 0.0),
            play_on_awake=True,
            particle_life=2.0,
            speed=5.0,
            size=0.12,
            color=Color.CYAN,
            size_over_lifetime=linear_size_over_lifetime(0.15, 0.01),
            color_over_lifetime=linear_color_over_lifetime(Color.WHITE, Color.CYAN),
            max_particles=200,
            burst=ParticleBurst2D(interval=0.8, count=20, randomize=True),
            gravity_scale=1.0,
            shape=CircleShape2D(),
            particle_shape_type="circle",
        )
        spark_go = GameObject("Spark Emitter")
        spark_go.transform.position = (0.0, 2.0, 0.0)
        spark_go.add_component(self.spark_ps)
        self.add_object(spark_go)

        # ---------- Emitter 3: rain (rect shape, gravity on) ----------
        self.rain_ps = ParticleSystem2D(
            position=(0.0, 0.0),
            play_on_awake=True,
            particle_life=3.0,
            speed=0.5,
            size=0.08,
            color=(0.5, 0.6, 1.0),
            max_particles=400,
            burst=ParticleBurst2D(interval=0.03, count=3),
            gravity_scale=2.0,
            shape=RectShape2D(size=(10.0, 0.5), direction=(0.0, -1.0)),
            particle_shape_type="rect",
        )
        rain_go = GameObject("Rain Emitter")
        rain_go.transform.position = (4.0, 5.0, 0.0)
        rain_go.add_component(self.rain_ps)
        self.add_object(rain_go)

        self._active_ps = self.fire_ps

    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()
        elif key == Keys.KEY_1:
            self._active_ps.shape = CircleShape2D()
        elif key == Keys.KEY_2:
            self._active_ps.shape = ConeShape2D(angle_degrees=30.0, direction=(0.0, 1.0))
        elif key == Keys.KEY_3:
            self._active_ps.shape = RectShape2D(size=(3.0, 0.5), direction=(1.0, 0.0))
        elif key == Keys.G:
            current = self._active_ps.gravity_scale
            self._active_ps.gravity_scale = 0.0 if current != 0.0 else 1.0
        elif key == Keys.SPACE:
            if self._active_ps.is_playing:
                self._active_ps.stop()
            else:
                self._active_ps.play()

    def on_update(self):
        # Slowly rotate the spark emitter position for visual interest
        t = Time.delta_time
        go = self.spark_ps.game_object
        if go:
            px = float(go.transform.position.x)
            py = float(go.transform.position.y)
            angle = math.atan2(py, px) + t * 0.5
            r = 2.5
            go.transform.position = (r * math.cos(angle), r * math.sin(angle), 0)

    def on_draw(self):
        super().on_draw()
        active_count = sum(
            sum(1 for p in ps._particles if p.active)
            for ps in (self.fire_ps, self.spark_ps, self.rain_ps)
        )
        self.window.draw_text(
            f"2D Particles: {active_count} active | [1-3] shape  [G] gravity  [SPACE] play/stop",
            10, 10, Color.WHITE, font_size=18,
        )


if __name__ == "__main__":
    window = Window2D(900, 700, "2D Particle Demo", project_root=".")
    scene = ParticleDemo()
    window.show_scene(scene)
    window.run()
