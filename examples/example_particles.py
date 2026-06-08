import sys
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import (
    GameObject,
    Window3D, 
    Scene3D,
    Object3D, 
    ParticleSystem, 
    ParticleBurst, 
    create_plane,
    linear_size_over_lifetime,
    linear_color_over_lifetime,
    linear_velocity_over_lifetime,
    SphereShape,
    ConeShape,
    BoxShape,
)
from engine.physics3d import BoxCollider3D, SphereCollider3D, CollisionMode, Rigidbody3D
from engine.types import Color


class ParticleScene(Scene3D):
    def setup(self):
        super().setup()
        self.camera.position = (0, 10, 20)
        self.camera.look_at((0, 0, 0))
        self.light.direction = (1, -1, -1)
        
        # Ground
        self.ground = create_plane(width=30, height=30, position=(0, -2, 0), color=Color.SAND)
        self.ground.add_component(Rigidbody3D(is_static=True))
        ground_collider = self.ground.add_component(BoxCollider3D())
        ground_collider.size = [30, 1, 30]
        self.add_object(self.ground)

        # Initial particle system
        self.burst = ParticleBurst(interval=1.0, count=10, randomize=False)
        self.size_curve = linear_size_over_lifetime(0.4, 0.05)
        self.color_curve = linear_color_over_lifetime(Color.CYAN, Color.PURPLE)
        self.velocity_curve = linear_velocity_over_lifetime(6.0, 1.0)

        self.collider_template = SphereCollider3D(radius=0.4)
        self.collider_template.collision_mode = CollisionMode.IGNORE
        
        self.shape = SphereShape()

        self.ps = ParticleSystem(
            position=(0, 2, 0),
            play_on_awake=True,
            particle_life=2.0,
            speed=5.0,
            size=0.2,
            color=Color.CYAN,
            size_over_lifetime=self.size_curve,
            color_over_lifetime=self.color_curve,
            velocity_over_lifetime=self.velocity_curve,
            max_particles=100,
            burst=self.burst,
            gravity_scale=1.0,
            collider=self.collider_template,
            shape=self.shape,
        )
        self.ps_go = GameObject()
        self.ps_go.add_component(self.ps)
        self.add_object(self.ps_go)
        
        # UI state
        self.buttons = [
            {"label": "Burst: 1s", "action": self.set_burst_1s, "rect": (20, 50, 120, 30)},
            {"label": "Burst: Rand", "action": self.set_burst_rand, "rect": (20, 90, 120, 30)},
            {"label": "Obj: Cube", "action": self.set_obj_cube, "rect": (160, 50, 120, 30)},
            {"label": "Obj: Sphere", "action": self.set_obj_sphere, "rect": (160, 90, 120, 30)},
            {"label": "Grav: -1", "action": self.set_grav_neg, "rect": (300, 50, 120, 30)},
            {"label": "Grav: 0", "action": self.set_grav_zero, "rect": (300, 90, 120, 30)},
            {"label": "Grav: 1", "action": self.set_grav_pos, "rect": (300, 130, 120, 30)},
            {"label": "Max: 100", "action": self.set_max_100, "rect": (440, 50, 120, 30)},
            {"label": "Max: 1000", "action": self.set_max_1000, "rect": (440, 90, 120, 30)},
            {"label": "Curve: Linear", "action": self.set_curve_linear, "rect": (580, 50, 140, 30)},
            {"label": "Curve: Pulse", "action": self.set_curve_pulse, "rect": (580, 90, 140, 30)},
            {"label": "Vel: Decay", "action": self.set_vel_decay, "rect": (740, 50, 140, 30)},
            {"label": "Vel: Wave", "action": self.set_vel_wave, "rect": (740, 90, 140, 30)},
            {"label": "Life: 3s", "action": self.set_life_short, "rect": (900, 50, 120, 30)},
            {"label": "Life: 7s", "action": self.set_life_long, "rect": (900, 90, 120, 30)},
            {"label": "Shape: Sphere", "action": self.set_shape_sphere, "rect": (20, 130, 120, 30)},
            {"label": "Shape: Cone", "action": self.set_shape_cone, "rect": (20, 170, 120, 30)},
            {"label": "Shape: Box", "action": self.set_shape_box, "rect": (20, 210, 120, 30)},
            {"label": "Shape: Cone -X", "action": self.set_shape_cone_neg_x, "rect": (160, 130, 120, 30)},
            {"label": "Shape: Box +X", "action": self.set_shape_box_pos_x, "rect": (160, 170, 120, 30)},
        ]

    def set_burst_1s(self):
        self.ps.burst.interval = 1.0
        self.ps.burst.randomize = False

    def set_burst_rand(self):
        self.ps.burst.interval = 0.1
        self.ps.burst.randomize = True

    def set_obj_cube(self):
        self.ps.particle_object = None # Defaults to cube
        self.rebuild_ps()

    def set_obj_sphere(self):
        def set_obj(filename="Example/stairs_modular_right.obj"):
            obj = Object3D(filename)
            go = GameObject()
            go.add_component(obj)
            return go
        self.ps.particle_object = lambda: set_obj()
        self.rebuild_ps()

    def set_grav_neg(self): self.ps.gravity_scale = -1.0
    def set_grav_zero(self): self.ps.gravity_scale = 0.0
    def set_grav_pos(self): self.ps.gravity_scale = 1.0

    def set_max_100(self):
        self.ps.max_particles = 100
        self.rebuild_ps()

    def set_max_1000(self):
        self.ps.max_particles = 1000
        self.rebuild_ps()

    def set_curve_linear(self):
        self.size_curve = linear_size_over_lifetime(0.4, 0.05)
        self.color_curve = linear_color_over_lifetime(Color.CYAN, Color.PURPLE)
        self.rebuild_ps()

    def set_curve_pulse(self):
        def pulse_size(t: float) -> float:
            return 0.1 + 0.3 * (1.0 - abs(2.0 * t - 1.0))

        def pulse_color(t: float):
            return (
                0.2 + 0.8 * (1.0 - t),
                0.6 * (1.0 - t) + 0.4 * t,
                0.2 + 0.8 * t,
                1.0,
            )

        self.size_curve = pulse_size
        self.color_curve = pulse_color
        self.rebuild_ps()

    def set_vel_decay(self):
        self.velocity_curve = linear_velocity_over_lifetime(6.0, 1.0)
        self.rebuild_ps()

    def set_vel_wave(self):
        def wave_velocity(t: float):
            return 1.5 + 4.0 * abs(2.0 * t - 1.0)

        self.velocity_curve = wave_velocity
        self.rebuild_ps()

    def set_life_short(self):
        self.ps.particle_life = 3.0
        self.rebuild_ps()

    def set_life_long(self):
        self.ps.particle_life = 7.0
        self.rebuild_ps()

    def set_shape_sphere(self):
        self.shape = SphereShape()
        self.rebuild_ps()

    def set_shape_cone(self):
        self.shape = ConeShape(angle_degrees=30.0)
        self.rebuild_ps()

    def set_shape_box(self):
        self.shape = BoxShape(size=(4.0, 1.0, 4.0), direction=(0.0, 1.0, 0.0))
        self.rebuild_ps()

    def set_shape_cone_neg_x(self):
        self.shape = ConeShape(angle_degrees=30.0, direction=(-1.0, 0.0, 0.0))
        self.rebuild_ps()

    def set_shape_box_pos_x(self):
        self.shape = BoxShape(size=(1.0, 4.0, 4.0), direction=(1.0, 0.0, 0.0))
        self.rebuild_ps()

    def rebuild_ps(self):
        # We need to recreate the pool
        old_ps = self.ps
        old_ps.destroy()
        self.remove_object(self.ps_go)

        self.ps = ParticleSystem(
            position=old_ps._position,
            play_on_awake=True,
            particle_life=old_ps.particle_life,
            speed=old_ps.speed,
            size=old_ps.size,
            particle_object=old_ps.particle_object,
            color=old_ps.color,
            size_over_lifetime=self.size_curve,
            color_over_lifetime=self.color_curve,
            velocity_over_lifetime=self.velocity_curve,
            max_particles=old_ps.max_particles,
            burst=old_ps.burst,
            gravity_scale=old_ps.gravity_scale,
            collider=old_ps.collider,
            shape=self.shape,
        )
        self.ps_go = GameObject()
        self.ps_go.add_component(self.ps)
        self.add_object(self.ps_go)

    def on_update(self):
        self.window.set_caption(f"Particle Test - {self.window.fps:.1f} FPS - Particles: {sum(1 for p in self.ps._particles if p.active)}")

    def on_draw(self):
        super().on_draw()
        # Draw UI
        self.draw_text("Particle System Controls", 20, 10, Color.WHITE, font_size=30)
        for btn in self.buttons:
            x, y, w, h = btn["rect"]
            self.draw_rectangle(x, y, w, h, Color.GRAY)
            self.draw_text(btn["label"], x + 10, y + 5, Color.BLACK, font_size=20)

    def on_mouse_press(self, x, y, button, mods):
        for btn in self.buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                btn["action"]()
                break

if __name__ == "__main__":
    window = Window3D(1000, 800, "Particle Test")
    scene = ParticleScene()
    window.show_scene(scene)
    window.run()