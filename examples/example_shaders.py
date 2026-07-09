"""
example_shaders.py — Demonstrates the shader / material system.

Creates several 3D objects, each with a different ShaderMaterial:

  1. **Rim-light cube** — a glowing Fresnel rim effect.
  2. **Dissolve sphere** — dissolves in and out over time.
  3. **Colour-cycle cube** — smoothly cycles through the rainbow.
  4. **Custom tint sphere** — a minimal unlit shader with a tint property.

Use **SPACE** to toggle dissolve direction.

Run:
    python examples/example_shaders.py
"""

import sys, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import Window3D, Scene3D, create_cube, create_sphere, create_plane, Object3D, GameObject
from engine.d3.light import DirectionalLight3D, PointLight3D
from engine.graphics.shader import Shader, ShaderProperty
from engine.graphics.shader_material import ShaderMaterial
from engine.types import Color
from engine.input import Keys
from engine.component import Time


class ShaderScene(Scene3D):
    """Showcase different built-in shader presets on 3D primitives."""

    def setup(self):
        super().setup()  # adds default directional light
        self.camera.position = (0, 4, 14)
        self.camera.look_at((0, 0, 0))

        # ── 1. Rim-light cube ──────────────────────────────────────
        rim_go = create_cube(position=(-5, 1, 0))
        rim_go.name = "Rim Cube"
        self.rim_mat = ShaderMaterial(Shader.rim_light())
        self.rim_mat.set_color("rim_color", (0.0, 0.8, 1.0, 1.0))
        self.rim_mat.set_float("rim_power", 3.0)
        self.rim_mat.set_float("rim_intensity", 1.5)
        rim_go.get_component(Object3D).material = self.rim_mat
        self.add_object(rim_go)

        # ── 2. Dissolve sphere ─────────────────────────────────────
        dissolve_go = create_sphere(position=(-1.5, 1, 0))
        dissolve_go.name = "Dissolve Sphere"
        self.dissolve_mat = ShaderMaterial(Shader.dissolve())
        self.dissolve_mat.set_color("tint_color", (1.0, 0.3, 0.1, 1.0))
        self.dissolve_mat.set_color("edge_color", (1.0, 0.9, 0.0, 1.0))
        dissolve_go.get_component(Object3D).material = self.dissolve_mat
        self.add_object(dissolve_go)

        # ── 3. Colour-cycle cube ───────────────────────────────────
        cycle_go = create_cube(position=(2, 1, 0))
        cycle_go.name = "Cycle Cube"
        self.cycle_mat = ShaderMaterial(Shader.color_cycle())
        self.cycle_mat.set_float("speed", 0.5)
        cycle_go.get_component(Object3D).material = self.cycle_mat
        self.add_object(cycle_go)

        # ── 4. Custom unlit tint sphere ────────────────────────────
        tint_go = create_sphere(position=(5.5, 1, 0))
        tint_go.name = "Tint Sphere"
        self.tint_mat = ShaderMaterial(Shader.unlit())
        self.tint_mat.set_color("tint_color", (0.2, 1.0, 0.4, 1.0))
        tint_go.get_component(Object3D).material = self.tint_mat
        self.add_object(tint_go)

        # ── Floor ──────────────────────────────────────────────────
        floor = create_plane(width=20, height=20, position=(0, -0.5, 0),
                             color=(0.25, 0.25, 0.28))
        self.add_object(floor)

        # ── Point light for visual interest ────────────────────────
        pl_go = GameObject("PointLight")
        pl_go.add_component(PointLight3D(intensity=6, range=20.0))
        pl_go.transform.position = (0, 5, 3)
        self.add_object(pl_go)

        self.dissolve_dir = 1  # +1 = dissolving, -1 = appearing

    # ── Per-frame update ────────────────────────────────────────────
    def on_update(self):
        t = Time.time
        dt = Time.delta_time

        # Rotate objects slowly
        for obj in self.objects:
            if "Cube" in obj.name or "Sphere" in obj.name:
                obj.transform.rotation_y += dt * 30

        # Animate rim intensity (pulse)
        self.rim_mat.set_float(
            "rim_intensity", 1.0 + 1.5 * abs(math.sin(t * 2.0))
        )

        # Animate dissolve threshold
        current = self.dissolve_mat.get_float("threshold")
        current += self.dissolve_dir * dt * 0.3
        if current >= 1.0:
            current = 1.0
            self.dissolve_dir = -1
        elif current <= 0.0:
            current = 0.0
            self.dissolve_dir = 1
        self.dissolve_mat.set_float("threshold", current)

        # Drive colour-cycle time
        self.cycle_mat.set_float("time", t)

        # Pulse the tint colour
        g = 0.5 + 0.5 * math.sin(t * 3.0)
        self.tint_mat.set_color("tint_color", (0.2, g, 1.0 - g, 1.0))

    # ── Input ───────────────────────────────────────────────────────
    def on_key_press(self, key, modifiers):
        if key == Keys.SPACE:
            self.dissolve_dir *= -1
        if key == Keys.ESCAPE:
            self.window.close()


if __name__ == "__main__":
    window = Window3D(1024, 768, "Shader Material Demo", project_root=".")
    window.show_scene(ShaderScene())
    window.run()
