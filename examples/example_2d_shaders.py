"""
Example: 2D Shaders

Demonstrates using ShaderMaterial with Object2D in a Scene2D.

Shows:
  - Shader.unlit_2d() with animated tint
  - Shader.flash_2d() hit/flash effect
  - A custom 2D shader (pulsing stripes)

Controls:
  SPACE - Toggle flash effect on the middle object
  ESC   - Quit

Run:
    python examples/example_2d_shaders.py
"""

import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d2 import Window2D, Scene2D, create_rect, create_circle, Object2D
from engine.graphics.shader import Shader, ShaderProperty
from engine.graphics.shader_material import ShaderMaterial
from engine.types import Color
from engine.input import Keys
from engine.component import Time


class Shader2DScene(Scene2D):
    """2D shader material showcase."""

    def setup(self):
        super().setup()

        # Camera setup for 2D
        self.camera.orthographic_size = 6.0
        self.camera.position = (0, 0)

        # 1. Pulsing tint using unlit_2d
        tint_go = create_rect(2.5, 2.5, color=Color.WHITE)
        tint_go.transform.position = (-4, 0, 0)
        tint_go.name = "Pulse Tint"
        self.tint_mat = ShaderMaterial(Shader.unlit_2d())
        self.tint_mat.set_color("tint_color", (0.2, 0.8, 1.0, 1.0))
        tint_go.get_component(Object2D).material = self.tint_mat
        self.add_object(tint_go)

        # 2. Flash effect
        flash_go = create_circle(1.5, color=Color.WHITE)
        flash_go.transform.position = (0, 0, 0)
        flash_go.name = "Flash Circle"
        self.flash_mat = ShaderMaterial(Shader.flash_2d())
        self.flash_mat.set_color("flash_color", (1.0, 0.2, 0.2, 1.0))
        self.flash_mat.set_float("flash_amount", 0.0)
        flash_go.get_component(Object2D).material = self.flash_mat
        self.add_object(flash_go)
        self.flash_active = False

        # 3. Custom striped shader
        stripe_go = create_rect(2.5, 2.5, color=Color.WHITE)
        stripe_go.transform.position = (4, 0, 0)
        stripe_go.name = "Stripe Rect"

        # Define a simple custom 2D shader (striped pulsing effect)
        stripe_shader = Shader(
            name="Stripe2D",
            vertex_source="""
                #version 330 core
                in vec2 in_position;
                in vec2 in_texcoord;
                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = projection * view * model * vec4(in_position, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            """,
            fragment_source="""
                #version 330 core
                in vec2 v_texcoord;
                uniform vec4 tint_color;
                uniform float time;
                uniform float stripe_width;
                out vec4 frag_color;

                void main() {
                    vec2 uv = v_texcoord;
                    float stripe = step(0.5, fract(uv.x * stripe_width + time * 2.0));
                    vec4 base = mix(vec4(0.2, 0.6, 1.0, 1.0), vec4(1.0, 0.3, 0.8, 1.0), stripe);
                    frag_color = base * tint_color;
                }
            """,
            properties=[
                ShaderProperty("tint_color", "color", (1.0, 1.0, 1.0, 1.0)),
                ShaderProperty("time", "float", 0.0),
                ShaderProperty("stripe_width", "float", 8.0),
            ],
        )

        self.stripe_mat = ShaderMaterial(stripe_shader)
        self.stripe_mat.set_color("tint_color", (1.0, 1.0, 1.0, 1.0))
        self.stripe_mat.set_float("stripe_width", 6.0)
        stripe_go.get_component(Object2D).material = self.stripe_mat
        self.add_object(stripe_go)

        # Simple floor / background rect
        bg = create_rect(18, 1, color=(0.15, 0.15, 0.18))
        bg.transform.position = (0, -3, 0)
        self.add_object(bg)

        self.flash_dir = 1

    def on_update(self):
        t = Time.time
        dt = Time.delta_time

        # Animate the tint pulse
        g = 0.6 + 0.4 * math.sin(t * 3.0)
        self.tint_mat.set_color("tint_color", (0.2, g, 1.0 - g * 0.5, 1.0))

        # Animate flash
        if self.flash_active:
            amount = self.flash_mat.get_float("flash_amount")
            amount += self.flash_dir * dt * 3.0
            if amount >= 1.0:
                amount = 1.0
                self.flash_dir = -1
            elif amount <= 0.0:
                amount = 0.0
                self.flash_dir = 1
            self.flash_mat.set_float("flash_amount", amount)

        # Animate custom stripe shader
        self.stripe_mat.set_float("time", t)

        # Slow rotation on all objects for visual interest
        for obj in self.objects:
            if "Rect" in obj.name or "Circle" in obj.name:
                obj.transform.rotation_z += dt * 20

    def on_key_press(self, key, modifiers):
        if key == Keys.SPACE:
            self.flash_active = not self.flash_active
            if not self.flash_active:
                self.flash_mat.set_float("flash_amount", 0.0)
        if key == Keys.ESCAPE:
            self.window.close()


if __name__ == "__main__":
    print("2D Shader Example")
    print("SPACE - Toggle flash effect")
    print("ESC   - Quit")
    print()

    window = Window2D(1024, 768, "2D Shader Demo")
    window.show_scene(Shader2DScene())
    window.run()
