import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3.window import Window3D
from engine.d3 import create_cube, create_sphere, create_plane, Object3D, PointLight3D, GameObject
from engine.graphics.material import UnlitMaterial, LitMaterial, SpecularMaterial, EmissiveMaterial, TransparentMaterial
from engine.types import Color
from engine.input import Keys

class MaterialExample(Window3D):
    def setup(self):
        super().setup()
        self.camera.position = (0, 5, 15)
        self.camera.look_at((0, 0, 0))

        # 1. Unlit Material (White)
        unlit_go = create_cube(position=(-6, 0, 0))
        unlit_go.name = "Unlit Cube"
        unlit_go.get_component(Object3D).material = UnlitMaterial(color=Color.WHITE)
        self.add_object(unlit_go)

        self.pl = GameObject()
        self.pl.add_component(PointLight3D(intensity=10))
        self.pl.transform.position = (0, 3, 0)
        self.add_object(self.pl)

        # 2. Lit Material (Default - Red)
        lit_go = create_cube(position=(-3, 0, 0))
        lit_go.name = "Lit Cube"
        lit_go.get_component(Object3D).material = LitMaterial(color=Color.RED)
        self.add_object(lit_go)

        # 3. Specular Material (Blue, shiny)
        spec_go = create_sphere(position=(0, 0, 0))
        spec_go.name = "Specular Sphere"
        spec_go.get_component(Object3D).material = SpecularMaterial(
            color=Color.BLUE, 
            specular_color=Color.WHITE, 
            shininess=64.0
        )
        self.add_object(spec_go)

        # 4. Emissive Material (Green glow)
        emissive_go = create_cube(position=(3, 0, 0))
        emissive_go.name = "Emissive Cube"
        emissive_go.get_component(Object3D).material = EmissiveMaterial(color=Color.GREEN, intensity=2.0)
        emissive_go.add_component(PointLight3D())
        self.add_object(emissive_go)

        # 5. Transparent Material (Yellow, see-through)
        trans_go = create_cube(position=(6, 0, 0))
        trans_go.name = "Transparent Cube"
        trans_go.get_component(Object3D).material = TransparentMaterial(color=Color.YELLOW, alpha=0.3)
        self.add_object(trans_go)

        # Add a floor to see shadows/reflections (not real reflections yet but context)
        floor = create_plane(width=20, height=20, position=(0, -1, 0), color=(0.2, 0.2, 0.2))
        self.add_object(floor)
        self.light.intensity = 0

    def on_update(self):
        # Rotate objects
        for obj in self.objects:
            if "Cube" in obj.name or "Sphere" in obj.name:
                obj.transform.rotation_y += self.delta_time * 45
        if self.is_key_pressed(Keys.SPACE):
            pl = self.pl.get_component(PointLight3D)
            pl.intensity = 10 - pl.intensity

if __name__ == "__main__":
    MaterialExample().run()
