"""
Example: Shadow mapping demonstration
Shows how shadows work with a directional light.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import (
    Window3D, Scene3D, GameObject, DirectionalLight3D,
    create_cube, create_sphere, create_plane, Time
)
from engine.input import Keys
from engine.types import Color


class ShadowScene(Scene3D):
    """Demonstrates shadow mapping with various objects."""
    
    def setup(self):
        """Called once at startup."""
        super().setup()
        
        # Create a ground plane (receives shadows)
        self.ground = create_plane(width=30, height=30, position=(0, 0, 0))
        self.ground.name = "Ground"
        self.add_object(self.ground)
        
        # Create some cubes that cast shadows
        self.cube1 = create_cube(size=2, position=(-4, 1, 0), color=Color.RED)
        self.cube1.name = "Red Cube"
        self.add_object(self.cube1)
        
        self.cube2 = create_cube(size=1.5, position=(0, 0.75, -2), color=Color.BLUE)
        self.cube2.name = "Blue Cube"
        self.add_object(self.cube2)
        
        self.cube3 = create_cube(size=2.5, position=(4, 1.25, 1), color=Color.GREEN)
        self.cube3.name = "Green Cube"
        self.add_object(self.cube3)
        
        # Create a sphere
        self.sphere = create_sphere(radius=1.5, position=(0, 1.5, 3), color=Color.YELLOW)
        self.sphere.name = "Yellow Sphere"
        self.add_object(self.sphere)
        
        # Configure the default directional light for shadows
        # (The light is created by super().setup())
        self.light.cast_shadows = True
        self.light.shadow_resolution = 2048
        self.light.shadow_distance = 50.0
        self.light.shadow_bias = 0.002
        # Angle the light
        self.light.game_object.transform.rotation = (-50, -30, 0)
        
        # Set up camera
        self.camera.position = (8, 8, 12)
        self.camera.look_at((0, 0, 0))
        
        # Movement settings
        self.rotation_speed = 20  # degrees per second
        self.light_rotation_speed = 15
        
        # Track which object is rotating
        self.rotating_object = None
    
    def on_update(self):
        """Called every frame."""
        delta_time = Time.delta_time
        
        # Rotate cubes slowly
        self.cube1.transform.rotation_y += self.rotation_speed * 0.5 * delta_time
        self.cube3.transform.rotation_y -= self.rotation_speed * 0.3 * delta_time
        
        # Camera orbit with A/D keys
        if self.window.is_key_pressed(Keys.A):
            self.camera.orbit(-delta_time * 0.5, 0)
        if self.window.is_key_pressed(Keys.D):
            self.camera.orbit(delta_time * 0.5, 0)
        
        # Camera vertical orbit with W/S keys
        if self.window.is_key_pressed(Keys.W):
            self.camera.orbit(0, -delta_time * 0.3)
        if self.window.is_key_pressed(Keys.S):
            self.camera.orbit(0, delta_time * 0.3)
        
        # Zoom with Q/E
        if self.window.is_key_pressed(Keys.Q):
            self.camera.zoom(-5 * delta_time)
        if self.window.is_key_pressed(Keys.E):
            self.camera.zoom(5 * delta_time)
        
        # Rotate light with arrow keys
        if self.window.is_key_pressed(Keys.LEFT):
            light_go = self.light.game_object
            light_go.transform.rotation_y -= self.light_rotation_speed * delta_time
        if self.window.is_key_pressed(Keys.RIGHT):
            light_go = self.light.game_object
            light_go.transform.rotation_y += self.light_rotation_speed * delta_time
        if self.window.is_key_pressed(Keys.UP):
            light_go = self.light.game_object
            light_go.transform.rotation_x -= self.light_rotation_speed * delta_time
        if self.window.is_key_pressed(Keys.DOWN):
            light_go = self.light.game_object
            light_go.transform.rotation_x += self.light_rotation_speed * delta_time
        
        # Update window title
        self.window.set_caption(
            f"Shadow Example - FPS: {self.window.fps:.0f} | "
            f"Shadows: {'ON' if self.light.cast_shadows else 'OFF'}"
        )
    
    def on_key_press(self, key, modifiers):
        """Called when a key is pressed."""
        if key == Keys.ESCAPE:
            self.window.close()
        elif key == Keys.SPACE:
            # Toggle shadows
            self.light.cast_shadows = not self.light.cast_shadows
            print(f"Shadows: {'ON' if self.light.cast_shadows else 'OFF'}")
        elif key == Keys.R:
            # Reset camera
            self.camera.position = (8, 8, 12)
            self.camera.look_at((0, 0, 0))
        elif key == Keys.KEY_1:
            # Change shadow resolution
            self.light.shadow_resolution = 512
            print(f"Shadow resolution: 512")
        elif key == Keys.KEY_2:
            self.light.shadow_resolution = 1024
            print(f"Shadow resolution: 1024")
        elif key == Keys.KEY_3:
            self.light.shadow_resolution = 2048
            print(f"Shadow resolution: 2048")
        elif key == Keys.KEY_4:
            self.light.shadow_resolution = 4096
            print(f"Shadow resolution: 4096")
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Called when mouse wheel is scrolled."""
        self.camera.zoom(-scroll_y * 1.5)


if __name__ == "__main__":
    print("=== Engine3D Shadow Example ===")
    print("Controls:")
    print("  A/D - Orbit camera horizontally")
    print("  W/S - Orbit camera vertically")
    print("  Q/E - Zoom in/out")
    print("  Arrow Keys - Rotate light direction")
    print("  SPACE - Toggle shadows on/off")
    print("  1/2/3/4 - Change shadow resolution (512/1024/2048/4096)")
    print("  R - Reset camera position")
    print("  ESC - Exit")
    print()
    
    # Create and run the application
    window = Window3D(1024, 768, "Engine3D - Shadow Example")
    scene = ShadowScene()
    window.show_scene(scene)
    window.run()
