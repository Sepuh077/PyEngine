"""
Example: Many objects with GPU acceleration
Demonstrates rendering 100+ objects at 60 FPS.
Tests position update speed by moving ALL objects with arrow keys.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import Window3D, Scene3D, Time, create_cube
from engine.input import Keys
from engine.types import Color


class ManyObjectsScene(Scene3D):
    """Render many objects efficiently using GPU."""
    
    def setup(self):
        """Create a grid of objects."""
        super().setup()
        
        self.num_objects = 500
        grid_size = 25
        spacing = 4.0
        
        print(f"Loading {self.num_objects} objects...")
        
        for i in range(self.num_objects):
            row = i // grid_size
            col = i % grid_size
            
            # Position in grid, centered
            x = (col - grid_size / 2) * spacing
            z = (row - grid_size / 2) * spacing
            y = 0
            
            # Create object with random color
            # obj = self.load_object(
            #     "example/stairs_modular_right.obj",
            #     position=(x, y, z),
            #     scale=0.5,
            #     color=Color.random_bright()
            # )
            obj = create_cube(position=(x, y, z), color=Color.random_bright())
            
            # Store original position for reset
            obj.tag = f"{x},{y},{z}"
        
        print(f"Loaded {len(self.objects)} objects!")
        
        # Camera setup
        self.camera.position = (0, 30, 40)
        self.camera.look_at((0, 0, 0))
        
        # Light
        self.light.direction = (0.3, -0.7, -0.5)
        
        # Animation
        self.time = 0
        
        # Movement speed
        self.move_speed = 20.0
        
        # Track total offset for display
        self.offset_x = 0.0
        self.offset_z = 0.0
    
    def on_update(self):
        """Animate all objects."""
        delta_time = Time.delta_time
        self.time += delta_time
        
        # Movement speed
        move = self.move_speed * delta_time
        
        # Arrow keys move ALL objects - stress test for position updates!
        dx, dz = 0, 0
        
        if self.window.is_key_pressed(Keys.LEFT):
            dx = -move
        if self.window.is_key_pressed(Keys.RIGHT):
            dx = move
        if self.window.is_key_pressed(Keys.UP):
            dz = -move
        if self.window.is_key_pressed(Keys.DOWN):
            dz = move
        
        # Apply movement to all objects
        if dx != 0 or dz != 0:
            self.offset_x += dx
            self.offset_z += dz
            for obj in self.objects:
                obj.x += dx
                obj.z += dz
        
        # Rotate each object at slightly different speed
        for i, obj in enumerate(self.objects):
            obj.transform.rotation_y = self.time * 30 + i * 10
        
        # Camera orbit with A/D
        if self.window.is_key_pressed(Keys.A):
            self.camera.orbit(-delta_time * 0.5, 0)
        if self.window.is_key_pressed(Keys.D):
            self.camera.orbit(delta_time * 0.5, 0)
        
        # Camera zoom with W/S
        if self.window.is_key_pressed(Keys.W):
            self.camera.zoom(-move)
        if self.window.is_key_pressed(Keys.S):
            self.camera.zoom(move)
        
        # Update window title with FPS and offset
        self.window.set_caption(
            f"PyEngine - {self.num_objects} objects - "
            f"Offset: ({self.offset_x:.1f}, {self.offset_z:.1f}) - "
            f"{self.window.fps:.1f} FPS"
        )
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()
        elif key == Keys.R:
            # Reset all objects to original positions
            print("Resetting positions...")
            self.offset_x = 0
            self.offset_z = 0
            for obj in self.objects:
                # Parse original position from tag
                x, y, z = map(float, obj.tag.split(','))
                obj.transform.position = (x, y, z)
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        self.camera.zoom(-scroll_y * 3)


if __name__ == "__main__":
    print("=== PyEngine Many Objects Example ===")
    print("Tests position update speed with 100 objects")
    print()
    print("Controls:")
    print("  Arrow Keys - Move ALL objects (Left/Right = X, Up/Down = Z)")
    print("  A/D - Orbit camera")
    print("  W/S - Zoom camera")
    print("  R - Reset all positions")
    print("  Mouse Scroll - Zoom")
    print("  ESC - Exit")
    print()
    
    window = Window3D(800, 600, "PyEngine - Many Objects")
    scene = ManyObjectsScene()
    window.show_scene(scene)
    window.run()
