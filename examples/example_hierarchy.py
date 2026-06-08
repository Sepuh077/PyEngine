"""
Example: Hierarchical Transforms
Demonstrates parent-child transform relationships where children inherit
position, rotation, and scale from their parents.

Controls:
  Arrow Keys - Move parent object (Left/Right = X, Up/Down = Z)
  Q/E - Rotate parent around Y axis
n  A/D - Rotate parent around X axis
  W/S - Rotate parent around Z axis
  SPACE - Toggle auto-rotation
  R - Reset all transforms
  ESC - Exit
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import Window3D, Scene3D, GameObject, ParticleSystem, Time
from engine.d3.object3d import create_cube, create_sphere
from engine.input import Keys
from engine.types import Color


class HierarchyScene(Scene3D):
    """Example demonstrating hierarchical transforms."""
    
    def setup(self):
        """Called once at startup."""
        super().setup()
        # Create parent object (central cube)
        self.parent_obj = create_cube(size=1.0, position=(0, 0, 0), color=Color.RED)
        self.add_object(self.parent_obj)
        
        # Create child objects that will orbit around the parent
        # Child 1: Orbiting on X axis
        self.child1 = create_cube(size=0.4, position=(2, 0, 0), color=Color.GREEN)
        self.child1.transform.parent = self.parent_obj.transform
        self.add_object(self.child1)
        
        # Child 2: Orbiting on Z axis
        self.child2 = create_cube(size=0.4, position=(0, 0, 2), color=Color.BLUE)
        self.child2.transform.parent = self.parent_obj.transform
        self.add_object(self.child2)
        
        # Child 3: Orbiting diagonally
        self.child3 = GameObject()
        self.child3.add_component(ParticleSystem(is_local=True, color=Color.BLACK, size=0.1))
        self.child3.add_component(ParticleSystem(is_local=False, color=Color.WHITE, size=0.1))
        self.child3.transform.parent = self.parent_obj.transform
        self.add_object(self.child3)
        
        # Grandchild: Child of child1 (nested hierarchy)
        self.grandchild = create_sphere(radius=0.2, position=(0.8, 0, 0), color=Color.CYAN)
        self.grandchild.transform.parent = self.child1.transform
        self.add_object(self.grandchild)
        
        # Set up camera
        self.camera.position = (8, 6, 8)
        self.camera.look_at((0, 0, 0))
        
        # Set up light
        self.light.direction = (0.5, -1, -0.5)
        self.light.ambient = 0.3
        
        # Movement and rotation speeds
        self.move_speed = 5.0
        self.rotation_speed = 45.0  # degrees per second
        self.auto_rotate = True
        
        # For display
        self.show_info = True
    
    def on_update(self):
        """Called every frame."""
        delta_time = Time.delta_time
        # Auto-rotate parent if enabled
        if self.auto_rotate:
            self.parent_obj.transform.rotation_y += self.rotation_speed * delta_time * 0.5
        
        # Manual rotation controls
        if self.window.is_key_pressed(Keys.Q):
            self.parent_obj.transform.rotation_y += self.rotation_speed * delta_time
        if self.window.is_key_pressed(Keys.E):
            self.parent_obj.transform.rotation_y -= self.rotation_speed * delta_time
        if self.window.is_key_pressed(Keys.A):
            self.parent_obj.transform.rotation_x += self.rotation_speed * delta_time
        if self.window.is_key_pressed(Keys.D):
            self.parent_obj.transform.rotation_x -= self.rotation_speed * delta_time
        if self.window.is_key_pressed(Keys.W):
            self.parent_obj.transform.rotation_z += self.rotation_speed * delta_time
        if self.window.is_key_pressed(Keys.S):
            self.parent_obj.transform.rotation_z -= self.rotation_speed * delta_time
        
        # Move parent with arrow keys
        move_dist = self.move_speed * delta_time
        if self.window.is_key_pressed(Keys.LEFT):
            self.parent_obj.transform.x -= move_dist
        if self.window.is_key_pressed(Keys.RIGHT):
            self.parent_obj.transform.x += move_dist
        if self.window.is_key_pressed(Keys.UP):
            self.parent_obj.transform.z -= move_dist
        if self.window.is_key_pressed(Keys.DOWN):
            self.parent_obj.transform.z += move_dist
        
        # Rotate the grandchild independently
        self.grandchild.transform.rotation_y += self.rotation_speed * delta_time * 2
        
        # Update window title with transform info
        if self.show_info:
            parent_pos = self.parent_obj.transform.position
            parent_rot = self.parent_obj.transform.rotation
            child1_world = self.child1.transform.world_position
            self.window.set_caption(
                f"Hierarchy - Parent: pos({parent_pos[0]:.1f}, {parent_pos[1]:.1f}, {parent_pos[2]:.1f}) "
                f"rot({parent_rot[1]:.0f}°) - Child1 world: ({child1_world[0]:.1f}, {child1_world[1]:.1f}, {child1_world[2]:.1f}) "
                f"- {self.window.fps:.0f} FPS"
            )
    
    def on_key_press(self, key, modifiers):
        """Called when a key is pressed."""
        if key == Keys.ESCAPE:
            self.window.close()
        elif key == Keys.SPACE:
            # Toggle auto-rotation
            self.auto_rotate = not self.auto_rotate
        elif key == Keys.R:
            # Reset all transforms
            self.parent_obj.transform.position = (0, 0, 0)
            self.parent_obj.transform.rotation = (0, 0, 0)
            self.parent_obj.transform.scale = 1.0
            self.camera.position = (8, 6, 8)
            self.camera.look_at((0, 0, 0))


if __name__ == "__main__":
    print("=== PyEngine Hierarchy Example ===")
    print("This example demonstrates parent-child transform relationships.")
    print()
    print("Scene:")
    print("  - Red cube: Parent object (center)")
    print("  - Green cube: Child orbiting on X axis")
    print("  - Blue cube: Child orbiting on Z axis")
    print("  - Yellow cube: Child orbiting diagonally")
    print("  - Cyan sphere: Grandchild (child of green cube)")
    print()
    print("Controls:")
    print("  Arrow Keys - Move parent object")
    print("  Q/E - Rotate parent around Y axis")
    print("  A/D - Rotate parent around X axis")
    print("  W/S - Rotate parent around Z axis")
    print("  SPACE - Toggle auto-rotation")
    print("  R - Reset all transforms")
    print("  ESC - Exit")
    print()
    
    window = Window3D(1024, 768, "PyEngine - Hierarchy Example")
    scene = HierarchyScene()
    window.show_scene(scene)
    window.run()
