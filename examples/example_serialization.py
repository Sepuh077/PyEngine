"""
Example: Testing GameObject Prefab and Scene Serialization
Demonstrates saving/loading GameObject prefabs and complete scenes.

Controls:
    WASD/Arrows - Move player
    SPACE - Jump (if Rigidbody3D added)
    
    F1 - Save current player as prefab (player.prefab)
    F2 - Load player prefab at random position
    F3 - Save current scene (test_scene.json)
    F4 - Load saved scene (replaces current scene)
    F5 - Clear all objects
    
    ESC - Quit
"""
import os
import sys
import random

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)

from engine3d.input import Keys
from engine3d.types import Color
from engine3d.engine3d import Window3D, GameObject, Time, Script
from engine3d.engine3d.scene import Scene3D
from engine3d.physics3d import Rigidbody3D, BoxCollider3D


PREFAB_PATH = "player.prefab"
SCENE_PATH = "test_scene.json"


class PlayerMovement(Script):
    """Basic player movement controller."""

    def __init__(self, speed: float = 5.0):
        super().__init__()
        self.speed = speed

    def start(self):
        super().start()
        self.rb = self.game_object.get_component(Rigidbody3D)

    def update(self):
        if not self.game_object:
            return

        from engine3d.drawing import get_window
        window = get_window()
        if not window:
            return

        delta_time = Time.delta_time
        dx = dy = dz = 0.0
        step = self.speed

        if window.is_key_pressed(Keys.W) or window.is_key_pressed(Keys.UP):
            dz -= step
        if window.is_key_pressed(Keys.S) or window.is_key_pressed(Keys.DOWN):
            dz += step
        if window.is_key_pressed(Keys.A) or window.is_key_pressed(Keys.LEFT):
            dx -= step
        if window.is_key_pressed(Keys.D) or window.is_key_pressed(Keys.RIGHT):
            dx += step
        # if window.is_key_pressed(Keys.SPACE):
        #     dy += step
        if window.is_key_pressed(Keys.L):
            self.game_object.transform.scale += 0.1
        
        self.rb.velocity[0] = dx
        self.rb.velocity[2] = dz
        print(self.rb.velocity)

        # if dx or dy or dz:
        #     self.game_object.transform.move(dx, dy, dz)


class SerializationScene(Scene3D):
    """Demo for testing GameObject and Scene serialization."""
    
    def setup(self):
        super().setup()
        self.player = None
        self._create_player()
        
        # Create some environment
        self._create_environment()
        
        # Camera
        self.camera.position = (0, 15, 20)
        self.camera.look_at((0, 0, 0))
        
        # UI state
        self.message = "Press F1-F5 for save/load tests"
        self.message_timer = 0.0
        
    def _create_player(self):
        """Create the player object."""
        if self.player:
            self.remove_object(self.player)
        
        self.player = self.load_object("example/stairs_modular_right.obj", color=Color.BLUE)
        self.player.transform.position = (0, 2, 0)
        self.player.name = "Player"
        self.player.add_component(Rigidbody3D(use_gravity=False))
        self.player.add_component(BoxCollider3D())
        self.player.add_component(PlayerMovement(speed=5.0))
        
    def _create_environment(self):
        """Create some environment objects."""
        # Floor
        from engine3d.engine3d.object3d import create_plane
        floor = self.add_object(create_plane(30, 30, color=Color.DARK_GRAY))
        floor.transform.position = (0, -0.5, 0)
        floor.add_component(Rigidbody3D(is_static=True))
        floor.add_component(BoxCollider3D())
        floor.name = "Floor"
        
        # Some random cubes
        for i in range(5):
            cube = self.load_object("example/stairs_modular_right.obj", color=Color.BROWN)
            cube.transform.position = (random.uniform(-10, 10), 0.5, random.uniform(-10, 10))
            cube.transform.rotation = (0, random.uniform(0, 360), 0)
            cube.name = f"Cube_{i}"
    
    def on_update(self):
        delta_time = Time.delta_time
        
        # Update message timer
        if self.message_timer > 0:
            self.message_timer -= delta_time
            if self.message_timer <= 0:
                self.message = "Press F1-F5 for save/load tests"
        
        # Camera follows player
        if self.player:
            px, py, pz = self.player.transform.position
            self.camera.position = (px, py + 15, pz + 20)
            self.camera.target = (px, py, pz)
    
    def _show_message(self, text: str, duration: float = 3.0):
        """Show a message on screen."""
        self.message = text
        self.message_timer = duration
        print(text)
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()
            return
        
        # F1: Save player as prefab
        if key == Keys.F1:
            if self.player:
                try:
                    self.player.save(PREFAB_PATH)
                    self._show_message(f"Player saved to {PREFAB_PATH}")
                except Exception as e:
                    self._show_message(f"Error saving player: {e}")
        
        # F2: Load player prefab at random position
        elif key == Keys.F2:
            if os.path.exists(PREFAB_PATH):
                try:
                    # Remove old player
                    if self.player:
                        self.remove_object(self.player)
                    
                    # Load at random position
                    pos = (random.uniform(-5, 5), 1.0, random.uniform(-5, 5))
                    rot = (0, random.uniform(0, 360), 0)
                    self.player = GameObject.load(PREFAB_PATH, position=pos, rotation=rot)
                    self.add_object(self.player)
                    self._show_message(f"Player loaded at {pos}")
                except Exception as e:
                    self._show_message(f"Error loading player: {e}")
            else:
                self._show_message(f"No prefab found at {PREFAB_PATH}")
        
        # F3: Save scene
        elif key == Keys.F3:
            try:
                # Save this scene
                self.save(SCENE_PATH)
                self._show_message(f"Scene saved to {SCENE_PATH}")
            except Exception as e:
                self._show_message(f"Error saving scene: {e}")
        
        # F4: Load scene
        elif key == Keys.F4:
            if os.path.exists(SCENE_PATH):
                try:
                    # Clear current objects
                    self.clear_objects()
                    self.player = None
                    
                    # Load scene
                    scene = Scene3D.load(SCENE_PATH)
                    
                    # Copy scene data to this scene
                    self.camera = scene.camera
                    for obj in scene.objects:
                        self.add_object(obj)
                    
                    # Find player
                    for obj in self.objects:
                        if obj.name == "Player":
                            self.player = obj
                            break
                    
                    self._show_message(f"Scene loaded from {SCENE_PATH}")
                except Exception as e:
                    self._show_message(f"Error loading scene: {e}")
            else:
                self._show_message(f"No scene found at {SCENE_PATH}")
        
        # F5: Clear all objects
        elif key == Keys.F5:
            self.clear_objects()
            self.player = None
            self._show_message("All objects cleared")
    
    def on_draw(self):
        super().on_draw()
        # Draw help text
        y = 10
        line_height = 25
        
        self.draw_text("=== Serialization Demo ===", 10, y, Color.WHITE, 20)
        y += line_height
        self.draw_text(self.message, 10, y, Color.YELLOW, 18)
        y += line_height * 2
        
        self.draw_text("Controls:", 10, y, Color.CYAN, 18)
        y += line_height
        self.draw_text("  WASD/Arrows - Move player", 10, y, Color.WHITE, 16)
        y += line_height
        self.draw_text("  SPACE - Move up", 10, y, Color.WHITE, 16)
        y += line_height
        self.draw_text("  F1 - Save player prefab", 10, y, Color.GREEN, 16)
        y += line_height
        self.draw_text("  F2 - Load player prefab (random pos)", 10, y, Color.GREEN, 16)
        y += line_height
        self.draw_text("  F3 - Save scene", 10, y, Color.GREEN, 16)
        y += line_height
        self.draw_text("  F4 - Load scene", 10, y, Color.GREEN, 16)
        y += line_height
        self.draw_text("  F5 - Clear all objects", 10, y, Color.GREEN, 16)
        y += line_height
        self.draw_text("  ESC - Quit", 10, y, Color.WHITE, 16)
        
        # Show object count
        y += line_height * 2
        self.draw_text(f"Objects: {len(self.objects)}", 10, y, Color.MAGENTA, 18)


if __name__ == "__main__":
    print("=== Serialization Demo ===")
    print("This example demonstrates:")
    print("  - Saving GameObjects as prefabs (.prefab files)")
    print("  - Loading GameObjects with custom position/rotation")
    print("  - Saving complete scenes (.json files)")
    print("  - Loading complete scenes")
    print()
    
    window = Window3D(900, 600, "Engine3D - Serialization Demo")
    scene = SerializationScene()
    window.show_scene(scene)
    window.run()
