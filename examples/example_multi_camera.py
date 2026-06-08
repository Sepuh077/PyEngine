"""
Example: Multiple Cameras - Rear-view Mirror and Minimap

Demonstrates how to use multiple cameras for:
- Rear-view mirror (like in racing games)
- Minimap (top-down view)
- Picture-in-picture displays

Controls:
- WASD: Move the main camera
- Mouse: Look around
- ESC: Exit
"""
import sys
from pathlib import Path
import math
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import (
    Window3D, Scene3D, Time, Camera3D,
    Viewport, ClearFlags, RenderLayer,
    create_cube, create_sphere, create_plane
)
from engine.d3.camera import ClearFlags as CF
from engine.d3.physics import BoxCollider3D, Rigidbody3D
from engine.input import Keys
from engine.types import Color


class MultiCameraScene(Scene3D):
    """Scene demonstrating multiple cameras with viewports."""
    
    def setup(self):
        super().setup()
        
        # Create a floor
        floor = self.add_object(create_plane(50, 50, color=Color.DARK_GRAY))
        floor.transform.position = (0, 0, 0)
        floor.add_component(Rigidbody3D(is_static=True))
        
        # Create some objects scattered around
        for x in range(-20, 21, 5):
            for z in range(-20, 21, 5):
                if x == 0 and z == 0:
                    continue
                cube = self.add_object(create_cube(1.0, color=Color.random_bright()))
                cube.transform.position = (x, 0.5, z)
                cube.add_component(Rigidbody3D(is_static=True))
        
        # Create a "player" object (a red cube that we control)
        self.player = self.add_object(create_cube(1.0, color=Color.RED))
        self.player.transform.position = (0, 0.5, 0)
        self.player.name = "Player"
        
        # Setup main camera (first person view)
        # The main camera is already created by Scene3D.__init__
        self.main_camera.position = (0, 3, 8)
        self.main_camera.look_at((0, 0, 0))
        self.main_camera.priority = 0  # Render first (default)
        self.main_camera.is_main = True
        self.main_camera.clear_flags = CF.SOLID_CLEAR  # Clear with solid color
        self.main_camera.background_color = (0.1, 0.1, 0.15)
        
        # Create minimap camera (top-down view)
        self.minimap_camera = self.create_minimap_camera(
            position=(0, 40, 0),
            look_at=(0, 0, 0),
            corner='top-right',
            size=0.22
        )
        self.minimap_camera.background_color = (0.1, 0.1, 0.2)
        self.minimap_camera.clear_flags = CF.SOLID_CLEAR
        self.minimap_camera.fov = 50
        
        # Create rear-view mirror camera
        self.mirror_camera = self.create_mirror_camera(
            position=(0, 3, -8),
            look_at=(0, 1, 10),
            position_str='top',
            width=0.35,
            height=0.12
        )
        self.mirror_camera.background_color = (0.05, 0.05, 0.1)
        self.mirror_camera.clear_flags = CF.SOLID_CLEAR
        self.mirror_camera.fov = 70
        
        # Movement settings
        self.move_speed = 10.0
        self.mouse_sensitivity = 0.002
        self.yaw = 0
        self.pitch = 0
        
        # Hide mouse cursor and capture it
        import pygame
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        print(f"Setup complete. Cameras: {len(self.cameras)}")
        for i, cam in enumerate(self.get_cameras_sorted()):
            print(f"  Camera {i}: priority={cam.priority}, viewport={cam.viewport}")
    
    def on_update(self):
        delta_time = Time.delta_time
        speed = self.move_speed * delta_time
        
        # Move the main camera
        if self.window.is_key_pressed(Keys.W):
            self.main_camera.move_forward(speed)
        if self.window.is_key_pressed(Keys.S):
            self.main_camera.move_forward(-speed)
        if self.window.is_key_pressed(Keys.A):
            self.main_camera.move_right(-speed)
        if self.window.is_key_pressed(Keys.D):
            self.main_camera.move_right(speed)
        
        # Update player position to follow main camera
        self.player.transform.position = self.main_camera.position
        
        # Update mirror camera to be behind the player looking backward
        player_pos = self.main_camera.position
        forward = self.main_camera.forward
        
        # Mirror camera is behind player, looking in opposite direction
        mirror_offset = -forward * 5  # 5 units behind
        mirror_pos = (
            player_pos[0] + mirror_offset[0],
            player_pos[1] + 1,  # Slightly above
            player_pos[2] + mirror_offset[2]
        )
        self.mirror_camera.position = mirror_pos
        self.mirror_camera.look_at((
            player_pos[0] - forward[0] * 10,
            player_pos[1],
            player_pos[2] - forward[2] * 10
        ))
        
        # Update minimap camera to follow player
        self.minimap_camera.position = (player_pos[0], 40, player_pos[2])
        self.minimap_camera.look_at(player_pos)
        
        # Update title
        pos = self.main_camera.position
        self.window.set_caption(
            f"Multi-Camera Demo - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) - "
            f"FPS: {self.window.fps:.0f}"
        )
    
    def on_mouse_motion(self, x, y, dx, dy):
        # Update yaw and pitch
        self.yaw -= dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        
        # Clamp pitch
        self.pitch = max(-1.5, min(1.5, self.pitch))
        
        # Calculate new look direction
        look_x = -math.cos(self.pitch) * math.sin(self.yaw)
        look_y = math.sin(self.pitch)
        look_z = -math.cos(self.pitch) * math.cos(self.yaw)
        
        # Update camera target
        pos = self.main_camera.position
        self.main_camera.target = (
            pos[0] + look_x,
            pos[1] + look_y,
            pos[2] + look_z
        )
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            import pygame
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            self.window.close()


if __name__ == "__main__":
    print("=== PyEngine Multiple Cameras Example ===")
    print()
    print("This example demonstrates multiple camera viewports:")
    print("  - Main camera: Full screen view (controlled by player)")
    print("  - Mirror camera: Top-center, shows rear view")
    print("  - Minimap: Top-right corner, top-down view")
    print()
    print("Controls:")
    print("  WASD - Move")
    print("  Mouse - Look around")
    print("  ESC - Exit")
    print()
    
    window = Window3D(1024, 768, "PyEngine - Multiple Cameras Demo")
    scene = MultiCameraScene()
    window.show_scene(scene)
    window.run()
