"""
Example: Collision Detection and Bounding Boxes
Demonstrates collision detection between objects and visual bounding boxes.
"""
import os
import sys
import math
import numpy as np
import pygame
import random

# Add the project root to sys.path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)

from engine.d3 import Window3D, Scene3D, Time
from engine.d3.object3d import create_cube, create_plane
from engine.physics3d import BoxCollider3D, SphereCollider3D, Collider3D, Rigidbody3D
from engine.input import Keys
from engine.types import Color


class CollisionScene(Scene3D):
    """Example demonstrating collision detection and bounding boxes."""

    def setup(self):
        """Called once at startup."""
        super().setup()
        
        # Create some static obstacles
        floor = self.add_object(create_plane(50, 50, color=Color.DARK_GRAY))
        floor.transform.position = (0, 0, 0)
        floor.add_component(Rigidbody3D(is_static=True))
        floor.add_component(BoxCollider3D())  # user adds

        self.obstacles = []
        # 2 cube obstacles, 2 sphere obstacles
        positions = [
            (-5, 1, 0),
            (5, 1, 0),
            (0, 1, -5),
            (0, 1, 5)
        ]
        colliders = [BoxCollider3D, BoxCollider3D, SphereCollider3D, SphereCollider3D]
        for i in range(4):
            obs = self.add_object(create_cube(2.0, color=Color.GREEN))
            obs.transform.position = positions[i]
            obs.add_component(colliders[i]())  # user adds
            self.obstacles.append(obs)

        # Create a moving player object (sphere collider for testing sphere-sphere)
        self.player = self.add_object(create_cube(1.0, color=Color.BLUE))
        self.player.transform.position = (0, 0.5, 0)
        self.player.add_component(SphereCollider3D())  # user adds

        # Create some moving enemies (use stairs OBJ for reliability; GLTF needs .bin)
        self.enemies = []
        obj_path = "example/stairs_modular_right.obj"
        for i in range(2):
            enemy = self.add_object(obj_path, scale=1, color=Color.RED)  # color tints if no vertex colors
            enemy.transform.position = (-3 + i * 6, 0.75, -3 + i * 6)
            enemy.speed = 2.0 + i * 0.5  # Slower movement for visibility
            self.enemies.append(enemy)

        # Set up camera
        self.camera.position = (0, 15, 15)
        self.camera.look_at((0, 0, 0))

        # Set up light
        self.light.direction = (0.5, -1, -0.5)
        self.light.ambient = 0.3

        # Movement speed
        self.move_speed = 10.0  # Increased for more visible movement

        # Toggle for bounding boxes
        self.show_bounding_boxes = True

        # Movement state
        self.move_dir = [0, 0, 0]  # x, y, z

    def on_update(self):
        """Called every frame."""
        delta_time = Time.delta_time
        # Move enemies in circles
        if not hasattr(self, 'time_elapsed'):
            self.time_elapsed = 0.0
        self.time_elapsed += delta_time

        for i, enemy in enumerate(self.enemies):
            angle = self.time_elapsed * enemy.speed + i * math.pi / 2
            radius = 3.0
            enemy.x = math.cos(angle) * radius
            enemy.z = math.sin(angle) * radius
            enemy.y = 0.75

            # Check collision with player (via colliders)
            pcoll = self.player.get_component(Collider3D)
            ecoll = enemy.get_component(Collider3D)
            if pcoll and ecoll and pcoll.check_collision(ecoll):
                # Collision detected - change color or something
                enemy._color = np.array(Color.YELLOW, dtype=np.float32)
            else:
                enemy._color = np.array(Color.RED, dtype=np.float32)

        # Update movement direction
        keys = pygame.key.get_pressed()
        self.move_dir = [0, 0, 0]
        if keys[pygame.K_w]:
            self.move_dir[2] = -1
        if keys[pygame.K_s]:
            self.move_dir[2] = 1
        if keys[pygame.K_a]:
            self.move_dir[0] = -1
        if keys[pygame.K_d]:
            self.move_dir[0] = 1
        self.move_dir[1] = -0.1

        # Move player based on input
        delta = self.move_speed * delta_time
        dx = self.move_dir[0] * delta
        dy = self.move_dir[1] * delta
        dz = self.move_dir[2] * delta
        if dx != 0 or dy != 0 or dz != 0:
            self.window.move_object(self.player, (dx, dy, dz))

        # Update window title
        pos = self.player.transform.position
        self.window.set_caption(
            f"Collision Demo - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) - "
            f"BBoxes: {'ON' if self.show_bounding_boxes else 'OFF'} - {self.window.fps:.0f} FPS"
        )

    def on_key_press(self, key, modifiers):
        """Called when a key is pressed."""
        if key == Keys.ESCAPE:
            self.window.close()
        elif key == Keys.SPACE:
            # Toggle bounding boxes
            self.show_bounding_boxes = not self.show_bounding_boxes
        elif key == Keys.R:
            # Reset player position
            self.player.transform.position = (0, 0.5, 0)

    def on_draw(self):
        super().on_draw()
        if self.show_bounding_boxes:
            for obj in self.objects:
                self.window.draw_collider(obj, Color.WHITE)


if __name__ == "__main__":
    print("=== Engine3D Collision Detection Example ===")
    print("Controls:")
    print("  WASD - Move blue cube (player)")
    print("  SPACE - Toggle bounding boxes")
    print("  R - Reset player position")
    print("  ESC - Exit")
    print()
    print("IMPORTANT: Click on the window to focus it, then use keyboard controls.")
    print()
    print("Green: Static obstacles (2 cubes + 2 spheres)")
    print("Red/Yellow stairs: Moving enemies (yellow when colliding with player)")
    print("Blue: Player (sphere collider) - moves with WASD (cannot pass through obstacles)")
    print("White lines: Bounding boxes (toggle with SPACE)")
    print()

    window = Window3D(800, 600, "Engine3D - Collision Demo")
    scene = CollisionScene()
    window.show_scene(scene)
    window.run()
