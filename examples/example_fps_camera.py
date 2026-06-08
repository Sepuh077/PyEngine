"""
Example: First-person camera controls
Demonstrates FPS-style camera movement with mouse look.
"""
import sys
from pathlib import Path
import math
import random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import Window3D, Scene3D, Time, Camera3D
from engine.d3.object3d import create_cube, create_plane
from engine.d3.physics import BoxCollider3D, SphereCollider3D, CapsuleCollider3D, Collider3D, Rigidbody3D
from engine.input import Keys
from engine.types import Color


class FPSCameraScene(Scene3D):
    """First-person camera example."""
    
    def setup(self):
        super().setup()
        # Create a floor
        floor = self.add_object(create_plane(50, 50, color=Color.DARK_GRAY))
        floor.transform.position = (0, 0, 0)
        floor.add_component(Rigidbody3D(is_static=True))
        collider_classes = [BoxCollider3D, SphereCollider3D, CapsuleCollider3D]
        self.dc = True
        
        # Create some objects to look at (user adds collider)
        for x in range(-40, 41, 10):
            for z in range(-40, 41, 10):
                if x == 0 and z == 0:
                    continue
                cube = self.add_object(create_cube(1.0, color=Color.random_bright()))
                if random.random() < 0.5:
                    cube.add_component(Rigidbody3D(is_static=True))
                cube.transform.position = (x, 0.5, z)
                cube.add_component(random.choice(collider_classes)())
        
        # Create taller pillars
        for i in range(4):
            pillar = self.add_object(create_cube(2.0, color=Color.BLUE))
            angle = i * math.pi / 2
            pillar.transform.position = (
                15 * math.cos(angle),
                2,
                15 * math.sin(angle)
            )
            pillar.transform.scale_xyz = (2, 4, 2)
            pillar.add_component(random.choice(collider_classes)())
        
        # Load the stairs model
        stairs = self.load_object(
            "example/stairs_modular_right.obj",
            position=(0, 0, 0),
            scale=2.0,
            color=Color.ORANGE
        )
        stairs.add_component(BoxCollider3D())  # user adds
        
        # Camera setup - first person style
        self.camera.position = (0, 2, 10)
        self.camera.look_at((0, 2, 0))
        self.camera_obj = create_cube(1, self.camera.position)
        self.camera_obj.add_component(SphereCollider3D())
        
        # Mouse look settings
        self.mouse_sensitivity = 0.002
        self.move_speed = 10.0
        
        # Hide mouse cursor and capture it
        import pygame
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        self.yaw = 0
        self.pitch = 0
    
    def on_update(self):        
        delta_time = Time.delta_time
        # Movement
        speed = self.move_speed * delta_time
        
        if self.window.is_key_pressed(Keys.W):
            self.camera.move_forward(speed)
        if self.window.is_key_pressed(Keys.S):
            self.camera.move_forward(-speed)
        if self.window.is_key_pressed(Keys.A):
            self.camera.move_right(-speed)
        if self.window.is_key_pressed(Keys.D):
            self.camera.move_right(speed)
        if self.window.is_key_pressed(Keys.SPACE):
            self.camera.move_up(speed)
        if self.window.is_key_pressed(Keys.LSHIFT):
            self.camera.move_up(-speed)

        self.camera_obj.transform.position = self.camera.position
        
        # Rotate all cubes
        for obj in self.objects:
            if obj.get_component(Camera3D):
                continue
            if not obj.get_component(Rigidbody3D) or not obj.get_component(Rigidbody3D).is_static:
                obj.transform.rotation_y += delta_time * 20
                obj.transform.rotation_x += delta_time * 10
                obj.transform.rotation_z += delta_time * 5
            # Check via colliders
            ocoll = obj.get_component(Collider3D)
            ccoll = self.camera_obj.get_component(Collider3D)
            if ocoll and ccoll and ocoll.check_collision(ccoll):
                print(obj)
        
        # Update title
        pos = self.camera.position
        self.window.set_caption(f"FPS Camera - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) - {self.window.fps:.0f} FPS")
    
    def on_mouse_motion(self, x, y, dx, dy):
        import math
        
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
        pos = self.camera.position
        self.camera.target = (
            pos[0] + look_x,
            pos[1] + look_y,
            pos[2] + look_z
        )
    
    def on_key_press(self, key, modifiers):
        if key == Keys.SPACE:
            self.dc = not self.dc

        if key == Keys.ESCAPE:
            import pygame
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            self.window.close()

    def on_draw(self):
        super().on_draw()
        if not self.dc:
            return
        for obj in self.objects:
            self.window.draw_collider(obj, color=(0, 1, 0))


if __name__ == "__main__":
    print("=== PyEngine FPS Camera Example ===")
    print("Controls:")
    print("  WASD - Move")
    print("  SPACE - Move up")
    print("  SHIFT - Move down")
    print("  Mouse - Look around")
    print("  ESC - Exit")
    print()
    
    window = Window3D(800, 600, "PyEngine - FPS Camera")
    scene = FPSCameraScene()
    window.show_scene(scene)
    window.run(200)
