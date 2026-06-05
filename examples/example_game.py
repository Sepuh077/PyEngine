"""
Example: First-person camera controls
Demonstrates FPS-style camera movement with mouse look.
"""
import sys
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine3d.engine3d import Window3D, Scene3D
from engine3d.engine3d.object3d import create_cube, create_plane
from engine3d.physics3d import BoxCollider3D, Rigidbody3D, Collider3D
from engine3d.input import Keys
from engine3d.types import Color


class FPSCameraScene(Scene3D):
    """First-person camera example."""
    
    def setup(self):
        super().setup()
        # Create a floor
        floor = self.add_object(create_plane(50, 50, color=Color.DARK_GRAY))
        floor.add_component(BoxCollider3D())
        floor.add_component(Rigidbody3D(is_static=True))
        floor.transform.position = (0, 0, 0)
        
        # Load the stairs model
        stairs = self.load_object(
            "example/Bush_Common_Flowers.gltf",
            position=(0, 2, 0),
            scale=2.0,
        )
        # stairs.get_component(Object3D).material = UnlitMaterial(color=Color.WHITE)
        stairs.add_component(BoxCollider3D(center=(0, 0.41, 0), size=(1, 0.18, 1)))  # user adds
        stairs.add_component(BoxCollider3D(center=(0.35, 0, 0), size=(0.03, 1, 1)))
        stairs.add_component(BoxCollider3D(center=(-0.35, 0, 0), size=(0.03, 1, 1)))
        stairs.add_component(Rigidbody3D(is_static=True))
        
        # Camera setup - first person style
        self.camera_obj = self.add_object(create_cube(1, (0, 50, 0), color=Color.WHITE))
        self.camera_obj.add_component(BoxCollider3D())
        self.rb = self.camera_obj.add_component(Rigidbody3D(use_gravity=True, drag=10.0))
        self.camera.look_at(self.camera_obj.transform.position)
        self.update_camera_position()
        
        # Mouse look settings
        self.mouse_sensitivity = 0.002
        self.move_speed = 10.0
        
        # Hide mouse cursor and capture it
        import pygame
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
        
        self.yaw = 0
        self.pitch = 0

    def jump(self):
        self.rb.velocity[1] = 10

    def update_camera_position(self):
        dist = 5
        height = 4

        pitch = self.camera_obj.transform.rotation_x
        yaw = self.camera_obj.transform.rotation_y

        # Forward direction from yaw & pitch
        dir_x = -math.cos(pitch) * math.sin(yaw)
        dir_y = math.sin(pitch)
        dir_z = -math.cos(pitch) * math.cos(yaw)

        # Camera goes behind that direction
        p = self.camera_obj.transform.position
        cam_x = p[0] - dir_x * dist
        cam_y = p[1] - dir_y * dist + height
        cam_z = p[2] - dir_z * dist

        self.camera.position = (cam_x, cam_y, cam_z)

        # 🔴 THIS is the missing part
        self.camera.look_at(self.camera_obj.transform.position)
    
    def on_update(self):        
        # Movement
        speed = self.move_speed
        yaw = self.camera_obj.transform.rotation_y

        forward_x = -math.sin(yaw)
        forward_z = -math.cos(yaw)
        right_x = math.cos(yaw)
        right_z = -math.sin(yaw)

        move_x = 0.0
        move_z = 0.0

        if self.window.is_key_pressed(Keys.W):
            move_x += forward_x
            move_z += forward_z
        if self.window.is_key_pressed(Keys.S):
            move_x -= forward_x
            move_z -= forward_z
        if self.window.is_key_pressed(Keys.A):
            move_x -= right_x
            move_z -= right_z
        if self.window.is_key_pressed(Keys.D):
            move_x += right_x
            move_z += right_z
        velocity = self.camera_obj.get_component(Rigidbody3D).velocity
        move_len = math.hypot(move_x, move_z)
        if move_len > 0:
            move_x = move_x / move_len * speed
            move_z = move_z / move_len * speed
            velocity[0] = move_x
            velocity[2] = move_z

            # self.camera_obj.transform.position = (
            #     self.camera_obj.transform.position[0] + move_x,
            #     self.camera_obj.transform.position[1],
            #     self.camera_obj.transform.position[2] + move_z,
            # )

        self.update_camera_position()
        
        # Rotate all cubes
        for obj in self.objects:
            # Check via colliders
            if obj != self.camera_obj:
                ocoll = obj.get_component(Collider3D)
                ccoll = self.camera_obj.get_component(Collider3D)
                if ocoll and ccoll and ocoll.check_collision(ccoll):
                    print(True)
        
        # Update title
        pos = self.camera.position
        self.window.set_caption(f"FPS Camera - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) - {self.window.fps:.0f} FPS")
    
    def on_mouse_motion(self, x, y, dx, dy):        
        # Update yaw and pitch
        self.yaw -= dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity
        
        # Clamp pitch
        self.pitch = max(-1.5, min(1.5, self.pitch))
        
        self.camera_obj.transform.rotation = (self.pitch, self.yaw, 0)
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            import pygame
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)
            self.window.close()
        if key == Keys.SPACE:
            self.jump()

    def on_draw(self):
        super().on_draw()
        for obj in self.objects:
            self.window.draw_collider(obj, Color.RED)


if __name__ == "__main__":
    print("=== Engine3D FPS Camera Example ===")
    print("Controls:")
    print("  WASD - Move cube on the ground")
    print("  Mouse - Rotate view around cube")
    print("  ESC - Exit")
    print()
    
    window = Window3D(800, 600, "Engine3D - FPS Camera")
    scene = FPSCameraScene()
    window.show_scene(scene)
    window.run(200)
