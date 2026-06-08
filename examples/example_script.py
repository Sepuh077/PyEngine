"""
Example demonstrating the Script component system similar to Unity.

Scripts inherit from Script class and are added via gameObject.add_component().
They receive automatic lifecycle callbacks:
- start(): Called once when the object is created
- update(): Called every frame
- on_collision_enter(other): Called when collision starts
- on_collision_stay(other): Called every frame while colliding
- on_collision_exit(other): Called when collision ends
"""
import os
import sys
import math

# Add project root to path
current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)
sys.path.insert(0, project_root)

from engine.d3 import Window3D, Scene3D, Script, WaitForSeconds, WaitEndOfFrame, Time
from engine.d3.object3d import create_cube, create_sphere, create_plane, Object3D
from engine.physics3d import Rigidbody3D, BoxCollider3D, SphereCollider3D
from engine.types import Color
from engine.input import Keys
from time import time


class CoroutineDemo(Script):
    """
    Demonstrates Unity-like coroutines with WaitForSeconds, WaitEndOfFrame and frame yielding.
    """
    
    def start(self):
        print(f"CoroutineDemo started on {self.game_object.name}")
        # Start coroutines for each wait type
        # self.start_coroutine(self.wait_one_frame_demo())
        # self.start_coroutine(self.wait_for_seconds_demo())
        self.start_coroutine(self.wait_end_of_frame_demo())
    
    def wait_one_frame_demo(self):
        print("Coroutine[None]: Starting...")
        s = time()
        yield None  # Wait one frame
        print("Coroutine[None]: Resumed after one frame.")
        print(time() - s, Time.delta_time)

        yield None
        print("Coroutine[None]: Resumed after another frame.")
        print(time() - s, Time.delta_time)
    
    def wait_for_seconds_demo(self):
        s = time()
        print("Coroutine[WaitForSeconds]: Waiting 2 seconds...")
        yield WaitForSeconds(2.0)
        print("Coroutine[WaitForSeconds]: 2 seconds passed! Scaling up...")
        self.game_object.transform.scale = 2.0
        print(time() - s, Time.delta_time)
        
        yield WaitForSeconds(1.0)
        print("Coroutine[WaitForSeconds]: Scaling back down.")
        self.game_object.transform.scale = 1.0
        print(time() - s, Time.delta_time)

        # Loop to show repeated timed waits
        count = 0
        while True:
            yield WaitForSeconds(0.5)
            count += 1
            print(f"Coroutine[WaitForSeconds]: Tick {count}")
            obj3d = self.game_object.get_component(Object3D)
            if obj3d:
                obj3d._visible = not obj3d._visible

    def wait_end_of_frame_demo(self):
        print("Coroutine[WaitEndOfFrame]: Waiting for end of frame...")
        s = time()
        yield WaitEndOfFrame()
        print("Coroutine[WaitEndOfFrame]: End of frame reached.")
        print(time() - s, Time.delta_time)
        s = time()
        print("Coroutine[WaitEndOfFrame]: Waiting for end of frame again...")
        yield WaitEndOfFrame()
        print(time() - s, Time.delta_time)
        print("Coroutine[WaitEndOfFrame]: End of frame reached again.")


class Rotator(Script):
    """
    Simple script that rotates an object continuously.
    Demonstrates the update() lifecycle method.
    """
    
    def __init__(self, rotation_speed=(0, 30, 0)):
        super().__init__()
        self.rotation_speed = rotation_speed
    
    def start(self):
        """Called once when the script is initialized."""
        print(f"Rotator started on {self.game_object.name}")
    
    def update(self):
        """Called every frame - rotate the object."""
        delta_time = Time.delta_time
        rx, ry, rz = self.rotation_speed
        self.game_object.transform.rotation = (
            self.game_object.transform.rotation[0] + rx * delta_time,
            self.game_object.transform.rotation[1] + ry * delta_time,
            self.game_object.transform.rotation[2] + rz * delta_time
        )


class Bouncer(Script):
    """
    Script that makes an object bounce up and down.
    Demonstrates time-based animation in update().
    """
    
    def __init__(self, height=1.0, speed=2.0):
        super().__init__()
        self.height = height
        self.speed = speed
        self.base_y = 0.0
        self.time = 0.0
    
    def start(self):
        """Store the initial Y position."""
        self.base_y = self.game_object.transform.position[1]
        print(f"Bouncer started on {self.game_object.name} at y={self.base_y}")
    
    def update(self):
        """Bounce the object using sine wave."""
        delta_time = Time.delta_time
        self.time += delta_time * self.speed
        new_y = self.base_y + math.sin(self.time) * self.height
        pos = self.game_object.transform.position
        self.game_object.transform.position = (pos[0], new_y, pos[2])


class CollisionLogger(Script):
    """
    Script that logs collision events.
    Demonstrates on_collision_enter, on_collision_stay, and on_collision_exit.
    """
    
    def __init__(self, color_on_hit=Color.RED, color_normal=Color.BLUE):
        super().__init__()
        self.color_on_hit = color_on_hit
        self.color_normal = color_normal
        self.is_colliding = False
        self.collision_timer = 0.0
    
    def start(self):
        print(f"CollisionLogger started on {self.game_object.name}")
    
    def update(self):
        """Flash color while colliding."""
        delta_time = Time.delta_time
        if self.is_colliding:
            self.collision_timer += delta_time * 10
            obj3d = self.game_object.get_component(Object3D)
            if obj3d:
                # Flash between hit color and normal color
                t = (math.sin(self.collision_timer) + 1) / 2
                obj3d.color = (
                    self.color_normal[0] * (1 - t) + self.color_on_hit[0] * t,
                    self.color_normal[1] * (1 - t) + self.color_on_hit[1] * t,
                    self.color_normal[2] * (1 - t) + self.color_on_hit[2] * t,
                )
        else:
            # Reset to normal color
            obj3d = self.game_object.get_component(Object3D)
            if obj3d:
                obj3d.color = self.color_normal
    
    def on_collision_enter(self, other):
        """Called when we start colliding with something."""
        self.is_colliding = True
        other_obj = other.game_object
        print(f"[COLLISION ENTER] {self.game_object.name} hit {other_obj.name}")
    
    def on_collision_stay(self, other):
        """Called every frame while colliding."""
        # Uncomment to see continuous collision logging
        # other_obj = other.game_object
        # print(f"[COLLISION STAY] {self.game_object.name} still touching {other_obj.name}")
        pass
    
    def on_collision_exit(self, other):
        """Called when we stop colliding."""
        self.is_colliding = False
        other_obj = other.game_object
        print(f"[COLLISION EXIT] {self.game_object.name} left {other_obj.name}")


class PlayerController(Script):
    """
    Player controller using WASD + Space/Shift for movement.
    Demonstrates input handling in a script.
    """
    
    def __init__(self, speed=5.0):
        super().__init__()
        self.speed = speed
        self.velocity_y = 0.0
        self.is_grounded = False
    
    def start(self):
        print(f"PlayerController started on {self.game_object.name}")
    
    def update(self):
        """Handle player movement input."""
        import pygame
        
        delta_time = Time.delta_time
        keys = pygame.key.get_pressed()
        dx = dy = dz = 0.0
        
        # WASD movement
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dz -= self.speed * delta_time
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dz += self.speed * delta_time
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dx -= self.speed * delta_time
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dx += self.speed * delta_time
        if keys[pygame.K_SPACE]:
            dy += self.speed * delta_time
        if keys[pygame.K_LSHIFT]:
            dy -= self.speed * delta_time
        
        if dx or dy or dz:
            pos = self.game_object.transform.position
            self.game_object.transform.position = (pos[0] + dx, pos[1] + dy, pos[2] + dz)


class ColorChanger(Script):
    """
    Script that gradually changes the object's color over time.
    """
    
    def start(self):
        self.time = 0.0
    
    def update(self):
        delta_time = Time.delta_time
        self.time += delta_time
        obj3d = self.game_object.get_component(Object3D)
        if obj3d:
            # Cycle through rainbow colors
            r = (math.sin(self.time * 0.5) + 1) / 2
            g = (math.sin(self.time * 0.7 + 2) + 1) / 2
            b = (math.sin(self.time * 0.9 + 4) + 1) / 2
            obj3d.color = (r, g, b)


# ============================================================================
# Serializable Enum Example
# ============================================================================

from enum import Enum
from engine.d3 import InspectorField


class GameState(Enum):
    """Example enum for game states - shows in inspector as dropdown."""
    MENU = 0
    PLAYING = 1
    PAUSED = 2
    GAME_OVER = 3


class WeaponType(Enum):
    """Example enum for weapon types - shows in inspector as dropdown."""
    SWORD = 1
    AXE = 2
    BOW = 3
    STAFF = 4
    DAGGER = 5


class Difficulty(Enum):
    """Example enum for difficulty levels - shows in inspector as dropdown."""
    EASY = "easy"
    NORMAL = "normal"
    HARD = "hard"
    NIGHTMARE = "nightmare"


class EnumDemoScript(Script):
    """
    Demonstrates serializable enum fields in the inspector.
    
    This script shows how to define enum fields that appear as dropdown
    menus in the editor inspector, similar to Unity's enum support.
    
    Features demonstrated:
    - Integer-based enums (GameState, WeaponType)
    - String-based enums (Difficulty)
    - Using enum values in game logic
    - Serialization/deserialization of enum values
    """
    
    # Integer-based enum fields
    current_state = InspectorField(GameState, default=GameState.MENU, tooltip="Current game state")
    equipped_weapon = InspectorField(WeaponType, default=WeaponType.SWORD, tooltip="Equipped weapon type")
    
    # String-based enum field
    difficulty = InspectorField(Difficulty, default=Difficulty.NORMAL, tooltip="Game difficulty")
    
    # Regular fields for comparison
    player_name = InspectorField(str, default="Hero", tooltip="Player name")
    health = InspectorField(int, default=100, min_value=0, max_value=100, tooltip="Player health")
    
    def start(self):
        """Called once when the script starts."""
        print(f"[EnumDemoScript] Started on {self.game_object.name}")
        print(f"  Current State: {self.current_state} (value: {self.current_state.value if isinstance(self.current_state, Enum) else self.current_state})")
        print(f"  Weapon: {self.equipped_weapon} (value: {self.equipped_weapon.value if isinstance(self.equipped_weapon, Enum) else self.equipped_weapon})")
        print(f"  Difficulty: {self.difficulty} (value: {self.difficulty.value if isinstance(self.difficulty, Enum) else self.difficulty})")
        print(f"  Player: {self.player_name}, Health: {self.health}")
    
    def update(self):
        """Called every frame - demonstrates using enum values."""
        # Example: Check game state and act accordingly
        # Note: The stored value is the enum's value (int or string), 
        # so we compare with the value or convert to enum member
        
        # Get the current state value
        state_value = self.current_state
        
        # If you need the enum member, you can get it:
        if isinstance(state_value, Enum):
            current_game_state = state_value
        else:
            current_game_state = GameState(state_value)
        
        # Now you can use it in your logic
        if current_game_state == GameState.PLAYING:
            # Game logic for playing state
            pass
        elif current_game_state == GameState.PAUSED:
            # Game logic for paused state
            pass


class ScriptScene(Scene3D):
    """Demo showcasing the Script component system."""
    
    def setup(self):
        super().setup()
        # Create floor
        floor = self.add_object(create_plane(20, 20, color=Color.DARK_GRAY))
        floor.transform.position = (0, -0.5, 0)
        floor.add_component(Rigidbody3D(is_static=True))
        floor.add_component(BoxCollider3D())
        floor.name = "Floor"
        
        # Create a rotating cube with collision detection
        self.rotating_cube = self.add_object(create_cube(1.5, color=Color.BLUE))
        self.rotating_cube.transform.position = (-3, 1, 0)
        self.rotating_cube.name = "RotatingCube"
        self.rotating_cube.add_component(Rigidbody3D(is_static=True))
        self.rotating_cube.add_component(BoxCollider3D())
        # Add scripts: rotation + collision logging
        self.rotating_cube.add_component(Rotator(rotation_speed=(0, 45, 0)))
        self.rotating_cube.add_component(CollisionLogger(
            color_on_hit=Color.RED, 
            color_normal=Color.BLUE
        ))
        
        # Create a bouncing sphere
        self.bouncing_sphere = self.add_object(create_sphere(0.8, color=Color.GREEN))
        self.bouncing_sphere.transform.position = (0, 2, -3)
        self.bouncing_sphere.name = "BouncingSphere"
        self.bouncing_sphere.add_component(Rigidbody3D(is_static=True))
        self.bouncing_sphere.add_component(SphereCollider3D())
        # Add scripts: bouncing + color changing
        self.bouncing_sphere.add_component(Bouncer(height=0.5, speed=3.0))
        self.bouncing_sphere.add_component(ColorChanger())
        
        # Create player cube that can be moved around
        self.player = self.add_object(create_cube(1.0, color=Color.YELLOW))
        self.player.transform.position = (3, 1, 0)
        self.player.name = "Player"
        self.player.add_component(Rigidbody3D())
        self.player.add_component(BoxCollider3D())
        # Add player controller script
        self.player.add_component(PlayerController(speed=5.0))
        # Add collision logging to player too
        self.player.add_component(CollisionLogger(
            color_on_hit=Color.ORANGE,
            color_normal=Color.YELLOW
        ))
        
        # Create some static obstacles
        for i in range(3):
            obstacle = self.add_object(create_cube(1.0, color=Color.GRAY))
            obstacle.transform.position = (0, 0.5, 3 + i * 2)
            obstacle.name = f"Obstacle{i}"
            obstacle.add_component(Rigidbody3D(is_static=True))
            obstacle.add_component(BoxCollider3D())

        # Create coroutine demo object
        self.coro_obj = self.add_object(create_cube(0.5, color=Color.CYAN))
        self.coro_obj.transform.position = (0, 2, 0)
        self.coro_obj.name = "CoroutineCube"
        self.coro_obj.add_component(CoroutineDemo())
        
        # Create enum demo object
        self.enum_demo_obj = self.add_object(create_cube(0.8, color=Color.PURPLE))
        self.enum_demo_obj.transform.position = (5, 1, 0)
        self.enum_demo_obj.name = "EnumDemo"
        self.enum_demo_obj.add_component(Rigidbody3D(is_static=True))
        self.enum_demo_obj.add_component(BoxCollider3D())
        self.enum_demo_obj.add_component(EnumDemoScript())
        
        # Camera setup
        self.camera.position = (0, 8, 12)
        self.camera.look_at((0, 0, 0))
        
        # Light setup
        self.light.direction = (0.3, -0.8, -0.5)
        self.light.ambient = 0.4
        
        print("\n=== Script System Demo ===")
        print("Controls:")
        print("  WASD/Arrows - Move player (yellow cube)")
        print("  Space - Move up")
        print("  Shift - Move down")
        print("  ESC - Quit")
        print("\nWatch the console for collision events!")
        print("- Blue cube rotates automatically")
        print("- Green sphere bounces and changes color")
        print("- Yellow player can be moved with WASD")
        print()
    
    def on_update(self):
        # Update camera to follow player slightly
        player_pos = self.player.transform.position
        self.camera.look_at((player_pos[0], 0, player_pos[2]))
    
    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()
    
    def on_draw(self):
        super().on_draw()
        # Draw some UI text
        self.draw_text("Script System Demo", 10, 10, Color.WHITE, 24)
        self.draw_text("WASD: Move Player | Watch console for collisions", 10, 40, Color.WHITE, 16)


if __name__ == "__main__":
    window = Window3D(900, 600, "Engine3D - Script System Demo")
    scene = ScriptScene()
    window.show_scene(scene)
    window.run()
