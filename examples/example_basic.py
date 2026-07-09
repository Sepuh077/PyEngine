"""
Example: Basic usage of PyEngine
Demonstrates loading objects, camera control, and input handling.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import Window3D, Scene3D, GameObject, ParticleSystem, ParticleBurst, Time
from engine.input import Keys
from engine.types import Color


class BasicScene(Scene3D):
    """Simple example with a rotating object."""
    
    def setup(self):
        """Called once at startup."""
        super().setup()
        
        # Load a 3D object
        self.stairs = self.load_object(
            "example/stairs_modular_right.obj",
            position=(0, 0, 0),
            scale=1.0,
            color=Color.ORANGE
        )
        
        # Set up camera
        self.camera.position = (0, 5, 15)
        self.camera.look_at((0, 0, 0))
        
        # Set up light
        self.light.direction = (0.5, -1, -0.5)
        self.light.ambient = 0.3
        
        # Rotation speed
        self.rotation_speed = 30  # degrees per second
        
        # Movement speed for entity
        self.entity_move_speed = 10.0  # units per second

        # Particle system setup
        burst = ParticleBurst(interval=1.0, count=12, randomize=True)
        self.particles = ParticleSystem(
            position=(0, 2, 0),
            play_on_awake=True,
            play_duration=0.0,
            particle_life=2.0,
            speed=3.5,
            size=0.2,
            particle_object=None,
            color=Color.CYAN,
            loop=True,
            max_particles=120,
            burst=burst,
            gravity_scale=0.3,
        )
        ps_go = GameObject()
        ps_go.add_component(self.particles)
        self.add_object(ps_go)
    
    def on_update(self):
        """Called every frame."""
        delta_time = Time.delta_time
        # Rotate the object
        self.stairs.transform.rotation_y += self.rotation_speed * delta_time
        
        # Entity movement with arrow keys
        move_speed = self.entity_move_speed * delta_time
        
        # Left/Right arrows: move horizontally (X-axis)
        if self.window.is_key_pressed(Keys.LEFT):
            self.stairs.transform.x -= move_speed
        if self.window.is_key_pressed(Keys.RIGHT):
            self.stairs.transform.x += move_speed
        
        # Up/Down arrows: move in Z-axis
        if self.window.is_key_pressed(Keys.UP):
            self.stairs.transform.z -= move_speed
        if self.window.is_key_pressed(Keys.DOWN):
            self.stairs.transform.z += move_speed
        
        # Camera orbit with A/D keys
        if self.window.is_key_pressed(Keys.A):
            self.camera.orbit(-delta_time, 0)
        if self.window.is_key_pressed(Keys.D):
            self.camera.orbit(delta_time, 0)
        
        # Camera zoom with W/S keys
        if self.window.is_key_pressed(Keys.W):
            self.camera.zoom(-move_speed)
        if self.window.is_key_pressed(Keys.S):
            self.camera.zoom(move_speed)
        
        # Update window title with position info
        pos = self.stairs.transform.position
        self.window.set_caption(
            f"PyEngine - Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) - {self.window.fps:.0f} FPS"
        )
    
    def on_key_press(self, key, modifiers):
        """Called when a key is pressed."""
        if key == Keys.ESCAPE:
            self.window.close()
        elif key == Keys.SPACE:
            # Toggle rotation direction
            self.rotation_speed = -self.rotation_speed
        elif key == Keys.R:
            # Reset object position and camera
            self.stairs.transform.position = (0, 0, 0)
            self.camera.position = (0, 5, 15)
            self.camera.look_at((0, 0, 0))
        elif key == Keys.P:
            if self.particles.is_playing:
                self.particles.stop(clear_particles=True)
            else:
                self.particles.play()
        elif key == Keys.E:
            self.particles.emit(20)
    
    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        """Called when mouse wheel is scrolled."""
        # Zoom in/out
        self.camera.zoom(-scroll_y * 2)


if __name__ == "__main__":
    print("=== PyEngine Basic Example ===")
    print("Controls:")
    print("  Arrow Keys - Move object (Left/Right = X, Up/Down = Z)")
    print("  A/D - Orbit camera")
    print("  W/S - Zoom camera")
    print("  SPACE - Toggle rotation direction")
    print("  R - Reset position")
    print("  ESC - Exit")
    print("  P - Toggle particles")
    print("  E - Emit burst")
    print()
    
    # Create and run the application
    window = Window3D(800, 600, "PyEngine - Basic Example", project_root=".")
    scene = BasicScene()
    window.show_scene(scene)
    window.run()
