"""
Example: Audio system in Engine3D
Demonstrates AudioClip, AudioSource, AudioListener, and 3D spatial audio.
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Generate test sounds if they don't exist yet
SOUNDS_DIR = Path(__file__).resolve().parent / "sounds"
if not SOUNDS_DIR.exists():
    import subprocess
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "generate_test_sounds.py")],
        check=True,
    )

from engine.d3 import (
    Window3D, Scene3D, GameObject, Time,
    AudioClip, AudioSource, AudioListener,
    create_cube, create_sphere,
)
from engine.input import Keys
from engine.types import Color


class AudioScene(Scene3D):
    """Scene demonstrating 2D and 3D audio sources."""

    def setup(self):
        super().setup()

        # -- load clips ------------------------------------------------------
        snd = str(SOUNDS_DIR)
        self.clip_440 = AudioClip.load(os.path.join(snd, "tone_440hz.wav"))
        self.clip_660 = AudioClip.load(os.path.join(snd, "tone_660hz.wav"))
        self.clip_square = AudioClip.load(os.path.join(snd, "square_220hz.wav"))
        self.clip_noise = AudioClip.load(os.path.join(snd, "noise_burst.wav"))
        self.clip_chirp = AudioClip.load(os.path.join(snd, "chirp_up.wav"))

        # -- camera ----------------------------------------------------------
        self.camera.position = (0, 5, 15)
        self.camera.look_at((0, 0, 0))

        # -- 2D background music (non-spatial) --------------------------------
        music_obj = GameObject("BackgroundMusic")
        self.music_source = AudioSource(
            clip=self.clip_440,
            volume=0.3,
            loop=True,
            spatial_blend=0.0,  # fully 2D
        )
        music_obj.add_component(self.music_source)
        self.add_object(music_obj)

        # -- 3D spatial source on a moving cube ------------------------------
        self.cube_go = create_cube(size=1.0)
        self.cube_go.name = "SpatialCube"
        self.cube_go.transform.position = (-8, 1, 0)
        self.spatial_source = AudioSource(
            clip=self.clip_square,
            volume=0.8,
            loop=True,
            spatial_blend=1.0,   # fully 3D
            min_distance=2.0,
            max_distance=25.0,
        )
        self.cube_go.add_component(self.spatial_source)
        self.add_object(self.cube_go)

        # -- one-shot SFX source on a sphere ---------------------------------
        self.sphere_go = create_sphere(radius=0.6)
        self.sphere_go.name = "SFXSphere"
        self.sphere_go.transform.position = (4, 1, -3)
        self.sfx_source = AudioSource(
            clip=self.clip_noise,
            volume=0.7,
            spatial_blend=0.5,   # half-3D
            min_distance=1.0,
            max_distance=30.0,
        )
        self.sphere_go.add_component(self.sfx_source)
        self.add_object(self.sphere_go)

        # -- light -----------------------------------------------------------
        self.light.direction = (0.5, -1, -0.5)
        self.light.ambient = 0.35

        # -- state -----------------------------------------------------------
        self.cube_speed = 6.0  # units/sec
        self.cube_direction = 1.0
        self._current_clip_idx = 0
        self._clip_list = [
            self.clip_440,
            self.clip_660,
            self.clip_square,
            self.clip_noise,
            self.clip_chirp,
        ]

    # -- update --------------------------------------------------------------

    def on_update(self):
        dt = Time.delta_time

        # Move the 3D cube back and forth along X
        pos = self.cube_go.transform.position
        new_x = pos.x + self.cube_speed * self.cube_direction * dt
        if new_x > 12:
            self.cube_direction = -1.0
        elif new_x < -12:
            self.cube_direction = 1.0
        self.cube_go.transform.position = (new_x, pos.y, pos.z)
        self.cube_go.transform.rotation_y += 60 * dt

        # Camera orbit
        if self.window.is_key_pressed(Keys.A):
            self.camera.orbit(-dt, 0)
        if self.window.is_key_pressed(Keys.D):
            self.camera.orbit(dt, 0)
        if self.window.is_key_pressed(Keys.W):
            self.camera.zoom(-dt * 8)
        if self.window.is_key_pressed(Keys.S):
            self.camera.zoom(dt * 8)

        # HUD
        cube_pos = self.cube_go.transform.position
        self.window.set_caption(
            f"Audio Demo  |  Cube X={cube_pos.x:+.1f}  |  "
            f"Music={'ON' if self.music_source.is_playing else 'OFF'}  |  "
            f"{self.window.fps:.0f} FPS"
        )

    def on_draw(self):
        self.draw_text("Audio Example", 10, 10, Color.WHITE, 20)
        self.draw_text("[M] toggle music   [SPACE] play SFX   [C] cycle clip", 10, 36, Color.YELLOW, 14)
        self.draw_text("[P] play/stop 3D cube   [A/D] orbit   [W/S] zoom", 10, 54, Color.YELLOW, 14)
        playing_clip = self._clip_list[self._current_clip_idx].file_path.split("/")[-1]
        self.draw_text(f"Current SFX clip: {playing_clip}", 10, 76, Color.CYAN, 14)

    # -- input ---------------------------------------------------------------

    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()

        elif key == Keys.M:
            if self.music_source.is_playing:
                self.music_source.stop()
            else:
                self.music_source.play()

        elif key == Keys.SPACE:
            self.sfx_source.play()

        elif key == Keys.C:
            self._current_clip_idx = (self._current_clip_idx + 1) % len(self._clip_list)
            self.sfx_source.clip = self._clip_list[self._current_clip_idx]

        elif key == Keys.P:
            if self.spatial_source.is_playing:
                self.spatial_source.stop()
            else:
                self.spatial_source.play()


if __name__ == "__main__":
    print("=== Engine3D Audio Example ===")
    print("Controls:")
    print("  M     - Toggle background music (2D)")
    print("  SPACE - Play one-shot SFX on sphere (half-3D)")
    print("  C     - Cycle SFX clip")
    print("  P     - Toggle 3D spatial source on moving cube")
    print("  A/D   - Orbit camera")
    print("  W/S   - Zoom camera")
    print("  ESC   - Exit")
    print()

    window = Window3D(800, 600, "Engine3D - Audio Example")
    scene = AudioScene()
    window.show_scene(scene)
    window.run()
