"""
Example: Animations with Animator, Clips and KeyFrames

Demonstrates:
- Creating KeyFrames with property bindings
- Building an AnimationClip (looping)
- Using Animator component with states + parameter-driven transitions
- Simple idle/walk state machine on a cube

Controls:
  SPACE - Toggle walking parameter (triggers transition)
  ESC   - Quit

Run:
    python examples/example_animation.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.d3 import Window3D, Scene3D, create_cube, Object3D, Time
from engine.animation import KeyFrame, AnimationClip, AnimatorState, Animator
from engine.component import Script
from engine.types import Color, Vector3
from engine.input import Keys


class _DummyScript(Script):
    """Minimal script so objects with only Animator get processed in the Cython game loop."""
    pass


class AnimationDemoScene(Scene3D):
    def setup(self):
        super().setup()

        self.camera.position = (0, 3, 8)
        self.camera.look_at((0, 1, 0))

        # Create a simple cube to animate
        cube = create_cube(position=(0, 1, 0), color=Color.ORANGE)
        self.cube = cube
        self.add_object(cube)

        transform = cube.transform

        # --- Create clips ---
        # Idle clip: small bob up/down using property binding
        idle_kf0 = KeyFrame(step=0)
        idle_kf0.bind_property(transform, "position", Vector3(0, 1, 0))

        idle_kf1 = KeyFrame(step=1)
        idle_kf1.bind_property(transform, "position", Vector3(0, 1.4, 0))

        idle_clip = AnimationClip(keyframes=[idle_kf0, idle_kf1], is_loop=True, frame_update_time=0.2)

        # Walk clip: move side to side
        walk_kf0 = KeyFrame(step=0)
        walk_kf0.bind_property(transform, "position", Vector3(-1.5, 1, 0))

        walk_kf1 = KeyFrame(step=1)
        walk_kf1.bind_property(transform, "position", Vector3(1.5, 1, 0))

        walk_clip = AnimationClip(keyframes=[walk_kf0, walk_kf1], is_loop=True, frame_update_time=0.25)

        # --- States ---
        idle_state = AnimatorState("idle", idle_clip)
        walk_state = AnimatorState("walk", walk_clip)

        # --- Animator ---
        self.animator = Animator()
        self.animator.register_state(idle_state, is_initial=True)
        self.animator.register_state(walk_state)

        self.animator.register_parameter("is_walking", is_trigger=False, value=False)

        # Transition: idle -> walk when is_walking becomes True
        idle_state.add_transition(
            walk_state,
            "is_walking",
            lambda p: p.value is True
        )

        # Transition: walk -> idle when is_walking becomes False
        walk_state.add_transition(
            idle_state,
            "is_walking",
            lambda p: p.value is False
        )

        cube.add_component(self.animator)
        cube.add_component(_DummyScript())
        self.animator.start()

        self.is_walking = False

    def on_update(self):
        # The Animator updates its current clip automatically
        pass

    def on_key_press(self, key, modifiers):
        if key == Keys.SPACE:
            self.is_walking = not self.is_walking
            self.animator.set_parameter("is_walking", self.is_walking)
            print(f"Walking: {self.is_walking}")

        if key == Keys.ESCAPE:
            self.window.close()


if __name__ == "__main__":
    print("Animation Example")
    print("SPACE - Toggle walk state")
    print("ESC   - Quit")
    print()

    window = Window3D(800, 600, "PyEngine - Animation Demo", project_root=".")
    window.show_scene(AnimationDemoScene())
    window.run()
