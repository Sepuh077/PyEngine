# PyEngine - Open Source 2D/3D Game Engine for Python

PyEngine is an open-source, beginner-friendly 2D/3D game engine for Python inspired by [Arcade](https://arcade.academy/). It provides a simple, batteries-included API built on a component-entity architecture with GPU acceleration via ModernGL and an included PySide6 editor.

## Features

- **Arcade-like API** - Simple, intuitive interface similar to arcade.Window
- **GPU Accelerated** - ModernGL backend for 100-1000x faster rendering than software
- **Component-Entity System** - GameObjects with attachable Components (Object3D, Camera3D, Light, Scripts, etc.)
- **Transform Hierarchy** - Parent-child relationships with world/local position, rotation, and scale
- **Shadow Mapping** - Directional and omnidirectional point light shadows with configurable resolution and bias
- **Multi-Camera** - Multiple cameras with viewports, render priorities, and clear flags (minimaps, mirrors, split-screen)
- **Materials** - Unlit, Lit, Specular, Emissive, Transparent, and Skybox materials
- **Custom Shaders** - Powerful `Shader` + `ShaderMaterial` system. Includes built-in effects (unlit, rim light, dissolve, flash, color cycle) and easy support for fully custom GLSL shaders on both 2D (`Object2D`) and 3D (`Object3D`) objects.
- **Animations** - `KeyFrame` + `AnimationClip` system for step-based animation with callbacks and direct property binding (`bind_property`). Includes a full state-machine `Animator` (Component) supporting parameters (bools, floats, triggers) and conditional transitions (Unity Animator style). See `examples/example_animation.py`.
- **Particle System** - Pooled particles with lifetime curves, burst emission, shapes (sphere, cone, box), collision, and shadow support
- **Physics** - Colliders (box, sphere, mesh), rigidbodies, collision detection/response, and raycasting
- **2D UI System** - Labels, buttons, checkboxes, sliders, progress bars, and panels
- **Scene Management** - Scene loading/saving (JSON), async loading with progress callbacks, scene switching
- **Scriptable Objects** - Data containers that can be saved as `.asset` files and referenced by name
- **Prefabs** - Reusable object templates with serialization
- **Input System** - Keyboard, mouse, and scroll input with per-frame state tracking
- **Resource Management** - Centralized loading of meshes and textures
- **Build System** - Package games as executables via PyInstaller or Nuitka
- **Editor** - PySide6-based editor with hierarchy, inspector, viewport, gizmos, undo/redo, and project browser

## Installation

### Requirements
- Python 3.8–3.12
- A working OpenGL driver

⚠️ Python 3.13+ is not supported yet.
Some dependencies (notably pygame) fail to build due to changes in Python's packaging system (distutils removal).

### Install (editable)

```bash
pip install -e .
```

This installs the engine in editable mode **and compiles the Cython modules** (if you have a C compiler).

**For full Cython acceleration** (recommended for development and benchmarking):

```bash
pip install -e .
```

After this, the compiled `.so`/`.pyd` files should be present and
`python bench_cython.py` should show the speedups (typically 5-10x+ in
physics, math, game loop, etc.) without any extra commands.

PyEngine follows the same model as NumPy/Pillow/etc.:

- `pip install pyengine` (from PyPI) → pre-built wheel with compiled extensions (no compiler needed).
- `pip install -e .` (from source clone) → setuptools + the `[build-system]` deps (Cython + numpy) will compile the extensions during install if a C compiler is available on your machine.

If you lack a C compiler the install still succeeds (pure-Python fallback).

To force pure-Python mode:

```bash
PYENGINE_PURE_PYTHON=1 python your_script.py
```

To force pure-Python mode:

```bash
PYENGINE_PURE_PYTHON=1 python your_script.py
```

#### System requirements for full Cython speed (when building from source)

You need a C compiler + Python headers on your machine (this is true for *any* package that ships C extensions: numpy, Pillow, etc.):

- **Linux**: `sudo apt-get install build-essential python3-dev` (or equivalent)
- **macOS**: `xcode-select --install`
- **Windows**: Install "Microsoft C++ Build Tools" (from Visual Studio)

Then simply:

```bash
pip install -e .
```

This is the only command you should need. The Cython modules will be built as part of the install.

For additional features:

```bash
pip install -e .[editor]   # + PySide6 editor
pip install -e .[dev]      # + pytest
pip install -e .[all]      # editor + dev tools
```

### Install from requirements

```bash
pip install -r requirements.txt
```

> **Note**: Cython + numpy are only needed at *build time* when compiling the accelerated modules from source. They are declared in `pyproject.toml` `[build-system]`. `pip install pyengine` on supported platforms gives you the compiled version automatically (like NumPy).

### Selective extras

```bash
pip install -e .           # core (includes GPU via moderngl)
pip install -e .[editor]   # + PySide6 editor
pip install -e .[dev]      # + pytest
pip install -e .[all]      # editor + dev tools
```

## CLI - Create and Manage Projects

After installing (`pip install -e .`), the `pyengine` command is available globally.

### Create a new project

```bash
pyengine startproject mygame
cd mygame
```

This scaffolds a ready-to-run project:

```
mygame/
  main.py              # Entry point (MainScene + window setup)
  settings.py          # Game config (title, resolution, FPS, initial scene)
  requirements.txt     # Dependencies
  pyproject.toml       # Build config ([tool.pyengine.build])
  build.py             # Standalone build script
  assets/              # 3D models, textures, sounds
  scenes/              # Scene files (.scene)
  scripts/             # Custom scripts and components
  .gitignore
```

### Run the project

```bash
pyengine run
# or: python main.py
```

### Build an executable

```bash
pyengine build                     # default (PyInstaller, directory)
pyengine build --onefile           # single .exe / binary
pyengine build --backend nuitka    # use Nuitka instead
pyengine build --debug             # keep console window open
pyengine build --clean             # remove build artifacts
```

> **Note about Cython modules**: If you installed the engine with Cython acceleration enabled (recommended), the built executable will contain native `.pyd` (Windows) / `.so` files. 
> - Nuitka (`--backend nuitka`) usually handles this more reliably.
> - On Windows the resulting EXE will typically require the **Visual C++ Redistributable** on the end-user's machine.
> - Always test your final `.exe` on a clean Windows installation (or VM) that does not have your development tools.

### Launch the editor

```bash
pyengine editor
```

### Other

```bash
pyengine --version        # print version
pyengine --help           # list all commands
pyengine <command> --help # help for a specific command
```

## Quick Start

```python
from engine.d3 import Window3D, Scene3D, Keys, Color, Time

class MyGame(Scene3D):
    def setup(self):
        super().setup()
        self.cube = self.add_object("model.obj")
        self.cube.transform.position = (0, 0, 0)
        self.cube.get_component(Object3D).color = Color.ORANGE

        self.camera.position = (0, 5, 10)
        self.camera.look_at((0, 0, 0))

    def on_update(self):
        self.cube.transform.rotation_y += 30 * Time.delta_time

    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()

window = Window3D(800, 600, "My 3D Game")
window.show_scene(MyGame())
window.run()
```

## Editor

PyEngine includes a PySide6-based editor for inspecting and building scenes.

```bash
python -c "from engine.d3 import run_editor; run_editor('.')"
```

Editor layout:
- **Left** - Scene hierarchy (add/remove objects)
- **Center** - Viewport (render, axis gizmos, selection overlays)
- **Right** - Inspector (selected object, components, InspectorFields)
- **Bottom** - Project/files browser

## Examples

### Basic Example
```python
from engine.d3 import Window3D, Scene3D, Keys, Time
from engine.input import Input

class BasicGame(Scene3D):
    def setup(self):
        super().setup()
        self.obj = self.add_object("example/stairs_modular_right.obj")
        self.camera.position = (0, 5, 15)

    def on_update(self):
        self.obj.transform.rotation_y += 30 * Time.delta_time
        if Input.get_key(Keys.LEFT):
            self.camera.orbit(-Time.delta_time, 0)
        if Input.get_key(Keys.RIGHT):
            self.camera.orbit(Time.delta_time, 0)

    def on_mouse_scroll(self, x, y, sx, sy):
        self.camera.zoom(-sy * 2)

window = Window3D(800, 600, "Basic Example")
window.show_scene(BasicGame())
window.run()
```

### Scene Switching
```python
from engine.d3 import Window3D, Scene3D, Keys, Time
from engine.input import Input

class MenuScene(Scene3D):
    def setup(self):
        super().setup()
        self.title = self.add_object("title.obj")
        self.camera.position = (0, 5, 10)

    def on_key_press(self, key, mods):
        if key == Keys.ENTER:
            self.window.show_scene(GameScene())

class GameScene(Scene3D):
    def setup(self):
        super().setup()
        self.player = self.add_object("player.obj")

    def on_update(self):
        if Input.get_key(Keys.W):
            self.player.transform.position += (0, 0, -5 * Time.delta_time)

    def on_key_press(self, key, mods):
        if key == Keys.ESCAPE:
            self.window.show_scene(MenuScene())

window = Window3D(800, 600, "Game with Scenes")
window.show_scene(MenuScene())
window.run()
```

### Animations
```python
from engine.d3 import Window3D, Scene3D, create_cube, Object3D, Time
from engine.animation import KeyFrame, AnimationClip, AnimatorState, Animator
from engine.types import Vector3
from engine.input import Input, Keys

class AnimScene(Scene3D):
    def setup(self):
        super().setup()
        cube = create_cube()
        self.add_object(cube)

        # Simple looping clip that moves the cube up/down
        kf0 = KeyFrame(step=0)
        kf0.bind_property(cube.transform, "position", Vector3(0, 1, 0))
        kf1 = KeyFrame(step=4)
        kf1.bind_property(cube.transform, "position", Vector3(0, 3, 0))

        clip = AnimationClip(keyframes=[kf0, kf1], is_loop=True)

        state = AnimatorState("bob", clip)
        anim = Animator()
        anim.register_state(state, is_initial=True)
        cube.add_component(anim)
        anim.start()

    def on_key_press(self, key, mods):
        if key == Keys.ESCAPE:
            self.window.close()

window = Window3D(800, 600, "Animation Demo", project_root=".")
window.show_scene(AnimScene())
window.run()
```

See also the dedicated `examples/example_animation.py` (state machine + transitions).

### Shaders (2D & 3D)
```python
from engine.d3 import Window3D, Scene3D, create_cube, Object3D
from engine.graphics import Shader, ShaderMaterial
from engine.input import Keys

class ShaderScene(Scene3D):
    def setup(self):
        super().setup()
        cube = create_cube()
        mat = ShaderMaterial(Shader.rim_light())
        mat.set_color("rim_color", (0, 0.8, 1, 1))
        cube.get_component(Object3D).material = mat
        self.add_object(cube)

    def on_key_press(self, key, mods):
        if key == Keys.ESCAPE:
            self.window.close()

window = Window3D(800, 600, "Shaders")
window.show_scene(ShaderScene())
window.run()
```
See `examples/example_shaders.py` and `examples/example_2d_shaders.py` for more.

### Shadows
```python
from engine.d3 import (
    Window3D, Scene3D, GameObject, Object3D,
    DirectionalLight3D, create_cube, create_plane,
)

class ShadowDemo(Scene3D):
    def setup(self):
        super().setup()

        # Ground plane that receives shadows
        ground = create_plane(size=20)
        ground.get_component(Object3D).receive_shadows = True
        self.add_object(ground)

        # Cube that casts shadows
        cube = create_cube(size=2, position=(0, 1, 0))
        cube.get_component(Object3D).cast_shadows = True
        self.add_object(cube)

        # Directional light with shadows
        light_obj = GameObject("Sun")
        light = DirectionalLight3D()
        light.cast_shadows = True
        light.shadow_resolution = 2048
        light_obj.add_component(light)
        light_obj.transform.rotation = (-45, 30, 0)
        self.add_object(light_obj)

        self.camera.position = (0, 10, 15)
        self.camera.look_at((0, 0, 0))

window = Window3D(800, 600, "Shadow Demo")
window.shadows_enabled = True
window.show_scene(ShadowDemo())
window.run()
```

### Multi-Camera (Minimap)
```python
from engine.d3 import Window3D, Scene3D

class GameScene(Scene3D):
    def setup(self):
        super().setup()
        self.add_object("level.obj")

        # Add a minimap camera (top-down view in the top-right corner)
        self.create_minimap_camera(
            position=(0, 50, 0),
            look_at=(0, 0, 0),
            corner='top-right',
            size=0.25,
        )

window = Window3D(1280, 720, "Multi-Camera Demo")
window.show_scene(GameScene())
window.run()
```

### Particle System
```python
from engine.d3 import (
    Window3D, Scene3D, GameObject, ParticleSystem, ParticleBurst,
    ConeShape, linear_size_over_lifetime, Color,
)

class ParticleDemo(Scene3D):
    def setup(self):
        super().setup()

        emitter = GameObject("Fire")
        ps = ParticleSystem(
            speed=4.0,
            particle_life=1.5,
            size=0.3,
            color=Color.ORANGE,
            shape=ConeShape(angle_degrees=15),
            burst=ParticleBurst(interval=0.05, count=3),
            size_over_lifetime=linear_size_over_lifetime(0.3, 0.0),
            gravity_scale=0.0,
            cast_shadows=False,
            receive_shadows=False,
        )
        emitter.add_component(ps)
        self.add_object(emitter)

        self.camera.position = (0, 3, 8)
        self.camera.look_at((0, 1, 0))

window = Window3D(800, 600, "Particle Demo")
window.show_scene(ParticleDemo())
window.run()
```

### Physics
```python
from engine.d3 import Window3D, Scene3D, create_cube, Script, Time
from engine.d3.physics import BoxCollider3D, SphereCollider3D, Rigidbody3D
from engine.input import Input, Keys

class PlayerController(Script):
    def start(self):
        self.rb = self.get_component(Rigidbody3D)

    def update(self):
        if Input.get_key(Keys.SPACE):
            self.rb.add_force((0, 10, 0))

class PhysicsDemo(Scene3D):
    def setup(self):
        super().setup()

        # Ground with box collider
        ground = create_cube(size=1, position=(0, -1, 0))
        ground.transform.scale = (20, 1, 20)
        ground.add_component(BoxCollider3D(size=(20, 1, 20)))
        self.add_object(ground)

        # Falling sphere
        ball = create_cube(size=1, position=(0, 5, 0))
        ball.add_component(SphereCollider3D(radius=0.5))
        ball.add_component(Rigidbody3D())
        ball.add_component(PlayerController())
        self.add_object(ball)

        self.camera.position = (0, 5, 15)
        self.camera.look_at((0, 2, 0))

window = Window3D(800, 600, "Physics Demo")
window.show_scene(PhysicsDemo())
window.run()
```

### 2D UI Overlay
```python
from engine.d3 import (
    Window3D, Scene3D, Label, Button, Slider, ProgressBar,
)

class UIDemo(Scene3D):
    def setup(self):
        super().setup()

        # UI elements are added to the scene's canvas
        self.canvas.add(Label("Health: 100", x=10, y=10))
        self.canvas.add(ProgressBar(x=10, y=40, width=200, value=0.75))
        self.canvas.add(Button("Restart", x=10, y=80, on_click=lambda: print("restart")))
        self.canvas.add(Slider(x=10, y=120, width=200, min_val=0, max_val=100))

window = Window3D(800, 600, "UI Demo")
window.show_scene(UIDemo())
window.run()
```

### Scriptable Objects
```python
from engine import ScriptableObject, InspectorField

class WeaponData(ScriptableObject):
    damage = InspectorField(float, default=10.0, min_value=0.0)
    attack_speed = InspectorField(float, default=1.0)
    weapon_name = InspectorField(str, default="Sword")

# Create and save
sword = WeaponData.create("Iron Sword")
sword.damage = 25.0
sword.save("assets/iron_sword.asset")

# Load from anywhere
sword = WeaponData.load("assets/iron_sword.asset")
print(sword.damage)  # 25.0
```

### Async Scene Loading
```python
from engine.d3 import SceneManager

manager = SceneManager()

def on_progress(p):
    print(f"Loading: {p*100:.0f}%")

def on_complete(scene):
    window.show_scene(scene)

manager.load_scene_async(
    "Scenes/level1.scene",
    on_progress=on_progress,
    on_complete=on_complete,
)

# Call manager.poll() in your update loop to dispatch callbacks on the main thread
```

## API Reference

### Core

| Class | Description |
|-------|-------------|
| `Window3D` | Application window. Creates the OpenGL context, runs the main loop, and hosts scenes. |
| `Scene3D` | Primary subclass target. Override `setup`, `on_update`, `on_draw`, and input handlers. Has its own objects, cameras, lights, and UI canvas. |
| `GameObject` | Entity container. Holds a `Transform` and any number of `Component` instances. |
| `Transform` | Position, rotation, scale with parent-child hierarchy and dirty-flag caching. |
| `Component` | Base class for all attachable components. |
| `Script` | Component with `start`, `update`, and coroutine support (`WaitForSeconds`, `WaitEndOfFrame`). |
| `Time` | Global `delta_time`, `time`, and `scale`. |
| `Tag` | Named tags for categorizing GameObjects (e.g. `"Player"`, `"Enemy"`). |

### Rendering

| Class | Description |
|-------|-------------|
| `Object3D` | Mesh component. Loads OBJ/GLTF, stores color, material, visibility, `cast_shadows`, `receive_shadows`. |
| `Camera3D` | Camera component with FOV, near/far, viewport, priority, clear flags, render mask, and skybox. |
| `Viewport` | Normalized screen region for a camera. Helpers: `full_screen()`, `minimap()`, `mirror()`. |
| `ClearFlags` | What a camera clears before rendering: `SKYBOX`, `COLOR`, `DEPTH`, `NOTHING`. |
| `RenderLayer` | Layer flags for selective rendering: `DEFAULT`, `UI`, `MIRROR`, `MINIMAP`, etc. |

### Lighting and Shadows

| Class | Description |
|-------|-------------|
| `DirectionalLight3D` | Directional light with color, intensity, ambient, and shadow settings. |
| `PointLight3D` | Point light with position, range, attenuation, and omnidirectional shadow support. |
| `ShadowMap` | Manages depth framebuffer for directional light shadow mapping. |
| `OmnidirectionalShadowMap` | Cubemap-based shadows for point lights (6-face rendering). |

Shadow properties on lights: `cast_shadows`, `shadow_resolution`, `shadow_bias`, `shadow_distance`.
Per-object control: `Object3D.cast_shadows` and `Object3D.receive_shadows`.

### Materials

| Class | Description |
|-------|-------------|
| `UnlitMaterial` | No lighting, flat color. |
| `LitMaterial` | Diffuse lighting (default). |
| `SpecularMaterial` | Diffuse + specular highlight with `shininess`. |
| `EmissiveMaterial` | Self-illuminating with `intensity`. |
| `TransparentMaterial` | Lit material with alpha blending. |
| `SkyboxMaterial` | Equirectangular or cubemap environment background. |

### Particle System

| Class | Description |
|-------|-------------|
| `ParticleSystem` | Component that emits pooled GameObjects as particles. |
| `ParticleBurst` | Burst configuration: `interval`, `count`, `randomize`. |
| `SphereShape` | Emit in all directions from center. |
| `ConeShape` | Emit within a cone angle around a direction. |
| `BoxShape` | Emit from one side of a box toward the other. |

Key parameters: `speed`, `particle_life`, `size`, `color`, `gravity_scale`, `max_particles`,
`size_over_lifetime`, `color_over_lifetime`, `velocity_over_lifetime`, `cast_shadows`, `receive_shadows`.

### Physics

| Class | Description |
|-------|-------------|
| `BoxCollider` | Axis-aligned or oriented box collider. |
| `SphereCollider` | Sphere collider with center and radius. |
| `MeshCollider` | Collider that uses the object's mesh geometry. |
| `Rigidbody` | Physics body with velocity, gravity, and forces. |
| `Raycast` | Cast rays into the scene and get hit information. |
| `CollisionGroup` | Group colliders for filtering (e.g. `"Player"` vs `"Enemy"`). |

### UI System

| Class | Description |
|-------|-------------|
| `UIManager` | Manages UI elements for a scene (accessed via `scene.canvas`). |
| `Label` | Text display. |
| `Button` | Clickable button with `on_click` callback. |
| `CheckBox` | Toggle checkbox. |
| `Slider` | Draggable value slider. |
| `ProgressBar` | Progress indicator. |
| `Panel` | Container for grouping UI elements. |

### Scene Management

| Class | Description |
|-------|-------------|
| `SceneManager` | Instance-based manager with `load_scene()` and `load_scene_async()`. Call `poll()` from your update loop to dispatch async callbacks on the main thread. |

### Data and Serialization

| Class | Description |
|-------|-------------|
| `Prefab` | Reusable GameObject template. Save/load with `_to_prefab_dict` / `_from_prefab_dict`. |
| `ScriptableObject` | Named data container saved as `.asset` files. Supports `InspectorField` for editor editing. |
| `InspectorField` | Descriptor that exposes a field in the editor with type, range, tooltip, etc. |
| `@serializable` | Decorator to mark a `Script` subclass as serializable. |

### Input

```python
from engine.input import Input, Keys

if Input.get_key(Keys.W):        # held down
    ...
if Input.get_key_down(Keys.SPACE):  # pressed this frame
    ...
pos = Input.mouse_position        # (x, y)
delta = Input.mouse_delta          # (dx, dy)
```

### Primitives

```python
from engine.d3 import create_cube, create_sphere, create_plane

cube = create_cube(size=2.0, position=(0, 1, 0), color=Color.RED)
sphere = create_sphere(radius=1.0, position=(3, 1, 0))
plane = create_plane(size=10.0)
```

### Color

```python
from engine.types import Color

Color.WHITE, Color.RED, Color.ORANGE, Color.SKY_BLUE  # predefined
Color.from_rgb(255, 128, 0)   # from 0-255 values
Color.from_hex("#FF8800")     # from hex string
Color.random_bright()         # random saturated color
Color.lerp(Color.RED, Color.BLUE, 0.5)  # interpolation
```

### Performance Helpers

```python
window.enable_culling = True
window.enable_instancing = True
window.instancing_min = 2
window.instancing_auto = True
window.instancing_auto_min_objects = 64
window.culling_auto = True
window.culling_auto_min_objects = 64

window.show_profiler = True
window.profiler_interval = 0.25  # seconds
```

For large scenes, mark objects as `static = True` and call
`window.build_static_batches()` after setup to merge static geometry.

## Project Structure

```
engine/
├── __init__.py                 # Core exports (GameObject, Script, Time, Tag, ...)
├── d2/                         # 2D engine
│   ├── window2d.py             # Window2D
│   ├── scene2d.py              # Scene2D
│   ├── object2d.py, sprite.py  # 2D renderables
│   ├── camera2d.py
│   └── physics/
│       ├── __init__.py
│       ├── collider.py         # Collider2D, BoxCollider2D, CircleCollider2D, ...
│       ├── rigidbody.py        # Rigidbody2D
│       ├── raycast.py
│       ├── collision.py
│       ├── collision_bool.py
│       ├── collision_manifold.py
│       ├── geometry.py
│       └── types.py
├── d3/                         # 3D engine
│   ├── window.py               # Window3D (main loop, rendering, shadows)
│   ├── scene.py                # Scene3D
│   ├── object3d.py             # Object3D + create_cube/sphere/plane
│   ├── camera.py               # Camera3D, Viewport, ClearFlags, RenderLayer
│   ├── light.py                # Lights (Directional, Point) + shadow support
│   ├── particle.py             # ParticleSystem + shapes + curves
│   └── physics/
│       ├── __init__.py
│       ├── collider.py         # Collider3D, BoxCollider3D, SphereCollider3D, CapsuleCollider3D, ...
│       ├── rigidbody.py        # Rigidbody3D
│       ├── raycast.py
│       ├── group.py            # CollisionGroup filtering
│       ├── collision.py
│       ├── collision_bool.py
│       ├── collision_manifold.py
│       ├── geometry.py
│       └── types.py
├── animation/                  # KeyFrame, AnimationClip, Animator (state machine)
├── graphics/                   # Materials and custom shaders
│   ├── material.py             # Unlit, Lit, Specular, Transparent, Skybox, ...
│   ├── shader.py
│   ├── shader_material.py
│   └── shadow.py               # Shadow mapping
├── ui/                         # 2D UI system
│   ├── widgets.py              # Label, Button, Slider, ProgressBar, Panel, ...
│   └── manager.py              # UIManager (accessed via scene.canvas)
├── editor/                     # PySide6 editor
│   ├── window.py
│   ├── viewport.py
│   ├── scene.py
│   ├── gizmo.py
│   └── ...
├── input/                      # Input, Keys, MouseButtons
├── types/                      # Vector2, Vector3, Color, Quaternion
├── component.py                # Component, Script, InspectorField, serializable, ...
├── gameobject.py               # GameObject, Prefab
├── transform.py                # Position/rotation/scale + parent-child hierarchy
├── scene.py                    # Base Scene + SceneManager (loading/saving)
├── resources.py                # Centralized mesh/texture loading
├── scriptable_object.py        # ScriptableObject + InspectorField (data assets)
├── drawing.py                  # Arcade-style immediate-mode 2D drawing
├── audio.py                    # AudioClip, AudioSource, AudioListener
├── build.py                    # Packaging helpers
├── cli.py                      # `pyengine` CLI (startproject, run, build, editor)
├── window_base.py              # Shared window infrastructure
└── cython/                     # Performance-critical Cython extensions (.pyx)
```

Top-level in the repository:
- `examples/` – runnable demos
- `tests/`
- `Scenes/`, `example/`
- `pyproject.toml`, `setup_cython.py`

## Performance

| Configuration | FPS with 100 Objects |
|---------------|----------------------|
| Software (pygame) | ~0.3 FPS |
| PyEngine (GPU) | 200+ FPS |

## License

MIT License
