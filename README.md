<table>
<tr>
<td width="220" valign="middle">
  <img src="images/logo.jpg" alt="PyEngine logo" width="200">
</td>
<td valign="middle">

# PyEngine — Open Source 2D/3D Game Engine for Python

PyEngine is a beginner-friendly **2D/3D game engine for Python**, inspired by [Arcade](https://arcade.academy/) and Unity-style workflows. It uses a component–entity model, GPU rendering via **ModernGL**, optional **Cython** acceleration, and a **PySide6** editor.

</td>
</tr>
</table>

**Status:** Alpha (`0.1.0`) · **License:** [MIT](LICENSE) · **Python:** 3.8–3.12

```bash
pip install -e ".[dev]"   # from a clone
pyengine --version
python -m pytest tests/ -q
```

---

## Features

| Area | Highlights |
|------|------------|
| **API** | Arcade-like `Window` / `Scene` style; `GameObject` + `Component` / `Script` |
| **Rendering** | ModernGL (OpenGL 3.3+), materials, custom GLSL, shadows, multi-camera viewports, instancing |
| **2D & 3D** | Parallel stacks (`engine.d2` / `engine.d3`): sprites, cameras, physics, scenes |
| **Scripts** | Unity-like lifecycle: `awake` / `start` / `fixed_update` / `update` / `late_update` (empty hooks are **not** called) |
| **Physics** | Colliders, rigidbodies, friction/bounce materials, continuous collision, raycasts, collider groups |
| **Animation** | Keyframes, clips, Animator state machine (parameters & transitions) |
| **Particles** | Pooled emitters, shapes (sphere/cone/box), lifetime curves, bursts |
| **UI** | Labels, buttons, checkboxes, sliders, progress bars, panels (`scene.canvas`) |
| **Audio** | `AudioClip`, `AudioSource`, `AudioListener` (pygame.mixer, 2D/3D blend) |
| **Data** | Prefabs, ScriptableObjects (`.asset`), scene JSON, resources |
| **Tooling** | `pyengine` CLI (startproject / run / build / editor), Cython hot paths, tests + window integration tests |
| **Editor** | Hierarchy, inspector, viewport, gizmos, undo/redo, project browser (optional `[editor]`) |

---

## Requirements

- **Python 3.8–3.12** (3.13+ not supported yet — pygame/packaging gaps)
- A working **OpenGL** driver (for games and window tests)
- Optional: C compiler for Cython acceleration when installing from source

---

## Installation

### From source (recommended for development)

```bash
git clone https://github.com/Sepuh077/3D-engine.git
cd 3D-engine
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux / macOS
# source venv/bin/activate

pip install -e .
```

This installs the package and **builds Cython extensions** when a C compiler is available. Without a compiler the install still succeeds (pure-Python fallbacks).

```bash
pip install -e ".[editor]"   # + PySide6 editor
pip install -e ".[dev]"      # + pytest (+ Cython for rebuilds)
pip install -e ".[all]"      # editor + dev
```

### Pure-Python mode (no native extensions)

```bash
# Windows PowerShell
$env:PYENGINE_PURE_PYTHON = "1"
python your_script.py

# Linux / macOS
PYENGINE_PURE_PYTHON=1 python your_script.py
```

### Building Cython from source (optional)

You need a C compiler + Python headers:

| OS | Tools |
|----|--------|
| Linux | `build-essential`, `python3-dev` |
| macOS | `xcode-select --install` |
| Windows | [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) |

Then `pip install -e .` is enough. Benchmark:

```bash
python bench_cython.py
```

### Dependencies

Declared in `pyproject.toml`: **pygame**, **numpy**, **pillow**, **trimesh**, **moderngl**.  
Optional: **PySide6** (editor), **pytest** / **ruff** / **pytest-cov** (`[dev]`).  
`requirements.txt` pins useful versions for local installs.

---

## CLI

After install, the `pyengine` command is available:

```bash
pyengine startproject mygame      # scaffold a 3D project
pyengine --2d startproject my2d  # scaffold a 2D project
cd mygame
pyengine run
pyengine editor
pyengine build                   # PyInstaller directory
pyengine build --onefile
pyengine build --backend nuitka
pyengine --version
```

Scaffold layout:

```
mygame/
  main.py
  settings.py
  requirements.txt
  pyproject.toml
  build.py
  assets/
  scenes/
  scripts/
  .gitignore
```

> Built executables that include Cython modules may need the **Visual C++ Redistributable** on end-user Windows machines. Test on a clean PC/VM.

---

## Quick start

```python
from engine.d3 import Window3D, Scene3D, Object3D, create_cube, Time
from engine.input import Keys
from engine.types import Color

class MyGame(Scene3D):
    def setup(self):
        super().setup()
        self.cube = create_cube(size=1.0, color=Color.ORANGE)
        self.add_object(self.cube)
        self.camera.position = (0, 5, 10)
        self.camera.look_at((0, 0, 0))

    def on_update(self):
        self.cube.transform.rotation_y += 30 * Time.delta_time

    def on_key_press(self, key, modifiers):
        if key == Keys.ESCAPE:
            self.window.close()

window = Window3D(800, 600, "My 3D Game")
window.show_scene(MyGame())
window.run()  # e.g. window.run(fps=200) for a higher render cap
```

2D entry point: `from engine.d2 import Window2D, Scene2D` (see `examples/example_2d_game.py`).

---

## Scripts & timing

### Lifecycle (Unity-like)

```python
from engine.component import Script

class Player(Script):
    def awake(self): ...
    def start(self): ...

    def fixed_update(self):
        # Physics rate (default 60 Hz). Time.delta_time == Time.fixed_delta_time
        ...

    def update(self):
        # Once per rendered frame (variable dt)
        ...

    def late_update(self):
        # After update + fixed steps (camera follow, etc.)
        ...
```

**Frame order:** `fixed_update` × N → `update` → `late_update` → end-of-frame coroutines → render.

Empty base methods are **not** invoked: a script is only registered for a phase if it overrides that method.

Coroutines: `yield WaitForSeconds(1)`, `WaitForFrames(n)`, `WaitEndOfFrame()` via `game_object.start_coroutine(...)`.

### Time

| Setting | Default | Meaning |
|---------|---------|---------|
| `Time.delta_time` | — | Last frame duration (scaled) |
| `Time.maximum_delta_time` | `0.1` | Ceiling for **slow** frames only (hitches). Does **not** limit high FPS |
| `Time.fixed_delta_time` | `1/60` | Physics / `fixed_update` step. Set to `0` for one variable-dt physics step per frame |
| `Time.maximum_physics_steps` | `8` | Max fixed sub-steps per frame |

At 200 FPS, `delta_time ≈ 0.005` and is never clamped by `maximum_delta_time`. Cap render rate with `window.run(fps=200)`.

---

## Editor

```bash
pip install -e ".[editor]"
pyengine editor
# or: python -c "from engine.d3 import run_editor; run_editor('.')"
```

- **Hierarchy** — scene graph  
- **Viewport** — play view, gizmos, selection  
- **Inspector** — components & `InspectorField`s  
- **Project browser** — assets, prefabs, scriptable objects  
- **Undo / redo**, console  

Editor modules live under `engine/editor/` (`window`, `hierarchy`, `project_browser`, `console`, `widgets`, …).

---

## Examples

Runnable demos in `examples/`:

| Script | Topic |
|--------|--------|
| `example_basic.py` | Load mesh, camera, particles |
| `example_game.py` / `example_fps_camera.py` | Gameplay-style input |
| `example_collision.py` / `example_collision_groups.py` | Physics & groups |
| `example_shadows.py` / `example_materials.py` / `example_shaders.py` | Lighting & materials |
| `example_multi_camera.py` | Minimap / multi-view |
| `example_particles.py` | Particle system |
| `example_animation.py` | Animator state machine |
| `example_2d_game.py` / `example_2d_ui.py` / `example_2d_shaders.py` | 2D stack |
| `example_audio.py` | Audio |
| `example_scriptable_object.py` / `example_serialization.py` | Data & save/load |
| `example_scene_loading.py` | Scene files |

```bash
python examples/example_basic.py
```

Sample meshes live under `example/` (e.g. `stairs_modular_right.obj`).

---

## Snippets

### Physics

```python
from engine.d3 import create_cube
from engine.d3.physics import BoxCollider3D, Rigidbody3D

ground = create_cube(position=(0, -1, 0))
ground.transform.scale = (20, 1, 20)
ground.add_component(BoxCollider3D())
ground.add_component(Rigidbody3D(is_static=True, use_gravity=False))

ball = create_cube(position=(0, 5, 0))
ball.add_component(BoxCollider3D())
ball.add_component(Rigidbody3D(use_gravity=True))
```

Use `ColliderGroup` + `CollisionRelation` for ignore/trigger/solid filtering (`examples/example_collision_groups.py`).

### Multi-camera

```python
self.create_minimap_camera(position=(0, 50, 0), corner="top-right", size=0.25)
```

### UI

```python
from engine.ui import Label, Button, ProgressBar

self.canvas.add(Label(10, 10, text="Health"))
self.canvas.add(ProgressBar(10, 40, width=200, value=75, max_value=100))
btn = Button(10, 80, text="Restart")
btn.on("click", lambda *_: print("restart"))
self.canvas.add(btn)
```

### Scriptable objects

```python
from engine import ScriptableObject, InspectorField

class WeaponData(ScriptableObject):
    damage = InspectorField(float, default=10.0)

sword = WeaponData.create("IronSword")
sword.damage = 25.0
sword.save("assets/iron_sword.asset")
```

### Input

```python
from engine.input import Input, Keys

if Input.get_key(Keys.W):
    ...
if Input.get_key_down(Keys.SPACE):
    ...
```

### Primitives & color

```python
from engine.d3 import create_cube, create_sphere, create_plane
from engine.types import Color

create_cube(size=2.0, position=(0, 1, 0), color=Color.RED)
Color.from_hex("#FF8800")
```

---

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

- Unit tests cover physics, scripts lifecycle, UI, audio, prefabs, math, coroutines, etc.
- **Window integration tests** (`tests/test_window_integration.py`, mark `window`) open a small OpenGL window, tick a few frames, and close it.
  - They **run by default** when a real GL context can be created (normal desktop Windows/Linux/macOS).
  - They **skip** only if GL cannot open, or if you set `PYENGINE_SKIP_WINDOW_TESTS=1` (headless CI).
  - Other tests must not leave `SDL_VIDEODRIVER=dummy` set; the suite cleans that up before window tests.

```bash
# Window tests only
python -m pytest tests/test_window_integration.py -v
# or
python -m pytest -m window -v
```

Benchmark Cython vs pure Python:

```bash
python bench_cython.py
```

---

## API overview

### Core

| Symbol | Role |
|--------|------|
| `Window3D` / `Window2D` | App window, main loop, rendering |
| `Scene3D` / `Scene2D` | Scene content + lifecycle hooks |
| `GameObject` | Entity; always has `Transform` |
| `Component` / `Script` | Attachable behaviour |
| `Time` | `delta_time`, fixed step, hitch clamp |
| `Tag` | Named tags for queries |
| `Resources` | Mesh / texture loading |
| `Prefab` | Reusable object templates |
| `ScriptableObject` | Data assets (`.asset`) |

### Rendering & lights

`Object3D`, `Camera3D`, `Viewport`, `ClearFlags`, `RenderLayer`, materials (`Unlit`, `Lit`, `Specular`, `Emissive`, `Transparent`, `Skybox`), `Shader` / `ShaderMaterial`, `DirectionalLight3D`, `PointLight3D`, shadow maps.

### Physics (3D)

`Rigidbody3D`, `BoxCollider3D`, `SphereCollider3D`, `CapsuleCollider3D`, `ColliderGroup`, raycast helpers, continuous collision modes, physics materials (bounciness / friction).

### Physics (2D)

`Rigidbody2D`, `BoxCollider2D`, `CircleCollider2D`, … under `engine.d2.physics`.

### UI

`UIManager` (`scene.canvas`), `Label`, `Button`, `CheckBox`, `Slider`, `ProgressBar`, `Panel`.

### Audio

`AudioClip`, `AudioSource`, `AudioListener`.

Version: `from engine import __version__` (also `engine.version`).

---

## Project layout

```
engine/
  version.py, log.py, window_base.py, cli.py, build.py
  component.py, component_registry.py, gameobject.py, transform.py
  scene.py, resources.py, scriptable_object.py, drawing.py, audio.py
  d2/                 # Window2D, Scene2D, sprites, 2D physics
  d3/                 # Window3D, Scene3D, Object3D, lights, particles, 3D physics
    shaders/          # default pipeline GLSL (.vert / .frag)
  rendering/          # shared RenderLayer, Viewport, ClearFlags
  physics/            # PhysicsWorld + shared solver constants
  animation/          # clips + Animator
  graphics/           # materials, shaders, shadows
  ui/                 # canvas widgets
  editor/             # PySide6 editor + mixins/ (play, inspector, prefab, …)
  input/              # Input, Keys
  types/              # Vector2/3, Color, Quaternion
  cython/             # accelerated .pyx modules (generated .c gitignored)
examples/             # demos
tests/                # pytest suite (+ optional window tests)
example/              # sample assets (meshes, glTF)
bench_cython.py
pyproject.toml
setup.py / setup_cython.py
```

---

## Packaging games

```bash
pyengine build
pyengine build --onefile
pyengine build --backend nuitka
```

Uses project `pyproject.toml` `[tool.pyengine.build]` when present. Prefer testing the build on a machine without your toolchain installed.

---

## Development tooling

```bash
pip install -e ".[dev]"
ruff check engine tests          # lint
python -m pytest tests/ -q      # unit + window tests
python -m pytest --cov=engine --cov-report=term-missing -q
pyengine --version              # includes Cython acceleration status
```

Generated Cython ``.c`` / ``.so`` / ``.pyd`` files are gitignored; ``pip install -e .``
compiles from ``.pyx`` when a C compiler is available.

---

## Performance notes

- Prefer GPU path (ModernGL) over software drawing for many objects.
- Cython speeds up math, transforms, collision, game loop, particles when built.
- Large static scenes: `static` batches / instancing helpers on `Window3D` (`enable_instancing`, `build_static_batches`, profiler caption via `show_profiler`).
- Scripts: only override the lifecycle methods you need.

---

## Contributing / development

```bash
pip install -e ".[all]"
python -m pytest tests/ -q
python bench_cython.py
```

Issues: [GitHub Issues](https://github.com/Sepuh077/3D-engine/issues).

---

## License

[MIT License](LICENSE) — © PyEngine contributors.
