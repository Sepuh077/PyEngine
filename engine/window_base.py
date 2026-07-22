"""
WindowBase — shared base class for Window3D and Window2D.

Provides: Pygame + ModernGL init, 2D HUD overlay, event handling, timing,
drawing helpers, input, lifecycle stubs, and main loop.

Subclasses must implement:
    _init_gpu()           — compile subclass-specific shaders / create VAOs
    _render()             — full per-frame rendering pipeline
    _process_collisions() — physics collision detection / resolution
    _cleanup_gpu()        — release subclass-specific GPU resources
"""
import time
from pathlib import Path
import pygame

try:
    from pygame import gfxdraw
    HAS_GFXDRAW = True
except ImportError:
    HAS_GFXDRAW = False

import numpy as np
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from engine.gameobject import GameObject
from engine.types import Color, ColorType
from engine.input import Input
from engine.component import Script, Time

# Cython-accelerated game loop (falls back to pure-Python iteration).
# Import the module directly — do not gate on global CYTHON_ENABLED, so a
# partial Cython install still accelerates the loop when cy_gameloop loads.
try:
    import os as _os
    if _os.environ.get("PYENGINE_PURE_PYTHON", "0").lower() in ("1", "true", "yes"):
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_gameloop import (
        cy_update_objects as _cy_update_objects,
        cy_update_end_of_frame as _cy_update_end_of_frame,
    )
    _USE_CYTHON_GAMELOOP = True
except Exception:
    _USE_CYTHON_GAMELOOP = False

try:
    import moderngl
    HAS_MODERNGL = True
except ImportError:
    HAS_MODERNGL = False


class WindowBase:
    """Abstract base for both 2D and 3D engine windows.

    Common parameters include project_root for asset location and
    auto_load_scriptable_assets (default True) which controls whether
    ScriptableObject assets are loaded on startup or lazily on first access
    via ScriptableObject.get() etc.
    """

    # -- Overlay shaders (shared by both 2D and 3D) -------------------------

    OVERLAY_VERTEX_SHADER = '''
    #version 330 core
    in vec2 in_pos;
    in vec2 in_tex;
    out vec2 frag_tex;
    void main() {
        gl_Position = vec4(in_pos, 0.0, 1.0);
        frag_tex = in_tex;
    }
    '''

    OVERLAY_FRAGMENT_SHADER = '''
    #version 330 core
    in vec2 frag_tex;
    uniform sampler2D tex;
    out vec4 frag_color;
    void main() {
        frag_color = texture(tex, frag_tex);
    }
    '''

    # =======================================================================
    # Init
    # =======================================================================

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "Engine",
        resizable: bool = False,
        project_root: Union[str, Path] = ".",
        auto_load_scriptable_assets: bool = True,
        background_color: ColorType = (0.1, 0.1, 0.15),
        use_pygame_window: bool = True,
        use_pygame_events: bool = True,
    ):
        if not HAS_MODERNGL:
            raise ImportError(
                "ModernGL is required. Install with: pip install moderngl"
            )

        self.width = width
        self.height = height
        self.title = title
        self.background_color = background_color
        self.project_root = project_root if isinstance(project_root, Path) else Path(project_root).resolve()
        self.auto_load_scriptable_assets = auto_load_scriptable_assets

        # Ensure project root is on sys.path so that project packages like "scripts"
        # (and modules saved in scene files as e.g. "scripts.player") can be imported.
        self._ensure_project_on_sys_path()

        # -- Pygame ----------------------------------------------------------
        self._use_pygame_window = use_pygame_window
        self._use_pygame_events = use_pygame_events and use_pygame_window

        pygame.init()

        if self._use_pygame_window:
            flags = pygame.OPENGL | pygame.DOUBLEBUF
            if resizable:
                flags |= pygame.RESIZABLE
            pygame.display.set_mode((width, height), flags)
            pygame.display.set_caption(title)
            pygame.event.set_allowed([
                pygame.QUIT,
                pygame.KEYDOWN,
                pygame.KEYUP,
                pygame.MOUSEBUTTONDOWN,
                pygame.MOUSEBUTTONUP,
                pygame.MOUSEMOTION,
                pygame.VIDEORESIZE,
            ])

        # -- ModernGL context ------------------------------------------------
        try:
            self._ctx = moderngl.create_context(require=330)
        except Exception as exc:
            from engine.log import get_logger
            get_logger("window").debug(
                "OpenGL 3.3 context failed (%s); retrying without require", exc
            )
            self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self._ctx.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )

        # -- Overlay shader --------------------------------------------------
        self._overlay_program = self._ctx.program(
            vertex_shader=self.OVERLAY_VERTEX_SHADER,
            fragment_shader=self.OVERLAY_FRAGMENT_SHADER,
        )

        # -- 2D HUD surface + fullscreen quad --------------------------------
        self._2d_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        self._fonts: dict = {}
        self._image_cache: dict = {}

        quad = np.array([
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
            -1.0, -1.0, 0.0, 1.0,
             1.0,  1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 0.0,
        ], dtype=np.float32)
        self._2d_vbo = self._ctx.buffer(quad.tobytes())
        self._2d_vao = self._ctx.vertex_array(
            self._overlay_program,
            [(self._2d_vbo, '2f 2f', 'in_pos', 'in_tex')],
        )
        self._2d_texture = self._ctx.texture((width, height), 4)

        # -- Profiler / caption ----------------------------------------------
        self.show_profiler = False
        self.profiler_interval = 0.25
        self._profiler_text = ""
        self._last_profiler_time = 0.0
        self._caption_base = title

        # -- Scene / objects -------------------------------------------------
        self.objects: List[GameObject] = []
        self._current_scene = None

        # -- Timing ----------------------------------------------------------
        self._clock = pygame.time.Clock()
        self._running = False
        self._fps = 60
        self._delta_time = 0.0
        Time.delta_time = 0.0

        self._setup_done = False

        # Always register project root (for lazy loading support)
        from engine.scriptable_object import ScriptableObject
        ScriptableObject.set_project_root(str(self.project_root))

        if self.auto_load_scriptable_assets:
            # Load all ScriptableObject assets upfront (default behavior)
            self._load_scriptable_objects()

        # -- Register for global draw funcs ----------------------------------
        from engine import drawing
        drawing.set_window(self)

        # -- Subclass-specific GPU init --------------------------------------
        self._init_gpu()

    # =======================================================================
    # Abstract / hook methods (override in subclass)
    # =======================================================================

    def _init_gpu(self):
        """Called once at end of __init__. Compile shaders, create VAOs, etc."""

    def _load_scriptable_objects(self) -> None:
        """
        Load all ScriptableObject assets from the project directory.
        Called automatically if auto_load_scriptable_assets=True (the default).
        When False, loading happens lazily on first ScriptableObject.get() etc.
        """
        try:
            from engine.scriptable_object import ScriptableObject

            loaded = ScriptableObject.load_all_assets(str(self.project_root))
            if loaded:
                print(f"Loaded {len(loaded)} ScriptableObject assets from {self.project_root}")

        except Exception as e:
            # Assets may be missing in empty projects; still log for diagnostics
            from engine.log import get_logger
            get_logger("window").debug(
                "ScriptableObject auto-load skipped/failed: %s", e, exc_info=True
            )

    def _render(self):
        """Full per-frame rendering pipeline. Must be implemented by subclass."""
        raise NotImplementedError

    def _process_collisions(self):
        """Physics collision detection / resolution."""

    def _cleanup_gpu(self):
        """Release subclass-specific GPU resources."""

    def _ensure_project_on_sys_path(self) -> None:
        """Ensure the project root directory is at the front of sys.path.

        This makes it possible to do ``import scripts.xxx`` and similar for any
        Python package/module living directly under the project root. This is
        required when deserializing scenes (or ScriptableObjects) that were saved
        with references to user-defined component classes.
        """
        import sys
        import types
        proj = str(self.project_root)
        if proj not in sys.path:
            sys.path.insert(0, proj)

        # Pre-register the conventional "scripts" package (and potentially others)
        # so that "import scripts.player" works even before Python walks the path
        # in certain reload or partially-initialized states.
        for pkg_name in ("scripts",):
            if pkg_name not in sys.modules:
                pkg_dir = self.project_root / pkg_name
                if pkg_dir.is_dir():
                    pkg = types.ModuleType(pkg_name)
                    pkg.__path__ = [str(pkg_dir)]
                    sys.modules[pkg_name] = pkg

    # =======================================================================
    # Properties
    # =======================================================================

    @property
    def current_scene(self):
        return self._current_scene

    @property
    def fps(self) -> float:
        return self._clock.get_fps()

    @property
    def delta_time(self) -> float:
        return self._delta_time

    @property
    def size(self) -> Tuple[int, int]:
        return (self.width, self.height)

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 1.0

    @property
    def mouse_position(self) -> Tuple[int, int]:
        return Input.get_mouse_position()

    def set_caption(self, title: str):
        self.title = title
        self._caption_base = title
        self._apply_caption()

    def _apply_caption(self):
        title = self._caption_base
        if self.show_profiler and self._profiler_text:
            title = f"{title} | {self._profiler_text}"
        pygame.display.set_caption(title)

    # =======================================================================
    # Input helpers
    # =======================================================================

    def is_key_pressed(self, key: int) -> bool:
        return Input.get_key(key)

    def is_key_down(self, key: int) -> bool:
        return Input.get_key_down(key)

    def is_key_up(self, key: int) -> bool:
        return Input.get_key_up(key)

    def is_mouse_button_pressed(self, button: int) -> bool:
        return Input.get_mouse_button(button)

    def is_mouse_button_down(self, button: int) -> bool:
        return Input.get_mouse_button_down(button)

    def is_mouse_button_up(self, button: int) -> bool:
        return Input.get_mouse_button_up(button)

    # =======================================================================
    # Internal helpers
    # =======================================================================

    def _active_objects(self) -> List[GameObject]:
        return self._current_scene.objects if self._current_scene else self.objects

    def _get_font(self, name: Optional[str], size: int) -> pygame.font.Font:
        key = (name or 'default', size)
        if key not in self._fonts:
            if name and (name.lower().endswith(('.ttf', '.otf')) or '/' in name or '\\' in name):
                self._fonts[key] = pygame.font.Font(name, size)
            else:
                self._fonts[key] = pygame.font.SysFont(name, size)
        return self._fonts[key]

    def bind_context(self):
        """Ensure this window's GL context is active before rendering."""
        if self._use_pygame_window:
            return
        try:
            self._ctx.use()
        except Exception as exc:
            from engine.log import get_logger
            get_logger("window").debug("ctx.use() failed: %s", exc)

    # =======================================================================
    # 2D overlay rendering (HUD)
    # =======================================================================

    def _render_2d_overlay(self):
        """Upload HUD surface to GPU texture and render as fullscreen quad."""
        if self._current_scene:
            self._current_scene.canvas.draw(self._2d_surface)

        data = pygame.image.tostring(self._2d_surface, "RGBA", False)
        self._2d_texture.write(data)

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._2d_texture.use(location=0)
        self._overlay_program['tex'].value = 0
        self._2d_vao.render(moderngl.TRIANGLES)
        self._ctx.enable(moderngl.DEPTH_TEST)

    # =======================================================================
    # Screen-space drawing helpers (all draw onto self._2d_surface)
    # =======================================================================

    def draw_text(
        self, text: str, x: float, y: float,
        color: ColorType = Color.WHITE,
        font_size: int = 24,
        font_name: Optional[str] = None,
        anchor_x: str = 'left',
        anchor_y: str = 'top',
        baseline_adjust: bool = True,
    ) -> None:
        font = self._get_font(font_name, font_size)
        if len(color) == 3:
            rgb = tuple(int(c * 255) for c in color)
            alpha = 255
        else:
            rgb = tuple(int(c * 255) for c in color[:3])
            alpha = int(color[3] * 255)
        text_surf = font.render(text, True, rgb)
        if alpha < 255:
            text_surf = text_surf.convert_alpha()
            arr = pygame.surfarray.pixels_alpha(text_surf)
            arr[:] = (arr[:] * (alpha / 255)).astype(np.uint8)
            del arr
        w, h = text_surf.get_size()
        # Accept floats (from project_point etc.) and cast after adjustments
        x = float(x)
        y = float(y)
        if anchor_x == 'center':
            x -= w // 2
        elif anchor_x == 'right':
            x -= w
        if anchor_y == 'center':
            y -= h // 2
        elif anchor_y == 'bottom':
            y -= h
        if baseline_adjust:
            y -= font.get_ascent() // 6
        # Final cast to int for blit (Pygame can be picky with float32/ float in some versions/platforms)
        x = int(x)
        y = int(y)
        self._2d_surface.blit(text_surf, (x, y))

    def draw_rectangle(
        self, x: int, y: int, width: int, height: int,
        color: ColorType, border_width: int = 0,
    ) -> None:
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self._2d_surface, col, rect, border_width)

    def draw_circle(
        self, x: int, y: int, radius: int, color: ColorType,
        border_width: int = 2, aa: bool = True,
    ) -> None:
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        x, y, radius = int(x), int(y), int(radius)
        if border_width > 0 and HAS_GFXDRAW and aa:
            if -32768 <= x <= 32767 and -32768 <= y <= 32767 and 0 < radius <= 32767:
                gfxdraw.aacircle(self._2d_surface, x, y, radius, col[:3])
                if border_width > 1:
                    gfxdraw.aacircle(self._2d_surface, x, y, radius - 1, col[:3])
            else:
                pygame.draw.circle(self._2d_surface, col, (x, y), max(1, abs(radius)), border_width)
        else:
            pygame.draw.circle(self._2d_surface, col, (x, y), max(1, abs(radius)), border_width)

    def draw_ellipse(
        self, x: int, y: int, width: int, height: int,
        color: ColorType, border_width: int = 2, aa: bool = True,
    ) -> None:
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        rect = pygame.Rect(x, y, width, height)
        if border_width > 0 and HAS_GFXDRAW and aa:
            cx, cy, rx, ry = x + width // 2, y + height // 2, width // 2, height // 2
            if (-32768 <= cx <= 32767 and -32768 <= cy <= 32767
                    and 0 < rx <= 32767 and 0 < ry <= 32767):
                gfxdraw.aaellipse(self._2d_surface, cx, cy, rx, ry, col[:3])
            else:
                pygame.draw.ellipse(self._2d_surface, col, rect, border_width)
        else:
            pygame.draw.ellipse(self._2d_surface, col, rect, border_width)

    def draw_polygon(
        self, points: List[Tuple[int, int]], color: ColorType,
        border_width: int = 2, aa: bool = True,
    ) -> None:
        if len(points) < 3:
            return
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        if border_width > 0 and HAS_GFXDRAW and aa:
            gfxdraw.aapolygon(self._2d_surface, points, col[:3])
        else:
            pygame.draw.polygon(self._2d_surface, col, points, border_width)

    def draw_line(
        self, start: Tuple[int, int], end: Tuple[int, int],
        color: ColorType, width: int = 2, aa: bool = True,
    ) -> None:
        if len(color) == 3:
            col = tuple(int(c * 255) for c in color) + (255,)
        else:
            col = tuple(int(c * 255) for c in color)
        if aa:
            pygame.draw.aaline(self._2d_surface, col[:3], start, end)
        else:
            pygame.draw.line(self._2d_surface, col, start, end, width)

    def draw_image(
        self, image: Union[str, pygame.Surface], x: int, y: int,
        scale: float = 1.0, alpha: float = 1.0,
    ) -> None:
        if isinstance(image, str):
            if image not in self._image_cache:
                surf = pygame.image.load(image).convert_alpha()
                self._image_cache[image] = surf
            surf = self._image_cache[image]
        else:
            surf = image
        if scale != 1.0:
            new_size = (int(surf.get_width() * scale), int(surf.get_height() * scale))
            surf = pygame.transform.scale(surf, new_size)
        if alpha < 1.0:
            surf = surf.copy()
            surf.set_alpha(int(alpha * 255))
        self._2d_surface.blit(surf, (x, y))

    # =======================================================================
    # Lifecycle stubs (override in subclass)
    # =======================================================================

    def setup(self):
        pass

    def on_update(self):
        pass

    def on_draw(self):
        pass

    def on_key_press(self, key: int, modifiers: int):
        pass

    def on_key_release(self, key: int, modifiers: int):
        pass

    def on_mouse_press(self, x: int, y: int, button: int, modifiers: int):
        pass

    def on_mouse_release(self, x: int, y: int, button: int, modifiers: int):
        pass

    def on_mouse_motion(self, x: int, y: int, dx: int, dy: int):
        pass

    def on_mouse_scroll(self, x: int, y: int, scroll_x: int, scroll_y: int):
        pass

    def on_resize(self, width: int, height: int):
        self.width = max(1, width)
        self.height = max(1, height)
        self._2d_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        if hasattr(self, '_2d_texture'):
            self._2d_texture.release()
            self._2d_texture = self._ctx.texture((self.width, self.height), 4)
        if hasattr(self, '_ctx'):
            self._ctx.viewport = (0, 0, self.width, self.height)

    # =======================================================================
    # Event handling
    # =======================================================================

    def _handle_events(self):
        if not self._use_pygame_events:
            return

        Input._update_frame_start()

        for event in pygame.event.get():
            if self._current_scene and self._current_scene.canvas.process_pygame_event(event):
                continue

            if event.type == pygame.QUIT:
                self._running = False

            elif event.type == pygame.KEYDOWN:
                Input._keys_pressed.add(event.key)
                Input._keys_down_this_frame.add(event.key)
                mods = pygame.key.get_mods()
                if self._current_scene:
                    self._current_scene.on_key_press(event.key, mods)
                self.on_key_press(event.key, mods)

            elif event.type == pygame.KEYUP:
                Input._keys_pressed.discard(event.key)
                Input._keys_up_this_frame.add(event.key)
                mods = pygame.key.get_mods()
                if self._current_scene:
                    self._current_scene.on_key_release(event.key, mods)
                self.on_key_release(event.key, mods)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                Input._mouse_buttons.add(event.button)
                Input._mouse_down_this_frame.add(event.button)
                x, y = event.pos
                mods = pygame.key.get_mods()
                if event.button == 4:
                    Input._mouse_scroll = (0, 1)
                    if self._current_scene:
                        self._current_scene.on_mouse_scroll(x, y, 0, 1)
                    self.on_mouse_scroll(x, y, 0, 1)
                elif event.button == 5:
                    Input._mouse_scroll = (0, -1)
                    if self._current_scene:
                        self._current_scene.on_mouse_scroll(x, y, 0, -1)
                    self.on_mouse_scroll(x, y, 0, -1)
                else:
                    if self._current_scene:
                        self._current_scene.on_mouse_press(x, y, event.button, mods)
                    self.on_mouse_press(x, y, event.button, mods)

            elif event.type == pygame.MOUSEBUTTONUP:
                Input._mouse_buttons.discard(event.button)
                Input._mouse_up_this_frame.add(event.button)
                x, y = event.pos
                mods = pygame.key.get_mods()
                if self._current_scene:
                    self._current_scene.on_mouse_release(x, y, event.button, mods)
                self.on_mouse_release(x, y, event.button, mods)

            elif event.type == pygame.MOUSEMOTION:
                x, y = event.pos
                dx, dy = event.rel
                Input._mouse_position = (x, y)
                Input._mouse_delta = (dx, dy)
                if self._current_scene:
                    self._current_scene.on_mouse_motion(x, y, dx, dy)
                self.on_mouse_motion(x, y, dx, dy)

            elif event.type == pygame.VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self._ctx.viewport = (0, 0, event.w, event.h)
                if self._current_scene:
                    self._current_scene.on_resize(event.w, event.h)
                self.on_resize(event.w, event.h)

    # =======================================================================
    # Main loop
    # =======================================================================

    def start(self, start_components: bool = True):
        if not self._setup_done:
            self.setup()
            self._setup_done = True
        if start_components:
            # One-time lifecycle: run on every object (cheap, and some non-behavior
            # components may still implement start/awake).
            for obj in self._active_objects():
                obj.start_components()

    def tick(self, fps: Optional[int] = None, simulate: bool = True) -> bool:
        if fps is not None:
            self._fps = fps
        if not self._running:
            self.start(start_components=False)
            self._running = True

        raw_dt = self._clock.tick(self._fps) / 1000.0
        Time.set(raw_dt)  # clamps via Time.maximum_delta_time
        self._delta_time = Time.delta_time
        frame_dt = Time.delta_time

        self._handle_events()

        if simulate:
            if self._current_scene:
                self._current_scene.on_update()
            self.on_update()

            # Use the fast Cython entity container's updatables list when available.
            # For scenes with thousands of passive objects this avoids touching
            # the vast majority of them every frame.
            scene = self._current_scene
            if scene is not None and getattr(scene, '_updatables', None):
                active = scene._updatables
            else:
                active = self._active_objects()

            fixed_list = (
                scene._fixed_updatables
                if scene is not None and getattr(scene, '_fixed_updatables', None)
                else ()
            )
            late_list = (
                scene._late_updatables
                if scene is not None and getattr(scene, '_late_updatables', None)
                else ()
            )

            # Unity-like order:
            #   FixedUpdate × N  →  Update  →  LateUpdate  →  (EOF coroutines) → Render
            #
            # Fixed-step physics: accumulate frame time and step scripts + RBs
            # + collisions at a constant rate (default 60 Hz).
            fixed_dt = Time.fixed_delta_time
            if fixed_dt > 0.0:
                Time._physics_accumulator += frame_dt
                max_steps = max(1, int(Time.maximum_physics_steps))
                steps = 0
                while Time._physics_accumulator >= fixed_dt and steps < max_steps:
                    Time.delta_time = fixed_dt
                    self._run_fixed_updates(fixed_list)
                    self._update_rigidbodies(active)
                    self._process_collisions()
                    Time.fixed_time += fixed_dt
                    Time._physics_accumulator -= fixed_dt
                    steps += 1
                # Drop leftover if we hit the step cap (prevent spiral of death)
                if steps >= max_steps:
                    Time._physics_accumulator = min(
                        Time._physics_accumulator, fixed_dt
                    )
                # Restore frame delta for Update / LateUpdate / render
                Time.delta_time = frame_dt
            else:
                # fixed_delta_time <= 0: one variable-dt physics step, still
                # call fixed_update once so gameplay can rely on the hook.
                self._run_fixed_updates(fixed_list)
                self._update_rigidbodies(active)
                self._process_collisions()

            # Frame update: scripts (opt-in) / animators / particles / coroutines.
            # Rigidbodies already integrated above.
            Time._skip_rigidbody_frame_update = True
            try:
                if _USE_CYTHON_GAMELOOP:
                    _cy_update_objects(active, frame_dt)
                else:
                    for obj in active:
                        obj.update()
            finally:
                Time._skip_rigidbody_frame_update = False

            if self._current_scene:
                self._current_scene.canvas.update(self._delta_time)

            # LateUpdate — only objects whose scripts override late_update
            self._run_late_updates(late_list)

            if _USE_CYTHON_GAMELOOP:
                _cy_update_end_of_frame(active, frame_dt)
            else:
                for obj in active:
                    obj.update_end_of_frame()

        self._render()
        return self._running

    def _run_fixed_updates(self, fixed_objs) -> None:
        """Call Script.fixed_update on opt-in lists only (empty list = free)."""
        if not fixed_objs:
            return
        for obj in fixed_objs:
            scripts = getattr(obj, '_scripts_fixed', None)
            if not scripts:
                continue
            for script in scripts:
                if not getattr(script, '_awoken', False):
                    script.awake()
                    script._awoken = True
                if not getattr(script, '_started', False):
                    script.start()
                    script._started = True
                script.fixed_update()

    def _run_late_updates(self, late_objs) -> None:
        """Call Script.late_update on opt-in lists only (empty list = free)."""
        if not late_objs:
            return
        for obj in late_objs:
            scripts = getattr(obj, '_scripts_late', None)
            if not scripts:
                continue
            for script in scripts:
                if not getattr(script, '_awoken', False):
                    script.awake()
                    script._awoken = True
                if not getattr(script, '_started', False):
                    script.start()
                    script._started = True
                script.late_update()

    def _update_rigidbodies(self, active) -> None:
        """Integrate all non-sleeping rigidbodies for the current Time.delta_time."""
        for obj in active:
            rb = getattr(obj, '_rigidbody', None)
            if rb is None:
                continue
            if getattr(rb, 'is_sleeping', False):
                continue
            try:
                rb.update()
            except Exception:
                from engine.log import get_logger
                get_logger("physics").exception(
                    "Rigidbody.update failed on %r", obj
                )

    def run(self, fps: int = 60):
        self._fps = fps
        self._running = True
        while self._running:
            self.tick(fps)
        self._cleanup()

    def close(self):
        self._running = False

    def _cleanup(self):
        self._cleanup_gpu()
        if hasattr(self, '_2d_texture'):
            self._2d_texture.release()
        if hasattr(self, '_2d_vao'):
            self._2d_vao.release()
        if hasattr(self, '_2d_vbo'):
            self._2d_vbo.release()
        if hasattr(self, '_overlay_program'):
            self._overlay_program.release()
        self._ctx.release()
        pygame.quit()
