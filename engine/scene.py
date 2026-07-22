"""
Scene - Base scene class shared by both 2D and 3D engines.
Provides object management, lifecycle, input handling, and UI canvas.
"""
from typing import List, Optional, Tuple, Union, Callable, TYPE_CHECKING
import json
import threading
import time

from engine.gameobject import GameObject
from engine.component import Script
from engine.types import Color, ColorType

if TYPE_CHECKING:
    from engine.ui.manager import UIManager


class Scene:
    """
    Base scene that can hold GameObjects and receive lifecycle / input callbacks.

    Subclass this (or Scene3D / Scene2D) to create different scenes
    (menu, game, pause screen, etc.).
    """

    def __init__(self):
        self.window = None
        self.objects: List[GameObject] = []
        # Updatables is a (usually much smaller) list of GameObjects that have
        # Scripts, Rigidbodies, Animators or active coroutines.  The Cython
        # game loop and other simulation code iterate only this list for very
        # large scenes full of passive objects.
        self._updatables: List[GameObject] = []
        # Opt-in phase lists (only objects whose scripts override the method)
        self._fixed_updatables: List[GameObject] = []
        self._late_updatables: List[GameObject] = []
        # Optional Cython-backed fast container (used for scans/rebuilds and
        # future direct C-level iteration).
        self._entity_container = None
        self._setup_done = False
        # Lazy-init canvas to avoid import issues at module level
        self._canvas: Optional['UIManager'] = None

    # -- UI canvas (lazy) ---------------------------------------------------

    @property
    def canvas(self):
        if self._canvas is None:
            from engine.ui.manager import UIManager
            self._canvas = UIManager(self)
        return self._canvas

    @canvas.setter
    def canvas(self, value):
        self._canvas = value

    # -- Window attachment --------------------------------------------------

    def _attach_window(self, window):
        self.window = window
        if not self._setup_done:
            self.setup()
            self._setup_done = True

    def _detach_window(self):
        self.on_hide()

    # -- Object management --------------------------------------------------

    def add_object(self, obj: GameObject, **kwargs) -> GameObject:
        position = kwargs.pop('position', None)
        rotation = kwargs.pop('rotation', None)
        scale = kwargs.pop('scale', None)

        if not isinstance(obj, GameObject):
            obj = GameObject(str(obj))

        if position is not None:
            obj.transform.position = position
        if rotation is not None:
            obj.transform.rotation = rotation
        if scale is not None:
            obj.transform.scale = scale

        self.objects.append(obj)
        obj._scene = self
        # Register for fast simulation path if this object carries behavior
        self._register_updatable_if_needed(obj)
        return obj

    def remove_object(self, obj: GameObject):
        if obj not in self.objects:
            return
        # Remove descendants first
        descendants = []
        def _collect(transform):
            for child in transform.children:
                if child.game_object in self.objects:
                    descendants.append(child.game_object)
                    _collect(child)
        _collect(obj.transform)
        for desc in descendants:
            if desc in self.objects:
                self.objects.remove(desc)
                if hasattr(desc, '_scene'):
                    desc._scene = None
                self._unregister_updatable(desc)
                self._unregister_fixed_updatable(desc)
                self._unregister_late_updatable(desc)
        self.objects.remove(obj)
        if hasattr(obj, '_scene'):
            obj._scene = None
        self._unregister_updatable(obj)
        self._unregister_fixed_updatable(obj)
        self._unregister_late_updatable(obj)

    def clear_objects(self):
        for obj in self.objects:
            if hasattr(obj, '_scene'):
                obj._scene = None
        self.objects.clear()
        if hasattr(self, '_updatables'):
            self._updatables.clear()
        if hasattr(self, '_fixed_updatables'):
            self._fixed_updatables.clear()
        if hasattr(self, '_late_updatables'):
            self._late_updatables.clear()

    # -- Fast entity/component container support ---------------------------

    def _register_updatable(self, obj: 'GameObject'):
        """Register a GameObject as requiring per-frame simulation work.

        This is called automatically when Scripts, Rigidbodies, Animators or
        coroutines are added.  The Cython fast path will iterate only the
        much smaller _updatables list instead of every object in the scene.
        """
        if not hasattr(self, '_updatables'):
            self._updatables = []
        if obj not in self._updatables:
            self._updatables.append(obj)

        # Also feed the Cython container when present (for advanced use / rebuilds)
        if self._entity_container is not None:
            try:
                self._entity_container._ensure_updatable(obj)
            except Exception as exc:
                from engine.log import get_logger
                get_logger("scene").debug("entity_container ensure failed: %s", exc)

    def _unregister_updatable(self, obj: 'GameObject'):
        """Remove an object from the updatables fast list (called on remove)."""
        if hasattr(self, '_updatables') and obj in self._updatables:
            self._updatables.remove(obj)

    def _register_fixed_updatable(self, obj: 'GameObject'):
        if not hasattr(self, '_fixed_updatables'):
            self._fixed_updatables = []
        if obj not in self._fixed_updatables:
            self._fixed_updatables.append(obj)

    def _unregister_fixed_updatable(self, obj: 'GameObject'):
        if hasattr(self, '_fixed_updatables') and obj in self._fixed_updatables:
            self._fixed_updatables.remove(obj)

    def _register_late_updatable(self, obj: 'GameObject'):
        if not hasattr(self, '_late_updatables'):
            self._late_updatables = []
        if obj not in self._late_updatables:
            self._late_updatables.append(obj)

    def _unregister_late_updatable(self, obj: 'GameObject'):
        if hasattr(self, '_late_updatables') and obj in self._late_updatables:
            self._late_updatables.remove(obj)

    def _register_updatable_if_needed(self, obj: 'GameObject'):
        """Check object state and register only if it has behavioral components."""
        # Delegate to GameObject so phase lists stay consistent
        if hasattr(obj, '_refresh_updatable_registration'):
            # Temporarily attach scene if not set (add_object sets it first)
            obj._refresh_updatable_registration()
            return
        if (getattr(obj, '_scripts_update', None) and len(obj._scripts_update) > 0) or \
           (getattr(obj, '_scripts', None) and len(obj._scripts) > 0) or \
           getattr(obj, '_active_coroutines', None) or \
           getattr(obj, '_end_of_frame_coroutines', None) or \
           getattr(obj, '_rigidbody', None) is not None or \
           getattr(obj, '_animator', None) is not None or \
           getattr(obj, '_particle_system', None) is not None:
            self._register_updatable(obj)

    def _ensure_entity_container(self):
        """Lazily create the Cython fast entity container if acceleration is on."""
        if self._entity_container is not None:
            return
        try:
            from engine.cython import CYTHON_ENABLED
            if not CYTHON_ENABLED:
                return
            from engine.cython.cy_entities import EntityContainer
            self._entity_container = EntityContainer()
            # Seed it with whatever we already have
            for o in self._updatables:
                self._entity_container._ensure_updatable(o)
        except Exception as exc:
            from engine.log import get_logger
            get_logger("scene").debug("entity container init failed: %s", exc)
            self._entity_container = None

    def get_objects_by_name(self, name: str) -> List[GameObject]:
        return [o for o in self.objects if o.name == name]

    def get_objects_by_tag(self, tag: str) -> List[GameObject]:
        return [o for o in self.objects if o.tag == tag]

    # -- Script lifecycle ---------------------------------------------------

    def start_components(self):
        for obj in self.objects:
            obj.start_components()

    def awake_components(self):
        for obj in self.objects:
            obj.awake_components()

    # -- Lifecycle (override) -----------------------------------------------

    def setup(self):
        pass

    def on_show(self):
        pass

    def on_hide(self):
        pass

    def on_update(self):
        pass

    def on_draw(self):
        pass

    # -- Input (override) ---------------------------------------------------

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
        pass

    # -- 2D drawing helpers (delegate to window) ----------------------------

    def draw_text(self, text, x, y, color=Color.WHITE, font_size=24,
                  font_name=None, anchor_x='left', anchor_y='top',
                  baseline_adjust=True):
        if self.window:
            self.window.draw_text(text, x, y, color, font_size, font_name,
                                  anchor_x, anchor_y, baseline_adjust)

    def draw_rectangle(self, x, y, width, height, color, border_width=0):
        if self.window:
            self.window.draw_rectangle(x, y, width, height, color, border_width)

    def draw_circle(self, x, y, radius, color, border_width=2, aa=True):
        if self.window:
            self.window.draw_circle(x, y, radius, color, border_width, aa)

    def draw_ellipse(self, x, y, width, height, color, border_width=2, aa=True):
        if self.window:
            self.window.draw_ellipse(x, y, width, height, color, border_width, aa)

    def draw_polygon(self, points, color, border_width=2, aa=True):
        if self.window:
            self.window.draw_polygon(points, color, border_width, aa)

    def draw_line(self, start, end, color, width=2, aa=True):
        if self.window:
            self.window.draw_line(start, end, color, width, aa)

    def draw_image(self, image, x, y, scale=1.0, alpha=1.0):
        if self.window:
            self.window.draw_image(image, x, y, scale, alpha)

    # -- Fast Cython entity/component container support ---------------------

    def rebuild_updatables(self):
        """
        Rebuild the _updatables list by scanning all objects.

        Uses the Cython EntityContainer (from cy_entities) for a fast C-level
        scan when Cython acceleration is available.  This is the main hook
        for the "fast container for very large numbers of objects".
        """
        self._ensure_entity_container()
        if self._entity_container is not None:
            try:
                self._updatables = self._entity_container.collect_updatables(self.objects)
                return
            except Exception:
                pass

        # Pure-Python fallback scan
        self._updatables = [
            obj for obj in self.objects
            if (getattr(obj, '_scripts', None) and len(obj._scripts) > 0) or
               getattr(obj, '_active_coroutines', None) or
               getattr(obj, '_end_of_frame_coroutines', None) or
               getattr(obj, '_rigidbody', None) is not None or
               getattr(obj, '_animator', None) is not None or
               getattr(obj, '_particle_system', None) is not None
        ]


class SceneManager:
    """
    Manages scene loading with both synchronous and asynchronous options.
    """
    ProgressCallback = Callable[[float], None]

    def __init__(self):
        self._current_load: Optional[threading.Thread] = None
        self._loading_progress: float = 0.0
        self._loaded_scene: Optional[Scene] = None
        self._loading_error: Optional[Exception] = None
        self._pending_callbacks: list = []
        self._lock = threading.Lock()

    def poll(self):
        with self._lock:
            callbacks = list(self._pending_callbacks)
            self._pending_callbacks.clear()
        for cb in callbacks:
            cb()

    @staticmethod
    def load_scene(path: str) -> 'Scene':
        # Delegate to Scene3D.load for backward compat
        from engine.d3.scene import Scene3D
        return Scene3D.load(path)

    def load_scene_async(self, path, on_progress=None, on_complete=None, on_error=None):
        from engine.d3.scene import Scene3D
        self._loading_progress = 0.0
        self._loaded_scene = None
        self._loading_error = None

        def _load():
            try:
                self._loading_progress = 0.1
                if on_progress:
                    on_progress(0.1)
                scene = Scene3D.load(path)
                self._loading_progress = 1.0
                self._loaded_scene = scene
                if on_progress:
                    on_progress(1.0)
                if on_complete:
                    with self._lock:
                        self._pending_callbacks.append(lambda: on_complete(scene))
            except Exception as e:
                self._loading_error = e
                if on_error:
                    with self._lock:
                        self._pending_callbacks.append(lambda: on_error(e))

        self._current_load = threading.Thread(target=_load, daemon=True)
        self._current_load.start()

    def get_loading_progress(self) -> float:
        if self._current_load is None or not self._current_load.is_alive():
            return 1.0 if self._loaded_scene else 0.0
        return self._loading_progress

    def is_loading(self) -> bool:
        return self._current_load is not None and self._current_load.is_alive()

    def get_loaded_scene(self):
        return self._loaded_scene

    def get_loading_error(self):
        return self._loading_error
