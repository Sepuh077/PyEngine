"""
Scene2D - A 2D scene that extends the shared Scene base.
"""
from typing import List, Optional, Tuple, TYPE_CHECKING

from engine.scene import Scene
from engine.gameobject import GameObject
from engine.d2.camera2d import Camera2D
from engine.d2.object2d import Object2D

if TYPE_CHECKING:
    from engine.d2.window2d import Window2D


class Scene2D(Scene):
    """
    A 2D scene that can be displayed in a Window2D.

    Subclass this to create different scenes (menu, game, pause screen, etc.)

    Example::

        class GameScene(Scene2D):
            def setup(self):
                self.player = self.add_object(create_rect(32, 32, color=(0, 0.8, 1)))

            def on_update(self):
                speed = 200 * Time.delta_time
                if self.window.is_key_pressed(Keys.RIGHT):
                    self.player.transform.position += (speed, 0, 0)

            def on_key_press(self, key, modifiers):
                if key == Keys.ESCAPE:
                    self.window.close()
    """

    def __init__(self):
        super().__init__()
        self._cameras: List[Camera2D] = []
        self._main_camera: Optional[Camera2D] = None

        # Create default orthographic camera at the origin (Unity-like orthographic_size for viewport)
        cam_obj = GameObject("Main Camera")
        camera = Camera2D(orthographic_size=5.0, is_main=True)
        cam_obj.add_component(camera)
        self.add_object(cam_obj)
        self._main_camera = camera

    # -- Camera management --------------------------------------------------

    @property
    def main_camera(self) -> Camera2D:
        """Get the main camera."""
        if self._main_camera:
            return self._main_camera
        for cam in self._cameras:
            if cam.is_main:
                self._main_camera = cam
                return cam
        if self._cameras:
            return self._cameras[0]
        # Fallback
        return Camera2D()

    @main_camera.setter
    def main_camera(self, camera: Camera2D):
        for cam in self._cameras:
            cam._is_main = False
        if camera in self._cameras:
            camera._is_main = True
            self._main_camera = camera
        elif camera.game_object and camera.game_object in self.objects:
            self._cameras.append(camera)
            camera._is_main = True
            self._main_camera = camera

    @property
    def camera(self) -> Camera2D:
        """Alias for main_camera."""
        return self.main_camera

    @camera.setter
    def camera(self, value: Camera2D):
        self.main_camera = value

    @property
    def cameras(self) -> List[Camera2D]:
        """All cameras in this scene."""
        return self._cameras.copy()

    # -- Object management (2D override) ------------------------------------

    def add_object(self, obj, **kwargs) -> GameObject:
        """Add a GameObject (or Object2D) to the scene."""
        if isinstance(obj, Object2D):
            go = GameObject()
            go.add_component(obj)
        elif not isinstance(obj, GameObject):
            go = GameObject(str(obj))
        else:
            go = obj

        go = super().add_object(go, **kwargs)

        # Register Camera2D components
        for cam in go.get_components(Camera2D):
            if cam not in self._cameras:
                self._cameras.append(cam)
                if self._main_camera is None:
                    self._main_camera = cam

        return go

    def remove_object(self, obj: GameObject):
        if obj not in self.objects:
            return

        # Unregister cameras
        for cam in obj.get_components(Camera2D):
            if cam in self._cameras:
                self._cameras.remove(cam)
                if self._main_camera == cam:
                    self._main_camera = None
                    for c in self._cameras:
                        if c.is_main:
                            self._main_camera = c
                            break
                    if self._main_camera is None and self._cameras:
                        self._main_camera = self._cameras[0]

        super().remove_object(obj)

    def clear_objects(self):
        super().clear_objects()
        self._cameras.clear()
        self._main_camera = None

    # -- Render helpers -----------------------------------------------------

    def get_sorted_renderables(self) -> List[Object2D]:
        """Get all visible Object2D components sorted by (layer_id, sorting_order)."""
        renderables = []
        for obj in self.objects:
            obj2d = obj.get_component(Object2D)
            if obj2d and obj2d.visible:
                renderables.append(obj2d)
        renderables.sort(key=lambda o: o.sort_key)
        return renderables
