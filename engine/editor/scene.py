import json
from typing import Dict

from engine.d3.scene import Scene3D
from engine.d2.scene2d import Scene2D
from engine.d2.camera2d import Camera2D
from engine.gameobject import GameObject


class EditorScene(Scene3D):
    def __init__(self) -> None:
        super().__init__()
        self.editor_label = "Untitled Scene"


class EditorScene2D(Scene2D):
    """2D editor scene with save/load support."""

    def __init__(self) -> None:
        super().__init__()
        self.editor_label = "Untitled Scene"

    # -- Serialization ------------------------------------------------------

    def save(self, path: str) -> None:
        data = self._to_scene_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "EditorScene2D":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_scene_dict(data)

    def _to_scene_dict(self) -> dict:
        cam = self.main_camera
        cam_pos = cam.position
        # Exclude particle system particles and the main camera object
        # (camera is saved separately in the "camera" key)
        cam_go = cam.game_object if cam else None
        visible_objects = [
            obj for obj in self.objects
            if not getattr(obj, '_is_particle_system_particle', False)
            and obj is not cam_go
        ]
        return {
            "_mode": "2d",
            "camera": {
                "x": float(cam_pos.x),
                "y": float(cam_pos.y),
                "zoom": float(cam.zoom),
                "orthographic_size": float(getattr(cam, 'orthographic_size', 5.0)),
            },
            "objects": [obj._to_prefab_dict() for obj in visible_objects],
        }

    @classmethod
    def _from_scene_dict(cls, data: dict) -> "EditorScene2D":
        scene = cls()
        scene.clear_objects()

        cam_data = data.get("camera", {})
        if cam_data:
            cam_obj = GameObject("Main Camera")
            zoom_val = cam_data.get("zoom", 1.0)
            size_val = cam_data.get("orthographic_size", 5.0)
            cam = Camera2D(zoom=zoom_val, orthographic_size=size_val, is_main=True)
            cam_obj.add_component(cam)
            cam_obj.transform.position = (
                cam_data.get("x", 0.0),
                cam_data.get("y", 0.0),
                0.0,
            )
            scene.add_object(cam_obj)

        go_registry: Dict[str, GameObject] = {}
        for obj_data in data.get("objects", []):
            obj = GameObject._from_prefab_dict(obj_data)
            # Skip camera-only objects (avoid duplicates from old saves)
            if obj.get_component(Camera2D) and len(obj.components) <= 2:
                continue
            obj._scene = scene
            scene.objects.append(obj)
            go_registry[obj._id] = obj

        # Register any additional cameras from loaded objects
        for obj in scene.objects:
            for cam in obj.get_components(Camera2D):
                if cam not in scene._cameras:
                    scene._cameras.append(cam)

        return scene
