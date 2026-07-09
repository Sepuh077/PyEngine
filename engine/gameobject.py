from typing import List, Optional, Type, TypeVar, Generator, Any, Dict, Tuple, Union, TYPE_CHECKING
import importlib
import json
import uuid
from enum import Enum

import numpy as np

from engine.component import Component, Script, WaitForSeconds, WaitEndOfFrame, WaitForFrames, Time, Tag
from engine.transform import Transform
from engine.types import Vector3, Vector2

if TYPE_CHECKING:
    from engine.d3 import Scene3D

T = TypeVar('T', bound=Component)

# Global registry for resolving component references during deserialization
_component_ref_registry: Dict[str, 'GameObject'] = {}

# Global registry of all scenes (for static query methods)
_scenes_registry: List['Scene3D'] = []

class GameObject:
    def __init__(self, name: str = "GameObject", _id: Optional[str] = None):
        self.name = name
        self._tag: Optional[Union[str, Tag]] = None
        self.components: List[Component] = []
        
        # Unique ID for serialization (auto-generated, not exposed to users)
        self._id = _id if _id else str(uuid.uuid4())
        
        # Every GameObject has a Transform
        self.transform = Transform()
        self.add_component(self.transform)

        # Fast-path list: only Script subclasses (components with real update logic).
        # Populated automatically by add_component(); used by the Cython game loop
        # to skip objects whose components are all no-op (Transform, Object2D, …).
        self._scripts: List[Script] = []

        # Coroutines state
        self._active_coroutines: List[Dict[str, Any]] = []
        self._end_of_frame_coroutines: List[Dict[str, Any]] = []
        
        # Reference to the scene this object belongs to
        self._scene: Optional['Scene3D'] = None
        
        # Render layer for selective camera rendering
        # Import here to avoid circular imports at module level
        from engine.d3.camera import RenderLayer
        self._render_layer: RenderLayer = RenderLayer.DEFAULT
    
    @property
    def scene(self) -> Optional['Scene3D']:
        """Get the scene this GameObject belongs to."""
        return self._scene
    
    @property
    def render_layer(self):
        """Get the render layer for this object (used for selective camera rendering)."""
        return self._render_layer
    
    @render_layer.setter
    def render_layer(self, value):
        """Set the render layer for this object."""
        from engine.d3.camera import RenderLayer
        if isinstance(value, RenderLayer):
            self._render_layer = value
        else:
            self._render_layer = RenderLayer.DEFAULT

    @property
    def tag(self) -> Optional[str]:
        """Get the tag as a string (works with both Tag objects and strings)."""
        if self._tag is None:
            return None
        if isinstance(self._tag, Tag):
            return self._tag.name
        return str(self._tag)
    
    @tag.setter
    def tag(self, value: Optional[Union[str, Tag]]):
        """Set the tag (accepts string, Tag object, or None)."""
        if value is None:
            self._tag = None
        elif isinstance(value, Tag):
            self._tag = value
        else:
            # Store as string, but also register with Tag system
            self._tag = str(value)
            Tag.create(str(value))  # Auto-register string tags

    def add_component(self, component: Component) -> Component:
        component.game_object = self
        self.components.append(component)
        if isinstance(component, Script):
            self._scripts.append(component)
        component.on_attach()
        return component

    def start_coroutine(self, routine: Generator) -> Generator:
        """Starts a coroutine on this GameObject."""
        # Initial step to get first yield value
        try:
            yield_val = next(routine)
            self._queue_coroutine(routine, yield_val)
        except StopIteration:
            pass
        return routine

    def _queue_coroutine(self, routine: Generator, wait_instruction: Any):
        """Queues a coroutine based on wait instruction."""
        wait = wait_instruction
        # Treat None as wait one frame
        if wait is None:
            wait = WaitForFrames(1)
        
        entry = {
            "generator": routine,
            "wait_instruction": wait
        }
        if isinstance(wait, WaitEndOfFrame):
            self._end_of_frame_coroutines.append(entry)
        else:
            self._active_coroutines.append(entry)

    def _step_coroutines(
        self,
        coroutines: List[Dict[str, Any]],
        delta_time: float,
        allow_end_of_frame: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Advance coroutines and return active + end-of-frame queues."""
        still_active = []
        end_of_frame_queue = []
        for coro in coroutines:
            gen = coro["generator"]
            wait = coro["wait_instruction"]

            is_ready = True
            if isinstance(wait, WaitForSeconds):
                if not wait.is_done(delta_time):
                    is_ready = False
            elif isinstance(wait, WaitForFrames):
                if not wait.step():
                    is_ready = False

            if is_ready:
                try:
                    new_wait = next(gen)
                except StopIteration:
                    continue

                if new_wait is None:
                    new_wait = WaitForFrames(1)

                entry = {
                    "generator": gen,
                    "wait_instruction": new_wait,
                }
                if isinstance(new_wait, WaitEndOfFrame):
                    if allow_end_of_frame:
                        end_of_frame_queue.append(entry)
                    else:
                        still_active.append(entry)
                else:
                    still_active.append(entry)
            else:
                still_active.append(coro)
        
        return still_active, end_of_frame_queue

    def _update_coroutines(self, delta_time: float):
        """Processes active coroutines during the main update phase."""
        self._active_coroutines, new_end_of_frame = self._step_coroutines(
            self._active_coroutines,
            delta_time,
            allow_end_of_frame=True,
        )
        if new_end_of_frame:
            self._end_of_frame_coroutines.extend(new_end_of_frame)

    def _update_end_of_frame_coroutines(self, delta_time: float):
        """Processes coroutines waiting for end of frame."""
        self._end_of_frame_coroutines, deferred_end_of_frame = self._step_coroutines(
            self._end_of_frame_coroutines,
            delta_time,
            allow_end_of_frame=False,
        )
        if deferred_end_of_frame:
            self._end_of_frame_coroutines.extend(deferred_end_of_frame)

    def update(self):
        # Update components
        for comp in self.components:
            comp.update()
        
        # Update coroutines (main phase)
        self._update_coroutines(Time.delta_time)

    def update_end_of_frame(self):
        """Called by Window3D to process end-of-frame coroutines."""
        self._update_end_of_frame_coroutines(Time.delta_time)

    def start_components(self):
        """Call awake() and start() on all components that haven't been started yet."""
        for comp in self.components:
            # Call awake() first if not already awoken
            if not comp._awoken:
                comp.awake()
                comp._awoken = True
            # Then call start() if not already started
            if not comp._started:
                comp.start()
                comp._started = True

    def awake_components(self):
        """Call awake() on all components that haven't been awoken yet."""
        for comp in self.components:
            if not comp._awoken:
                comp.awake()
                comp._awoken = True

    def get_component(self, component_type: Type[T]) -> Optional[T]:
        for comp in self.components:
            if isinstance(comp, component_type):
                return comp
        return None

    def get_components(self, component_type: Type[T]) -> List[T]:
        return [comp for comp in self.components if isinstance(comp, component_type)]

    def __repr__(self):
        return f"GameObject(name='{self.name}', position={self.transform.position})"

    # =========================================================================
    # Prefab serialization
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Save this GameObject (components + start values) to a prefab file.
        """
        data = self._to_prefab_dict()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float]] = None,
    ) -> "GameObject":
        """
        Load a GameObject prefab from disk and return the created GameObject.
        
        Args:
            path: Path to the prefab file
            position: Optional position to set (x, y, z). Defaults to (0, 0, 0).
            rotation: Optional rotation to set (rx, ry, rz) in degrees. Defaults to (0, 0, 0).
        """
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        obj = cls._from_prefab_dict(data)
        
        # Apply position and rotation
        if position is not None:
            obj.transform.position = position
        if rotation is not None:
            obj.transform.rotation = rotation
        
        return obj

    def _to_prefab_dict(self) -> Dict[str, Any]:
        data = {
            "_id": self._id,
            "name": self.name,
            "tag": self.tag,
            "components": [self._component_to_prefab(comp, idx) for idx, comp in enumerate(self.components)],
        }
        # Store prefab source path if this GameObject is from a prefab
        if hasattr(self, '_prefab') and self._prefab is not None:
            if hasattr(self._prefab, 'path'):
                data["_prefab_path"] = self._prefab.path
        return data

    @classmethod
    def _from_prefab_dict(cls, data: Dict[str, Any]) -> "GameObject":
        game_object = cls(name=data.get("name", "GameObject"), _id=data.get("_id"))
        game_object.tag = data.get("tag")

        # Store prefab path for later restoration
        prefab_path = data.get("_prefab_path")
        if prefab_path:
            game_object._prefab_path = prefab_path

        components_data = data.get("components", [])
        for comp_data in components_data:
            component = cls._component_from_prefab(comp_data)
            if component is None:
                continue
            if isinstance(component, Transform):
                game_object.transform = component
                game_object.components[0] = component
                component.game_object = game_object
            else:
                game_object.add_component(component)

        return game_object

    @staticmethod
    def _component_to_prefab(component: Component, component_index: int = 0) -> Dict[str, Any]:
        comp_cls = component.__class__
        module_name = comp_cls.__module__
        class_name = comp_cls.__name__

        skip_keys = {
            "game_object",
            "_mesh",
            "mesh",
            "_vao",
            "_vbo",
            "_gl_texture",
            "_gpu_initialized",
            "_mesh_key",
            "_mesh_cache",
            "_texture_image",
            "_started",  # Script lifecycle state - should always start fresh
            "_awoken",   # Script lifecycle state - should always start fresh
        }

        is_object3d = module_name in {"src.engine.object3d", "engine.d3.object3d"} and class_name == "Object3D"
        if is_object3d:
            skip_keys = set(skip_keys)
            skip_keys.update({
                "_local_min",
                "_local_max",
                "_local_radius",
                "_uv",
            })

        is_collider = module_name.startswith("src.physics")
        if is_collider:
            skip_keys = set(skip_keys)
            skip_keys.update({
                "_current_collisions",
                "mesh_data",
                "sphere",
                "obb",
                "aabb",
                "cylinder",
                "_transform_dirty",
            })

        is_particle_system = module_name in {"src.engine.particle", "engine.d3.particle"} and class_name == "ParticleSystem"
        if is_particle_system:
            skip_keys = set(skip_keys)
            skip_keys.update({
                "_particles",
                "_container",
                "_playing",
                "_elapsed",
                "_emit_timer",
                "_rng",
            })

        is_transform = module_name in {"src.engine.transform", "engine.d3.transform", "engine.transform"} and class_name == "Transform"
        if is_transform:
            skip_keys = set(skip_keys)
            skip_keys.update({
                "_children",  # Rebuilt automatically when _parent is set on children
            })

        is_object2d = module_name in {"src.engine.d2.object2d", "engine.d2.object2d"} and class_name == "Object2D"
        if is_object2d:
            skip_keys = set(skip_keys)
            skip_keys.update({
                "_sprite_surface",  # live pygame Surface, not serializable; path is saved via 'sprite' InspectorField
                "_texture_dirty",
            })

        # Get the game object ID for component references
        game_object_id = component.game_object._id if component.game_object else None

        state = {
            key: GameObject._serialize_value(value, game_object_id, component_index)
            for key, value in component.__dict__.items()
            if key not in skip_keys
        }

        data = {
            "module": module_name,
            "class": class_name,
            "state": state,
        }
        return data

    @staticmethod
    def _component_from_prefab(data: Dict[str, Any]) -> Optional[Component]:
        module_name = data.get("module")
        class_name = data.get("class")
        state = data.get("state", {})
        if not module_name or not class_name:
            return None

        module = importlib.import_module(module_name)
        comp_cls = getattr(module, class_name, None)
        if comp_cls is None:
            raise ValueError(f"Component class '{class_name}' not found in {module_name}")

        component: Component = comp_cls()
        
        # First pass: deserialize without resolving component refs
        # Store the raw state for later resolution
        restored_state = GameObject._deserialize_value(state, None)  # None = no registry yet
        component.__dict__.update(restored_state)
        
        # Store the raw serialized state for second-pass resolution
        component._serialized_state = state

        if module_name in {"src.engine.object3d", "engine.d3.object3d"} and class_name == "Object3D":
            GameObject._restore_object3d_geometry(component)

        if module_name in {"src.engine.d2.object2d", "engine.d2.object2d"} and class_name == "Object2D":
            GameObject._restore_object2d_sprite(component)

        return component

    @staticmethod
    def _serialize_value(value: Any, source_go_id: str = None, source_comp_idx: int = 0) -> Any:
        # Handle Enum members
        if isinstance(value, Enum):
            return {
                "__type__": "enum",
                "enum_class": f"{value.__class__.__module__}.{value.__class__.__name__}",
                "name": value.name,
                "value": value.value,
            }
        
        # Handle GameObject references
        if isinstance(value, GameObject):
            return {
                "__type__": "gameobject_ref",
                "game_object_id": value._id,
                "game_object_name": value.name,
            }
        
        # Handle component references
        if isinstance(value, Component):
            if value.game_object:
                # Find the component index
                comp_idx = -1
                for idx, comp in enumerate(value.game_object.components):
                    if comp is value:
                        comp_idx = idx
                        break
                if comp_idx >= 0:
                    return {
                        "__type__": "component_ref",
                        "game_object_id": value.game_object._id,
                        "component_index": comp_idx,
                        "component_class": f"{value.__class__.__module__}.{value.__class__.__name__}",
                    }
            return None  # Component without game_object, can't reference
        
        if isinstance(value, np.ndarray):
            return {
                "__type__": "ndarray",
                "dtype": str(value.dtype),
                "value": value.tolist(),
            }
        # Quaternion serialization
        from engine.types.quaternion import Quaternion
        if isinstance(value, Quaternion):
            return {
                "__type__": "Quaternion",
                "value": value.to_list(),
            }
        if isinstance(value, Vector3):
            return {
                "__type__": "Vector3",
                "value": value.to_list(),
            }
        if isinstance(value, Vector2):
            return {
                "__type__": "Vector2",
                "value": [value.x, value.y],
            }
        if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            return value.item()
        try:
            from engine.d3.physics.group import ColliderGroup
        except ImportError:
            ColliderGroup = None
        if ColliderGroup is not None and isinstance(value, ColliderGroup):
            return {
                "__type__": "ColliderGroup",
                "name": value.name,
            }
        
        # Handle Viewport objects
        try:
            from engine.d3.camera import Viewport
        except ImportError:
            Viewport = None
        if Viewport is not None and isinstance(value, Viewport):
            return {
                "__type__": "Viewport",
                "x": value.x,
                "y": value.y,
                "width": value.width,
                "height": value.height,
            }

        try:
            from engine.graphics.material import Material
        except ImportError:
            Material = None
        if Material is not None and isinstance(value, Material):
            state = {
                k: GameObject._serialize_value(v, source_go_id, source_comp_idx)
                for k, v in value.__dict__.items()
            }
            return {
                "__type__": "Material",
                "class": value.__class__.__name__,
                "state": state,
            }
        
        # Handle ScriptableObject references
        try:
            from engine.scriptable_object import ScriptableObject
        except ImportError:
            ScriptableObject = None
        if ScriptableObject is not None and isinstance(value, ScriptableObject):
            return {
                "__type__": "ScriptableObject",
                "so_type": f"{value.__class__.__module__}.{value.__class__.__name__}",
                "so_name": value.name,
                "so_path": value.source_path,
            }

        # Handle serializable instances (classes decorated with @serializable)
        if hasattr(value, '__serializable__') and value.__serializable__:
            # Serialize all InspectorField values of this serializable instance
            serializable_fields = {}
            if hasattr(value, 'get_inspector_fields'):
                for field_name, _ in value.get_inspector_fields():
                    field_value = value.get_inspector_field_value(field_name)
                    serializable_fields[field_name] = GameObject._serialize_value(
                        field_value, source_go_id, source_comp_idx
                    )
            return {
                "__type__": "serializable",
                "class": f"{value.__class__.__module__}.{value.__class__.__name__}",
                "fields": serializable_fields,
            }

        if isinstance(value, dict):
            return {key: GameObject._serialize_value(val, source_go_id, source_comp_idx) for key, val in value.items()}
        if isinstance(value, list):
            return [GameObject._serialize_value(val, source_go_id, source_comp_idx) for val in value]
        if isinstance(value, set):
            return {
                "__type__": "set",
                "value": [GameObject._serialize_value(val, source_go_id, source_comp_idx) for val in value],
            }
        if isinstance(value, tuple):
            return {
                "__type__": "tuple",
                "value": [GameObject._serialize_value(val, source_go_id, source_comp_idx) for val in value],
            }
        if isinstance(value, bytes):
            return {
                "__type__": "bytes",
                "value": list(value),
            }
        
        # Handle ParticleBurst dataclass
        try:
            from engine.d3.particle import ParticleBurst
        except ImportError:
            ParticleBurst = None
        if ParticleBurst is not None and isinstance(value, ParticleBurst):
            return {
                "__type__": "ParticleBurst",
                "interval": value.interval,
                "count": value.count,
                "randomize": value.randomize,
            }
        
        # Handle ParticleShape subclasses
        try:
            from engine.d3.particle import ParticleShape, SphereShape, ConeShape, BoxShape
        except ImportError:
            ParticleShape = SphereShape = ConeShape = BoxShape = None
        if ParticleShape is not None and isinstance(value, ParticleShape):
            if isinstance(value, SphereShape):
                return {"__type__": "SphereShape"}
            elif isinstance(value, ConeShape):
                angle_deg = value.angle_rad * 180.0 / 3.14159265359 if hasattr(value, 'angle_rad') else 25.0
                return {
                    "__type__": "ConeShape",
                    "angle_degrees": angle_deg,
                    "direction": tuple(value.direction) if hasattr(value, 'direction') else (0.0, 1.0, 0.0),
                }
            elif isinstance(value, BoxShape):
                return {
                    "__type__": "BoxShape",
                    "size": tuple(value.size) if hasattr(value, 'size') else (1.0, 1.0, 1.0),
                    "direction": tuple(value.direction) if hasattr(value, 'direction') else (0.0, 1.0, 0.0),
                }
            else:
                return {"__type__": "SphereShape"}  # Default fallback
        
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            return {
                "__type__": "repr",
                "class": f"{value.__class__.__module__}.{value.__class__.__name__}",
                "value": repr(value),
            }
        return value

    @staticmethod
    def _deserialize_value(value: Any, go_registry: Dict[str, 'GameObject'] = None) -> Any:
        if isinstance(value, dict):
            # Handle GameObject references
            if value.get("__type__") == "gameobject_ref":
                if go_registry is None:
                    return None  # Can't resolve without registry
                go_id = value.get("game_object_id")
                return go_registry.get(go_id)
            
            # Handle component references
            if value.get("__type__") == "component_ref":
                if go_registry is None:
                    return None  # Can't resolve without registry
                go_id = value.get("game_object_id")
                comp_idx = value.get("component_index", 0)
                go = go_registry.get(go_id)
                if go and 0 <= comp_idx < len(go.components):
                    return go.components[comp_idx]
                return None
            
            # Handle Enum members
            if value.get("__type__") == "enum":
                enum_class_path = value.get("enum_class", "")
                enum_name = value.get("name")
                enum_value = value.get("value")
                
                try:
                    # Parse the class path (module.ClassName)
                    if '.' in enum_class_path:
                        module_name, class_name = enum_class_path.rsplit('.', 1)
                        module = importlib.import_module(module_name)
                        enum_cls = getattr(module, class_name, None)
                        if enum_cls and issubclass(enum_cls, Enum):
                            # Return the enum member by value
                            return enum_cls(enum_value)
                except Exception:
                    pass
                
                # Fallback: return the raw value if we can't reconstruct the enum
                return enum_value
            
            if value.get("__type__") == "ndarray":
                return np.array(value.get("value", []), dtype=value.get("dtype", None))
            if value.get("__type__") == "Quaternion":
                from engine.types.quaternion import Quaternion
                v = value.get("value", [1, 0, 0, 0])
                return Quaternion(v[0], v[1], v[2], v[3])
            if value.get("__type__") == "Vector3":
                return Vector3(value.get("value", [0, 0, 0]))
            if value.get("__type__") == "Vector2":
                v = value.get("value", [0, 0])
                return Vector2(v[0], v[1])
            if value.get("__type__") == "tuple":
                return tuple(GameObject._deserialize_value(val, go_registry) for val in value.get("value", []))
            if value.get("__type__") == "set":
                return set(GameObject._deserialize_value(val, go_registry) for val in value.get("value", []))
            if value.get("__type__") == "ColliderGroup":
                from engine.d3.physics.group import ColliderGroup
                name = value.get("name", "default")
                return ColliderGroup._registry.get(name) or ColliderGroup(name)
            if value.get("__type__") == "Viewport":
                from engine.d3.camera import Viewport
                return Viewport(
                    x=value.get("x", 0.0),
                    y=value.get("y", 0.0),
                    width=value.get("width", 1.0),
                    height=value.get("height", 1.0),
                )
            if value.get("__type__") == "Material":
                from engine.graphics import material
                class_name = value.get("class", "LitMaterial")
                state = value.get("state", {})
                mat_cls = getattr(material, class_name, material.LitMaterial)
                mat = mat_cls()
                restored_state = GameObject._deserialize_value(state, go_registry)
                mat.__dict__.update(restored_state)
                return mat
            
            # Handle ScriptableObject references
            if value.get("__type__") == "ScriptableObject":
                from engine.scriptable_object import ScriptableObject, ScriptableObjectMeta
                so_name = value.get("so_name")
                so_path = value.get("so_path")
                so_type = value.get("so_type")
                
                # First try to get from registry by name
                if so_name:
                    instance = ScriptableObject.get(so_name)
                    if instance:
                        return instance
                
                # Try to load from file path
                if so_path:
                    try:
                        import os
                        if os.path.exists(so_path):
                            instance = ScriptableObject.load(so_path)
                            return instance
                    except Exception:
                        pass
                
                # Try to find by type and name
                if so_type:
                    so_class = ScriptableObjectMeta.get_type(so_type)
                    if so_class:
                        instances = ScriptableObject.get_by_type(so_class)
                        for inst in instances:
                            if inst.name == so_name:
                                return inst
                
                return None  # Could not resolve
            
            if value.get("__type__") == "bytes":
                return bytes(value.get("value", []))
            if value.get("__type__") == "ParticleBurst":
                from engine.d3.particle import ParticleBurst
                return ParticleBurst(
                    interval=value.get("interval", 1.0),
                    count=value.get("count", 10),
                    randomize=value.get("randomize", False),
                )
            if value.get("__type__") == "SphereShape":
                from engine.d3.particle import SphereShape
                return SphereShape()
            if value.get("__type__") == "ConeShape":
                from engine.d3.particle import ConeShape
                return ConeShape(
                    angle_degrees=value.get("angle_degrees", 25.0),
                    direction=value.get("direction", (0.0, 1.0, 0.0)),
                )
            if value.get("__type__") == "BoxShape":
                from engine.d3.particle import BoxShape
                return BoxShape(
                    size=value.get("size", (1.0, 1.0, 1.0)),
                    direction=value.get("direction", (0.0, 1.0, 0.0)),
                )
            
            # Handle serializable instances (classes decorated with @serializable)
            if value.get("__type__") == "serializable":
                serializable_class_name = value.get("class", "")
                serializable_fields = value.get("fields", {})
                
                # Try to find and instantiate the serializable class
                serializable_instance = None
                try:
                    # Parse the class path (module.ClassName)
                    if '.' in serializable_class_name:
                        module_name, class_name = serializable_class_name.rsplit('.', 1)
                        module = importlib.import_module(module_name)
                        serializable_cls = getattr(module, class_name, None)
                    else:
                        serializable_cls = None
                    
                    if serializable_cls and hasattr(serializable_cls, '__serializable__') and serializable_cls.__serializable__:
                        # Create a new instance
                        serializable_instance = serializable_cls()
                        
                        # Restore field values
                        for field_name, field_value in serializable_fields.items():
                            deserialized_value = GameObject._deserialize_value(field_value, go_registry)
                            try:
                                serializable_instance.set_inspector_field_value(field_name, deserialized_value)
                            except (AttributeError, ValueError):
                                pass  # Field might not exist anymore
                except Exception:
                    pass  # If we can't reconstruct, return None
                
                return serializable_instance
            
            if value.get("__type__") == "repr":
                return value.get("value")
            return {key: GameObject._deserialize_value(val, go_registry) for key, val in value.items()}
        if isinstance(value, list):
            return [GameObject._deserialize_value(val, go_registry) for val in value]
        return value

    @staticmethod
    def _restore_object3d_geometry(component: Component) -> None:
        try:
            from engine.d3.object3d import Object3D
        except ImportError:
            return
        if not isinstance(component, Object3D):
            return

        source_type = getattr(component, "_source_type", "none")
        if source_type == "file":
            source_path = getattr(component, "_source_path", None)
            if source_path:
                component.load(source_path)
        elif source_type == "primitive":
            prim_type = getattr(component, "_primitive_type", None)
            params = getattr(component, "_primitive_params", {}) or {}
            if prim_type == "cube":
                size = params.get("size", 1.0)
                from engine.d3.object3d import create_cube
                temp_go = create_cube(size)
                temp_obj = temp_go.get_component(Object3D)
                if temp_obj:
                    component.mesh = temp_obj.mesh
                    component._post_process_geometry(f"primitive_cube_{size}")
            elif prim_type == "sphere":
                radius = params.get("radius", 1.0)
                subdivisions = params.get("subdivisions", 2)
                from engine.d3.object3d import create_sphere
                temp_go = create_sphere(radius, subdivisions=subdivisions)
                temp_obj = temp_go.get_component(Object3D)
                if temp_obj:
                    component.mesh = temp_obj.mesh
                    component._post_process_geometry(f"primitive_sphere_{radius}")
            elif prim_type == "plane":
                width = params.get("width", 10.0)
                height = params.get("height", 10.0)
                from engine.d3.object3d import create_plane
                temp_go = create_plane(width, height)
                temp_obj = temp_go.get_component(Object3D)
                if temp_obj:
                    component.mesh = temp_obj.mesh
                    component._post_process_geometry(f"primitive_plane_{width}_{height}")

    @staticmethod
    def _restore_object2d_sprite(component: Component) -> None:
        """After deserialization, reload the pygame Surface for an Object2D if it has a sprite path.
        The surface itself is not serialized (runtime resource); the path comes from the 'sprite' InspectorField
        or the descriptor's private storage. This prevents 'str' object in _sprite_surface after load.
        """
        try:
            from engine.d2.object2d import Object2D
        except ImportError:
            return
        if not isinstance(component, Object2D):
            return

        path = None
        try:
            if hasattr(component, "get_inspector_field_value"):
                path = component.get_inspector_field_value("sprite")
        except Exception:
            path = None

        if not path:
            path = getattr(component, "_inspector_sprite", None)

        if path:
            try:
                component._load_sprite(str(path))
            except Exception:
                # Missing image or load error: leave without surface (will use color only)
                pass

    # =========================================================================
    # Static Query Methods (Unity-like Find functions)
    # =========================================================================

    @classmethod
    def get_by_tag(cls, scene: 'Scene3D', tag: Union[str, Tag]) -> Optional['GameObject']:
        """
        Find the first GameObject with the specified tag in a scene.
        
        Args:
            scene: The scene to search in
            tag: Tag name (string) or Tag object to search for
            
        Returns:
            First GameObject with matching tag, or None if not found
        """
        tag_name = tag.name if isinstance(tag, Tag) else str(tag)
        for obj in scene.objects:
            if obj.tag == tag_name:
                return obj
        return None

    @classmethod
    def get_all_by_tag(cls, scene: 'Scene3D', tag: Union[str, Tag]) -> List['GameObject']:
        """
        Find all GameObjects with the specified tag in a scene.
        
        Args:
            scene: The scene to search in
            tag: Tag name (string) or Tag object to search for
            
        Returns:
            List of all GameObjects with matching tag
        """
        tag_name = tag.name if isinstance(tag, Tag) else str(tag)
        return [obj for obj in scene.objects if obj.tag == tag_name]

    @classmethod
    def get_by_type(cls, scene: 'Scene3D', component_type: Type[T]) -> Optional['GameObject']:
        """
        Find the first GameObject that has a component of the specified type.
        
        Args:
            scene: The scene to search in
            component_type: Component class to search for (e.g., Collider, Rigidbody, Camera3D)
            
        Returns:
            First GameObject with matching component type, or None if not found
        """
        for obj in scene.objects:
            if obj.get_component(component_type) is not None:
                return obj
        return None

    @classmethod
    def get_all_by_type(cls, scene: 'Scene3D', component_type: Type[T]) -> List['GameObject']:
        """
        Find all GameObjects that have a component of the specified type.
        
        Args:
            scene: The scene to search in
            component_type: Component class to search for (e.g., Collider, Rigidbody, Camera3D)
            
        Returns:
            List of all GameObjects with matching component type
        """
        return [obj for obj in scene.objects if obj.get_component(component_type) is not None]

    @classmethod
    def find_by_name(cls, scene: 'Scene3D', name: str) -> Optional['GameObject']:
        """
        Find the first GameObject with the specified name in a scene.
        
        Args:
            scene: The scene to search in
            name: Name to search for
            
        Returns:
            First GameObject with matching name, or None if not found
        """
        for obj in scene.objects:
            if obj.name == name:
                return obj
        return None

    @classmethod
    def find_all_by_name(cls, scene: 'Scene3D', name: str) -> List['GameObject']:
        """
        Find all GameObjects with the specified name in a scene.
        
        Args:
            scene: The scene to search in
            name: Name to search for
            
        Returns:
            List of all GameObjects with matching name
        """
        return [obj for obj in scene.objects if obj.name == name]


# =========================================================================
# Prefab System
# =========================================================================

class Prefab:
    """
    A prefab represents a template GameObject that can be instantiated multiple times.
    
    When a prefab is modified, all instances are automatically updated (except position).
    Each instance maintains its own position but shares all other properties with the prefab.
    
    Usage:
        # Create a prefab from a GameObject
        prefab = Prefab.create_from_gameobject(my_object, "path/to/prefab.prefab")
        
        # Instantiate the prefab in a scene
        instance = prefab.instantiate(scene, position=(1, 2, 3))
        
        # When you modify the prefab, all instances update
        prefab.update_from_gameobject(another_object)
        # Now all instances reflect the changes
    """
    
    # Global registry of all loaded prefabs (path -> Prefab)
    _registry: Dict[str, 'Prefab'] = {}
    
    def __init__(self, path: str):
        """
        Initialize a prefab from a file path.
        
        Args:
            path: Path to the .prefab file
        """
        self.path = path
        self._data: Optional[Dict[str, Any]] = None
        self._instances: List[GameObject] = []  # All instances created from this prefab
        
        # Load the data
        self._load()
        
        # Register this prefab
        Prefab._registry[path] = self
    
    @classmethod
    def create_from_gameobject(cls, game_object: GameObject, path: str) -> 'Prefab':
        """
        Create a new prefab from a GameObject and save it to disk.
        
        Args:
            game_object: The GameObject to create a prefab from
            path: Path to save the .prefab file
            
        Returns:
            The created Prefab instance
        """
        # Ensure .prefab extension
        if not path.endswith('.prefab'):
            path = path + '.prefab'
        
        # Create a copy of the game object data for the prefab
        # We need to create a fresh ID for the prefab template
        prefab_data = {
            "_id": str(uuid.uuid4()),
            "name": game_object.name,
            "tag": game_object.tag,
            "components": [GameObject._component_to_prefab(comp, idx) for idx, comp in enumerate(game_object.components)],
        }
        
        # Save to disk
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(prefab_data, handle, indent=2)
        
        # Create and return the prefab
        prefab = cls(path)
        prefab._data = prefab_data
        return prefab
    
    @classmethod
    def load(cls, path: str) -> 'Prefab':
        """
        Load a prefab from a file path. Returns cached instance if already loaded.
        
        Args:
            path: Path to the .prefab file
            
        Returns:
            The Prefab instance
        """
        # Check registry first
        if path in cls._registry:
            return cls._registry[path]
        
        return cls(path)
    
    def _load(self) -> None:
        """Load the prefab data from disk."""
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                self._data = json.load(handle)
        except FileNotFoundError:
            self._data = None
        except Exception as e:
            print(f"Error loading prefab {self.path}: {e}")
            self._data = None
    
    def reload(self) -> None:
        """Reload the prefab from disk and update all instances."""
        self._load()
        self._update_all_instances()
    
    def _update_all_instances(self) -> None:
        """Update all instances to reflect the current prefab data."""
        if self._data is None:
            return
        
        for instance in self._instances[:]:  # Copy list to avoid modification during iteration
            if instance is not None:
                self._apply_to_instance(instance)
    
    def _apply_to_instance(self, instance: GameObject) -> None:
        """
        Apply the prefab data to an instance, preserving its position.
        
        Args:
            instance: The GameObject instance to update
        """
        if self._data is None:
            return
        
        # Store current position (not shared)
        current_position = instance.transform.position
        current_parent = instance.transform.parent
        current_name = instance.name
        
        # Rebuild the instance from prefab data
        # We need to be careful not to break existing references
        
        # Update tag (shared across instances)
        instance.tag = self._data.get("tag")
        
        # Update components (except Transform which we handle specially)
        # Remove all non-Transform components
        instance.components = [c for c in instance.components if isinstance(c, Transform)]
        
        # Re-add components from prefab data
        components_data = self._data.get("components", [])
        for comp_data in components_data:
            component = GameObject._component_from_prefab(comp_data)
            if component is None:
                continue
            if isinstance(component, Transform):
                # Keep existing transform but update its properties (except position)
                # Actually we want to preserve the instance's transform entirely
                # So skip the transform from prefab
                continue
            else:
                instance.add_component(component)
        
        # Restore instance-specific fields (not shared)
        instance.transform.position = current_position
        instance.transform.parent = current_parent
        instance.name = current_name
        
        # Resolve component references
        # This is tricky - we need the scene's go_registry
        # For now, we'll skip this during instance update
        # It will be resolved when the scene is fully loaded
    
    def instantiate(self, scene=None, position: Optional[Tuple[float, float, float]] = None,
                    rotation: Optional[Tuple[float, float, float]] = None,
                    parent: Optional[Transform] = None) -> GameObject:
        """
        Create a new instance of this prefab.
        
        Args:
            scene: Optional scene to add the instance to
            position: Optional position for the instance (defaults to origin)
            rotation: Optional rotation for the instance
            parent: Optional parent transform
            
        Returns:
            The new GameObject instance
        """
        if self._data is None:
            raise ValueError(f"Cannot instantiate prefab: data not loaded from {self.path}")
        
        # Create a new GameObject from prefab data
        obj = GameObject._from_prefab_dict(self._data)
        
        # Set position
        if position is not None:
            obj.transform.position = position
        else:
            obj.transform.position = (0, 0, 0)
        
        # Set rotation
        if rotation is not None:
            obj.transform.rotation = rotation
        
        # Set parent
        if parent is not None:
            obj.transform.parent = parent
        
        # Register as an instance
        self._instances.append(obj)
        obj._prefab = self  # Store reference to prefab
        
        # Add to scene if provided
        if scene is not None:
            scene.add_object(obj)
        
        return obj
    
    def register_instance(self, instance: GameObject) -> None:
        """
        Register an existing GameObject as an instance of this prefab.
        
        Args:
            instance: The GameObject to register as an instance
        """
        if instance not in self._instances:
            self._instances.append(instance)
            instance._prefab = self
    
    def unregister_instance(self, instance: GameObject) -> None:
        """
        Unregister a GameObject from being an instance of this prefab.
        
        Args:
            instance: The GameObject to unregister
        """
        if instance in self._instances:
            self._instances.remove(instance)
            if hasattr(instance, '_prefab'):
                delattr(instance, '_prefab')
    
    def save(self) -> None:
        """Save the current prefab data to disk."""
        if self._data is None:
            return
        
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2)
    
    def update_from_gameobject(self, game_object: GameObject) -> None:
        """
        Update this prefab from a GameObject, save to disk, and update all instances.
        
        Args:
            game_object: The GameObject to use as the new prefab template
        """
        # Create new prefab data
        self._data = {
            "_id": self._data.get("_id", str(uuid.uuid4())) if self._data else str(uuid.uuid4()),
            "name": game_object.name,
            "tag": game_object.tag,
            "components": [GameObject._component_to_prefab(comp, idx) for idx, comp in enumerate(game_object.components)],
        }
        
        # Save to disk
        self.save()
        
        # Update all instances
        self._update_all_instances()
    
    def apply_field_to_instances(
        self,
        component_class_name: str,
        field_name: str,
        value: Any,
        exclude_instance: Optional[GameObject] = None,
    ) -> None:
        """
        Apply a specific field change to all instances without rebuilding components.
        
        This is much more efficient than _update_all_instances() because it only
        changes one field value instead of destroying and recreating all components.
        
        Args:
            component_class_name: The class name of the component to update
            field_name: The name of the field to change
            value: The new value to set
            exclude_instance: Optional instance to skip (e.g. the one being edited)
        """
        for instance in self._instances[:]:
            if instance is None or instance is exclude_instance:
                continue
            for comp in instance.components:
                if type(comp).__name__ == component_class_name:
                    try:
                        comp.set_inspector_field_value(field_name, value)
                    except Exception:
                        try:
                            setattr(comp, field_name, value)
                        except Exception:
                            pass
                    break
    
    @property
    def name(self) -> str:
        """Get the name of this prefab."""
        if self._data:
            return self._data.get("name", "Prefab")
        return "Prefab"
    
    @property
    def instances(self) -> List[GameObject]:
        """Get all instances of this prefab."""
        return self._instances[:]
    
    @classmethod
    def get_prefab_for_path(cls, path: str) -> Optional['Prefab']:
        """Get a prefab by its file path."""
        return cls._registry.get(path)
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear the prefab registry."""
        cls._registry.clear()
