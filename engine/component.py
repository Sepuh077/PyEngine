from typing import Optional, TYPE_CHECKING, Generator, Type, TypeVar, List, Any, Generic, Tuple, Union, overload, Set, Dict
from dataclasses import dataclass
from enum import Enum

from engine.types import Color as ColorType
from engine.types import Vector3

if TYPE_CHECKING:
    from engine.gameobject import GameObject


T = TypeVar('T', bound="Component")
_T = TypeVar('_T')


# =========================================================================
# Tag System (Unity-like)
# =========================================================================

class Tag:
    """
    A tag that can be assigned to GameObjects for categorization.
    
    Similar to Unity's Tag system, tags allow you to identify groups of
    GameObjects for filtering, querying, and game logic.
    
    Tags can be created dynamically or registered with TagManager for
    centralized management.
    
    Example:
        # Create a tag
        player_tag = Tag("Player")
        enemy_tag = Tag("Enemy")
        
        # Assign to GameObject
        player.tag = player_tag
        # Or just use string
        player.tag = "Player"
        
        # Query by tag
        players = GameObject.get_all_by_tag(scene, "Player")
    """
    
    # Registry of all known tags (name -> Tag instance)
    _registry: Dict[str, "Tag"] = {}
    
    def __init__(self, name: str):
        """
        Create a new tag with the given name.
        
        Args:
            name: The tag name (e.g., "Player", "Enemy", "Collectible")
        """
        self.name = name
        # Auto-register
        Tag._registry[name] = self
    
    @classmethod
    def create(cls, name: str) -> "Tag":
        """Create and register a new tag."""
        return cls(name)
    
    @classmethod
    def get_or_create(cls, name: str) -> "Tag":
        """Get existing tag or create new one."""
        if name in cls._registry:
            return cls._registry[name]
        return cls(name)
    
    @classmethod
    def all_tags(cls) -> List[str]:
        """Get list of all registered tag names."""
        return sorted(cls._registry.keys())
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered tags."""
        cls._registry.clear()
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"Tag('{self.name}')"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Tag):
            return self.name == other.name
        if isinstance(other, str):
            return self.name == other
        return False
    
    def __hash__(self) -> int:
        return hash(self.name)


# Convenience: Pre-defined common tags
Tag.create("Player")
Tag.create("Enemy")
Tag.create("Collectible")
Tag.create("Ground")
Tag.create("UI")
Tag.create("Camera")


def serializable(cls: type = None, *, name: Optional[str] = None) -> type:
    """
    Decorator to mark a class as serializable for use in InspectorField.
    
    Classes decorated with @serializable can be used as types in InspectorField,
    and their InspectorField members will be shown as nested/expandable fields
    in the inspector.
    
    Example:
        @serializable
        class WeaponStats:
            damage = InspectorField(float, default=10.0)
            attack_speed = InspectorField(float, default=1.0)
        
        class MyScript(Script):
            weapon = InspectorField(WeaponStats, default=None)
            # In the inspector, 'weapon' will show 'damage' and 'attack_speed' as sub-fields
    
    Args:
        cls: The class to decorate (when used without parentheses)
        name: Optional custom name for the serializable type (defaults to class name)
    
    Returns:
        The decorated class with __serializable__ marker and helper methods
    """
    def decorator(cls: type) -> type:
        # Mark the class as serializable
        cls.__serializable__ = True
        cls.__serializable_name__ = name or cls.__name__
        
        # Add get_inspector_fields method if not already present
        if not hasattr(cls, 'get_inspector_fields'):
            @classmethod
            def get_inspector_fields(cls_inner: type) -> List[Tuple[str, InspectorFieldInfo]]:
                """
                Get all inspector fields defined on this serializable class.
                
                Returns a list of (field_name, InspectorFieldInfo) tuples.
                """
                fields = []
                seen = set()
                
                # Walk through the MRO to get inherited fields
                for klass in reversed(cls_inner.__mro__):
                    for attr_name, value in vars(klass).items():
                        if attr_name in seen:
                            continue
                        if isinstance(value, InspectorField):
                            fields.append((attr_name, value.get_info()))
                            seen.add(attr_name)
                
                return fields
            
            cls.get_inspector_fields = get_inspector_fields
        
        # Add get_inspector_field_value and set_inspector_field_value if not present
        if not hasattr(cls, 'get_inspector_field_value'):
            def get_inspector_field_value(self, field_name: str) -> Any:
                """Get the value of an inspector field."""
                descriptor = getattr(type(self), field_name, None)
                if isinstance(descriptor, InspectorField):
                    return getattr(self, field_name)
                return None
            
            cls.get_inspector_field_value = get_inspector_field_value
        
        if not hasattr(cls, 'set_inspector_field_value'):
            def set_inspector_field_value(self, field_name: str, value: Any) -> None:
                """Set the value of an inspector field."""
                descriptor = getattr(type(self), field_name, None)
                if isinstance(descriptor, InspectorField):
                    setattr(self, field_name, value)
            
            cls.set_inspector_field_value = set_inspector_field_value
        
        return cls
    
    # Handle both @serializable and @serializable() usage
    if cls is not None:
        return decorator(cls)
    return decorator


class InspectorFieldType(Enum):
    """Types of inspector fields supported."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    COLOR = "color"
    VECTOR3 = "vector3"
    ENUM = "enum"
    LIST = "list"
    COMPONENT_REF = "component_ref"
    GAMEOBJECT_REF = "gameobject_ref"
    MATERIAL_REF = "material_ref"
    SCRIPTABLE_OBJECT_REF = "scriptable_object_ref"
    SERIALIZABLE = "serializable"


@dataclass
class InspectorFieldInfo:
    """
    Metadata for an inspector field.
    
    Attributes:
        name: Display name in the inspector
        field_type: Type of the field (float, int, bool, etc.)
        default_value: Default value for the field
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        step: Step increment (for numeric types)
        decimals: Number of decimal places (for float types)
        enum_options: List of (value, label) tuples for enum types
        enum_type: The Enum subclass type (for ENUM type)
        tooltip: Optional tooltip text
        list_item_type: The type of items in a list field (for LIST type)
        scriptable_object_type: The ScriptableObject subclass type (for SCRIPTABLE_OBJECT_REF type)
        serializable_type: The serializable class type (for SERIALIZABLE type)
    """
    name: str
    field_type: InspectorFieldType
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    decimals: Optional[int] = None
    enum_options: Optional[List[Tuple[Any, str]]] = None
    enum_type: Optional[type] = None
    tooltip: Optional[str] = None
    list_item_type: Optional[Union[type, InspectorFieldType]] = None
    scriptable_object_type: Optional[type] = None
    serializable_type: Optional[type] = None


class InspectorField(Generic[_T]):
    """
    Descriptor for defining inspector-visible fields on components.
    
    This allows fields to be used naturally in code while providing
    metadata for the inspector UI.
    
    Supported types:
        - float, int, bool, str: Basic types
        - list: List of items (use list_item_type to specify item type)
        - Color: RGB/RGBA tuple in 0-1 range
        - Vector3: 3D position/direction
        - GameObject: Reference to a GameObject
        - Component subclasses: Reference to a component
        - Enum subclasses: Automatically generates dropdown options
    
    Example:
        from engine.d3 import Script, InspectorField, Color, Vector3, GameObject
        
        class MyScript(Script):
            # Basic types
            speed = InspectorField(float, default=5.0, min_value=0.0, max_value=100.0)
            health = InspectorField(int, default=100)
            enabled = InspectorField(bool, default=True)
            name = InspectorField(str, default="Player")
            
            # Color and Vector3
            player_color = InspectorField(Color, default=(1.0, 0.0, 0.0))
            spawn_pos = InspectorField(Vector3, default=(0.0, 0.0, 0.0))
            
            # List fields
            scores = InspectorField(list, default=[], list_item_type=int)
            
            # Component reference
            target_transform = InspectorField(Transform, default=None)
            
            # GameObject reference
            target_object = InspectorField(GameObject, default=None)
            
            def update(self):
                self.transform.position[0] += self.speed * Time.delta_time
    """
    
    def __init__(
        self,
        field_type: Union[Type[_T], type, InspectorFieldType],
        default: Any = None,
        *,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: Optional[float] = None,
        decimals: Optional[int] = None,
        enum_options: Optional[List[Tuple[Any, str]]] = None,
        tooltip: Optional[str] = None,
        list_item_type: Optional[Union[type, InspectorFieldType]] = None,
    ):
        """
        Initialize an inspector field.
        
        Args:
            field_type: The type of the field (float, int, bool, str, Color, Vector3, 
                       GameObject, Component subclass, list, or InspectorFieldType enum value).
            default: Default value for the field (if None, a sensible default is used)
            min_value: Minimum value for numeric types
            max_value: Maximum value for numeric types
            step: Step increment for numeric types
            decimals: Decimal places for float types
            enum_options: List of (value, label) tuples for enum types
            tooltip: Tooltip text for the inspector
            list_item_type: The type of items in a list (for list fields)
        """
        # Store the original field type for component references
        self._original_field_type = None
        
        # Convert Python type to InspectorFieldType
        if isinstance(field_type, InspectorFieldType):
            self.field_type = field_type
        elif field_type == float:
            self.field_type = InspectorFieldType.FLOAT
        elif field_type == int:
            self.field_type = InspectorFieldType.INT
        elif field_type == bool:
            self.field_type = InspectorFieldType.BOOL
        elif field_type == str:
            self.field_type = InspectorFieldType.STRING
        elif field_type == list:
            self.field_type = InspectorFieldType.LIST
        elif field_type is ColorType:
            self.field_type = InspectorFieldType.COLOR
        elif field_type is Vector3:
            self.field_type = InspectorFieldType.VECTOR3
        elif isinstance(field_type, type):
            # Check for GameObject first (before Component, since we need to import it)
            # Use string comparison to avoid circular import
            if field_type.__name__ == 'GameObject' and 'gameobject' in field_type.__module__:
                self.field_type = InspectorFieldType.GAMEOBJECT_REF
            elif field_type.__name__ == 'Material' and 'graphics.material' in field_type.__module__:
                self.field_type = InspectorFieldType.MATERIAL_REF
                self._original_field_type = field_type
            elif field_type.__name__ in ('SkyboxMaterial',) and 'graphics.material' in field_type.__module__:
                self.field_type = InspectorFieldType.MATERIAL_REF
                self._original_field_type = field_type
            elif issubclass(field_type, Enum):
                self.field_type = InspectorFieldType.ENUM
                # Store the enum class for serialization and type info
                self._original_field_type = field_type
                # Auto-generate enum options if not provided
                if enum_options is None:
                    enum_options = [(e.value, e.name) for e in field_type]
            elif issubclass(field_type, Component):
                self.field_type = InspectorFieldType.COMPONENT_REF
                # Store the original component type for reference
                self._original_field_type = field_type
            else:
                # Check if it's a serializable class (marked with @serializable decorator)
                if hasattr(field_type, '__serializable__') and field_type.__serializable__:
                    self.field_type = InspectorFieldType.SERIALIZABLE
                    self._original_field_type = field_type
                else:
                    # Check if it's a ScriptableObject subclass
                    # Use string comparison to avoid circular import
                    try:
                        from engine.scriptable_object import ScriptableObject
                        if issubclass(field_type, ScriptableObject):
                            self.field_type = InspectorFieldType.SCRIPTABLE_OBJECT_REF
                            self._original_field_type = field_type
                        else:
                            raise ValueError(f"Unsupported field type: {field_type}")
                    except ImportError:
                        raise ValueError(f"Unsupported field type: {field_type}")
        elif isinstance(field_type, str):
            # String type name for deferred/lazy resolution (e.g., "SkyboxMaterial")
            # Store for editor to resolve later
            self._custom_type_name = field_type
            if "Material" in field_type or "Material" in field_type.lower():
                self.field_type = InspectorFieldType.MATERIAL_REF
            else:
                # Default to component ref for unknown custom types
                self.field_type = InspectorFieldType.COMPONENT_REF
        else:
            raise ValueError(f"Unsupported field type: {field_type}")
        
        # Set sensible default values if not provided
        if default is None:
            default = self._get_type_default()
        
        self.default_value = default
        self.min_value = min_value
        self.max_value = max_value
        self.step = step
        self.decimals = decimals
        self.enum_options = enum_options
        self.tooltip = tooltip
        self.list_item_type = list_item_type
        
        # Will be set by __set_name__
        self.name: Optional[str] = None
        self.private_name: Optional[str] = None
    
    def _get_type_default(self) -> Any:
        """Get the default value for this field type when none is specified."""
        if self.field_type == InspectorFieldType.FLOAT:
            return 0.0
        elif self.field_type == InspectorFieldType.INT:
            return 0
        elif self.field_type == InspectorFieldType.BOOL:
            return False
        elif self.field_type == InspectorFieldType.STRING:
            return ""
        elif self.field_type == InspectorFieldType.COLOR:
            return (1.0, 1.0, 1.0)  # White
        elif self.field_type == InspectorFieldType.VECTOR3:
            return (0.0, 0.0, 0.0)
        elif self.field_type == InspectorFieldType.ENUM:
            # Return first enum option value if available
            if self.enum_options:
                return self.enum_options[0][0]
            return None
        elif self.field_type == InspectorFieldType.LIST:
            return []  # Empty list by default
        elif self.field_type == InspectorFieldType.COMPONENT_REF:
            return None  # No component reference by default
        elif self.field_type == InspectorFieldType.GAMEOBJECT_REF:
            return None  # No GameObject reference by default
        elif self.field_type == InspectorFieldType.MATERIAL_REF:
            return None  # No material reference by default
        elif self.field_type == InspectorFieldType.SCRIPTABLE_OBJECT_REF:
            return None  # No ScriptableObject reference by default
        elif self.field_type == InspectorFieldType.SERIALIZABLE:
            # For serializable types, return None by default (user creates instance)
            # The editor will create an instance when needed
            return None
        return None
    
    def __set_name__(self, owner: type, name: str):
        """Called when the descriptor is assigned to a class attribute."""
        self.name = name
        self.private_name = f"_inspector_{name}"
    
    @overload
    def __get__(self, obj: None, objtype: Optional[type] = None) -> "InspectorField[_T]": ...

    @overload
    def __get__(self, obj: Any, objtype: Optional[type] = None) -> _T: ...

    def __get__(self, obj: Any, objtype: Optional[type] = None) -> Union[_T, "InspectorField[_T]"]:
        """Get the field value, returning the default if not set."""
        if obj is None:
            return self
        
        value = getattr(obj, self.private_name, None)
        if value is None:
            return self.default_value
        return value
    
    def __set__(self, obj: Any, value: _T):
        """Set the field value.
        
        For enum fields, convert int values to enum members automatically.
        """
        # Convert int to enum member for ENUM fields
        if self.field_type == InspectorFieldType.ENUM and isinstance(value, int):
            if self._original_field_type:
                try:
                    value = self._original_field_type(value)
                except ValueError:
                    pass  # Invalid value, keep as-is
        setattr(obj, self.private_name, value)
    
    def get_info(self) -> InspectorFieldInfo:
        """Get the metadata for this inspector field."""
        return InspectorFieldInfo(
            name=self.name or "",
            field_type=self.field_type,
            default_value=self.default_value,
            min_value=self.min_value,
            max_value=self.max_value,
            step=self.step,
            decimals=self.decimals,
            enum_options=self.enum_options,
            enum_type=self._original_field_type if self.field_type == InspectorFieldType.ENUM else None,
            tooltip=self.tooltip,
            list_item_type=self.list_item_type,
            scriptable_object_type=self._original_field_type if self.field_type == InspectorFieldType.SCRIPTABLE_OBJECT_REF else None,
            serializable_type=self._original_field_type if self.field_type == InspectorFieldType.SERIALIZABLE else None,
        )
    
    @property
    def component_type(self) -> Optional[type]:
        """Get the component type for COMPONENT_REF fields."""
        return self._original_field_type
    
    @property
    def scriptable_object_type(self) -> Optional[type]:
        """Get the ScriptableObject type for SCRIPTABLE_OBJECT_REF fields."""
        if self.field_type == InspectorFieldType.SCRIPTABLE_OBJECT_REF:
            return self._original_field_type
        return None
    
    @property
    def serializable_type(self) -> Optional[type]:
        """Get the serializable type for SERIALIZABLE fields."""
        if self.field_type == InspectorFieldType.SERIALIZABLE:
            return self._original_field_type
        return None
    
    @property
    def enum_type(self) -> Optional[type]:
        """Get the enum type for ENUM fields."""
        if self.field_type == InspectorFieldType.ENUM:
            return self._original_field_type
        return None


def inspector_field(
    field_type: Union[type, InspectorFieldType],
    default: Any = None,
    *,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    step: Optional[float] = None,
    decimals: Optional[int] = None,
    enum_options: Optional[List[Tuple[Any, str]]] = None,
    tooltip: Optional[str] = None,
    list_item_type: Optional[Union[type, InspectorFieldType]] = None,
) -> InspectorField:
    """
    Convenience function to create an InspectorField.
    
    This is equivalent to using InspectorField directly but may be more
    readable in some contexts.
    """
    return InspectorField(
        field_type,
        default,
        min_value=min_value,
        max_value=max_value,
        step=step,
        decimals=decimals,
        enum_options=enum_options,
        tooltip=tooltip,
        list_item_type=list_item_type,
    )


class Time:
    """
    Global time utility similar to Unity's Time class.

    Frame timing:
        - ``delta_time`` / ``unscaled_delta_time`` — last frame duration.
        - ``time`` — total scaled elapsed time since start.

    Hitch clamp (``maximum_delta_time``):
        This is a **ceiling on large dt only** (slow frames / freezes). It does
        **not** limit high FPS. At 200 FPS, raw dt ≈ 0.005 s, which is far
        below the ceiling and is never touched.

        Only when a frame takes longer than the ceiling (e.g. asset load hitch)
        is dt cut down so scripts/physics don't simulate a multi-second jump
        in one update.

        Set ``maximum_delta_time = 0`` (or None) to disable the ceiling entirely.

    Physics (fixed step):
        - ``fixed_delta_time`` — physics sub-step size (default 1/60 s).
          Independent of render FPS: at 200 FPS you still integrate physics
          at ~60 Hz (accumulator). Set to 0 for one variable-dt physics step
          per rendered frame.
        - ``maximum_physics_steps`` — max sub-steps per frame (spiral-of-death).
        - ``fixed_time`` — total fixed-step simulation time.
        - ``_skip_rigidbody_frame_update`` — when True, Rigidbody.update()
          no-ops so the window can run physics only in fixed sub-steps.

    Target render FPS is controlled by ``Window.run(fps=…)`` / ``tick(fps=…)``,
    not by these Time fields.
    """
    delta_time: float = 0.0
    unscaled_delta_time: float = 0.0
    scale: float = 1.0
    time: float = 0.0  # Total elapsed time since game start

    # Ceiling for slow frames only (does NOT affect high FPS).
    # 0.1 s ≈ ignore freezes worse than 10 FPS for one frame; Unity uses ~0.33.
    maximum_delta_time: float = 0.1

    # Fixed-step physics
    fixed_delta_time: float = 1.0 / 60.0
    maximum_physics_steps: int = 8
    fixed_time: float = 0.0
    _physics_accumulator: float = 0.0
    _skip_rigidbody_frame_update: bool = False

    @staticmethod
    def set(raw_dt: float):
        """Record frame delta; optionally clamp *large* dt via maximum_delta_time.

        Small dt (high FPS) is never increased or reduced by this — only values
        above ``maximum_delta_time`` are capped. Pass 0 / None for max to skip.
        """
        if raw_dt < 0.0:
            raw_dt = 0.0
        max_dt = Time.maximum_delta_time
        if max_dt is not None and max_dt > 0.0 and raw_dt > max_dt:
            raw_dt = max_dt
        Time.unscaled_delta_time = raw_dt
        Time.delta_time = raw_dt * Time.scale
        Time.time += Time.delta_time


class Component:
    """Base for attachable components like Transform, Object3D, Collider, Rigidbody."""

    def __init__(self):
        self.game_object: Optional['GameObject'] = None
        self._started = False
        self._awoken = False
        self.enabled: bool = True
    
    def awake(self):
        """
        Called once when the script is first created, before start().
        Use this for initialization that doesn't depend on other objects.
        Override to set up initial state.
        """
        pass
    
    def start(self):
        """
        Called once when play mode begins.
        Override to set up initial state that may depend on other objects.
        """
        pass

    def on_attach(self):
        pass

    def update(self):
        pass

    @property
    def transform(self):
        if self.game_object:
            return self.game_object.transform
        return None

    @property
    def scene(self):
        """Get the scene this component's GameObject belongs to."""
        if self.game_object:
            return self.game_object.scene
        return None

    def add_component(self, component: "Component") -> "Component":
        if not self.game_object:
            raise AttributeError("Current component must contain 'game_object' before adding a new component!")
        return self.game_object.add_component(component)

    def get_component(self, component_type: Type[T]) -> Optional[T]:
        if not self.game_object:
            return None
        return self.game_object.get_component(component_type)

    def get_components(self, component_type: Type[T]) -> List[T]:
        if not self.game_object:
            return []
        return self.game_object.get_components(component_type)

    @classmethod
    def get_inspector_fields(cls) -> List[Tuple[str, InspectorFieldInfo]]:
        """
        Get all inspector fields defined on this component class.
        
        Returns a list of (attribute_name, InspectorFieldInfo) tuples for all
        InspectorField descriptors defined on this class and its parents.
        
        Example:
            for name, info in component.get_inspector_fields():
                print(f"{name}: {info.field_type} = {info.default_value}")
        """
        fields = []
        seen = set()
        
        # Walk through the MRO (Method Resolution Order) to get inherited fields
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if name in seen:
                    continue
                if isinstance(value, InspectorField):
                    fields.append((name, value.get_info()))
                    seen.add(name)
        
        return fields

    def get_inspector_field_value(self, name: str) -> Any:
        """
        Get the current value of an inspector field by name.
        
        Args:
            name: The name of the inspector field
            
        Returns:
            The current value of the field
        """
        # Get the descriptor from the class
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, InspectorField):
            return getattr(self, name)
        return None

    def set_inspector_field_value(self, name: str, value: Any) -> None:
        """
        Set the value of an inspector field by name.
        
        Args:
            name: The name of the inspector field
            value: The new value to set
        """
        # Get the descriptor from the class
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, InspectorField):
            setattr(self, name, value)
        else:
            # Fallback for regular properties/attributes (e.g., Transform.position)
            setattr(self, name, value)
        # Mark dirty for visual refresh (e.g., collider bounds recalculation)
        if hasattr(self, '_transform_dirty'):
            self._transform_dirty = True


class WaitForSeconds:
    """
    Yield instruction to wait for a specified number of seconds.
    
    Example:
        yield WaitForSeconds(2.0)
    """
    def __init__(self, seconds: float):
        self.seconds = seconds
        self.elapsed = 0.0

    def is_done(self, delta_time: float) -> bool:
        self.elapsed += delta_time
        return self.elapsed >= self.seconds


class WaitForFrames:
    """
    Yield instruction to wait for a number of frames.
    
    Used internally to support `yield None` as "wait one frame".
    """
    def __init__(self, frames: int = 1):
        self.frames = frames

    def step(self) -> bool:
        self.frames -= 1
        return self.frames <= 0


class WaitEndOfFrame:
    """
    Yield instruction to resume at the end of the current frame.

    Example:
        yield WaitEndOfFrame()
    """
    pass


class Script(Component):
    """
    Base class for user-defined scripts/components.
    
    Similar to Unity's MonoBehaviour, scripts can be attached to GameObjects
    and receive lifecycle callbacks:
    - awake(): Called once when the script is first created (before start)
    - start(): Called once when play mode begins
    - fixed_update(): Fixed timestep (physics rate); only called if overridden
    - update(): Once per rendered frame; only called if overridden
    - late_update(): After update + physics; only called if overridden
    - on_collision_enter/stay/exit(other)

    Empty hooks are **not** called every frame: GameObject only registers a
    script for a phase when that method is actually overridden, so large
    scenes with few FixedUpdate/LateUpdate users pay almost nothing.

    Frame order (Unity-like)::

        fixed_update × N  (Time.delta_time == fixed_delta_time)
        update            (frame delta)
        late_update       (frame delta)
        render
    
    Example:
        class PlayerController(Script):
            def awake(self):
                self.speed = 5.0
                
            def start(self):
                self.start_coroutine(self.delayed_action())

            def fixed_update(self):
                # Forces / physics-friendly movement
                ...
                
            def update(self):
                # Input, animation triggers
                ...

            def late_update(self):
                # Camera follow after all motion
                ...
                
            def on_collision_enter(self, other):
                print(f"Collided with {other.game_object.name}")

            def delayed_action(self):
                print("Waiting...")
                yield WaitForSeconds(2.0)
                print("Done waiting!")
    """
    
    def __init__(self):
        super().__init__()
    
    def update(self):
        """
        Called every rendered frame when overridden.
        Uses ``Time.delta_time`` (variable frame delta).
        """
        pass

    def fixed_update(self):
        """
        Called on each physics step when overridden.

        Runs at ``Time.fixed_delta_time`` (default 60 Hz). During this call,
        ``Time.delta_time`` equals the fixed step size. Prefer this for
        applying forces and other physics-coupled logic.
        """
        pass

    def late_update(self):
        """
        Called once per frame after ``update`` and fixed physics when overridden.

        Uses frame ``Time.delta_time``. Ideal for camera follow and other
        logic that should run after all movement for the frame.
        """
        pass
    
    def on_collision_enter(self, other):
        """
        Called when this object's collider starts touching another collider.
        
        Args:
            other: The other Collider involved in the collision
        """
        pass
    
    def on_collision_stay(self, other):
        """
        Called every frame while this object's collider is touching another collider.
        
        Args:
            other: The other Collider involved in the collision
        """
        pass
    
    def on_collision_exit(self, other):
        """
        Called when this object's collider stops touching another collider.
        
        Args:
            other: The other Collider involved in the collision
        """
        pass

    def start_coroutine(self, routine: Generator) -> Generator:
        """
        Starts a coroutine.
        
        Args:
            routine: A generator function (using yield)
            
        Returns:
            The routine generator
        """
        if self.game_object:
            return self.game_object.start_coroutine(routine)
        return routine
