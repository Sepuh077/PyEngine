"""
Scriptable Objects - Data containers that can be saved as assets and used from anywhere.

Similar to Unity's ScriptableObject, these are data-only classes that:
- Can be saved as .asset files
- Can be referenced by name from anywhere in the code
- Have their fields editable in the inspector
- Can be created from the editor's context menu

Example:
    from engine import ScriptableObject, InspectorField
    
    class PlayerData(ScriptableObject):
        max_health = InspectorField(int, default=100, min_value=0)
        move_speed = InspectorField(float, default=5.0, min_value=0.0)
        player_name = InspectorField(str, default="Player")
        
    # Create and save
    data = PlayerData.create("Player1Data")
    data.save("assets/player_data.asset")
    
    # Load and use from anywhere
    data = PlayerData.load("assets/player_data.asset")
    # Or get by name from registry
    data = ScriptableObject.get("Player1Data")
    print(data.max_health)  # 100
"""

from typing import Dict, List, Any, Optional, Type, TypeVar, Tuple
import json
import uuid
from pathlib import Path
from dataclasses import dataclass

from engine.component import InspectorField, InspectorFieldInfo

T = TypeVar('T', bound='ScriptableObject')

# File extension for scriptable object assets
SCRIPTABLE_OBJECT_EXT = ".asset"


@dataclass
class ScriptableObjectTypeInfo:
    """Metadata about a ScriptableObject type."""
    type_class: Type['ScriptableObject']
    type_name: str
    module_name: str
    description: Optional[str] = None


class ScriptableObjectMeta(type):
    """
    Metaclass for ScriptableObject that automatically registers types.
    """
    
    # Registry of all ScriptableObject types (class_name -> TypeInfo)
    _types: Dict[str, ScriptableObjectTypeInfo] = {}
    
    def __new__(mcs, name: str, bases: tuple, namespace: dict):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Don't register the base class itself
        if name != 'ScriptableObject':
            # Register this type
            type_info = ScriptableObjectTypeInfo(
                type_class=cls,
                type_name=name,
                module_name=cls.__module__,
                description=cls.__doc__
            )
            mcs._types[name] = type_info
            
            # Also register with full module path for uniqueness
            full_name = f"{cls.__module__}.{name}"
            if full_name != name:
                mcs._types[full_name] = type_info
        
        return cls
    
    @classmethod
    def get_all_types(mcs) -> Dict[str, ScriptableObjectTypeInfo]:
        """Get all registered ScriptableObject types."""
        return mcs._types.copy()
    
    @classmethod
    def get_type(mcs, type_name: str) -> Optional[Type['ScriptableObject']]:
        """Get a ScriptableObject type by name."""
        info = mcs._types.get(type_name)
        return info.type_class if info else None


class ScriptableObject(metaclass=ScriptableObjectMeta):
    """
    Base class for scriptable objects - data containers that can be saved as assets.
    
    ScriptableObjects are similar to Unity's ScriptableObject concept:
    - They are data-only classes (no GameObject attachment)
    - Can be saved to disk as .asset files
    - Can be loaded and accessed from anywhere by name
    - Support InspectorField for editor-visible properties
    
    Example:
        class WeaponData(ScriptableObject):
            damage = InspectorField(float, default=10.0, min_value=0.0)
            attack_speed = InspectorField(float, default=1.0, min_value=0.1)
            weapon_name = InspectorField(str, default="Sword")
            
        # Create a new instance
        weapon = WeaponData.create("Iron Sword")
        weapon.damage = 15.0
        weapon.save("weapons/iron_sword.asset")
        
        # Load from anywhere
        weapon = WeaponData.load("weapons/iron_sword.asset")
        
        # Get by name from registry
        weapon = ScriptableObject.get("Iron Sword")
    """
    
    # Instance registry (name -> instance)
    _instances: Dict[str, 'ScriptableObject'] = {}

    # For lazy loading support
    _project_root: Optional[str] = None
    _assets_loaded: bool = False
    _asset_name_to_path: Dict[str, str] = {}  # name -> full path, built on demand for lazy
    
    def __init__(self, name: str = "ScriptableObject"):
        """
        Initialize a ScriptableObject.
        
        Args:
            name: The name of this instance (used for registry lookup)
        """
        self._name = name
        self._id = str(uuid.uuid4())
        self._source_path: Optional[str] = None
    
    @property
    def name(self) -> str:
        """Get the name of this ScriptableObject."""
        return self._name
    
    @name.setter
    def name(self, value: str):
        """Set the name of this ScriptableObject."""
        # Update registry if name changes
        old_name = self._name
        self._name = value
        
        # Update registry
        if old_name in ScriptableObject._instances:
            if ScriptableObject._instances[old_name] is self:
                del ScriptableObject._instances[old_name]
        ScriptableObject._instances[value] = self
    
    @property
    def source_path(self) -> Optional[str]:
        """Get the source file path for this ScriptableObject."""
        return self._source_path
    
    @classmethod
    def create(cls: Type[T], name: str = "ScriptableObject") -> T:
        """
        Create a new instance of this ScriptableObject type.
        
        Args:
            name: The name for the new instance
            
        Returns:
            A new ScriptableObject instance
        """
        instance = cls(name)
        # Register in the global registry
        ScriptableObject._instances[name] = instance
        ScriptableObject._assets_loaded = True
        return instance
    
    @classmethod
    def load(cls: Type[T], path: str) -> T:
        """
        Load a ScriptableObject from a file.
        
        If an instance with the same name already exists in the registry,
        it will be updated in place to preserve existing references.
        
        Args:
            path: Path to the .asset file
            
        Returns:
            The loaded ScriptableObject instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        instance_name = data.get("_name", "")
        
        # Populate lazy index
        if instance_name:
            ScriptableObject._asset_name_to_path[instance_name] = str(path)
        
        # Check if an instance with this name already exists in the registry
        # If so, update it in place to preserve references from components
        existing_instance = ScriptableObject._instances.get(instance_name)
        if existing_instance is not None:
            # Update the existing instance with new data
            existing_instance._source_path = path
            
            # Restore field values
            fields_data = data.get("fields", {})
            for field_name, value in fields_data.items():
                deserialized = cls._deserialize_value(value)
                try:
                    existing_instance.set_inspector_field_value(field_name, deserialized)
                except (AttributeError, ValueError):
                    # Field might not exist anymore, skip it
                    pass
            
            # Update _id if present
            if "_id" in data:
                existing_instance._id = data["_id"]
            
            return existing_instance
        
        # No existing instance, create a new one
        instance = cls._from_dict(data)
        instance._source_path = path
        
        # Register in the global registry
        ScriptableObject._instances[instance.name] = instance
        ScriptableObject._assets_loaded = True
        
        return instance
    
    @classmethod
    def load_from_name(cls: Type[T], name: str) -> Optional[T]:
        """
        Load a ScriptableObject by searching for its file.
        
        This searches the project for a .asset file with matching name.
        
        Args:
            name: The name of the ScriptableObject to find
            
        Returns:
            The loaded ScriptableObject or None if not found
        """
        # First check registry
        if name in cls._instances:
            instance = cls._instances[name]
            if isinstance(instance, cls):
                return instance
        
        # Try to find the file (this would need project root context)
        # For now, return None if not in registry
        return None
    
    def save(self, path: str) -> None:
        """
        Save this ScriptableObject to a file.
        
        Args:
            path: Path to save the .asset file
        """
        # Ensure .asset extension
        if not path.endswith(SCRIPTABLE_OBJECT_EXT):
            path = path + SCRIPTABLE_OBJECT_EXT
        
        data = self._to_dict()
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        
        self._source_path = path
    
    def _to_dict(self) -> Dict[str, Any]:
        """Serialize this ScriptableObject to a dictionary."""
        data = {
            "_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "_name": self._name,
            "_id": self._id,
            "fields": {}
        }
        
        # Serialize all InspectorField values
        for field_name, field_info in self.get_inspector_fields():
            value = self.get_inspector_field_value(field_name)
            data["fields"][field_name] = self._serialize_value(value)
        
        return data
    
    @classmethod
    def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Deserialize a ScriptableObject from a dictionary."""
        # Get the type
        type_name = data.get("_type", "")
        
        # Try to find the correct class
        actual_cls = ScriptableObjectMeta.get_type(type_name)
        if actual_cls is None:
            # Fall back to the called class
            actual_cls = cls
        
        name = data.get("_name", "ScriptableObject")
        instance = actual_cls(name)
        instance._id = data.get("_id", str(uuid.uuid4()))
        
        # Restore field values
        fields_data = data.get("fields", {})
        for field_name, value in fields_data.items():
            deserialized = cls._deserialize_value(value)
            try:
                instance.set_inspector_field_value(field_name, deserialized)
            except (AttributeError, ValueError):
                # Field might not exist anymore, skip it
                pass
        
        return instance
    
    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a value for storage."""
        import numpy as np
        from engine.types import Vector3
        
        if isinstance(value, np.ndarray):
            return {
                "__type__": "ndarray",
                "dtype": str(value.dtype),
                "value": value.tolist(),
            }
        if isinstance(value, Vector3):
            return {
                "__type__": "Vector3",
                "value": value.to_list(),
            }
        if isinstance(value, tuple):
            return {
                "__type__": "tuple",
                "value": list(value),
            }
        if isinstance(value, list):
            return [ScriptableObject._serialize_value(v) for v in value]
        if isinstance(value, dict):
            return {k: ScriptableObject._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        # Fallback to repr for unknown types
        return {
            "__type__": "repr",
            "value": repr(value),
        }
    
    @staticmethod
    def _deserialize_value(value: Any) -> Any:
        """Deserialize a value from storage."""
        import numpy as np
        from engine.types import Vector3
        
        if isinstance(value, dict):
            if value.get("__type__") == "ndarray":
                return np.array(value.get("value", []), dtype=value.get("dtype", None))
            if value.get("__type__") == "Vector3":
                return Vector3(value.get("value", [0, 0, 0]))
            if value.get("__type__") == "tuple":
                return tuple(value.get("value", []))
            if value.get("__type__") == "repr":
                return value.get("value")
            return {k: ScriptableObject._deserialize_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ScriptableObject._deserialize_value(v) for v in value]
        return value
    
    @classmethod
    def get_inspector_fields(cls) -> List[Tuple[str, InspectorFieldInfo]]:
        """
        Get all inspector fields defined on this class.
        
        Returns a list of (field_name, InspectorFieldInfo) tuples.
        """
        fields = []
        seen = set()
        
        # Walk through the MRO to get inherited fields
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if name in seen:
                    continue
                if isinstance(value, InspectorField):
                    fields.append((name, value.get_info()))
                    seen.add(name)
        
        return fields
    
    def get_inspector_field_value(self, name: str) -> Any:
        """Get the current value of an inspector field."""
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, InspectorField):
            return getattr(self, name)
        return None
    
    def set_inspector_field_value(self, name: str, value: Any) -> None:
        """Set the value of an inspector field."""
        descriptor = getattr(type(self), name, None)
        if isinstance(descriptor, InspectorField):
            setattr(self, name, value)
    
    # =========================================================================
    # Global Registry Methods
    # =========================================================================
    
    @classmethod
    def get(cls, name: str) -> Optional['ScriptableObject']:
        """
        Get a ScriptableObject instance by name from the global registry.
        
        Lazy loading: If the name is not found and a project root is known
        (set via window's auto_load_scriptable_assets=False or set_project_root),
        this will locate **only the requested asset** by name and load just that
        one file (not the entire directory).
        
        Once loaded, the instance is kept in the registry.
        
        Args:
            name: The name of the ScriptableObject to find
            
        Returns:
            The ScriptableObject instance or None if not found
        """
        if name not in cls._instances:
            path = cls._find_asset_path_by_name(name)
            if path:
                try:
                    cls.load(path)  # load() registers it
                except Exception:
                    pass
        return cls._instances.get(name)
    
    @classmethod
    def get_all(cls) -> List['ScriptableObject']:
        """
        Get all registered ScriptableObject instances.
        
        In lazy mode, this returns only the assets that have been requested so far
        (via get() etc.). Call load_all_assets() explicitly to load everything.
        """
        return list(cls._instances.values())
    
    @classmethod
    def get_by_type(cls, scriptable_type: Type[T]) -> List[T]:
        """
        Get all ScriptableObject instances of a specific type.
        
        In lazy mode, this returns only the assets of that type that have been
        requested so far. Call load_all_assets() to load everything.
        
        Args:
            scriptable_type: The type to filter by
            
        Returns:
            List of ScriptableObject instances of the specified type
        """
        type_name = scriptable_type.__name__
        
        # Collect instances that match either by isinstance or by class name
        # (handles dynamic loading with different module names)
        seen_ids = set()
        instances = []
        
        for inst in cls._instances.values():
            # Check if already added
            if id(inst) in seen_ids:
                continue
            
            # Match by isinstance or by class name
            if isinstance(inst, scriptable_type) or type(inst).__name__ == type_name:
                seen_ids.add(id(inst))
                instances.append(inst)
        
        return instances
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Remove a ScriptableObject from the registry.
        
        Args:
            name: The name of the ScriptableObject to remove
        """
        if name in cls._instances:
            del cls._instances[name]
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered ScriptableObject instances."""
        cls._instances.clear()
        cls._assets_loaded = False
        cls._asset_name_to_path.clear()

    @classmethod
    def set_project_root(cls, directory: str) -> None:
        """Set the project root directory used for lazy asset loading.
        
        When lazy loading is enabled (auto_load_scriptable_assets=False on window),
        the first call to get(name) will locate and load **only** that specific
        asset (not the whole directory).
        """
        cls._project_root = str(Path(directory).resolve())
        cls._assets_loaded = False
        cls._asset_name_to_path.clear()

    @classmethod
    def _build_asset_index(cls, directory: str) -> None:
        """Scan the directory for .asset files and build a name -> path index.
        This is cheap (only reads _name from JSON, does not fully load or register).
        """
        cls._asset_name_to_path.clear()
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return
        for asset_file in dir_path.rglob(f"*{SCRIPTABLE_OBJECT_EXT}"):
            try:
                with open(asset_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("_name")
                if name:
                    cls._asset_name_to_path[name] = str(asset_file)
            except Exception:
                # Ignore bad JSON or unreadable files
                continue

    @classmethod
    def _find_asset_path_by_name(cls, name: str) -> Optional[str]:
        """Return the full path for a specific asset name, building index if needed."""
        if not cls._asset_name_to_path and cls._project_root:
            cls._build_asset_index(cls._project_root)
        return cls._asset_name_to_path.get(name)
    
    @classmethod
    def get_all_types(cls) -> Dict[str, ScriptableObjectTypeInfo]:
        """
        Get all registered ScriptableObject types.
        
        Returns:
            Dictionary mapping type names to type info
        """
        return ScriptableObjectMeta._types.copy()
    
    @classmethod
    def find_scriptable_object_files(cls, directory: str) -> List[str]:
        """
        Find all .asset files in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of paths to .asset files
        """
        asset_files = []
        dir_path = Path(directory)
        
        if dir_path.exists() and dir_path.is_dir():
            for asset_file in dir_path.rglob(f"*{SCRIPTABLE_OBJECT_EXT}"):
                asset_files.append(str(asset_file))
        
        return asset_files
    
    @classmethod
    def load_all_assets(cls, directory: str, scan_for_types: bool = True) -> List['ScriptableObject']:
        """
        Load all .asset files from a directory and register them in the instance registry.
        
        This method is called automatically by Window2D/Window3D (unless
        auto_load_scriptable_assets=False). It can also be called manually
        to ensure all ScriptableObject assets are available via ScriptableObject.get().
        
        Args:
            directory: The root directory to search for .asset files (recursively)
            scan_for_types: If True, scan for ScriptableObject type definitions in .py files
                           before loading assets (ensures types are registered)
            
        Returns:
            List of all loaded ScriptableObject instances
        """
        cls._project_root = str(Path(directory).resolve())
        loaded_instances = []
        dir_path = Path(directory)
        
        if not dir_path.exists() or not dir_path.is_dir():
            cls._assets_loaded = True
            return loaded_instances
        
        # Optionally scan for type definitions first
        if scan_for_types:
            cls._scan_and_register_types(dir_path)
        
        # Find all .asset files
        asset_files = cls.find_scriptable_object_files(directory)
        
        for asset_path in asset_files:
            try:
                # Load the asset file
                with open(asset_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Get the type from the stored type name
                type_name = data.get("_type", "")
                instance_name = data.get("_name", "Unknown")
                
                # Populate lazy index
                if instance_name:
                    cls._asset_name_to_path[instance_name] = str(asset_path)
                
                # Check if already loaded (avoid duplicates)
                if instance_name in cls._instances:
                    existing = cls._instances[instance_name]
                    # If same source path, skip loading but return the existing instance
                    if hasattr(existing, '_source_path') and existing._source_path == asset_path:
                        loaded_instances.append(existing)
                        continue
                    # If different source path, update the existing instance in place
                    # to preserve references from components
                    existing._source_path = asset_path
                    fields_data = data.get("fields", {})
                    for field_name, value in fields_data.items():
                        deserialized = cls._deserialize_value(value)
                        try:
                            existing.set_inspector_field_value(field_name, deserialized)
                        except (AttributeError, ValueError):
                            pass
                    if "_id" in data:
                        existing._id = data["_id"]
                    loaded_instances.append(existing)
                    continue
                
                # Try to find the type class
                so_class = ScriptableObjectMeta.get_type(type_name)
                
                if so_class is None:
                    # Type not found, try to import the module
                    if '.' in type_name:
                        module_name = type_name.rsplit('.', 1)[0]
                        try:
                            import importlib
                            importlib.import_module(module_name)
                            so_class = ScriptableObjectMeta.get_type(type_name)
                        except (ImportError, ModuleNotFoundError):
                            pass
                
                if so_class is None:
                    # Still can't find type - log warning and skip
                    print(f"Warning: Could not find ScriptableObject type '{type_name}' for asset '{asset_path}'")
                    continue
                
                # Create instance from data
                instance = so_class._from_dict(data)
                instance._source_path = asset_path
                
                # Register in the global registry
                cls._instances[instance.name] = instance
                loaded_instances.append(instance)
                
            except Exception as e:
                print(f"Warning: Failed to load asset '{asset_path}': {e}")
                continue
        
        cls._assets_loaded = True
        return loaded_instances
    
    @classmethod
    def _scan_and_register_types(cls, directory: Path) -> None:
        """
        Scan Python files in a directory for ScriptableObject subclass definitions
        and import them to register the types.
        
        Args:
            directory: The directory to scan
        """
        import re
        import sys
        import importlib.util
        
        if not directory.exists() or not directory.is_dir():
            return
        
        # Find all .py files
        for py_file in directory.rglob("*.py"):
            # Skip hidden directories, __pycache__, and virtual environments
            _skip = ('.', '__pycache__', 'venv', '.venv', 'env', '.env',
                     'node_modules', 'site-packages', '.git')
            if any(part.startswith('.') or part in _skip for part in py_file.parts):
                continue
            
            # Skip src directory (engine code - already imported)
            if 'src' in py_file.parts:
                continue
            
            try:
                # Read the file to check for ScriptableObject subclasses
                content = py_file.read_text(encoding='utf-8', errors='replace')
                
                # Look for ScriptableObject subclass definitions
                pattern = r'class\s+(\w+)\s*\(\s*ScriptableObject\s*\)'
                matches = re.findall(pattern, content)
                
                if not matches:
                    continue
                
                # Import the module to register the types
                try:
                    # Create a unique module name
                    relative_path = py_file.relative_to(directory)
                    module_name = '.'.join(relative_path.with_suffix('').parts)
                    
                    # Skip if already imported
                    if module_name in sys.modules:
                        continue
                    
                    # Import the module
                    spec = importlib.util.spec_from_file_location(module_name, str(py_file))
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        
                except Exception as e:
                    print(f"Warning: Could not import module '{py_file}': {e}")
                    continue
                    
            except Exception as e:
                print(f"Warning: Could not scan file '{py_file}': {e}")
                continue
    
    @classmethod
    def register_instance(cls, instance: 'ScriptableObject') -> None:
        """
        Register an existing ScriptableObject instance in the global registry.
        
        Args:
            instance: The ScriptableObject instance to register
        """
        if instance and instance.name:
            cls._instances[instance.name] = instance
            cls._assets_loaded = True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}')"
