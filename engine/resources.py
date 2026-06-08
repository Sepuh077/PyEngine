"""
Resources - Unity-like resource loading system.

Provides a simple API for loading game resources (prefabs, scriptable objects,
materials, scenes) from the Assets folder.

Example:
    from engine.d3 import Resources, GameObject, ScriptableObject
    
    # Load a single prefab
    player = Resources.load(GameObject, "prefabs/player")
    
    # Load all prefabs from a folder
    enemies = Resources.load_all(GameObject, "prefabs/enemies/")
    
    # Load a scriptable object
    settings = Resources.load(MySettings, "data/game_settings")
    
    # Load all materials from a folder
    materials = Resources.load_all(Material, "materials/")
"""
from pathlib import Path
from typing import Type, TypeVar, List, Optional, Any, Dict, Union
import os

T = TypeVar('T')


class Resources:
    """
    Unity-like resource loading system.
    
    Resources are loaded from the 'Assets' folder relative to the project root.
    The project root is determined by:
    1. Explicitly set via Resources.set_assets_path()
    2. Current working directory
    
    Supported resource types and their file extensions:
    - GameObject: .prefab
    - ScriptableObject: .asset
    - Material: .mat3d
    - Scene3D: .scene
    
    Example:
        # Set the assets path (optional, defaults to cwd/Assets)
        Resources.set_assets_path("/path/to/project")
        
        # Load a single prefab (extension optional)
        player = Resources.load(GameObject, "prefabs/player")
        
        # Load all prefabs from a folder
        all_enemies = Resources.load_all(GameObject, "prefabs/enemies/")
        
        # Load a specific scriptable object type
        weapon = Resources.load(WeaponData, "weapons/iron_sword")
    """
    
    # The root path for assets (defaults to cwd/Assets)
    _assets_path: Optional[Path] = None
    
    # Mapping of types to their file extensions
    _type_extensions: Dict[type, str] = {}
    
    @classmethod
    def _get_type_extension(cls, resource_type: type) -> str:
        """Get the file extension for a resource type."""
        # Check cache first
        if resource_type in cls._type_extensions:
            return cls._type_extensions[resource_type]
        
        # Determine extension based on type
        extension = None
        
        # Import types locally to avoid circular imports
        try:
            from engine.gameobject import GameObject
            if resource_type == GameObject or (isinstance(resource_type, type) and issubclass(resource_type, GameObject)):
                extension = ".prefab"
        except ImportError:
            pass
        
        try:
            from engine.scriptable_object import ScriptableObject
            if extension is None and isinstance(resource_type, type) and issubclass(resource_type, ScriptableObject):
                extension = ".asset"
        except ImportError:
            pass
        
        try:
            from engine.graphics.material import Material
            if extension is None and isinstance(resource_type, type) and issubclass(resource_type, Material):
                extension = ".mat3d"
        except ImportError:
            pass
        
        try:
            from engine.d3.scene import Scene3D
            if extension is None and resource_type == Scene3D:
                extension = ".scene"
        except ImportError:
            pass
        
        if extension is None:
            extension = ""  # Unknown type, no extension
        
        # Cache the result
        cls._type_extensions[resource_type] = extension
        return extension
    
    @classmethod
    def set_assets_path(cls, path: Union[str, Path]) -> None:
        """
        Set the root path for assets.
        
        Args:
            path: Path to the Assets folder or project root.
                  If path ends with 'Assets', uses it directly.
                  Otherwise, appends 'Assets' to the path.
        """
        path = Path(path).resolve()
        if path.name != "Assets":
            path = path / "Assets"
        cls._assets_path = path
    
    @classmethod
    def get_assets_path(cls) -> Path:
        """
        Get the current assets path.
        
        Returns:
            Path to the Assets folder.
        """
        if cls._assets_path is not None:
            return cls._assets_path
        
        # Default to cwd/Assets
        cls._assets_path = Path.cwd() / "Assets"
        return cls._assets_path
    
    @classmethod
    def load(
        cls,
        resource_type: Type[T],
        path: str,
        **kwargs
    ) -> Optional[T]:
        """
        Load a single resource from the Assets folder.
        
        Args:
            resource_type: The type of resource to load (GameObject, ScriptableObject, Material, Scene3D)
            path: Relative path from Assets folder (without extension)
            **kwargs: Additional arguments passed to the resource loader
        
        Returns:
            The loaded resource instance, or None if not found
        
        Example:
            # Load Assets/prefabs/player.prefab
            player = Resources.load(GameObject, "prefabs/player")
            
            # Load Assets/data/settings.asset as MySettings
            settings = Resources.load(MySettings, "data/settings")
        """
        assets_path = cls.get_assets_path()
        extension = cls._get_type_extension(resource_type)
        
        # Build full path
        resource_path = assets_path / path
        
        # Add extension if not already present
        if extension and not str(resource_path).endswith(extension):
            resource_path = Path(str(resource_path) + extension)
        
        # Check if file exists
        if not resource_path.exists():
            return None
        
        # Load based on type
        return cls._load_resource(resource_type, resource_path, **kwargs)
    
    @classmethod
    def load_all(
        cls,
        resource_type: Type[T],
        path: str = "",
        recursive: bool = True,
        **kwargs
    ) -> List[T]:
        """
        Load all resources of a type from a folder in Assets.
        
        Args:
            resource_type: The type of resources to load
            path: Relative path from Assets folder (empty string for root)
            recursive: If True, search subfolders recursively. If False, only search the given folder.
            **kwargs: Additional arguments passed to each resource loader
        
        Returns:
            List of loaded resource instances
        
        Example:
            # Load all prefabs from Assets/prefabs/ (including subfolders)
            prefabs = Resources.load_all(GameObject, "prefabs/")
            
            # Load prefabs only from the root of prefabs/ folder (no subfolders)
            prefabs = Resources.load_all(GameObject, "prefabs/", recursive=False)
            
            # Load all scriptable objects from Assets/data/weapons/
            weapons = Resources.load_all(WeaponData, "data/weapons/")
        """
        assets_path = cls.get_assets_path()
        extension = cls._get_type_extension(resource_type)
        
        # Build folder path
        folder_path = assets_path / path if path else assets_path
        
        if not folder_path.exists() or not folder_path.is_dir():
            return []
        
        results = []
        
        # Find all files with matching extension
        if recursive:
            file_iterator = folder_path.rglob(f"*{extension}")
        else:
            file_iterator = folder_path.glob(f"*{extension}")
        
        for file_path in file_iterator:
            if file_path.is_file():
                resource = cls._load_resource(resource_type, file_path, **kwargs)
                if resource is not None:
                    results.append(resource)
        
        return results
    
    @classmethod
    def _load_resource(
        cls,
        resource_type: Type[T],
        file_path: Path,
        **kwargs
    ) -> Optional[T]:
        """
        Internal method to load a resource based on its type.
        
        Args:
            resource_type: The type of resource to load
            file_path: Absolute path to the resource file
            **kwargs: Additional arguments for the loader
        
        Returns:
            The loaded resource instance, or None if loading failed or type mismatch
        """
        str_path = str(file_path)
        
        # GameObject / Prefab
        try:
            from engine.gameobject import GameObject
            if resource_type == GameObject or (isinstance(resource_type, type) and issubclass(resource_type, GameObject)):
                return GameObject.load(str_path, **kwargs)
        except ImportError:
            pass
        
        # ScriptableObject - need to check actual type after loading
        try:
            from engine.scriptable_object import ScriptableObject
            if isinstance(resource_type, type) and issubclass(resource_type, ScriptableObject):
                # Load using the base ScriptableObject.load which resolves the actual type
                resource = ScriptableObject.load(str_path)
                if resource is None:
                    return None
                # Check if the loaded resource is of the requested type (or a subclass)
                if not isinstance(resource, resource_type):
                    return None
                return resource
        except ImportError:
            pass
        
        # Material
        try:
            from engine.graphics.material import Material
            if isinstance(resource_type, type) and issubclass(resource_type, Material):
                return Material.load(str_path)
        except ImportError:
            pass
        
        # Scene3D
        try:
            from engine.d3.scene import Scene3D
            if resource_type == Scene3D:
                return Scene3D.load(str_path)
        except ImportError:
            pass
        
        return None
    
    @classmethod
    def exists(cls, path: str, resource_type: Optional[type] = None) -> bool:
        """
        Check if a resource exists at the given path.
        
        Args:
            path: Relative path from Assets folder
            resource_type: Optional type to determine extension
        
        Returns:
            True if the resource file exists
        """
        assets_path = cls.get_assets_path()
        resource_path = assets_path / path
        
        if resource_type is not None:
            extension = cls._get_type_extension(resource_type)
            if extension and not str(resource_path).endswith(extension):
                resource_path = Path(str(resource_path) + extension)
        
        return resource_path.exists()
    
    @classmethod
    def get_full_path(cls, path: str, resource_type: Optional[type] = None) -> Path:
        """
        Get the full path for a resource.
        
        Args:
            path: Relative path from Assets folder
            resource_type: Optional type to determine extension
        
        Returns:
            Full path to the resource
        """
        assets_path = cls.get_assets_path()
        resource_path = assets_path / path
        
        if resource_type is not None:
            extension = cls._get_type_extension(resource_type)
            if extension and not str(resource_path).endswith(extension):
                resource_path = Path(str(resource_path) + extension)
        
        return resource_path
