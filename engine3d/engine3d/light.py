"""
Light3D - Lighting for 3D scenes.
"""
import numpy as np
from typing import Tuple, Union
from engine3d.types import ColorType, Vector3, Color
from engine3d.component import Component, InspectorField


class Light3D(Component):
    """
    Base class for all lights.
    """
    
    # Inspector fields
    color = InspectorField(Color, default=(1.0, 1.0, 1.0), tooltip="Light color (RGB 0-1)")
    intensity = InspectorField(float, default=1.0, min_value=0.0, max_value=1000.0, step=0.1, decimals=2, tooltip="Light intensity multiplier")

    def __init__(self, 
                 color: ColorType = (1.0, 1.0, 1.0),
                 intensity: float = 1.0):
        """
        Initialize base light.
        
        Args:
            color: Light color (RGB 0-1)
            intensity: Light intensity multiplier
        """
        super().__init__()
        # InspectorField descriptors handle storage with defaults
        self.color = color
        self.intensity = intensity


class DirectionalLight3D(Light3D):
    """
    Directional light for 3D scenes.
    
    Example:
        light_go = GameObject("Light")
        light = DirectionalLight3D(color=Color.WHITE)
        light_go.add_component(light)
        # set direction by rotating the GameObject
        light_go.transform.rotation = (-45, 30, 0)
    """
    
    # Additional inspector fields for directional light
    ambient = InspectorField(float, default=0.2, min_value=0.0, max_value=1.0, step=0.05, decimals=2, tooltip="Ambient light level (0-1)")
    
    # Shadow properties
    cast_shadows = InspectorField(bool, default=True, tooltip="Enable shadow casting")
    shadow_resolution = InspectorField(int, default=1024, tooltip="Shadow map resolution")
    shadow_distance = InspectorField(float, default=50.0, min_value=1.0, max_value=500.0, step=1.0, decimals=1, tooltip="Maximum shadow distance")
    shadow_bias = InspectorField(float, default=0.001, min_value=0.0, max_value=0.1, step=0.0001, decimals=4, tooltip="Shadow depth bias to prevent acne")
    normal_bias = InspectorField(float, default=0.002, min_value=0.0, max_value=0.1, step=0.0001, decimals=4, tooltip="Normal bias to reduce acne while minimizing peter-panning for realistic shadows")

    def __init__(self, 
                 color: ColorType = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 ambient: float = 0.2,
                 cast_shadows: bool = True,
                 shadow_resolution: int = 4096,
                 shadow_distance: float = 50.0,
                 shadow_bias: float = 0.001,
                 normal_bias: float = 0.002):
        """
        Initialize directional light.
        
        Args:
            color: Light color (RGB 0-1)
            intensity: Light intensity multiplier
            ambient: Ambient light level (0-1)
            cast_shadows: Whether this light casts shadows
            shadow_resolution: Shadow map resolution (512, 1024, 2048, 4096)
            shadow_distance: Maximum distance for shadow rendering
            shadow_bias: Depth bias to prevent shadow acne
            normal_bias: Normal-based bias for realistic shadows (reduces acne/peter-panning)
        """
        super().__init__(color, intensity)
        self._fallback_direction = Vector3(0.3, -0.7, -0.5).normalized
        self._normalize_fallback_direction()
        
        self.ambient = ambient
        self.cast_shadows = cast_shadows
        self.shadow_resolution = shadow_resolution
        self.shadow_distance = shadow_distance
        self.shadow_bias = shadow_bias
        self.normal_bias = normal_bias
    
    def _normalize_fallback_direction(self):
        """Normalize the fallback direction vector."""
        self._fallback_direction = self._fallback_direction.normalized
    
    @property
    def direction(self) -> Vector3:
        """Get light direction. If attached to a GameObject, derives from transform rotation."""
        if self.game_object and self.game_object.transform:
            model = self.game_object.transform.get_model_matrix()
            # Forward vector is usually -Z in this coordinate system
            # 3rd column of the rotation matrix (index 2)
            fwd = -model[0:3, 2]
            norm = np.linalg.norm(fwd)
            if norm > 0:
                fwd = fwd / norm
            return Vector3(fwd)
        return Vector3(self._fallback_direction)
    
    @direction.setter
    def direction(self, value: Union[Tuple[float, float, float], Vector3]):
        """Set fallback light direction."""
        self._fallback_direction = Vector3(value).normalized
    
    def point_from(self, position: Union[Tuple[float, float, float], Vector3], 
                   target: Union[Tuple[float, float, float], Vector3] = (0, 0, 0)):
        """
        Set fallback light to point from a position towards a target.
        """
        if self.game_object and self.game_object.transform:
            self.game_object.transform.position = position
        pos = Vector3(position)
        tgt = Vector3(target)
        self._fallback_direction = (tgt - pos).normalized


class PointLight3D(Light3D):
    """
    Point light that emits in all directions from a position.
    """
    
    # Additional inspector fields for point light
    range = InspectorField(float, default=50.0, min_value=0.1, max_value=1000.0, step=0.5, decimals=2, tooltip="Maximum light range")

    # Shadow properties
    cast_shadows = InspectorField(bool, default=True, tooltip="Enable shadow casting")
    shadow_resolution = InspectorField(int, default=512, tooltip="Shadow cubemap resolution per face")
    shadow_bias = InspectorField(float, default=0.001, min_value=0.0, max_value=0.1, step=0.0001, decimals=4, tooltip="Shadow depth bias to prevent acne")
    shadow_near = InspectorField(float, default=0.1, min_value=0.01, max_value=10.0, step=0.1, decimals=2, tooltip="Near plane for shadow frustum")
    shadow_far = InspectorField(float, default=50.0, min_value=1.0, max_value=500.0, step=1.0, decimals=1, tooltip="Far plane for shadow frustum")

    def __init__(self,
                 color: ColorType = (1.0, 1.0, 1.0),
                 intensity: float = 1.0,
                 range: float = 50.0,
                 cast_shadows: bool = True,
                 shadow_resolution: int = 512,
                 shadow_bias: float = 0.001,
                 shadow_near: float = 0.1,
                 shadow_far: float = 50.0):
        """
        Initialize point light.
        
        Args:
            color: Light color (RGB 0-1)  
            intensity: Light intensity
            range: Maximum light range
            cast_shadows: Whether this light casts shadows
            shadow_resolution: Shadow cubemap resolution per face
            shadow_bias: Depth bias to prevent shadow acne
            shadow_near: Near plane for shadow frustum
            shadow_far: Far plane for shadow frustum
        """
        super().__init__(color, intensity)
        self._fallback_position = Vector3(0, 10, 0)
        self.range = range
        self.cast_shadows = cast_shadows
        self.shadow_resolution = shadow_resolution
        self.shadow_bias = shadow_bias
        self.shadow_near = shadow_near
        self.shadow_far = shadow_far
    
    @property
    def position(self) -> Vector3:
        if self.game_object and self.game_object.transform:
            return self.game_object.transform.world_position
        return Vector3(self._fallback_position)
    
    @position.setter
    def position(self, value: Union[Tuple[float, float, float], Vector3]):
        if self.game_object and self.game_object.transform:
            self.game_object.transform.position = value
        else:
            self._fallback_position = Vector3(value)
    
    @property
    def x(self) -> float:
        return float(self.position.x)
    
    @x.setter
    def x(self, value: float):
        p = self.position
        self.position = (value, p.y, p.z)
    
    @property
    def y(self) -> float:
        return float(self.position.y)
    
    @y.setter
    def y(self, value: float):
        p = self.position
        self.position = (p.x, value, p.z)
    
    @property
    def z(self) -> float:
        return float(self.position.z)
    
    @z.setter
    def z(self, value: float):
        p = self.position
        self.position = (p.x, p.y, value)
