from typing import Optional, List, Dict, Any, Type
import json
import numpy as np
from engine.types import Color, ColorType

# Material file extension
MATERIAL_FILE_EXT = ".mat3d"


class Material:
    """Base class for all materials."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0):
        self.color = color
        self.alpha = alpha

    # =========================================================================
    # File Save/Load (for .mat3d files)
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Save this material to a .mat3d file.
        
        Args:
            path: File path (will add .mat3d extension if not present)
        """
        if not path.endswith(MATERIAL_FILE_EXT):
            path = path + MATERIAL_FILE_EXT
        
        data = self._to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "Material":
        """
        Load a material from a .mat3d file.
        
        Args:
            path: File path to .mat3d file
            
        Returns:
            The loaded Material instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Serialize material to dict for saving."""
        # Get all serializable attributes (skip private, GPU resources, etc.)
        skip_keys = {
            "_gl_texture", "_gpu_initialized", "_texture_image",
            "_mesh", "_vao", "_vbo"
        }
        state = {}
        for key, value in self.__dict__.items():
            if key.startswith("_") and key not in ("_skybox",):
                continue
            if key in skip_keys:
                continue
            state[key] = self._serialize_value(value)
        
        return {
            "__class__": self.__class__.__name__,
            "__module__": self.__class__.__module__,
            "state": state,
        }
    
    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a value to JSON-compatible form."""
        if isinstance(value, np.ndarray):
            return {"__type__": "ndarray", "value": value.tolist(), "dtype": str(value.dtype)}
        if isinstance(value, (Color, tuple, list)):
            return list(value)
        if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
            return value.item()
        return value
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Material":
        """Deserialize material from dict."""
        class_name = data.get("__class__", "LitMaterial")
        state = data.get("state", {})
        
        # Import the correct class
        if class_name == "ShaderMaterial":
            from engine.graphics.shader_material import ShaderMaterial
            return ShaderMaterial._from_material_dict(dict(state))
        elif class_name == "SkyboxMaterial":
            from engine.graphics.material import SkyboxMaterial
            mat = SkyboxMaterial()
        elif class_name == "LitMaterial":
            mat = LitMaterial()
        elif class_name == "UnlitMaterial":
            mat = UnlitMaterial()
        elif class_name == "SpecularMaterial":
            mat = SpecularMaterial()
        elif class_name == "EmissiveMaterial":
            mat = EmissiveMaterial()
        elif class_name == "TransparentMaterial":
            mat = TransparentMaterial()
        else:
            mat = LitMaterial()  # Fallback
        
        # Restore state
        for key, value in state.items():
            setattr(mat, key, cls._deserialize_value(value))
        
        return mat
    
    @staticmethod
    def _deserialize_value(value: Any) -> Any:
        """Deserialize a value from JSON form."""
        if isinstance(value, dict) and value.get("__type__") == "ndarray":
            return np.array(value["value"], dtype=value.get("dtype", None))
        return value

    @property
    def color_vec4(self) -> np.ndarray:
        c = np.array(self.color, dtype=np.float32)
        if c.max() > 1.0:
            c /= 255.0
        if len(c) == 3:
            return np.append(c, self.alpha)
        return c


class UnlitMaterial(Material):
    """Material that ignores lighting and is always visible with its color."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0):
        super().__init__(color, alpha)


class LitMaterial(Material):
    """Lambert material with diffuse lighting."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0):
        super().__init__(color, alpha)


class SpecularMaterial(Material):
    """Phong / Blinn-Phong material for metal and plastic."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0, 
                 specular_color: ColorType = Color.WHITE, shininess: float = 32.0):
        super().__init__(color, alpha)
        self.specular_color = specular_color
        self.shininess = shininess

    @property
    def specular_vec3(self) -> np.ndarray:
        c = np.array(self.specular_color, dtype=np.float32)
        if c.max() > 1.0:
            c /= 255.0
        return c[:3]


class EmissiveMaterial(Material):
    """Material that glows and ignores lights around it."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 1.0, intensity: float = 1.0):
        super().__init__(color, alpha)
        self.intensity = intensity


class TransparentMaterial(Material):
    """Material with explicit alpha transparency."""
    def __init__(self, color: ColorType = Color.WHITE, alpha: float = 0.5):
        super().__init__(color, alpha)


class SkyboxMaterial(Material):
    """
    Skybox material for rendering a background environment around the camera.
    
    Can be assigned to Camera3D via the skybox inspector field.
    Supports either:
    - 6 separate cube map face textures
    - A single equirectangular texture (360° image)
    
    Example:
        # Create skybox with equirectangular texture
        skybox = SkyboxMaterial(texture_path="skybox.hdr")
        camera.skybox = skybox
        
        # Or create with 6 cubemap faces
        skybox = SkyboxMaterial(
            front="sky_front.png", back="sky_back.png",
            left="sky_left.png", right="sky_right.png",
            top="sky_top.png", bottom="sky_bottom.png"
        )
    """
    
    def __init__(self,
                 texture_path: Optional[str] = None,
                 front: Optional[str] = None,
                 back: Optional[str] = None,
                 left: Optional[str] = None,
                 right: Optional[str] = None,
                 top: Optional[str] = None,
                 bottom: Optional[str] = None,
                 color: ColorType = Color.WHITE):
        """
        Initialize skybox material.
        
        Args:
            texture_path: Path to equirectangular 360° texture (HDR or regular image)
            front/back/left/right/top/bottom: Paths to individual cubemap face textures
            color: Tint color for the skybox
        """
        super().__init__(color, 1.0)
        self.texture_path = texture_path  # Equirectangular texture
        self.front = front
        self.back = back
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        
        # GPU resources (initialized on first render)
        self._gl_texture = None
        self._gpu_initialized = False
        
    @property
    def is_cubemap(self) -> bool:
        """Check if this skybox uses 6 cubemap faces."""
        return any([self.front, self.back, self.left, self.right, self.top, self.bottom])
    
    @property
    def has_texture(self) -> bool:
        """Check if this skybox has any texture source."""
        return self.texture_path is not None or self.is_cubemap
    
    def get_texture_paths(self) -> List[Optional[str]]:
        """Get list of texture paths in cubemap order: right, left, top, bottom, front, back."""
        return [self.right, self.left, self.top, self.bottom, self.front, self.back]
    
    # =========================================================================
    # Gradient Support (Unity-like procedural skybox)
    # =========================================================================
    
    @property
    def is_gradient(self) -> bool:
        """Check if this skybox uses a gradient (no textures)."""
        return (not self.has_texture and 
                hasattr(self, '_gradient_colors') and 
                self._gradient_colors is not None)
    
    @classmethod
    def create_gradient(
        cls,
        top_color: ColorType = (0.35, 0.55, 0.95),     # Sky blue
        middle_color: ColorType = (0.85, 0.90, 0.98),  # Soft horizon
        bottom_color: ColorType = (0.45, 0.35, 0.25),  # Ground tint
        alpha: float = 1.0
    ) -> "SkyboxMaterial":
        """
        Create a Unity-like gradient skybox.
        
        Args:
            top_color: Color at the top (sky)
            middle_color: Color at the horizon
            bottom_color: Color at the bottom (ground)
            alpha: Overall alpha
            
        Returns:
            SkyboxMaterial with gradient colors
            
        Example:
            skybox = SkyboxMaterial.create_gradient(
                top_color=(0.4, 0.6, 1.0),      # Sky blue
                middle_color=(1.0, 1.0, 1.0),    # White
                bottom_color=(0.4, 0.25, 0.1)    # Brown
            )
            camera.skybox = skybox
        """
        mat = cls(color=(1.0, 1.0, 1.0))
        # Store gradient colors for rendering
        mat._gradient_colors = {
            'top': top_color,
            'middle': middle_color,
            'bottom': bottom_color
        }
        mat.alpha = alpha
        return mat

    @classmethod
    def create_default(cls) -> "SkyboxMaterial":
        """Default procedural sky used for new 3D cameras (Unity-like gradient).

        Prefer this over a solid clear color so scenes look like an outdoor
        environment out of the box. Replace with a textured skybox or set
        ``camera.skybox = None`` + ``ClearFlags.SOLID_CLEAR`` for solid color.
        """
        return cls.create_gradient()
    
    def get_gradient_colors(self) -> Optional[dict]:
        """Get gradient colors if this is a gradient skybox."""
        return getattr(self, '_gradient_colors', None)
