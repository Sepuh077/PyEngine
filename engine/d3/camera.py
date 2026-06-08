"""
Camera3D - First-person or orbit camera for 3D scenes.

Supports multiple cameras with viewports for:
- Picture-in-picture displays
- Rear-view mirrors in racing games
- Minimaps
- Split-screen multiplayer
"""
import numpy as np
import math
from typing import Tuple, TYPE_CHECKING, Union, Optional
from dataclasses import dataclass
from enum import Flag, auto

from engine.component import Component, InspectorField
from engine.types import Vector3, Color, ColorType

if TYPE_CHECKING:
    from .gameobject import GameObject
    from .graphics.material import SkyboxMaterial


class ClearFlags(Flag):
    """Flags controlling what a camera clears before rendering."""
    SKYBOX = auto()      # Render skybox
    COLOR = auto()       # Clear with solid color
    DEPTH = auto()       # Clear depth buffer
    NOTHING = auto()     # Don't clear (overlay cameras)
    
    # Common presets
    SKYBOX_CLEAR = SKYBOX | DEPTH
    SOLID_CLEAR = COLOR | DEPTH
    OVERLAY = NOTHING


class RenderLayer(Flag):
    """Layers for selective camera rendering."""
    DEFAULT = auto()
    UI = auto()
    MIRROR = auto()
    MINIMAP = auto()
    WATER = auto()
    PARTICLES = auto()
    
    # Common combinations
    ALL = DEFAULT | UI | MIRROR | MINIMAP | WATER | PARTICLES
    GAME = DEFAULT | WATER | PARTICLES


@dataclass
class Viewport:
    """
    Defines where a camera renders on screen.
    
    Coordinates are normalized (0.0 to 1.0) relative to window size.
    - (0, 0) is bottom-left
    - (1, 1) is top-right
    
    For example:
        Viewport(0, 0, 1, 1)       # Full screen (default)
        Viewport(0.7, 0.7, 0.3, 0.3)  # Top-right corner minimap
        Viewport(0, 0.8, 0.2, 0.2)    # Bottom-left rear-view mirror
    """
    x: float = 0.0
    y: float = 0.0
    width: float = 1.0
    height: float = 1.0
    
    def to_pixels(self, window_width: int, window_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to pixel coordinates."""
        px = int(self.x * window_width)
        py = int(self.y * window_height)
        pw = int(self.width * window_width)
        ph = int(self.height * window_height)
        return (px, py, pw, ph)
    
    def get_aspect_ratio(self, window_aspect: float) -> float:
        """Get the aspect ratio for this viewport."""
        if self.height == 0:
            return window_aspect
        return window_aspect * (self.width / self.height)
    
    @classmethod
    def full_screen(cls) -> 'Viewport':
        """Create a full-screen viewport."""
        return cls(0.0, 0.0, 1.0, 1.0)
    
    @classmethod
    def minimap(cls, corner: str = 'top-right', size: float = 0.25) -> 'Viewport':
        """Create a minimap viewport in a corner."""
        corners = {
            'top-right': cls(1.0 - size, 1.0 - size, size, size),
            'top-left': cls(0.0, 1.0 - size, size, size),
            'bottom-right': cls(1.0 - size, 0.0, size, size),
            'bottom-left': cls(0.0, 0.0, size, size),
        }
        return corners.get(corner, corners['top-right'])
    
    @classmethod
    def mirror(cls, position: str = 'top', width: float = 0.3, height: float = 0.15) -> 'Viewport':
        """Create a rear-view mirror viewport."""
        positions = {
            'top': cls((1.0 - width) / 2, 1.0 - height, width, height),
            'top-left': cls(0.0, 1.0 - height, width, height),
            'top-right': cls(1.0 - width, 1.0 - height, width, height),
        }
        return positions.get(position, positions['top'])


class Camera3D(Component):
    """
    3D Camera component.
    
    Position and rotation are controlled by the GameObject's transform.
    
    Multiple Camera Support:
        - viewport: Where on screen this camera renders (normalized coords)
        - priority: Render order (lower = rendered first)
        - is_main: Marks the primary camera (used for input, audio, etc.)
        - render_mask: Which render layers this camera sees
        - clear_flags: What to clear before rendering
    
    Attributes:
        skybox: Optional SkyboxMaterial for rendering environment background.
                Set this to show a skybox behind the scene.
    """
    
    # Inspector fields for editor support
    skybox = InspectorField(
        "SkyboxMaterial",  # String type name - resolved by editor
        default=None,
        tooltip="Skybox material for environment background (equirectangular or cubemap)"
    )
    fov = InspectorField(
        float,
        default=60.0,
        min_value=1.0,
        max_value=180.0,
        step=1.0,
        decimals=1,
        tooltip="Field of view in degrees"
    )
    near = InspectorField(
        float,
        default=0.1,
        min_value=0.001,
        max_value=100.0,
        step=0.01,
        decimals=3,
        tooltip="Near clipping plane distance"
    )
    far = InspectorField(
        float,
        default=1000.0,
        min_value=1.0,
        max_value=100000.0,
        step=10.0,
        decimals=1,
        tooltip="Far clipping plane distance"
    )
    priority = InspectorField(
        int,
        default=0,
        min_value=0,
        max_value=1000,
        tooltip="Render priority (lower = rendered first)"
    )
    # Note: is_main is a property (see below) - not an InspectorField to avoid conflict
    clear_flags = InspectorField(
        ClearFlags,
        default=ClearFlags.SKYBOX_CLEAR,
        tooltip="What to clear before rendering"
    )
    background_color = InspectorField(
        Color,
        default=(0.1, 0.1, 0.15),
        tooltip="Background color when clear_flags includes COLOR"
    )
    render_mask = InspectorField(
        RenderLayer,
        default=RenderLayer.ALL,
        tooltip="Which render layers this camera sees"
    )
    # Viewport fields (Viewport dataclass exposed as individual fields)
    viewport_x = InspectorField(
        float,
        default=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        decimals=3,
        tooltip="Viewport X position (0-1, normalized)"
    )
    viewport_y = InspectorField(
        float,
        default=0.0,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        decimals=3,
        tooltip="Viewport Y position (0-1, normalized)"
    )
    viewport_width = InspectorField(
        float,
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        decimals=3,
        tooltip="Viewport width (0-1, normalized)"
    )
    viewport_height = InspectorField(
        float,
        default=1.0,
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        decimals=3,
        tooltip="Viewport height (0-1, normalized)"
    )
    
    def __init__(self, 
                 fov: float = 60.0,
                 near: float = 0.1,
                 far: float = 1000.0,
                 viewport: Optional[Viewport] = None,
                 priority: int = 0,
                 is_main: bool = False,
                 clear_flags: Optional[ClearFlags] = None,
                 background_color: Optional[ColorType] = None):
        """
        Initialize camera.
        
        Args:
            fov: Field of view in degrees
            near: Near clipping plane
            far: Far clipping plane
            viewport: Screen region where this camera renders (normalized 0-1)
            priority: Render order (lower values render first)
            is_main: If True, this is the main camera for input/audio
            clear_flags: What to clear before rendering
            background_color: Background color when clear_flags includes COLOR
        """
        super().__init__()
        # InspectorField descriptors handle storage with defaults
        self.fov = fov
        self.near = near
        self.far = far
        self.priority = priority
        if clear_flags is not None:
            self.clear_flags = clear_flags
        if background_color is not None:
            self.background_color = background_color
        self.render_mask = RenderLayer.ALL  # Fixed default, not a constructor param
        
        # Multi-camera support
        vp = viewport if viewport is not None else Viewport.full_screen()
        self._viewport = vp
        # Sync viewport fields from dataclass (via InspectorField descriptors)
        self.viewport_x = vp.x
        self.viewport_y = vp.y
        self.viewport_width = vp.width
        self.viewport_height = vp.height
        
        self._is_main = is_main
        
        # Frustum cache
        self._frustum_cache = {
            "aspect": None,
            "fov": None,
            "tan_x": None,
            "tan_y": None,
        }
        self._target_cache = Vector3.zero()
    
    def __setattr__(self, name: str, value):
        """Override to sync viewport fields with viewport dataclass."""
        # Handle viewport_* fields syncing to viewport dataclass
        if name in ('viewport_x', 'viewport_y', 'viewport_width', 'viewport_height'):
            super().__setattr__(name, value)
            # Sync to _viewport if it exists
            if hasattr(self, '_viewport') and isinstance(self._viewport, Viewport):
                if name == 'viewport_x':
                    self._viewport.x = value
                elif name == 'viewport_y':
                    self._viewport.y = value
                elif name == 'viewport_width':
                    self._viewport.width = value
                elif name == 'viewport_height':
                    self._viewport.height = value
            return
        super().__setattr__(name, value)
    
    @property
    def is_main(self) -> bool:
        """Check if this is the main camera."""
        return self._is_main
    
    @is_main.setter
    def is_main(self, value: bool):
        """Set this camera as main (will update scene if attached)."""
        self._is_main = value
        # Scene will be notified to update its main_camera reference
    
    def set_full_screen(self):
        """Set viewport to full screen."""
        self.viewport = Viewport.full_screen()
        self._sync_viewport_fields()
    
    def set_minimap(self, corner: str = 'top-right', size: float = 0.25):
        """Set viewport as a minimap in a corner."""
        self.viewport = Viewport.minimap(corner, size)
        self._sync_viewport_fields()
        self.priority = 100  # Render on top
        self.clear_flags = ClearFlags.SOLID_CLEAR
    
    def set_mirror(self, position: str = 'top', width: float = 0.3, height: float = 0.15):
        """Set viewport as a rear-view mirror."""
        self.viewport = Viewport.mirror(position, width, height)
        self._sync_viewport_fields()
        self.priority = 50  # Render after main but before UI
        self.clear_flags = ClearFlags.SOLID_CLEAR
    
    def _sync_viewport_fields(self):
        """Sync viewport_* InspectorFields from viewport dataclass."""
        if hasattr(self, 'viewport') and isinstance(self.viewport, Viewport):
            self.viewport_x = self.viewport.x
            self.viewport_y = self.viewport.y
            self.viewport_width = self.viewport.width
            self.viewport_height = self.viewport.height

    @property
    def viewport(self) -> Viewport:
        """Get the viewport dataclass."""
        if not hasattr(self, '_viewport'):
            self._viewport = Viewport.full_screen()
        return self._viewport
    
    @viewport.setter
    def viewport(self, value: Viewport):
        """Set viewport and sync fields."""
        self._viewport = value
        self._sync_viewport_fields()

    @property
    def position(self) -> Vector3:
        """Get camera position from transform."""
        if self.game_object:
            return self.game_object.transform.position
        return Vector3.zero()
    
    @position.setter
    def position(self, value: Union[Tuple[float, float, float], Vector3]):
        """Set camera position via transform."""
        if self.game_object:
            self.game_object.transform.position = value
    
    @property
    def target(self) -> Vector3:
        """Get camera target (look at)."""
        return self._target_cache
    
    @target.setter
    def target(self, value: Union[Tuple[float, float, float], Vector3]):
        """Set camera target (look at)."""
        self.look_at(value)

    @property
    def forward(self) -> np.ndarray:
        """Get forward vector from transform rotation."""
        if self.game_object:
            return self.game_object.transform.forward
        return np.array([0, 0, -1], dtype=np.float32)

    def look_at(self, target: Union[Tuple[float, float, float], Vector3]):
        """Point camera at a target."""
        self._target_cache = Vector3(target)
        if self.game_object:
            self.game_object.transform.look_at(target)

    def orbit(self, dx: float, dy: float):
        """Orbit the camera around its target."""
        if not self.game_object:
            return
        target = self._target_cache
        pos = self.position
        offset = pos - target
        
        radius = offset.magnitude
        if radius < 0.001:
            return
            
        yaw = math.atan2(offset.x, offset.z)
        pitch = math.asin(np.clip(offset.y / radius, -1.0, 1.0))
        
        yaw += dx
        pitch += dy
        
        pitch = np.clip(pitch, -math.pi/2 + 0.01, math.pi/2 - 0.01)
        
        new_offset = Vector3(
            radius * math.sin(yaw) * math.cos(pitch),
            radius * math.sin(pitch),
            radius * math.cos(yaw) * math.cos(pitch)
        )
        
        self.position = target + new_offset
        self.look_at(target)

    def zoom(self, amount: float):
        """Zoom the camera towards or away from its target."""
        if not self.game_object:
            return
        target = self._target_cache
        pos = self.position
        offset = pos - target
        
        radius = offset.magnitude
        if radius < 0.001:
            direction = Vector3.forward()
        else:
            direction = offset.normalized
            
        new_radius = max(0.1, radius + amount)
        self.position = target + direction * new_radius
    
    # Movement methods delegated to transform
    def move(self, dx: float = 0, dy: float = 0, dz: float = 0):
        """Move camera by offset."""
        if self.game_object:
            self.game_object.transform.position = self.game_object.transform.position + Vector3(dx, dy, dz)

    def move_forward(self, distance: float):
        """Move camera forward."""
        if self.game_object:
            self.game_object.transform.position = self.game_object.transform.position + self.forward * distance

    def move_right(self, distance: float):
        """Move camera right."""
        if self.game_object:
            self.game_object.transform.position = self.game_object.transform.position + self.right * distance

    def move_up(self, distance: float):
        """Move camera up."""
        if self.game_object:
            self.game_object.transform.position = self.game_object.transform.position + self.up * distance

    @property
    def right(self) -> np.ndarray:
        """Get right vector."""
        if self.game_object:
            return self.game_object.transform.right
        return np.array([1, 0, 0], dtype=np.float32)

    @property
    def up(self) -> np.ndarray:
        """Get up vector."""
        if self.game_object:
            return self.game_object.transform.up
        return np.array([0, 1, 0], dtype=np.float32)

    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix for rendering."""
        if self.game_object:
            # View matrix is inverse of camera's world matrix
            # But we can construct it from position and rotation
            # Or use look_at with position + forward
            
            eye = self.game_object.transform.position
            # We need to handle rotation correctly.
            # Transform.get_model_matrix gives us the camera's position and orientation in world.
            # View matrix is the inverse of that.
            
            model_matrix = self.game_object.transform.get_model_matrix()
            try:
                return np.linalg.inv(model_matrix)
            except np.linalg.LinAlgError:
                return np.eye(4, dtype=np.float32)
        return np.eye(4, dtype=np.float32)
    
    def get_projection_matrix(self, aspect: float) -> np.ndarray:
        """Get the projection matrix for rendering."""
        return self._perspective_matrix(self.fov, aspect, self.near, self.far)

    # ... (Keep other methods like frustum culling, but update them to use transform) ...

    @staticmethod
    def _perspective_matrix(fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        """Create perspective projection matrix."""
        f = 1.0 / np.tan(np.radians(fov) / 2)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), -1],
            [0, 0, (2 * far * near) / (near - far), 0]
        ], dtype=np.float32)
