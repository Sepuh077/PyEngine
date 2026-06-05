"""
Shadow mapping support for the 3D engine.

Implements basic shadow mapping for directional lights using depth-only rendering
from the light's perspective.
Also supports omnidirectional shadow mapping for point lights using cubemap depth textures.
"""
import numpy as np
import moderngl
from typing import Optional, List, Tuple, TYPE_CHECKING


# Maximum number of shadow-casting lights supported
MAX_SHADOW_LIGHTS = 4


class ShadowMap:
    """
    Manages a shadow map framebuffer for shadow rendering.
    
    Creates a depth texture that stores the depth values from the light's perspective.
    This is then sampled during the main pass to determine if fragments are in shadow.
    """
    
    def __init__(self, ctx: 'moderngl.Context', resolution: int = 1024):
        """
        Initialize the shadow map.
        
        Args:
            ctx: ModernGL context
            resolution: Shadow map resolution (width and height)
        """
        self.ctx = ctx
        self.resolution = resolution
        
        # Create depth-only texture for shadow map
        self.depth_texture = ctx.depth_texture((resolution, resolution))
        self.depth_texture.compare_func = '<='  # Depth comparison for shadow sampling
        self.depth_texture.repeat_x = False
        self.depth_texture.repeat_y = False
        
        # Create framebuffer with depth attachment only
        self.framebuffer = ctx.framebuffer(depth_attachment=self.depth_texture)
        
        # Store previous viewport/FBO for restoration (supports editor custom FBO)
        self._prev_viewport = None
        self._prev_fbo = None
    
    def begin(self):
        """
        Begin shadow pass - bind framebuffer and set viewport.
        
        Call this before rendering objects to the shadow map.
        """
        self._prev_viewport = self.ctx.viewport
        self._prev_fbo = self.ctx.detect_framebuffer()
        self.framebuffer.use()
        self.ctx.viewport = (0, 0, self.resolution, self.resolution)
        
        # Clear depth buffer
        self.ctx.clear(depth=1.0)
        
        # depth writes enabled by default for depth fb
    
    def end(self):
        """
        End shadow pass - restore previous framebuffer (editor custom FBO) and viewport.
        
        Call this after rendering all shadow casters.
        """
        # Restore previous FBO (critical for editor embedding, not just screen)
        if self._prev_fbo:
            self._prev_fbo.use()
        else:
            self.ctx.screen.use()
        
        # Restore viewport
        if self._prev_viewport:
            self.ctx.viewport = self._prev_viewport
    
    def use(self, location: int = 1):
        """
        Bind the shadow map texture for sampling in shaders.
        
        Args:
            location: Texture unit to bind to (default 1, since 0 is often used for other textures)
        """
        self.depth_texture.use(location=location)
    
    def release(self):
        """Release GPU resources."""
        if self.depth_texture:
            self.depth_texture.release()
            self.depth_texture = None
        if self.framebuffer:
            self.framebuffer.release()
            self.framebuffer = None




def calculate_light_space_matrix(
    light_direction: np.ndarray,
    shadow_distance: float = 50.0,
    scene_center: np.ndarray = None,
    scene_radius: float = 20.0
) -> np.ndarray:
    """
    Calculate the light space matrix (projection * view) for shadow rendering.
    
    For directional lights, this uses an orthographic projection that encompasses
    the visible scene area. Uses centered ortho for stability.
    
    Args:
        light_direction: Normalized direction vector pointing FROM the light
        shadow_distance: Distance from camera for shadow rendering
        scene_center: Center of the scene to shadow (default: origin)
        scene_radius: Approximate radius of the scene to encompass
    
    Returns:
        4x4 light space matrix (projection @ view)
    """
    if scene_center is None:
        scene_center = np.array([0.0, 0.0, 0.0])
    
    # Normalize light direction
    light_dir = np.array(light_direction, dtype=np.float32)
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-6)
    
    # Position the light above the scene center
    light_pos = scene_center - light_dir * shadow_distance
    
    # Calculate view matrix (look at scene center from light position)
    # Choose up vector not collinear with light dir (handles vertical/horizontal)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(light_dir[1]) > abs(light_dir[0]) and abs(light_dir[1]) > abs(light_dir[2]):
        world_up = np.array([1.0, 0.0, 0.0])
    elif abs(light_dir[0]) > abs(light_dir[2]):
        world_up = np.array([0.0, 0.0, 1.0])
    
    # Calculate view matrix components
    forward = -light_dir  # Camera looks opposite to light direction
    right = np.cross(forward, world_up)
    right = right / (np.linalg.norm(right) + 1e-6)
    up = np.cross(right, forward)
    
    # Build view matrix
    view = np.eye(4, dtype=np.float32)
    view[0, :3] = right
    view[1, :3] = up
    view[2, :3] = forward
    view[0, 3] = -np.dot(right, light_pos)
    view[1, 3] = -np.dot(up, light_pos)
    view[2, 3] = -np.dot(forward, light_pos)
    
    # Orthographic projection for directional light - centered and stable
    ortho_size = max(scene_radius, shadow_distance * 0.6)
    
    left = -ortho_size
    right = ortho_size
    bottom = -ortho_size
    top = ortho_size
    near = 0.1
    far = shadow_distance * 2.0
    
    proj = np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2.0 / (far - near), -(far + near) / (far - near)],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    return (proj @ view).T


class OmnidirectionalShadowMap:
    """
    Manages an omnidirectional shadow map for point lights using multiple depth textures.
    
    Creates 6 depth textures (one per cubemap face direction) that store depth values
    from all 6 directions around the light. This allows point lights to cast shadows
    in all directions.
    
    Note: We use 6 separate 2D depth textures instead of a cubemap for better
    compatibility across different OpenGL drivers.
    """
    
    # Face indices and their corresponding directions
    # Order: +X, -X, +Y, -Y, +Z, -Z
    FACE_DIRECTIONS = [
        np.array([1.0, 0.0, 0.0]),   # +X (right)
        np.array([-1.0, 0.0, 0.0]),  # -X (left)
        np.array([0.0, 1.0, 0.0]),   # +Y (top)
        np.array([0.0, -1.0, 0.0]),  # -Y (bottom)
        np.array([0.0, 0.0, 1.0]),   # +Z (front)
        np.array([0.0, 0.0, -1.0]),  # -Z (back)
    ]
    
    # Up vectors for each face (to build view matrices)
    FACE_UPS = [
        np.array([0.0, -1.0, 0.0]),   # +X: look down -Y
        np.array([0.0, -1.0, 0.0]),   # -X: look down -Y
        np.array([0.0, 0.0, 1.0]),    # +Y: look forward +Z
        np.array([0.0, 0.0, -1.0]),   # -Y: look back -Z
        np.array([0.0, -1.0, 0.0]),   # +Z: look down -Y
        np.array([0.0, -1.0, 0.0]),   # -Z: look down -Y
    ]
    
    def __init__(self, ctx: 'moderngl.Context', resolution: int = 512, near: float = 0.1, far: float = 50.0):
        """
        Initialize the omnidirectional shadow map.
        
        Args:
            ctx: ModernGL context
            resolution: Shadow map resolution per face
            near: Near plane distance
            far: Far plane distance
        """
        self.ctx = ctx
        self.resolution = resolution
        self.near = near
        self.far = far
        
        # Create 6 separate depth textures and framebuffers (one per face)
        self.depth_textures = []
        self.framebuffers = []
        
        for _ in range(6):
            depth_tex = ctx.depth_texture((resolution, resolution))
            depth_tex.compare_func = '<='
            depth_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
            depth_tex.repeat_x = False
            depth_tex.repeat_y = False
            self.depth_textures.append(depth_tex)
            self.framebuffers.append(ctx.framebuffer(depth_attachment=depth_tex))
        
        # Store previous viewport/FBO for restoration
        self._prev_viewport = None
        self._prev_fbo = None
        
        # Light position (set during rendering)
        self._light_position = np.array([0.0, 0.0, 0.0])
        
        # Pre-compute perspective projection (90° FOV, aspect 1.0)
        self._projection = self._calculate_perspective_projection()
        
        # View matrices for each face (computed when light position is set)
        self._view_matrices: List[np.ndarray] = []
    
    def _calculate_perspective_projection(self) -> np.ndarray:
        """Calculate perspective projection matrix for 90° FOV."""
        fov = 90.0
        aspect = 1.0
        near = self.near
        far = self.far
        
        tan_half_fov = np.tan(np.radians(fov) / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 1.0 / (aspect * tan_half_fov)
        proj[1, 1] = 1.0 / tan_half_fov
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -(2.0 * far * near) / (far - near)
        proj[3, 2] = -1.0
        
        return proj
    
    def _calculate_view_matrix(self, face_index: int) -> np.ndarray:
        """Calculate view matrix for a specific cubemap face."""
        forward = self.FACE_DIRECTIONS[face_index]
        up = self.FACE_UPS[face_index]
        
        # Build look-at view matrix
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-6)
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward  # OpenGL convention: camera looks down -Z
        view[0, 3] = -np.dot(right, self._light_position)
        view[1, 3] = -np.dot(up, self._light_position)
        view[2, 3] = np.dot(forward, self._light_position)
        
        return view
    
    def set_light_position(self, position: np.ndarray):
        """Set the light position and pre-compute view matrices."""
        self._light_position = np.array(position, dtype=np.float32)
        self._view_matrices = [self._calculate_view_matrix(i) for i in range(6)]
    
    def get_view_projection_matrix(self, face_index: int) -> np.ndarray:
        """Get the combined view-projection matrix for a cubemap face."""
        view = self._view_matrices[face_index]
        return (self._projection @ view).T
    
    def begin_face(self, face_index: int):
        """
        Begin rendering to a specific cubemap face.
        
        Args:
            face_index: Index of the cubemap face (0-5)
        """
        if face_index == 0:
            # First face - save state
            self._prev_viewport = self.ctx.viewport
            self._prev_fbo = self.ctx.detect_framebuffer()
        
        self.framebuffers[face_index].use()
        self.ctx.viewport = (0, 0, self.resolution, self.resolution)
        
        # Clear depth buffer
        self.ctx.clear(depth=1.0)
    
    def end_face(self):
        """End rendering to current cubemap face."""
        pass  # No need to do anything between faces
    
    def begin(self):
        """Begin shadow pass - save state. Call before rendering all faces."""
        self._prev_viewport = self.ctx.viewport
        self._prev_fbo = self.ctx.detect_framebuffer()
    
    def end(self):
        """End shadow pass - restore previous framebuffer and viewport."""
        if self._prev_fbo:
            self._prev_fbo.use()
        else:
            self.ctx.screen.use()
        
        if self._prev_viewport:
            self.ctx.viewport = self._prev_viewport
    
    def use(self, location: int = 1):
        """
        Bind the shadow textures for sampling in shaders.
        
        Note: This binds the first face's texture. For proper cubemap sampling,
        the shader should use samplerCubeShadow with a proper cubemap texture.
        For simplicity with separate textures, we bind them sequentially.
        
        Args:
            location: Starting texture unit to bind to
        """
        # Bind the first depth texture (for compatibility with sampler2DShadow)
        # Full cubemap support would require assembling textures into a cubemap
        self.depth_textures[0].use(location=location)
    
    def get_depth_texture(self, face_index: int):
        """Get the depth texture for a specific face."""
        return self.depth_textures[face_index]
    
    def release(self):
        """Release GPU resources."""
        for tex in self.depth_textures:
            if tex:
                tex.release()
        for fb in self.framebuffers:
            if fb:
                fb.release()
        self.depth_textures = []
        self.framebuffers = []


def calculate_point_light_shadow_matrices(
    light_position: np.ndarray,
    near: float = 0.1,
    far: float = 50.0
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Calculate the 6 view-projection matrices for point light shadow rendering.
    
    Args:
        light_position: World position of the point light
        near: Near plane distance
        far: Far plane distance
    
    Returns:
        Tuple of (list of 6 view-projection matrices, projection matrix)
    """
    # Perspective projection with 90° FOV and aspect 1.0
    fov = 90.0
    aspect = 1.0
    tan_half_fov = np.tan(np.radians(fov) / 2.0)
    
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 1.0 / (aspect * tan_half_fov)
    proj[1, 1] = 1.0 / tan_half_fov
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -(2.0 * far * near) / (far - near)
    proj[3, 2] = -1.0
    
    # Face directions and up vectors
    face_dirs = OmnidirectionalShadowMap.FACE_DIRECTIONS
    face_ups = OmnidirectionalShadowMap.FACE_UPS
    
    matrices = []
    for i in range(6):
        forward = face_dirs[i]
        up = face_ups[i]
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-6)
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, light_position)
        view[1, 3] = -np.dot(up, light_position)
        view[2, 3] = np.dot(forward, light_position)
        
        matrices.append((proj @ view).T)
    
    return matrices, proj
