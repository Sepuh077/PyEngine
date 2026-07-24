"Object3D - 3D object that can be loaded, positioned, rotated, and scaled."
import hashlib
import numpy as np
import trimesh
from typing import Tuple, Optional, TYPE_CHECKING

from engine.component import Component
from engine.gameobject import GameObject
from trimesh.visual.texture import TextureVisuals
from engine.types import ColorType
from engine.graphics.material import Material, LitMaterial

if TYPE_CHECKING:
    import moderngl
    from .window import Window3D


class Object3D(Component):
    def __init__(
        self,
        filename: Optional[str] = None,
        color: Optional[ColorType] = None,
        cast_shadows: bool = True,
        receive_shadows: bool = True,
    ):
        super().__init__()
        # ---------------- Geometry ----------------
        self.mesh: Optional[trimesh.Trimesh] = None

        self._local_min = None
        self._local_max = None
        self._local_radius = None

        # Geometry source tracking (for serialization)
        self._source_type = "none"  # none | file | primitive
        self._source_path: Optional[str] = None
        self._primitive_type: Optional[str] = None
        self._primitive_params: dict = {}

        # Texture support (for GLTF etc)
        self._uses_texture = False
        self._texture_image = None
        self._uv = None

        # ---------------- Misc ----------------
        c = color if color is not None else (1, 1, 1)
        c = np.array(c, dtype=np.float32)
        if c.max() > 1.0:
            c /= 255.0
        if len(c) == 3:
            c = np.append(c, 1.0)
        self._color = c
        self._visible = True
        self.material: Material = LitMaterial(color=c)
        
        # Shadow properties
        self.cast_shadows = cast_shadows
        self.receive_shadows = receive_shadows

        # GPU handles (initialized later)
        self._vbo = None
        self._vao = None
        self._gpu_initialized = False

        # Mesh identity for batching/instancing
        self._mesh_key = None
        self._mesh = None

        if filename:
            self.load(filename)
    def load(self, filename: str):
        # Load with trimesh for full support including colors
        self._load_with_trimesh(filename)
        # Post-processing
        self._post_process_geometry(filename)

        # Track geometry source
        self._source_type = "file"
        self._source_path = filename
        self._primitive_type = None
        self._primitive_params = {}

    def _post_process_geometry(self, geometry_name: str):
        if self.mesh is not None and len(self.mesh.vertices) > 0:
            # Center geometry
            center = self.mesh.vertices.mean(axis=0)
            self.mesh.apply_translation(-center)

            self._local_min = self.mesh.vertices.min(axis=0)
            self._local_max = self.mesh.vertices.max(axis=0)
            self._local_radius = np.linalg.norm(self.mesh.vertices, axis=1).max()

            # Ensure normals (trimesh computes lazily on .vertex_normals access)
            _ = self.mesh.vertex_normals
            self._mesh_key = ("geom", geometry_name)
            self._transform_dirty = True

    def _load_with_trimesh(self, filename: str):
        loaded = trimesh.load(filename)

        if isinstance(loaded, trimesh.Scene):
            geometries = list(loaded.geometry.values())
            if not geometries:
                raise ValueError(f"No geometry found in {filename}")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = loaded

        self.mesh = mesh

        self._uses_texture = False
        self._texture_image = None
        self._uv = None

        if not hasattr(mesh, "visual"):
            return

        visual = mesh.visual

        # Vertex colors path (unified normalize/pad)
        if (
            hasattr(visual, "vertex_colors")
            and visual.vertex_colors is not None
            and len(visual.vertex_colors) == len(mesh.vertices)
        ):
            colors = visual.vertex_colors.astype(np.float32)
            if colors.max() > 1.0:
                colors /= 255.0
            if colors.shape[1] == 3:
                colors = np.pad(colors, ((0, 0), (0, 1)), constant_values=1.0)
            mesh.visual.vertex_colors = colors
            return

        # Textured path - always generate vertex colors from texture as fallback
        if isinstance(visual, TextureVisuals):
            material = visual.material
            uv = visual.uv
            img = (
                getattr(material, "baseColorTexture", None)
                or getattr(material, "image", None)
            )

            if img is not None and uv is not None:
                # Check if texture is valid (not a placeholder with all same color or transparent)
                img_arr = np.array(img)
                is_valid_texture = self._is_valid_texture(img_arr)

                # Always generate vertex colors from texture for fallback
                self._generate_vertex_colors_from_texture(mesh, img_arr, uv)

                # Only use texture rendering if it's valid and has transparency needs
                alpha_mode = getattr(material, "alphaMode", "OPAQUE")
                if alpha_mode != "OPAQUE" and is_valid_texture:
                    self._uses_texture = True
                    if img_arr.ndim == 2:
                        img_arr = np.stack([img_arr] * 3, axis=2)
                    img_float = img_arr.astype(np.float32) / 255.0
                    if img_float.shape[2] == 3:
                        img_float = np.pad(img_float, ((0, 0), (0, 0), (0, 1)), constant_values=1.0)
                    self._texture_image = img_float
                    self._uv = uv
                return

        # Simple material fallback
        material = getattr(visual, 'material', None)
        if material is not None:
            # Try various material color attributes
            base = getattr(material, 'baseColor', None)
            if base is None:
                base = getattr(material, 'main_color', None)
            if base is None:
                base = getattr(material, 'baseColorFactor', None)
            if base is None:
                base = getattr(material, 'diffuse', [1.0, 1.0, 1.0, 1.0])
                
            base = np.array(base, dtype=np.float32)
            # Normalize if needed (trimesh often uses 0-255 for main_color)
            if base.max() > 1.0:
                base /= 255.0
                
            if len(base) == 3:
                base = np.append(base, 1.0)
            colors = np.full((len(mesh.vertices), 4), base, dtype=np.float32)
            mesh.visual.vertex_colors = colors

    def _is_valid_texture(self, img_arr: np.ndarray) -> bool:
        """Check if a texture is valid (not a placeholder or empty)."""
        if img_arr is None or img_arr.size == 0:
            return False

        # Check if all pixels are the same (placeholder texture)
        if len(img_arr.shape) >= 3:
            pixels = img_arr.reshape(-1, img_arr.shape[-1])
        else:
            pixels = img_arr.reshape(-1)

        unique_pixels = np.unique(pixels, axis=0)

        # If only 1-2 unique colors and one is transparent, it's likely a placeholder
        if len(unique_pixels) <= 2:
            # Check if any pixel is fully transparent
            if len(unique_pixels.shape) > 1 and unique_pixels.shape[-1] >= 4:
                for pixel in unique_pixels:
                    if pixel[-1] == 0:  # Alpha is 0
                        return False
            # Check if it's a solid color placeholder (very small texture with uniform color)
            if img_arr.shape[0] <= 16 or img_arr.shape[1] <= 16:
                return False

        # Check if the texture is mostly transparent (e.g. broken or empty texture)
        if len(img_arr.shape) >= 3 and img_arr.shape[-1] == 4:
            alpha_channel = img_arr[..., 3]
            # If more than 90% of the texture is fully transparent, consider it invalid
            if (alpha_channel == 0).mean() > 0.9:
                return False

        return True

    def _generate_vertex_colors_from_texture(self, mesh, img_arr: np.ndarray, uv: np.ndarray):
        """Generate vertex colors by sampling the texture at UV coordinates."""
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr] * 3, axis=2)

        img = img_arr.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1

        def _sample_batch(uv_coords: np.ndarray) -> np.ndarray:
            """Vectorised texture lookup for an (N, 2) UV array -> (N, 4) RGBA."""
            us = uv_coords[:, 0] % 1.0
            vs = uv_coords[:, 1] % 1.0
            xs = np.clip((us * (w - 1)).astype(int), 0, w - 1)
            ys = np.clip(((1 - vs) * (h - 1)).astype(int), 0, h - 1)
            sampled = img[ys, xs]  # (N, channels)
            if channels == 3:
                rgba = np.ones((len(sampled), 4), dtype=np.float32)
                rgba[:, :3] = sampled
            else:
                rgba = sampled.copy()
                rgba[:, 3] = 1.0  # force alpha to 1.0
            return rgba

        num_uv = len(uv)
        num_vertices = len(mesh.vertices)
        num_faces = len(mesh.faces)

        if num_uv == num_vertices:
            v_colors = _sample_batch(uv)
        elif num_uv == num_faces * 3:
            face_arr = np.asarray(mesh.faces, dtype=np.intp)  # (F, 3)
            uv_indices = np.arange(num_faces * 3)
            vert_indices = face_arr.ravel()  # (F*3,)
            sampled = _sample_batch(uv[uv_indices])  # (F*3, 4)
            v_colors = np.zeros((num_vertices, 4), dtype=np.float32)
            counts = np.zeros(num_vertices, dtype=np.float32)
            np.add.at(v_colors, vert_indices, sampled)
            np.add.at(counts, vert_indices, 1.0)
            mask = counts > 0
            v_colors[mask] /= counts[mask, np.newaxis]
            v_colors[~mask] = [1, 1, 1, 1]
        else:
            v_colors = np.ones((num_vertices, 4), dtype=np.float32)

        mesh.visual.vertex_colors = v_colors

    # Dirty flag helper (marks transform dirty + colliders for runtime update)
    # =========================================================================
    # Position properties
    # =========================================================================
    @property
    def vertices(self):
        # Delegate to trimesh
        return self.mesh.vertices if self.mesh is not None else None

    @vertices.setter
    def vertices(self, v):
        # Update trimesh mesh
        if self.mesh is None:
            self.mesh = trimesh.Trimesh(vertices=v)
        else:
            self.mesh.vertices = v
        self._local_min = self.mesh.vertices.min(axis=0)
        self._local_max = self.mesh.vertices.max(axis=0)
        self._local_radius = np.linalg.norm(self.mesh.vertices, axis=1).max()
        self._mesh_key = None
    
    # =========================================================================
    # Appearance properties
    # =========================================================================
    
    @property
    def color(self) -> Tuple[float, float, float]:
        """Get color as tuple."""
        return tuple(self.material.color_vec4[:3])
    
    @color.setter
    def color(self, value: ColorType):
        """Set color."""
        self.material.color = value
    
    @property
    def visible(self) -> bool:
        """Is object visible?"""
        return self._visible
    
    @visible.setter
    def visible(self, value: bool):
        """Set visibility."""
        self._visible = value

    def show(self):
        """Make object visible."""
        self._visible = True
    
    def hide(self):
        """Make object invisible."""
        self._visible = False

    # Mesh identity for caching/instancing
    def get_mesh_key(self):
        if self._mesh_key is not None:
            return self._mesh_key

        if self.mesh is None:
            return None

        h = hashlib.blake2b(digest_size=16)
        h.update(self.mesh.vertices.tobytes())
        h.update(self.mesh.faces.tobytes())
        self._mesh_key = ("geom", h.hexdigest())
        return self._mesh_key
    
    # =========================================================================
    # Model matrix (for rendering; no collider)
    # =========================================================================
    
    def get_model_matrix(self) -> np.ndarray:
        return self.game_object.transform.get_model_matrix()

    def get_world_bounding_sphere(self) -> Tuple[np.ndarray, float]:
        """
        World-space bounding sphere (center, radius) for frustum culling.

        Uses local mesh radius scaled by max axis scale + world position.
        """
        go = self.game_object
        if go is None or go.transform is None:
            return np.zeros(3, dtype=np.float32), 1.0
        pos = np.asarray(go.transform.world_position, dtype=np.float32).reshape(3)
        scale = np.asarray(go.transform.world_scale, dtype=np.float32).reshape(3)
        max_s = float(np.max(np.abs(scale))) if scale.size else 1.0
        r = float(self._local_radius) if self._local_radius is not None else 1.0
        return pos, r * max_s * 1.05  # small pad for non-uniform scale / rotation

    # GPU helper for moderngl rendering
    def _get_flattened_geometry(self):
        if self.mesh is None:
            return None, None, None, None

        # Flatten using trimesh data (ensures correct colors from visual)
        faces = self.mesh.faces
        flat_vertices = self.mesh.vertices[faces.flatten()]
        flat_normals = self.mesh.vertex_normals[faces.flatten()]
        
        # Get colors from trimesh visual (fixed extraction + normalize 0-255->0-1)
        visual = self.mesh.visual
        if self._source_type == "primitive":
            # Primitives use white vertex colors so material base_color controls appearance
            # (trimesh assigns default gray 0.4 vertex colors which would tint the result)
            flat_colors = np.ones((len(flat_vertices), 4), dtype=np.float32)
        elif hasattr(visual, "vertex_colors") and visual.vertex_colors is not None:
            # Use per-vertex colors, repeat for face corners
            flat_colors = visual.vertex_colors[faces.flatten()].astype(np.float32)
            if flat_colors.max() > 1.0:
                flat_colors /= 255.0
        else:
            # Default white
            flat_colors = np.ones((len(flat_vertices), 4), dtype=np.float32)
        
        # UVs from visual
        if hasattr(visual, "uv") and visual.uv is not None:
            uv = visual.uv
            # UV layout: per-vertex or per-face-corner
            if len(uv) == len(self.mesh.vertices):
                flat_uvs = uv[faces.flatten()]
            elif len(uv) == len(faces) * 3:
                flat_uvs = uv
            else:
                flat_uvs = np.zeros((len(flat_vertices), 2), dtype=np.float32)
        else:
            # Fallback to _uv for compatibility
            if hasattr(self, "_uv") and self._uv is not None:
                uv = self._uv
                if len(uv) == len(self.mesh.vertices):
                    flat_uvs = uv[faces.flatten()]
                elif len(uv) == len(faces) * 3:
                    flat_uvs = uv
                else:
                    flat_uvs = np.zeros((len(flat_vertices), 2), dtype=np.float32)
            else:
                flat_uvs = np.zeros((len(flat_vertices), 2), dtype=np.float32)
            
        return flat_vertices, flat_normals, flat_colors, flat_uvs

    # =========================================================================
    # GPU methods (called by renderer)
    # =========================================================================
    
    def _init_gpu(self, ctx: 'moderngl.Context', program: 'moderngl.Program'):
        """Initialize GPU resources. Called by renderer."""
        flat_vertices, flat_normals, flat_colors, flat_uvs = self._get_flattened_geometry()
        
        if flat_vertices is None:
             raise RuntimeError("Object has no geometry loaded")
        
        # Interleave data
        vertex_data = np.hstack([flat_vertices, flat_normals, flat_colors, flat_uvs]).astype(np.float32)
        
        # Create GPU buffers
        self._vbo = ctx.buffer(vertex_data.tobytes())
        self._vao = ctx.vertex_array(
            program,
            [(self._vbo, '3f 3f 4f 2f', 'in_position', 'in_normal', 'in_color', 'in_uv')]
        )
        self._gpu_initialized = True
    
    def _release_gpu(self):
        """Release GPU resources."""
        if self._vao:
            self._vao.release()
            self._vao = None
        if self._vbo:
            self._vbo.release()
            self._vbo = None
        self._gpu_initialized = False

    def _rotation_matrix(self):
        return self.game_object.transform._local_quaternion.to_rotation_matrix()

    def __repr__(self):
        return f"Object3D(mesh={self._mesh_key})"



# =============================================================================
# Primitive factory functions
# =============================================================================

def create_cube(size: float = 1.0, 
                position: Tuple[float, float, float] = (0, 0, 0),
                color: Optional[ColorType] = None) -> GameObject:
    go = GameObject()
    go.transform.position = position
    go.transform.scale = size
    obj = Object3D(color=color)
    go.add_component(obj)
    obj._source_type = "primitive"
    obj._primitive_type = "cube"
    obj._primitive_params = {"size": size}
    
    s = 1.0 / 2 # Size handled by scale
    vertices = np.array([
        [-s, -s,  s], [ s, -s,  s], [ s,  s,  s], [-s,  s,  s],
        [-s, -s, -s], [-s,  s, -s], [ s,  s, -s], [ s, -s, -s],
        [-s,  s, -s], [-s,  s,  s], [ s,  s,  s], [ s,  s, -s],
        [-s, -s, -s], [ s, -s, -s], [ s, -s,  s], [-s, -s,  s],
        [ s, -s, -s], [ s,  s, -s], [ s,  s,  s], [ s, -s,  s],
        [-s, -s, -s], [-s, -s,  s], [-s,  s,  s], [-s,  s, -s],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [8, 9, 10], [8, 10, 11],
        [12, 13, 14], [12, 14, 15],
        [16, 17, 18], [16, 18, 19],
        [20, 21, 22], [20, 22, 23],
    ], dtype=np.int32)
    
    obj.mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    obj._post_process_geometry(f"primitive_cube_1")
    
    return go

def create_sphere(radius: float = 1.0, 
                  subdivisions: int = 2,
                  position: Tuple[float, float, float] = (0, 0, 0),
                  color: Optional[ColorType] = None) -> GameObject:
    go = GameObject()
    go.transform.position = position
    go.transform.scale = radius
    obj = Object3D(color=color)
    go.add_component(obj)
    obj._source_type = "primitive"
    obj._primitive_type = "sphere"
    obj._primitive_params = {"radius": radius, "subdivisions": subdivisions}
    
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    obj.mesh = mesh
    obj._post_process_geometry(f"primitive_sphere_1")
    return go

def create_plane(width: float = 10.0, 
                 height: float = 10.0,
                 position: Tuple[float, float, float] = (0, 0, 0),
                 color: Optional[ColorType] = None) -> GameObject:
    go = GameObject()
    go.transform.position = position
    go.transform.scale_xyz = (width, 1.0, height)
    obj = Object3D(color=color)
    go.add_component(obj)
    obj._source_type = "primitive"
    obj._primitive_type = "plane"
    obj._primitive_params = {"width": width, "height": height}
    
    w, h = 1.0 / 2, 1.0 / 2
    vertices = np.array([
        [-w, 0, -h],
        [ w, 0, -h],
        [ w, 0,  h],
        [-w, 0,  h],
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 2, 1],
        [0, 3, 2],
    ], dtype=np.int32)

    obj.mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    obj._post_process_geometry(f"primitive_plane_1_1")

    return go
