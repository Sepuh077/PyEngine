"""
Window2D - Main application window for 2D rendering.

Extends WindowBase — uses the *same* ModernGL pipeline as Window3D
with an orthographic projection and 2D sprite/shape rendering.

Example::

    from engine.d2 import Window2D
    from engine.d2.object2d import create_rect

    class MyGame(Window2D):
        def setup(self):
            self.player = self.add_object(create_rect(32, 32, color=(0, 0.8, 1)))

        def on_update(self):
            pass

        def on_key_press(self, key, modifiers):
            if key == pygame.K_ESCAPE:
                self.close()

    MyGame(800, 600, "My 2D Game").run()
"""
import math
import time
from pathlib import Path
import pygame
import numpy as np
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import moderngl

from engine.window_base import WindowBase
from engine.gameobject import GameObject
from engine.d2.object2d import Object2D
from engine.d2.camera2d import Camera2D
from engine.types import Color, ColorType
from engine.component import Script, Time
from engine.graphics.shader_material import ShaderMaterial

if TYPE_CHECKING:
    from engine.d2.scene2d import Scene2D


class Window2D(WindowBase):
    """
    Main application window for 2D rendering.

    Extends WindowBase (shared ModernGL/Pygame init, event loop, overlay
    drawing, timing).  Adds orthographic projection, Object2D sprite/shape
    rendering via ModernGL, sorting-order layers, and 2D collision.

    Subclass and override ``setup``, ``on_update``, ``on_draw`` etc.
    """

    # -- 2D sprite/shape shaders ------------------------------------------

    SPRITE_VERTEX_SHADER = '''
    #version 330 core
    in vec2 in_position;
    in vec2 in_texcoord;
    uniform mat4 projection;
    uniform mat4 view;
    uniform mat4 model;
    out vec2 v_texcoord;
    void main() {
        gl_Position = projection * view * model * vec4(in_position, 0.0, 1.0);
        v_texcoord = in_texcoord;
    }
    '''

    SPRITE_FRAGMENT_SHADER = '''
    #version 330 core
    in vec2 v_texcoord;
    uniform vec4 base_color;
    uniform sampler2D tex;
    uniform bool use_texture;
    uniform bool is_circle;
    out vec4 frag_color;
    void main() {
        if (is_circle) {
            vec2 center = v_texcoord - vec2(0.5);
            float dist = dot(center, center);
            if (dist > 0.25) discard;
        }
        vec4 color = base_color;
        if (use_texture) {
            color *= texture(tex, v_texcoord);
        }
        if (color.a < 0.001) discard;
        frag_color = color;
    }
    '''

    # -- Instanced 2D shaders ---------------------------------------------

    SPRITE_VERTEX_SHADER_INSTANCED = '''
    #version 330 core
    in vec2 in_position;
    in vec2 in_texcoord;

    // Per-instance attributes
    in vec4 in_model_0;
    in vec4 in_model_1;
    in vec4 in_model_2;
    in vec4 in_model_3;
    in vec4 in_inst_color;
    in float in_inst_flags;   // bit 0 = is_circle

    uniform mat4 projection;
    uniform mat4 view;

    out vec2 v_texcoord;
    out vec4 v_color;
    flat out int v_is_circle;

    void main() {
        mat4 model = mat4(in_model_0, in_model_1, in_model_2, in_model_3);
        gl_Position = projection * view * model * vec4(in_position, 0.0, 1.0);
        v_texcoord = in_texcoord;
        v_color = in_inst_color;
        v_is_circle = int(in_inst_flags) & 1;
    }
    '''

    SPRITE_FRAGMENT_SHADER_INSTANCED = '''
    #version 330 core
    in vec2 v_texcoord;
    in vec4 v_color;
    flat in int v_is_circle;

    uniform sampler2D tex;
    uniform bool use_texture;

    out vec4 frag_color;
    void main() {
        if (v_is_circle == 1) {
            vec2 center = v_texcoord - vec2(0.5);
            float dist = dot(center, center);
            if (dist > 0.25) discard;
        }
        vec4 color = v_color;
        if (use_texture) {
            color *= texture(tex, v_texcoord);
        }
        if (color.a < 0.001) discard;
        frag_color = color;
    }
    '''

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "2D Engine",
        resizable: bool = False,
        project_root: Union[str, Path] = ".",
        auto_load_scriptable_assets: bool = True,
        background_color: ColorType = (0.1, 0.1, 0.15),
        use_pygame_window: bool = True,
        use_pygame_events: bool = True,
    ):
        # WindowBase handles: pygame, moderngl context, overlay shader,
        # 2D HUD surface, timing, input, drawing helpers, main loop.
        # It also calls self._init_gpu() at the end.
        super().__init__(width, height, title, resizable, project_root, auto_load_scriptable_assets, background_color,
                         use_pygame_window, use_pygame_events)

        # Default 2D camera (orthographic_size controls viewport/world size like Unity)
        self._camera_go = GameObject("Default Camera")
        self.camera = Camera2D(orthographic_size=5.0)
        self._camera_go.add_component(self.camera)
        self.camera.set_screen_size(width, height)

        # GPU texture cache for Object2D sprites  {id(obj2d) -> moderngl.Texture}
        self._sprite_textures: dict = {}

        # ShaderMaterial program/VAO cache for 2D
        self._shader_2d_vao_cache: dict = {}

        # ── Editor compatibility attributes ──────────────────────────────
        self.active_camera_override: Optional[Camera2D] = None
        self.show_editor_overlays = False
        self.editor_selected_object: Optional[GameObject] = None
        self.editor_selected_objects: List[GameObject] = []
        self.editor_show_camera = False
        self.editor_show_axis = False
        self.editor_show_gizmo = True
        self.editor_show_colliders = False
        self._editor_gizmo = None

    # =====================================================================
    # GPU init / cleanup  (called by WindowBase)
    # =====================================================================

    # Simple collider/gizmo shader (same as Window3D's)
    _COLLIDER_VS = '''
    #version 330 core
    in vec3 in_position;
    uniform mat4 mvp;
    void main() { gl_Position = mvp * vec4(in_position, 1.0); }
    '''
    _COLLIDER_FS = '''
    #version 330 core
    uniform vec3 color;
    out vec4 frag_color;
    void main() { frag_color = vec4(color, 1.0); }
    '''

    def _init_gpu(self):
        """Compile the 2-D sprite shader and create a unit-quad VAO."""
        self._sprite_program = self._ctx.program(
            vertex_shader=self.SPRITE_VERTEX_SHADER,
            fragment_shader=self.SPRITE_FRAGMENT_SHADER,
        )

        # Instanced sprite program
        self._sprite_program_instanced = self._ctx.program(
            vertex_shader=self.SPRITE_VERTEX_SHADER_INSTANCED,
            fragment_shader=self.SPRITE_FRAGMENT_SHADER_INSTANCED,
        )

        # Collider / gizmo program (for editor overlays)
        self._collider_program = self._ctx.program(
            vertex_shader=self._COLLIDER_VS,
            fragment_shader=self._COLLIDER_FS,
        )

        # Unit quad  (centred at origin, 1×1 in world units)
        quad = np.array([
            # x     y      u    v
            -0.5, -0.5,   0.0, 1.0,
             0.5, -0.5,   1.0, 1.0,
             0.5,  0.5,   1.0, 0.0,
            -0.5, -0.5,   0.0, 1.0,
             0.5,  0.5,   1.0, 0.0,
            -0.5,  0.5,   0.0, 0.0,
        ], dtype=np.float32)
        self._quad_vbo = self._ctx.buffer(quad.tobytes())
        self._quad_vao = self._ctx.vertex_array(
            self._sprite_program,
            [(self._quad_vbo, '2f 2f', 'in_position', 'in_texcoord')],
        )

        # Instanced VAO + dynamic instance buffer
        # Per-instance data: model(4x4=16f) + color(4f) + flags(1f) = 21 floats = 84 bytes
        self._inst_floats_per = 21
        self._inst_bytes_per = self._inst_floats_per * 4
        self._inst_capacity = 64
        self._inst_vbo = self._ctx.buffer(reserve=self._inst_capacity * self._inst_bytes_per)
        self._inst_vao = self._ctx.vertex_array(
            self._sprite_program_instanced,
            [
                (self._quad_vbo, '2f 2f', 'in_position', 'in_texcoord'),
                (self._inst_vbo, '4f 4f 4f 4f 4f 1f /i',
                 'in_model_0', 'in_model_1', 'in_model_2', 'in_model_3',
                 'in_inst_color', 'in_inst_flags'),
            ],
        )

    def _cleanup_gpu(self):
        """Release 2-D GPU resources (called by WindowBase._cleanup)."""
        for tex in self._sprite_textures.values():
            tex.release()
        self._sprite_textures.clear()
        # Release cached ShaderMaterial VAOs
        for vao in getattr(self, '_shader_2d_vao_cache', {}).values():
            try:
                vao.release()
            except Exception:
                pass
        self._shader_2d_vao_cache = {}
        self._inst_vao.release()
        self._inst_vbo.release()
        self._quad_vao.release()
        self._quad_vbo.release()
        self._sprite_program_instanced.release()
        self._sprite_program.release()
        self._collider_program.release()

    def _get_shader_2d_vao(self, program) -> 'moderngl.VertexArray':
        """Get or create a quad VAO bound to a custom shader *program*."""
        key = id(program)
        vao = self._shader_2d_vao_cache.get(key)
        if vao is not None:
            return vao

        # Build attribute list matching the quad VBO layout: 2f(pos) 2f(uv)
        fmt_parts = []
        attr_names = []
        for fmt, name in [('2f', 'in_position'), ('2f', 'in_texcoord')]:
            if name in program:
                fmt_parts.append(fmt)
                attr_names.append(name)
            else:
                n_floats = int(fmt[0])
                fmt_parts.append(f'{n_floats}x4')

        format_str = ' '.join(fmt_parts)
        vao = self._ctx.vertex_array(
            program,
            [(self._quad_vbo, format_str, *attr_names)],
        )
        self._shader_2d_vao_cache[key] = vao
        return vao

    # =====================================================================
    # Scene management
    # =====================================================================

    def show_scene(self, scene: 'Scene2D', start_components: bool = True):
        """Switch to a different Scene2D."""
        if self._current_scene:
            self._current_scene._detach_window()

        self._current_scene = scene
        scene._attach_window(self)

        # Sync camera screen sizes
        if hasattr(scene, '_cameras'):
            for cam in scene._cameras:
                cam.set_screen_size(self.width, self.height)

        scene.on_show()
        self.start(start_components=start_components)

    # =====================================================================
    # Object management
    # =====================================================================

    def add_object(self, obj, **kwargs) -> GameObject:
        """Add a GameObject (or Object2D) to the window."""
        if isinstance(obj, Object2D):
            go = GameObject()
            go.add_component(obj)
        elif isinstance(obj, GameObject):
            go = obj
        else:
            go = GameObject(str(obj))

        position = kwargs.pop('position', None)
        if position is not None:
            go.transform.position = position

        self.objects.append(go)
        go._scene = self
        return go

    def remove_object(self, obj: GameObject):
        if obj in self.objects:
            # Release GPU texture if any
            obj2d = obj.get_component(Object2D)
            if obj2d and id(obj2d) in self._sprite_textures:
                self._sprite_textures.pop(id(obj2d)).release()
            self.objects.remove(obj)
            if hasattr(obj, '_scene'):
                obj._scene = None

    # =====================================================================
    # Coordinate helpers
    # =====================================================================

    def screen_to_world(self, screen_x: float, screen_y: float):
        """Convert screen-pixel position to world coordinates."""
        return self._get_active_camera().screen_to_world(screen_x, screen_y)

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Convert world position to screen-pixel coordinates."""
        return self._get_active_camera().world_to_screen(world_x, world_y)

    # =====================================================================
    # Resize hook
    # =====================================================================

    def on_resize(self, width: int, height: int):
        super().on_resize(width, height)
        cam = self._get_active_camera()
        cam.set_screen_size(self.width, self.height)

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _get_active_camera(self) -> Camera2D:
        if self.active_camera_override:
            return self.active_camera_override
        if self._current_scene:
            return self._current_scene.main_camera
        return self.camera

    def clear_objects(self):
        """Remove all objects from the current scene and the window."""
        if self._current_scene:
            self._current_scene.clear_objects()
        self.objects.clear()
        # Release textures
        for tex in self._sprite_textures.values():
            tex.release()
        self._sprite_textures.clear()

    def project_point(self, world_pos) -> Optional[Tuple[float, float, float]]:
        """Project a world-space point to screen-pixel coordinates.

        Returns ``(px, py, 0.0)`` or *None* if the camera is unavailable.
        Compatible with the Window3D signature used by the editor gizmo.
        """
        cam = self._get_active_camera()
        if cam is None:
            return None
        wx = float(world_pos[0]) if hasattr(world_pos, '__getitem__') else float(world_pos)
        wy = float(world_pos[1]) if hasattr(world_pos, '__getitem__') and len(world_pos) > 1 else 0.0
        sx, sy = cam.world_to_screen(wx, wy)
        return (sx, sy, 0.0)

    # =====================================================================
    # ModernGL 2-D rendering
    # =====================================================================

    def _render(self):
        """Render one frame via ModernGL: clear → sprites → HUD → flip."""
        self.bind_context()

        bg = self.background_color
        self._ctx.clear(float(bg[0]), float(bg[1]), float(bg[2]))

        cam = self._get_active_camera()
        cam.set_screen_size(self.width, self.height)

        # Set camera viewport rect (Unity Camera.rect style) before drawing this camera's content.
        # This allows changing the rendered area/size on screen (sub-viewports, split-screen, etc.).
        orig_viewport = getattr(self._ctx, 'viewport', (0, 0, self.width, self.height))
        try:
            vpw = max(1, int(self.width * getattr(cam, 'viewport_width', 1.0)))
            vph = max(1, int(self.height * getattr(cam, 'viewport_height', 1.0)))
            vpx = int(self.width * getattr(cam, 'viewport_x', 0.0))
            # y from top (2D convention) to GL bottom
            vpy = int(self.height * (1.0 - getattr(cam, 'viewport_y', 0.0) - getattr(cam, 'viewport_height', 1.0)))
            self._ctx.viewport = (vpx, vpy, vpw, vph)
        except Exception:
            pass

        # Build 4×4 view matrix from Camera2D's 3×3
        view3 = cam.get_view_matrix()          # 3×3
        view4 = np.eye(4, dtype=np.float32)
        view4[:2, :2] = view3[:2, :2]
        view4[:2, 3] = view3[:2, 2]

        proj = cam.get_projection_matrix()     # 4×4

        proj_bytes = proj.astype(np.float32).tobytes()
        # Upload in column-major order (GL expects column-major for mat4)
        view_bytes = view4.T.tobytes()

        # Upload camera matrices to both programs
        self._sprite_program['projection'].write(proj_bytes)
        self._sprite_program['view'].write(view_bytes)
        self._sprite_program_instanced['projection'].write(proj_bytes)
        self._sprite_program_instanced['view'].write(view_bytes)

        # Gather and sort visible Object2D by (layer_id, sorting_order)
        renderables: List[Object2D] = []
        for obj in self._active_objects():
            obj2d = obj.get_component(Object2D)
            if obj2d and obj2d.visible:
                renderables.append(obj2d)
        renderables.sort(key=lambda o: o.sort_key)

        # Disable depth test for 2D (sorting_order determines order)
        self._ctx.disable(moderngl.DEPTH_TEST)

        self._render_batched(renderables)

        self._ctx.enable(moderngl.DEPTH_TEST)

        # -- Editor gizmo overlay -----------------------------------------
        if self.editor_show_gizmo and self._editor_gizmo and self.editor_selected_objects:
            self._editor_gizmo.draw(self, self.editor_selected_objects)

        # -- Camera frustum visualisation ---------------------------------
        if self.editor_show_camera and self._current_scene:
            self._draw_camera_frustums(view4, proj)

        # -- Editor collider outlines (2D) ---------------------------------
        # Only draw collider wireframes when in editor mode with colliders enabled
        if self.show_editor_overlays and self.editor_show_colliders:
            self._draw_editor_colliders()

        # Restore full viewport after camera-specific content (for HUD etc.)
        try:
            self._ctx.viewport = orig_viewport
        except Exception:
            self._ctx.viewport = (0, 0, self.width, self.height)

        # -- HUD overlay (screen-space draw helpers + UI canvas) ----------
        self._2d_surface.fill((0, 0, 0, 0))
        if self._current_scene:
            self._current_scene.on_draw()
        self.on_draw()
        self._render_2d_overlay()

        # Clear HUD surface after present
        self._2d_surface.fill((0, 0, 0, 0))

        # Profiler
        if self.show_profiler:
            now = time.time()
            if now - self._last_profiler_time >= self.profiler_interval:
                fps_val = self._clock.get_fps()
                self._profiler_text = f"FPS: {fps_val:.0f}  objs: {len(renderables)}"
                self._apply_caption()
                self._last_profiler_time = now

        if self._use_pygame_window:
            pygame.display.flip()

    # -- Batched instanced rendering ---------------------------------------

    def _render_batched(self, renderables: List[Object2D]):
        """Draw all visible Object2D sprites using highly optimized instanced rendering.

        Objects with a :class:`ShaderMaterial` are rendered individually
        via the per-object path so their custom shader programs are used.
        """
        if not renderables:
            return

        # Separate objects that need custom shader rendering
        standard = []
        custom_shader = []
        for obj2d in renderables:
            shader_mat = getattr(obj2d, 'material', None)
            if isinstance(shader_mat, ShaderMaterial) and shader_mat.shader is not None:
                custom_shader.append(obj2d)
            else:
                standard.append(obj2d)

        # Render custom-shader objects individually
        for obj2d in custom_shader:
            self._render_object2d_gl(obj2d)

        renderables = standard
        if not renderables:
            return

        num_objects = len(renderables)
        
        # 1. Pre-allocate a single NumPy array for ALL instance data upfront.
        # 21 floats per object (4x4 matrix = 16, color = 4, flags = 1)
        all_inst_data = np.empty((num_objects, 21), dtype=np.float32)
        
        keys: List[Optional[int]] = []
        textures: dict = {}

        # Single unified pass to extract textures and build raw data
        for i, obj2d in enumerate(renderables):
            # Cache texture keys
            has_tex = (obj2d._sprite_surface is not None 
                    and not isinstance(obj2d._sprite_surface, (str, bytes)))
            if has_tex:
                tex = self._ensure_sprite_texture(obj2d)
                if tex is not None:
                    k = id(tex)
                    keys.append(k)
                    textures[k] = tex
                else:
                    keys.append(None)
            else:
                keys.append(None)
            
            # Directly write data into our pre-allocated NumPy block
            # Ensure _build_instance_data returns a flat list/tuple or numpy slice
            all_inst_data[i] = self._build_instance_data(obj2d) 

        # Handle dynamic GPU buffer resizing once before the draw sequence starts
        if self._inst_capacity < num_objects:
            self._inst_capacity = max(num_objects, self._inst_capacity * 2)
            self._inst_vbo.orphan(self._inst_capacity * self._inst_bytes_per)
            self._inst_vao.release()
            self._inst_vao = self._ctx.vertex_array(
                self._sprite_program_instanced,
                [
                    (self._quad_vbo, '2f 2f', 'in_position', 'in_texcoord'),
                    (self._inst_vbo, '4f 4f 4f 4f 4f 1f /i',
                    'in_model_0', 'in_model_1', 'in_model_2', 'in_model_3',
                    'in_inst_color', 'in_inst_flags'),
                ],
            )

        # Walk through and issue draws using slices of our master array
        current_key = keys[0]
        batch_start = 0

        for i in range(num_objects):
            if keys[i] != current_key:
                # Draw the current batch run
                count = i - batch_start
                if count > 0:
                    # Slice the pre-allocated array instead of casting a Python list
                    batch_slice = all_inst_data[batch_start:i]
                    self._inst_vbo.write(batch_slice.tobytes())
                    
                    use_tex = current_key is not None and current_key in textures
                    if use_tex:
                        textures[current_key].use(location=0)
                        self._sprite_program_instanced['tex'].value = 0
                    self._sprite_program_instanced['use_texture'].value = use_tex
                    
                    self._inst_vao.render(moderngl.TRIANGLES, instances=count)

                batch_start = i
                current_key = keys[i]

        # Flush the absolute final batch
        count = num_objects - batch_start
        if count > 0:
            batch_slice = all_inst_data[batch_start:num_objects]
            self._inst_vbo.write(batch_slice.tobytes())
            use_tex = current_key is not None and current_key in textures
            if use_tex:
                textures[current_key].use(location=0)
                self._sprite_program_instanced['tex'].value = 0
            self._sprite_program_instanced['use_texture'].value = use_tex
            self._inst_vao.render(moderngl.TRIANGLES, instances=count)

    def _build_instance_data(self, obj2d: Object2D) -> np.ndarray:
        """Return a flat float32 array (21 floats) for one sprite instance.

        Layout: model(16) + rgba(4) + flags(1)
        """
        go = obj2d.game_object
        if not go:
            return np.zeros(self._inst_floats_per, dtype=np.float32)

        pos = go.transform.position
        wx, wy = float(pos.x), float(pos.y)

        t_scale = go.transform.scale_xyz
        if hasattr(t_scale, 'x'):
            sx_f, sy_f = float(t_scale.x), float(t_scale.y)
        elif isinstance(t_scale, (tuple, list)):
            sx_f = float(t_scale[0])
            sy_f = float(t_scale[1]) if len(t_scale) > 1 else sx_f
        else:
            sx_f = sy_f = float(t_scale)

        size = obj2d.size
        w = size.x * sx_f
        h = size.y * sy_f

        rot = go.transform.rotation
        if hasattr(rot, '__getitem__'):
            rot_z = float(rot[2]) if len(rot) > 2 else 0.0
        else:
            rot_z = float(rot)
        rad = math.radians(rot_z)
        c_r, s_r = math.cos(rad), math.sin(rad)

        flip_x = getattr(obj2d, 'flip_x', False)
        flip_y = getattr(obj2d, 'flip_y', False)

        sx = w * (-1 if flip_x else 1)
        sy = h * (-1 if flip_y else 1) * (-1)

        # model matrix  (T * R * S) — column-major flat
        col = obj2d._color
        flags = 1.0 if getattr(obj2d, '_shape', None) == 'circle' else 0.0

        return np.array([
            sx * c_r,  sx * s_r, 0, 0,      # column 0
            -sy * s_r, sy * c_r, 0, 0,      # column 1
            0,         0,       1, 0,        # column 2
            wx,        wy,      0, 1,        # column 3
            float(col[0]), float(col[1]), float(col[2]), float(col[3]),
            flags,
        ], dtype=np.float32)

    # -- Per-object GL rendering (kept for editor / debug) -----------------

    def _render_object2d_gl(self, obj2d: Object2D):
        """Render one Object2D as a textured/colored quad via ModernGL.

        If the Object2D's ``game_object`` has a :class:`ShaderMaterial`
        attached (stored on the Object2D as ``material``), we compile the
        custom shader, upload its uniforms, and render through it instead
        of the standard sprite program.
        """
        go = obj2d.game_object
        if not go:
            return

        pos = go.transform.position
        wx, wy = float(pos.x), float(pos.y)

        # Scale
        t_scale = go.transform.scale_xyz
        if hasattr(t_scale, 'x'):
            sx_f, sy_f = float(t_scale.x), float(t_scale.y)
        elif isinstance(t_scale, (tuple, list)):
            sx_f = float(t_scale[0])
            sy_f = float(t_scale[1]) if len(t_scale) > 1 else sx_f
        else:
            sx_f = sy_f = float(t_scale)

        size = obj2d.size
        w = size.x * sx_f
        h = size.y * sy_f

        rot = go.transform.rotation
        if hasattr(rot, '__getitem__'):
            rot_z = float(rot[2]) if len(rot) > 2 else 0.0
        else:
            rot_z = float(rot)
        rad = math.radians(rot_z)
        c_r, s_r = math.cos(rad), math.sin(rad)

        flip_x = getattr(obj2d, 'flip_x', False)
        flip_y = getattr(obj2d, 'flip_y', False)

        sx = w * (-1 if flip_x else 1)
        sy = h * (-1 if flip_y else 1) * (-1)

        model = np.array([
            [sx * c_r,  -sy * s_r, 0, 0],
            [sx * s_r,   sy * c_r, 0, 0],
            [0,         0,       1, 0],
            [wx,        wy,      0, 1],
        ], dtype=np.float32)

        # ── Check for ShaderMaterial ────────────────────────────────
        shader_mat = getattr(obj2d, 'material', None)
        if isinstance(shader_mat, ShaderMaterial) and shader_mat.shader is not None:
            try:
                custom_prog = shader_mat.shader.compile(self._ctx)
            except Exception:
                custom_prog = None

            if custom_prog is not None:
                custom_vao = self._get_shader_2d_vao(custom_prog)

                # Camera matrices (already uploaded to standard prog above)
                cam = self._get_active_camera()
                view3 = cam.get_view_matrix()
                view4 = np.eye(4, dtype=np.float32)
                view4[:2, :2] = view3[:2, :2]
                view4[:2, 3] = view3[:2, 2]
                proj = cam.get_projection_matrix()

                if 'projection' in custom_prog:
                    custom_prog['projection'].write(proj.astype(np.float32).tobytes())
                if 'view' in custom_prog:
                    custom_prog['view'].write(view4.T.tobytes())
                if 'model' in custom_prog:
                    custom_prog['model'].write(model.T.tobytes())

                # Base colour
                col = obj2d._color
                if 'base_color' in custom_prog:
                    custom_prog['base_color'].value = (
                        float(col[0]), float(col[1]), float(col[2]), float(col[3]),
                    )

                # Texture
                has_texture = obj2d._sprite_surface is not None and not isinstance(obj2d._sprite_surface, (str, bytes))
                if has_texture:
                    tex = self._ensure_sprite_texture(obj2d)
                    if tex is not None:
                        tex.use(location=0)
                        if 'tex' in custom_prog:
                            custom_prog['tex'].value = 0
                    else:
                        has_texture = False
                if 'use_texture' in custom_prog:
                    custom_prog['use_texture'].value = has_texture
                if 'is_circle' in custom_prog:
                    custom_prog['is_circle'].value = getattr(obj2d, '_shape', None) == 'circle'

                # Upload shader-specific properties
                shader_mat.upload_uniforms(custom_prog)

                custom_vao.render(moderngl.TRIANGLES)
                return

        # ── Standard sprite path ───────────────────────────────────
        self._sprite_program['model'].write(model.T.tobytes())

        has_texture = obj2d._sprite_surface is not None and not isinstance(obj2d._sprite_surface, (str, bytes))
        if has_texture:
            tex = self._ensure_sprite_texture(obj2d)
            if tex is not None:
                tex.use(location=0)
                self._sprite_program['tex'].value = 0
            else:
                has_texture = False

        self._sprite_program['use_texture'].value = has_texture
        self._sprite_program['is_circle'].value = getattr(obj2d, '_shape', None) == 'circle'

        col = obj2d._color
        self._sprite_program['base_color'].value = (
            float(col[0]), float(col[1]), float(col[2]), float(col[3]),
        )

        self._quad_vao.render(moderngl.TRIANGLES)

    def _ensure_sprite_texture(self, obj2d: Object2D) -> 'moderngl.Texture':
        """Upload the Object2D's pygame Surface as a ModernGL texture (cached)."""
        key = id(obj2d)
        if key in self._sprite_textures and not obj2d._texture_dirty:
            return self._sprite_textures[key]

        surf = obj2d._sprite_surface

        # Defensive guard: after scene load / deserialization / editor state changes,
        # _sprite_surface can end up as a str (path) or other junk if the Surface
        # was serialized as repr and blindly restored into __dict__.
        # We treat any non-Surface as "no texture" and clear the bad state.
        if surf is None or not hasattr(surf, "get_size") or isinstance(surf, (str, bytes)):
            obj2d._sprite_surface = None
            obj2d._texture_dirty = True
            return None

        try:
            w, h = surf.get_size()
            data = pygame.image.tostring(surf, "RGBA", True)
        except Exception:
            obj2d._sprite_surface = None
            obj2d._texture_dirty = True
            return None

        if key in self._sprite_textures:
            old = self._sprite_textures[key]
            if old.size == (w, h):
                old.write(data)
                obj2d._texture_dirty = False
                return old
            old.release()

        tex = self._ctx.texture((w, h), 4, data)
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        self._sprite_textures[key] = tex
        obj2d._texture_dirty = False
        return tex

    # =====================================================================
    # Camera frustum drawing (editor)
    # =====================================================================

    def _draw_camera_frustums(self, view4: np.ndarray, proj: np.ndarray):
        """Draw a wireframe rectangle for each scene Camera2D showing its view bounds."""
        scene = self._current_scene
        if not scene or not hasattr(scene, '_cameras'):
            return

        mvp = (proj @ view4).astype(np.float32)

        for cam in scene._cameras:
            # Skip the editor override camera — only draw game cameras
            if cam is self.active_camera_override:
                continue

            pos = cam.position
            cx, cy = float(pos.x), float(pos.y)

            # With orthographic_size, the vertical half-extent is the size (Unity convention).
            # Horizontal follows aspect from the camera's (effective) screen size.
            hh = getattr(cam, 'orthographic_size', 5.0)
            aspect = cam._screen_width / max(1.0, cam._screen_height)
            hw = hh * aspect

            # Corners (CCW)
            corners = [
                (cx - hw, cy - hh),
                (cx + hw, cy - hh),
                (cx + hw, cy + hh),
                (cx - hw, cy + hh),
            ]

            # Build line-loop vertices (4 edges = 8 vertices)
            verts = []
            for i in range(4):
                x0, y0 = corners[i]
                x1, y1 = corners[(i + 1) % 4]
                verts.extend([x0, y0, 0.0, x1, y1, 0.0])

            vbo = self._ctx.buffer(np.array(verts, dtype=np.float32).tobytes())
            vao = self._ctx.vertex_array(
                self._collider_program,
                [(vbo, '3f', 'in_position')],
            )

            self._collider_program['mvp'].write(mvp.T.tobytes())
            self._collider_program['color'].value = (0.9, 0.9, 0.2)  # yellow
            vao.render(moderngl.LINES)

            # Draw small cross at camera position
            cross_size = max(hw, hh) * 0.03
            cross_verts = [
                cx - cross_size, cy, 0.0,  cx + cross_size, cy, 0.0,
                cx, cy - cross_size, 0.0,  cx, cy + cross_size, 0.0,
            ]
            vbo2 = self._ctx.buffer(np.array(cross_verts, dtype=np.float32).tobytes())
            vao2 = self._ctx.vertex_array(
                self._collider_program,
                [(vbo2, '3f', 'in_position')],
            )
            vao2.render(moderngl.LINES)

            vao.release()
            vbo.release()
            vao2.release()
            vbo2.release()

    # =====================================================================
    # Editor collider debug drawing (for 2D colliders in editor)
    # =====================================================================

    def _draw_editor_colliders(self):
        """Draw red wireframe outlines for Collider2D components (editor visualization)."""
        scene = self._current_scene
        if not scene:
            return
        objs = getattr(scene, "objects", []) or []
        if not objs:
            return

        cam = self._get_active_camera()
        if cam is None:
            return

        # Reuse the same view/proj construction as main render + frustum drawing
        view3 = cam.get_view_matrix()
        view4 = np.eye(4, dtype=np.float32)
        view4[:2, :2] = view3[:2, :2]
        view4[:2, 3] = view3[:2, 2]
        proj = cam.get_projection_matrix()
        mvp = (proj @ view4).astype(np.float32)

        import moderngl
        import math as _math  # local to avoid shadowing

        from engine.d2.physics.collider import (
            Collider2D, BoxCollider2D, CircleCollider2D, CapsuleCollider2D, PolygonCollider2D
        )

        color = (1.0, 0.0, 0.0)  # red, matching 3D editor collider color
        self._ctx.line_width = 1.5
        self._collider_program['color'].value = color

        # Draw on top (disable depth like gizmos do in editor)
        self._ctx.disable(moderngl.DEPTH_TEST)

        for obj in objs:
            for coll in obj.get_components(Collider2D):
                if not coll or not getattr(coll, 'game_object', None):
                    continue
                try:
                    coll.update_bounds()
                except Exception:
                    continue

                verts = []  # flat list of x,y,0 , x,y,0 ...

                if isinstance(coll, BoxCollider2D) and getattr(coll, 'obb', None):
                    c, angle, he = coll.obb
                    cx, cy = float(c[0]), float(c[1])
                    hx, hy = float(he[0]), float(he[1])
                    ca, sa = _math.cos(angle), _math.sin(angle)
                    def _rot(x, y):
                        return (cx + x * ca - y * sa, cy + x * sa + y * ca)
                    p0 = _rot(-hx, -hy)
                    p1 = _rot( hx, -hy)
                    p2 = _rot( hx,  hy)
                    p3 = _rot(-hx,  hy)
                    for a, b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
                        verts.extend([a[0], a[1], 0.0, b[0], b[1], 0.0])

                elif isinstance(coll, CircleCollider2D) and getattr(coll, 'circle', None):
                    c, r = coll.circle
                    cx, cy, rr = float(c[0]), float(c[1]), float(r)
                    segs = 24
                    for i in range(segs):
                        a0 = 2 * _math.pi * i / segs
                        a1 = 2 * _math.pi * (i + 1) / segs
                        verts.extend([
                            cx + rr * _math.cos(a0), cy + rr * _math.sin(a0), 0.0,
                            cx + rr * _math.cos(a1), cy + rr * _math.sin(a1), 0.0,
                        ])

                elif isinstance(coll, CapsuleCollider2D) and getattr(coll, 'capsule', None):
                    c, rad, hh, direc = coll.capsule
                    cx, cy = float(c[0]), float(c[1])
                    r, h = float(rad), float(hh)
                    segs = 16
                    # Connecting rect sides
                    if direc == 0:  # vertical
                        verts.extend([cx - r, cy - h, 0, cx - r, cy + h, 0])
                        verts.extend([cx + r, cy - h, 0, cx + r, cy + h, 0])
                        # two end circles
                        for sign in (-1, 1):
                            cy2 = cy + sign * h
                            for i in range(segs):
                                a0 = 2 * _math.pi * i / segs
                                a1 = 2 * _math.pi * (i + 1) / segs
                                verts.extend([
                                    cx + r * _math.cos(a0), cy2 + r * _math.sin(a0), 0,
                                    cx + r * _math.cos(a1), cy2 + r * _math.sin(a1), 0,
                                ])
                    else:  # horizontal
                        verts.extend([cx - h, cy - r, 0, cx + h, cy - r, 0])
                        verts.extend([cx - h, cy + r, 0, cx + h, cy + r, 0])
                        for sign in (-1, 1):
                            cx2 = cx + sign * h
                            for i in range(segs):
                                a0 = 2 * _math.pi * i / segs
                                a1 = 2 * _math.pi * (i + 1) / segs
                                verts.extend([
                                    cx2 + r * _math.cos(a0), cy + r * _math.sin(a0), 0,
                                    cx2 + r * _math.cos(a1), cy + r * _math.sin(a1), 0,
                                ])

                elif isinstance(coll, PolygonCollider2D) and getattr(coll, 'world_points', None):
                    pts = coll.world_points
                    n = len(pts)
                    for i in range(n):
                        x0, y0 = float(pts[i][0]), float(pts[i][1])
                        x1, y1 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
                        verts.extend([x0, y0, 0.0, x1, y1, 0.0])

                if len(verts) < 6:
                    continue

                vbo = self._ctx.buffer(np.array(verts, dtype=np.float32).tobytes())
                vao = self._ctx.vertex_array(
                    self._collider_program,
                    [(vbo, '3f', 'in_position')],
                )
                self._collider_program['mvp'].write(mvp.T.tobytes())
                vao.render(moderngl.LINES)
                vao.release()
                vbo.release()

        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.line_width = 1.0

    # =====================================================================
    # Public collider debug drawing
    # =====================================================================

    def draw_collider(self, obj, color=(0, 1, 0), line_width=1.0):
        """Draw wireframe outlines for all Collider2D components on a GameObject.

        Matches the Window3D ``draw_collider`` API.  Call from ``on_draw()``
        to visualise colliders at runtime.

        Args:
            obj:        A :class:`GameObject` whose Collider2D components will be drawn.
            color:      RGB tuple in 0-1 range.  Default is green.
            line_width: GL line width.  Default is 1.0.

        Example::

            def on_draw(self):
                self.window.draw_collider(self.player)
                self.window.draw_collider(self.enemy, color=(1, 0, 0), line_width=2.0)
        """
        import math as _math

        from engine.d2.physics.collider import (
            Collider2D, BoxCollider2D, CircleCollider2D,
            CapsuleCollider2D, PolygonCollider2D,
        )

        if not obj:
            return

        # Accept a bare Collider2D for convenience
        if isinstance(obj, Collider2D):
            colliders = [obj]
        else:
            colliders = obj.get_components(Collider2D)

        if not colliders:
            return

        cam = self._get_active_camera()
        if cam is None:
            return

        # Compute MVP once for all colliders on this object
        view3 = cam.get_view_matrix()
        view4 = np.eye(4, dtype=np.float32)
        view4[:2, :2] = view3[:2, :2]
        view4[:2, 3] = view3[:2, 2]
        proj = cam.get_projection_matrix()
        mvp = (proj @ view4).astype(np.float32)

        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.line_width = line_width
        self._collider_program['color'].value = tuple(float(c) for c in color[:3])
        self._collider_program['mvp'].write(mvp.tobytes())

        for collider in colliders:
            if not getattr(collider, 'game_object', None):
                continue
            try:
                collider.update_bounds()
            except Exception:
                continue

            verts = []

            if isinstance(collider, BoxCollider2D) and getattr(collider, 'obb', None):
                c, angle, he = collider.obb
                cx, cy = float(c[0]), float(c[1])
                hx, hy = float(he[0]), float(he[1])
                ca, sa = _math.cos(angle), _math.sin(angle)
                def _rot(x, y, _cx=cx, _cy=cy, _ca=ca, _sa=sa):
                    return (_cx + x * _ca - y * _sa, _cy + x * _sa + y * _ca)
                p0 = _rot(-hx, -hy)
                p1 = _rot( hx, -hy)
                p2 = _rot( hx,  hy)
                p3 = _rot(-hx,  hy)
                for a, b in [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]:
                    verts.extend([a[0], a[1], 0.0, b[0], b[1], 0.0])

            elif isinstance(collider, CircleCollider2D) and getattr(collider, 'circle', None):
                c, r = collider.circle
                cx, cy, rr = float(c[0]), float(c[1]), float(r)
                segs = 24
                for i in range(segs):
                    a0 = 2 * _math.pi * i / segs
                    a1 = 2 * _math.pi * (i + 1) / segs
                    verts.extend([
                        cx + rr * _math.cos(a0), cy + rr * _math.sin(a0), 0.0,
                        cx + rr * _math.cos(a1), cy + rr * _math.sin(a1), 0.0,
                    ])

            elif isinstance(collider, CapsuleCollider2D) and getattr(collider, 'capsule', None):
                c, rad, hh, direc = collider.capsule
                cx, cy = float(c[0]), float(c[1])
                r, h = float(rad), float(hh)
                segs = 16
                if direc == 0:
                    verts.extend([cx - r, cy - h, 0, cx - r, cy + h, 0])
                    verts.extend([cx + r, cy - h, 0, cx + r, cy + h, 0])
                    for sign in (-1, 1):
                        cy2 = cy + sign * h
                        for i in range(segs):
                            a0 = 2 * _math.pi * i / segs
                            a1 = 2 * _math.pi * (i + 1) / segs
                            verts.extend([
                                cx + r * _math.cos(a0), cy2 + r * _math.sin(a0), 0,
                                cx + r * _math.cos(a1), cy2 + r * _math.sin(a1), 0,
                            ])
                else:
                    verts.extend([cx - h, cy - r, 0, cx + h, cy - r, 0])
                    verts.extend([cx - h, cy + r, 0, cx + h, cy + r, 0])
                    for sign in (-1, 1):
                        cx2 = cx + sign * h
                        for i in range(segs):
                            a0 = 2 * _math.pi * i / segs
                            a1 = 2 * _math.pi * (i + 1) / segs
                            verts.extend([
                                cx2 + r * _math.cos(a0), cy + r * _math.sin(a0), 0,
                                cx2 + r * _math.cos(a1), cy + r * _math.sin(a1), 0,
                            ])

            elif isinstance(collider, PolygonCollider2D) and getattr(collider, 'world_points', None) is not None:
                pts = collider.world_points
                n = len(pts)
                for i in range(n):
                    x0, y0 = float(pts[i][0]), float(pts[i][1])
                    x1, y1 = float(pts[(i + 1) % n][0]), float(pts[(i + 1) % n][1])
                    verts.extend([x0, y0, 0.0, x1, y1, 0.0])

            if len(verts) < 6:
                continue

            vbo = self._ctx.buffer(np.array(verts, dtype=np.float32).tobytes())
            vao = self._ctx.vertex_array(
                self._collider_program,
                [(vbo, '3f', 'in_position')],
            )
            vao.render(moderngl.LINES)
            vao.release()
            vbo.release()

        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.line_width = 1.0

    # =====================================================================
    # 2D Physics collision processing
    # =====================================================================

    def _process_collisions(self):
        from engine.d2.physics import (
            Collider2D, CollisionMode, CollisionRelation,
            objects_collide_2d, get_collision_manifold_2d,
        )
        from engine.d2.physics.rigidbody import Rigidbody2D
        from engine.d2.physics.types import ColliderType2D
        from collections import defaultdict

        all_cols: List[Collider2D] = []
        for o in self._active_objects():
            all_cols.extend(o.get_components(Collider2D))
        if not all_cols:
            return

        n = len(all_cols)
        current_collisions: dict = defaultdict(set)
        # Bodies that rested on static / sleeping support this frame (for float wake)
        self._physics_supported_bodies: set = set()

        # --- Eagerly update bounds for every collider ----------------
        for c in all_cols:
            c.update_bounds()

        # =============================================================
        # Vectorised AABB broadphase  (N×N boolean overlap matrix)
        # =============================================================
        aabb_mins = np.empty((n, 2), dtype=np.float64)
        aabb_maxs = np.empty((n, 2), dtype=np.float64)
        valid_aabb = np.ones(n, dtype=bool)

        for i, c in enumerate(all_cols):
            if c.aabb is not None:
                aabb_mins[i] = c.aabb[0]
                aabb_maxs[i] = c.aabb[1]
            else:
                valid_aabb[i] = False

        # overlap[i,j] = True iff all four AABB conditions hold
        overlap = (
            np.all(aabb_maxs[:, None, :] >= aabb_mins[None, :, :], axis=2)
            & np.all(aabb_mins[:, None, :] <= aabb_maxs[None, :, :], axis=2)
        )
        overlap &= valid_aabb[:, None] & valid_aabb[None, :]
        np.fill_diagonal(overlap, False)

        # =============================================================
        # Initiator mask  (matches the original outer-loop skip)
        # =============================================================
        active = np.ones(n, dtype=bool)
        for i, c in enumerate(all_cols):
            if c.collision_mode == CollisionMode.IGNORE:
                active[i] = False
                continue
            if c.game_object:
                rb = getattr(c.game_object, '_rigidbody', None)
                if rb is None:
                    rb = c.game_object.get_component(Rigidbody2D)
                if rb and rb.is_static:
                    active[i] = False

        # =============================================================
        # Same-object mask  (skip pairs on the same GameObject)
        # =============================================================
        obj_ids = np.array(
            [id(c.game_object) if c.game_object else 0 for c in all_cols],
            dtype=np.int64,
        )
        same_obj = obj_ids[:, None] == obj_ids[None, :]

        # =============================================================
        # Group-relation lookup  (IGNORE=0 / TRIGGER=1 / SOLID=2)
        # =============================================================
        unique_groups = list({c.group for c in all_cols})
        gid_map = {id(g): idx for idx, g in enumerate(unique_groups)}
        ng = len(unique_groups)

        rel_lut = np.empty((ng, ng), dtype=np.int8)
        for gi, ga in enumerate(unique_groups):
            for gj, gb in enumerate(unique_groups):
                rel_lut[gi, gj] = ga.get_relation(gb).value

        col_gidx = np.array(
            [gid_map[id(c.group)] for c in all_cols], dtype=np.int32,
        )
        pair_rel = rel_lut[col_gidx[:, None], col_gidx[None, :]]  # (N,N)
        non_ignore = pair_rel != CollisionRelation.IGNORE.value

        # =============================================================
        # Combined candidate matrix
        # =============================================================
        candidates = active[:, None] & (~same_obj) & overlap & non_ignore

        # =============================================================
        # Vectorised circle-vs-circle narrowphase
        # =============================================================
        types = np.array([c.type.value for c in all_cols], dtype=np.int32)
        circ_mask = types == ColliderType2D.CIRCLE.value
        both_circ = circ_mask[:, None] & circ_mask[None, :]
        circ_cands = candidates & both_circ

        circ_idxs = np.where(circ_mask)[0]
        nc = len(circ_idxs)

        if nc > 1 and np.any(circ_cands):
            c_centers = np.array(
                [all_cols[k].circle[0] for k in circ_idxs], dtype=np.float64,
            )  # (nc, 2)
            c_radii = np.array(
                [all_cols[k].circle[1] for k in circ_idxs], dtype=np.float64,
            )  # (nc,)

            diff = c_centers[:, None, :] - c_centers[None, :, :]  # (nc,nc,2)
            dist_sq = np.sum(diff * diff, axis=2)                 # (nc,nc)
            r_sum = c_radii[:, None] + c_radii[None, :]           # (nc,nc)
            circle_hit = dist_sq <= r_sum * r_sum                 # (nc,nc)

            # Intersect with the candidate mask (mapped to local indices)
            local_cands = circ_cands[np.ix_(circ_idxs, circ_idxs)]
            local_hits = local_cands & circle_hit

            li, lj = np.nonzero(local_hits)
            for k in range(len(li)):
                i_g = int(circ_idxs[li[k]])
                j_g = int(circ_idxs[lj[k]])
                # Skip reverse pair only when the other body would also initiate
                # (two dynamics). Keep dynamic→static when static has lower index.
                if i_g > j_g and bool(active[j_g]):
                    continue
                ca, cb = all_cols[i_g], all_cols[j_g]
                current_collisions[ca].add(cb)
                current_collisions[cb].add(ca)
                if pair_rel[i_g, j_g] == CollisionRelation.SOLID.value:
                    manifold = get_collision_manifold_2d(ca, cb)
                    if manifold:
                        self._resolve_collision_2d(
                            ca.game_object, cb.game_object, manifold,
                            col_a=ca, col_b=cb,
                        )

        # =============================================================
        # Non-circle-circle pairs — per-pair narrowphase
        # =============================================================
        other_cands = candidates & ~both_circ
        ri, rj = np.nonzero(other_cands)
        for k in range(len(ri)):
            i, j = int(ri[k]), int(rj[k])
            # Avoid double impulses on dynamic–dynamic pairs (both directions
            # appear in the candidate matrix). Still allow dynamic→static when
            # the static collider has a lower index.
            if i > j and bool(active[j]):
                continue
            ca, cb = all_cols[i], all_cols[j]
            if objects_collide_2d(ca, cb):
                current_collisions[ca].add(cb)
                current_collisions[cb].add(ca)
                if pair_rel[i, j] == CollisionRelation.SOLID.value:
                    manifold = get_collision_manifold_2d(ca, cb)
                    if manifold:
                        self._resolve_collision_2d(
                            ca.game_object, cb.game_object, manifold,
                            col_a=ca, col_b=cb,
                        )

        # =============================================================
        # Fire collision events  (Enter / Stay / Exit)
        # =============================================================
        for c in all_cols:
            prev = c._current_collisions
            now = current_collisions.get(c, set())
            for oc in now - prev:
                c.OnCollisionEnter(oc)
                if c.game_object:
                    for script in c.game_object.get_components(Script):
                        script.on_collision_enter(oc)
            for oc in now & prev:
                c.OnCollisionStay(oc)
                if c.game_object:
                    for script in c.game_object.get_components(Script):
                        script.on_collision_stay(oc)
            for oc in prev - now:
                c.OnCollisionExit(oc)
                if c.game_object:
                    for script in c.game_object.get_components(Script):
                        script.on_collision_exit(oc)
            c._current_collisions = now.copy()

        # Wake gravity bodies that fell asleep without real support (floaters).
        # Propagate support through sleeping stacks a few times so tops of piles
        # count as supported when the base is on static geometry.
        supported = getattr(self, "_physics_supported_bodies", set())
        for _ in range(4):
            grew = False
            for c in all_cols:
                rb = None
                if c.game_object:
                    rb = getattr(c.game_object, "_rigidbody", None) or c.game_object.get_component(Rigidbody2D)
                if rb is None or rb.is_static or rb.is_kinematic:
                    continue
                if id(rb) in supported:
                    continue
                # Supported if any solid contact partner is static or supported
                for oc in current_collisions.get(c, ()):
                    orb = None
                    if oc.game_object:
                        orb = getattr(oc.game_object, "_rigidbody", None) or oc.game_object.get_component(Rigidbody2D)
                    if orb is None:
                        continue
                    if orb.is_static or id(orb) in supported:
                        supported.add(id(rb))
                        grew = True
                        break
            if not grew:
                break

        for o in self._active_objects():
            rb = getattr(o, "_rigidbody", None) or o.get_component(Rigidbody2D)
            if rb is None or rb.is_static or rb.is_kinematic:
                continue
            if not getattr(rb, "is_sleeping", False):
                continue
            if not getattr(rb, "use_gravity", True):
                continue
            if id(rb) not in supported:
                rb.wake()

    def _resolve_collision_2d(self, a: GameObject, b: GameObject, manifold,
                              col_a=None, col_b=None):
        """Separate overlapping objects and apply linear+angular collision response.

        Uses physics-material properties (bounciness, friction) on the colliders
        together with each rigidbody's mass **and** scalar inverse inertia so
        off-center contacts produce correct spin (same model as 3D, reduced to Z).

        Parameters
        ----------
        a, b : GameObjects involved in the collision.
        manifold : CollisionManifold2D with *normal* (from B towards A), *depth*,
            and optional *contact_point*.
        col_a, col_b : The Collider2D instances (optional; looked up if None).
        """
        from engine.d2.physics.rigidbody import Rigidbody2D
        from engine.d2.physics.collider import Collider2D
        from engine.d2.physics.response import (
            resolve_contact_2d,
            body_state_from_rigidbody,
            apply_body_state,
            estimate_contact_point,
            stabilize_contact_point,
            _as_np2,
            _face_align_from_rotation,
        )
        from engine.d3.physics.types import PhysicsMaterialCombine
        from engine.types import Vector3

        depth = getattr(manifold, "depth", 0.0)
        if depth <= 0:
            return
        normal = manifold.normal

        def _rb_of(go):
            rb = getattr(go, "_rigidbody", None)
            if rb is not None and not isinstance(rb, Rigidbody2D):
                rb = go.get_component(Rigidbody2D)
            if rb is None:
                rb = go.get_component(Rigidbody2D)
            return rb

        rb_a = _rb_of(a)
        rb_b = _rb_of(b)

        def _immovable(rb):
            if rb is None:
                return True
            return bool(getattr(rb, "is_static", False) or getattr(rb, "is_kinematic", False))

        a_static = _immovable(rb_a)
        b_static = _immovable(rb_b)

        if a_static and b_static:
            return

        if a_static or b_static:
            push = depth + 1e-6
        else:
            PENETRATION_SLOP = 0.001
            push = max(0.0, depth - PENETRATION_SLOP) * 0.95
            if push > 0.0:
                push += 1e-6

        a_sleep = rb_a is not None and getattr(rb_a, "is_sleeping", False)
        b_sleep = rb_b is not None and getattr(rb_b, "is_sleeping", False)

        def _speed(rb):
            if rb is None:
                return 0.0
            v = rb.velocity
            w = float(rb.angular_velocity)
            return float((v.x * v.x + v.y * v.y) ** 0.5 + 0.25 * abs(w))

        sa = 0.0 if a_static else _speed(rb_a)
        sb = 0.0 if b_static else _speed(rb_b)
        both_resting = sa < 0.12 and sb < 0.12

        # Sleeping stacks: skip micro-penetration contacts entirely so piles
        # don't wake and accumulate phantom spin every frame.
        if a_sleep and (b_sleep or b_static) and depth < 0.05 and both_resting:
            return
        if b_sleep and (a_sleep or a_static) and depth < 0.05 and both_resting:
            return

        # Only wake for deep sinks or real impacts — not resting overlap.
        deep = depth >= 0.05
        impact = max(sa, sb) > 0.25
        if deep or impact:
            if a_sleep and not a_static:
                rb_a.wake()
                a_sleep = False
            if b_sleep and not b_static:
                rb_b.wake()
                b_sleep = False
        elif both_resting and depth < 0.03 and not impact:
            # Soft rest contact: still resolve once for depenetration, but keep
            # sleep timers intact (do not wake).
            pass

        if col_a is None:
            col_a = a.get_component(Collider2D)
        if col_b is None:
            col_b = b.get_component(Collider2D)

        nx, ny = float(normal[0]), float(normal[1])
        if push > 0.0:
            if a_static:
                b.transform.position = Vector3(
                    b.transform.position.x - nx * push,
                    b.transform.position.y - ny * push,
                    b.transform.position.z,
                )
                if col_b is not None:
                    col_b._transform_dirty = True
                    col_b.update_bounds()
            elif b_static:
                a.transform.position = Vector3(
                    a.transform.position.x + nx * push,
                    a.transform.position.y + ny * push,
                    a.transform.position.z,
                )
                if col_a is not None:
                    col_a._transform_dirty = True
                    col_a.update_bounds()
            else:
                half = push * 0.5
                a.transform.position = Vector3(
                    a.transform.position.x + nx * half,
                    a.transform.position.y + ny * half,
                    a.transform.position.z,
                )
                b.transform.position = Vector3(
                    b.transform.position.x - nx * half,
                    b.transform.position.y - ny * half,
                    b.transform.position.z,
                )
                if col_a is not None:
                    col_a._transform_dirty = True
                    col_a.update_bounds()
                if col_b is not None:
                    col_b._transform_dirty = True
                    col_b.update_bounds()

        bounciness_a = getattr(col_a, "bounciness", 0.0) if col_a else 0.0
        bounciness_b = getattr(col_b, "bounciness", 0.0) if col_b else 0.0
        bm_a = getattr(col_a, "bounce_combine", PhysicsMaterialCombine.AVERAGE) if col_a else PhysicsMaterialCombine.AVERAGE
        bm_b = getattr(col_b, "bounce_combine", PhysicsMaterialCombine.AVERAGE) if col_b else PhysicsMaterialCombine.AVERAGE
        restitution = PhysicsMaterialCombine.combine(bounciness_a, bounciness_b, bm_a, bm_b)

        sf_a = getattr(col_a, "static_friction", 0.6) if col_a else 0.6
        sf_b = getattr(col_b, "static_friction", 0.6) if col_b else 0.6
        df_a = getattr(col_a, "dynamic_friction", 0.4) if col_a else 0.4
        df_b = getattr(col_b, "dynamic_friction", 0.4) if col_b else 0.4
        fc_a = getattr(col_a, "friction_combine", PhysicsMaterialCombine.AVERAGE) if col_a else PhysicsMaterialCombine.AVERAGE
        fc_b = getattr(col_b, "friction_combine", PhysicsMaterialCombine.AVERAGE) if col_b else PhysicsMaterialCombine.AVERAGE
        static_fric = PhysicsMaterialCombine.combine(sf_a, sf_b, fc_a, fc_b)
        dynamic_fric = PhysicsMaterialCombine.combine(df_a, df_b, fc_a, fc_b)

        n_arr = _as_np2(normal)
        n_len = float(np.linalg.norm(n_arr))
        if n_len > 1e-12:
            n_arr = n_arr / n_len
        face_align_a = 0.0 if a_static else _face_align_from_rotation(a, n_arr)
        face_align_b = 0.0 if b_static else _face_align_from_rotation(b, n_arr)

        cp = getattr(manifold, "contact_point", None)
        if cp is None:
            cp = estimate_contact_point(
                a.transform.position, b.transform.position, normal, depth
            )
        cp = stabilize_contact_point(
            a.transform.position, b.transform.position, cp, normal, depth,
            face_align_a=face_align_a, face_align_b=face_align_b,
        )

        pos_a, vel_a, omega_a, inv_mass_a, i_inv_a = body_state_from_rigidbody(
            rb_a, a, a_static
        )
        pos_b, vel_b, omega_b, inv_mass_b, i_inv_b = body_state_from_rigidbody(
            rb_b, b, b_static
        )

        result = resolve_contact_2d(
            pos_a=pos_a, vel_a=vel_a, omega_a=omega_a,
            inv_mass_a=inv_mass_a, i_inv_a=i_inv_a,
            pos_b=pos_b, vel_b=vel_b, omega_b=omega_b,
            inv_mass_b=inv_mass_b, i_inv_b=i_inv_b,
            contact_point=cp, normal=normal,
            restitution=restitution,
            static_friction=static_fric,
            dynamic_friction=dynamic_fric,
            face_align_a=face_align_a,
            face_align_b=face_align_b,
        )
        new_va, new_oa, new_vb, new_ob, unstable = result

        # Sleep only when resting on immovable geometry or an *already sleeping*
        # support. Never sleep two free-falling bodies that just bumped mid-air
        # (that froze floaters in multi-object piles).
        n_y = float(n_arr[1]) if n_arr is not None else 0.0
        support_up = n_y > 0.55   # normal from B toward A is mostly +Y
        support_down = n_y < -0.55

        def _partner_supports(other_rb, other_static, upward_for_self: bool) -> bool:
            if unstable:
                return False
            if other_static:
                return True
            if other_rb is not None and getattr(other_rb, "is_sleeping", False):
                return upward_for_self
            return False

        if rb_a is not None and not a_static:
            # A rests on B when normal points up toward A (B is below)
            can_sleep_a = _partner_supports(rb_b, b_static, support_up)
            apply_body_state(rb_a, new_va, new_oa, allow_sleep=can_sleep_a)
            if can_sleep_a:
                getattr(self, "_physics_supported_bodies", set()).add(id(rb_a))
        if rb_b is not None and not b_static:
            # B rests on A when normal points down toward B (A is below)
            can_sleep_b = _partner_supports(rb_a, a_static, support_down)
            apply_body_state(rb_b, new_vb, new_ob, allow_sleep=can_sleep_b)
            if can_sleep_b:
                getattr(self, "_physics_supported_bodies", set()).add(id(rb_b))

    # -- Pure-Python fallback for velocity resolution (linear only, legacy) --

    @staticmethod
    def _resolve_velocity_2d_py(vx_a, vy_a, vx_b, vy_b,
                                 nx, ny, inv_mass_a, inv_mass_b,
                                 restitution, static_fric, dynamic_fric):
        """Pure-Python linear-only impulse response (legacy / tests)."""
        import math

        rvx = vx_a - vx_b
        rvy = vy_a - vy_b
        vel_along_normal = rvx * nx + rvy * ny

        if vel_along_normal > 0:
            return (vx_a, vy_a, vx_b, vy_b)

        inv_mass_sum = inv_mass_a + inv_mass_b
        if inv_mass_sum < 1e-12:
            return (vx_a, vy_a, vx_b, vy_b)

        j = -(1.0 + restitution) * vel_along_normal / inv_mass_sum

        vx_a += j * inv_mass_a * nx
        vy_a += j * inv_mass_a * ny
        vx_b -= j * inv_mass_b * nx
        vy_b -= j * inv_mass_b * ny

        rvx = vx_a - vx_b
        rvy = vy_a - vy_b
        vt = rvx * nx + rvy * ny
        tx = rvx - vt * nx
        ty = rvy - vt * ny
        t_mag = math.sqrt(tx * tx + ty * ty)

        if t_mag < 1e-10:
            return (vx_a, vy_a, vx_b, vy_b)

        tx /= t_mag
        ty /= t_mag
        jt = -(rvx * tx + rvy * ty) / inv_mass_sum

        if abs(jt) < j * static_fric:
            vx_a += jt * inv_mass_a * tx
            vy_a += jt * inv_mass_a * ty
            vx_b -= jt * inv_mass_b * tx
            vy_b -= jt * inv_mass_b * ty
        else:
            jt_clamped = -j * dynamic_fric if jt < 0 else j * dynamic_fric
            vx_a += jt_clamped * inv_mass_a * tx
            vy_a += jt_clamped * inv_mass_a * ty
            vx_b -= jt_clamped * inv_mass_b * tx
            vy_b -= jt_clamped * inv_mass_b * ty

        return (vx_a, vy_a, vx_b, vy_b)
