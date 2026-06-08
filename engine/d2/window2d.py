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
import pygame
import numpy as np
from typing import List, Optional, Tuple, TYPE_CHECKING

import moderngl

from engine.window_base import WindowBase
from engine.gameobject import GameObject
from engine.d2.object2d import Object2D
from engine.d2.camera2d import Camera2D
from engine.types import Color, ColorType
from engine.component import Script, Time

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

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "2D Engine",
        resizable: bool = False,
        background_color: ColorType = (0.1, 0.1, 0.15),
        use_pygame_window: bool = True,
        use_pygame_events: bool = True,
    ):
        # WindowBase handles: pygame, moderngl context, overlay shader,
        # 2D HUD surface, timing, input, drawing helpers, main loop.
        # It also calls self._init_gpu() at the end.
        super().__init__(width, height, title, resizable, background_color,
                         use_pygame_window, use_pygame_events)

        # Default 2D camera (orthographic_size controls viewport/world size like Unity)
        self._camera_go = GameObject("Default Camera")
        self.camera = Camera2D(orthographic_size=5.0)
        self._camera_go.add_component(self.camera)
        self.camera.set_screen_size(width, height)

        # GPU texture cache for Object2D sprites  {id(obj2d) -> moderngl.Texture}
        self._sprite_textures: dict = {}

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

    def _cleanup_gpu(self):
        """Release 2-D GPU resources (called by WindowBase._cleanup)."""
        for tex in self._sprite_textures.values():
            tex.release()
        self._sprite_textures.clear()
        self._quad_vao.release()
        self._quad_vbo.release()
        self._sprite_program.release()
        self._collider_program.release()

    # =====================================================================
    # Scene management
    # =====================================================================

    def show_scene(self, scene: 'Scene2D', start_scripts: bool = True):
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
        self.start(start_scripts=start_scripts)

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
        view4[0, 0] = view3[0, 0]; view4[0, 1] = view3[0, 1]; view4[0, 3] = view3[0, 2]
        view4[1, 0] = view3[1, 0]; view4[1, 1] = view3[1, 1]; view4[1, 3] = view3[1, 2]

        proj = cam.get_projection_matrix()     # 4×4

        # Upload camera matrices
        self._sprite_program['projection'].write(proj.astype(np.float32).tobytes())
        self._sprite_program['view'].write(view4.astype(np.float32).tobytes())

        # Gather and sort visible Object2D by (layer_id, sorting_order)
        renderables: List[Object2D] = []
        for obj in self._active_objects():
            obj2d = obj.get_component(Object2D)
            if obj2d and obj2d.visible:
                renderables.append(obj2d)
        renderables.sort(key=lambda o: o.sort_key)

        # Disable depth test for 2D (sorting_order determines order)
        self._ctx.disable(moderngl.DEPTH_TEST)

        for obj2d in renderables:
            self._render_object2d_gl(obj2d)

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

    # -- Per-object GL rendering ------------------------------------------

    def _render_object2d_gl(self, obj2d: Object2D):
        """Render one Object2D as a textured/colored quad via ModernGL."""
        go = obj2d.game_object
        if not go:
            return

        pos = go.transform.position
        wx, wy = float(pos.x), float(pos.y)

        # Scale
        # Use scale_xyz so that non-uniform (x != y) scaling works for 2D objects.
        # The .scale property always returns a uniform float (x).
        t_scale = go.transform.scale_xyz
        if hasattr(t_scale, 'x'):
            sx_f, sy_f = float(t_scale.x), float(t_scale.y)
        elif isinstance(t_scale, (tuple, list)):
            sx_f = float(t_scale[0])
            sy_f = float(t_scale[1]) if len(t_scale) > 1 else sx_f
        else:
            sx_f = sy_f = float(t_scale)

        # Object size in world units
        size = obj2d.size
        w = size.x * sx_f
        h = size.y * sy_f

        # Z rotation (degrees → radians)
        rot = go.transform.rotation
        if hasattr(rot, '__getitem__'):
            rot_z = float(rot[2]) if len(rot) > 2 else 0.0
        else:
            rot_z = float(rot)
        rad = math.radians(rot_z)
        c_r, s_r = math.cos(rad), math.sin(rad)

        # Apply flip_x / flip_y by negating the corresponding effective scale.
        # This flips the geometry (which inverts the texture mapping for that axis).
        # We also apply a base Y sign correction (* -1 when not flip_y) so that
        # flip_y=False + positive scale produces the expected upright orientation
        # (addresses "sprites with positive scale are scaled negatively in y").
        flip_x = getattr(obj2d, 'flip_x', False)
        flip_y = getattr(obj2d, 'flip_y', False)

        sx = w * (-1 if flip_x else 1)
        sy = h * (-1 if flip_y else 1) * (-1)  # extra -1 corrects default Y for this engine's quad/UV/projection convention

        # Build 4×4 model matrix  (T * R * S)
        model = np.array([
            [sx * c_r,  -sy * s_r, 0, 0],
            [sx * s_r,   sy * c_r, 0, 0],
            [0,         0,       1, 0],
            [wx,        wy,      0, 1],
        ], dtype=np.float32)

        self._sprite_program['model'].write(model.tobytes())

        # Texture
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

        # Color (tint + alpha)
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

            self._collider_program['mvp'].write(mvp.tobytes())
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
        view4[0, 0] = view3[0, 0]; view4[0, 1] = view3[0, 1]; view4[0, 3] = view3[0, 2]
        view4[1, 0] = view3[1, 0]; view4[1, 1] = view3[1, 1]; view4[1, 3] = view3[1, 2]
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
                self._collider_program['mvp'].write(mvp.tobytes())
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
        view4[0, 0] = view3[0, 0]; view4[0, 1] = view3[0, 1]; view4[0, 3] = view3[0, 2]
        view4[1, 0] = view3[1, 0]; view4[1, 1] = view3[1, 1]; view4[1, 3] = view3[1, 2]
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
        from collections import defaultdict

        all_cols: List[Collider2D] = []
        for o in self._active_objects():
            all_cols.extend(o.get_components(Collider2D))
        if not all_cols:
            return

        current_collisions: dict = defaultdict(set)

        for ca in all_cols:
            rb_a = ca.game_object.get_component(Rigidbody2D) if ca.game_object else None
            if (rb_a and rb_a.is_static) or ca.collision_mode == CollisionMode.IGNORE:
                continue

            a = ca.game_object
            for cb in all_cols:
                if cb is ca or cb.game_object is a:
                    continue
                relation = ca.group.get_relation(cb.group)
                if relation == CollisionRelation.IGNORE:
                    continue

                if objects_collide_2d(ca, cb):
                    current_collisions[ca].add(cb)
                    current_collisions[cb].add(ca)

                    if relation == CollisionRelation.SOLID:
                        manifold = get_collision_manifold_2d(ca, cb)
                        if manifold:
                            self._resolve_collision_2d(a, cb.game_object, manifold)

        # Fire collision events
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

    def _resolve_collision_2d(self, a: GameObject, b: GameObject, manifold):
        """Separate overlapping objects along the collision normal."""
        from engine.d2.physics.rigidbody import Rigidbody2D
        from engine.types import Vector3

        rb_a = a.get_component(Rigidbody2D)
        rb_b = b.get_component(Rigidbody2D)

        a_static = (rb_a is None or rb_a.is_static or rb_a.is_kinematic)
        b_static = (rb_b is None or rb_b.is_static or rb_b.is_kinematic)

        normal = manifold.normal
        depth = manifold.depth

        if a_static and b_static:
            return

        if a_static:
            b.transform.position = Vector3(
                b.transform.position.x - normal[0] * depth,
                b.transform.position.y - normal[1] * depth,
                b.transform.position.z,
            )
        elif b_static:
            a.transform.position = Vector3(
                a.transform.position.x + normal[0] * depth,
                a.transform.position.y + normal[1] * depth,
                a.transform.position.z,
            )
        else:
            half = depth * 0.5
            a.transform.position = Vector3(
                a.transform.position.x + normal[0] * half,
                a.transform.position.y + normal[1] * half,
                a.transform.position.z,
            )
            b.transform.position = Vector3(
                b.transform.position.x - normal[0] * half,
                b.transform.position.y - normal[1] * half,
                b.transform.position.z,
            )
