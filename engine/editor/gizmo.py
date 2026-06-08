"""
Translation gizmo – three axis arrows (X / Y / Z) rendered in the 3D viewport.

When one or more game-objects are selected the gizmo appears at the average
world position.  The user can click-drag an arrow to move objects along that
single world axis – just like Unity's translate tool.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from engine.d3.camera import Camera3D
    from engine.gameobject import GameObject

# ── Axis identifiers ────────────────────────────────────────────────
AXIS_NONE = 0
AXIS_X    = 1
AXIS_Y    = 2
AXIS_Z    = 3

# Arrow colours                      normal              highlight
_COLORS = {
    AXIS_X: ((0.90, 0.20, 0.20), (1.00, 0.55, 0.55)),
    AXIS_Y: ((0.20, 0.85, 0.20), (0.55, 1.00, 0.55)),
    AXIS_Z: ((0.25, 0.55, 1.00), (0.60, 0.78, 1.00)),
}

# World-space unit directions
_DIRECTIONS = {
    AXIS_X: np.array([1, 0, 0], dtype=np.float32),
    AXIS_Y: np.array([0, 1, 0], dtype=np.float32),
    AXIS_Z: np.array([0, 0, 1], dtype=np.float32),
}

# ── Cone mesh (precomputed) ─────────────────────────────────────────
def _build_cone_mesh(segments: int = 12) -> np.ndarray:
    """Return (N, 3) float32 vertices for a cone of height=1, base-radius=1,
    tip at (0, 1, 0), base centred at origin.  Triangle soup (no indexing)."""
    verts = []
    tip = np.array([0, 1, 0], dtype=np.float32)
    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        b0 = np.array([math.cos(a0), 0, math.sin(a0)], dtype=np.float32)
        b1 = np.array([math.cos(a1), 0, math.sin(a1)], dtype=np.float32)
        # side triangle
        verts += [tip, b0, b1]
        # base triangle
        verts += [np.zeros(3, dtype=np.float32), b1, b0]
    return np.array(verts, dtype=np.float32)

_CONE_VERTS = _build_cone_mesh(12)


class TranslateGizmo:
    """Handles drawing and interaction for the 3-axis translate gizmo."""

    # Tuning constants
    SHAFT_LENGTH    = 1.0     # world-space length at scale=1
    CONE_HEIGHT     = 0.22    # relative to shaft
    CONE_RADIUS     = 0.07
    HIT_RADIUS_PX   = 18     # screen-space hit radius around the arrow
    SCREEN_SIZE_PX  = 120    # target screen-space pixel length of each arrow

    def __init__(self):
        # Interaction state
        self.active_axis: int = AXIS_NONE     # axis currently being dragged
        self.hovered_axis: int = AXIS_NONE    # axis under cursor (for highlight)
        self._dragging: bool = False
        self._drag_start_mouse: Optional[Tuple[int, int]] = None
        self._drag_objects: List[GameObject] = []
        self._drag_start_positions: List[np.ndarray] = []

        # GPU resources (lazily initialised on first draw)
        self._cone_vbo = None
        self._cone_vao = None
        self._line_vbo = None
        self._line_vao = None
        self._gpu_ready = False

    # ── GPU init ────────────────────────────────────────────────────
    def _ensure_gpu(self, ctx, program) -> None:
        if self._gpu_ready:
            return
        # Cone (triangle soup)
        self._cone_vbo = ctx.buffer(_CONE_VERTS.tobytes())
        self._cone_vao = ctx.vertex_array(
            program, [(self._cone_vbo, '3f', 'in_position')]
        )
        # A single line segment – rewritten every draw call
        self._line_vbo = ctx.buffer(reserve=6 * 4)  # 2 × vec3 float32
        self._line_vao = ctx.vertex_array(
            program, [(self._line_vbo, '3f', 'in_position')]
        )
        self._gpu_ready = True

    # ── Drawing ─────────────────────────────────────────────────────
    def draw(self, window, objects: List['GameObject']) -> None:
        """Draw the gizmo at the centre of *objects*.  Call inside GL context."""
        if not objects:
            return

        import moderngl

        ctx = window._ctx
        prog = window._collider_program
        self._ensure_gpu(ctx, prog)

        camera = window.active_camera_override or (
            window._current_scene.camera if window._current_scene else window.camera
        )
        if not camera:
            return

        # Gizmo centre = average world position of selected objects
        center = np.mean(
            [np.array(o.transform.world_position, dtype=np.float32) for o in objects],
            axis=0,
        )

        # Constant screen-space size
        scale = self._screen_scale(center, camera, window)

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix(window.aspect)
        vp = view @ proj

        # Draw on top of everything
        ctx.disable(0x0B71)   # GL_DEPTH_TEST

        for axis_id in (AXIS_X, AXIS_Y, AXIS_Z):
            direction = _DIRECTIONS[axis_id]
            is_hot = (axis_id == self.active_axis) or (axis_id == self.hovered_axis)
            color = _COLORS[axis_id][1] if is_hot else _COLORS[axis_id][0]
            lw = 3.0 if is_hot else 2.0

            shaft_end = center + direction * self.SHAFT_LENGTH * scale

            # ── shaft line ──
            pts = np.array([center, shaft_end], dtype=np.float32)
            self._line_vbo.write(pts.tobytes())
            identity = np.eye(4, dtype=np.float32)
            prog['mvp'].write((identity @ vp).astype(np.float32).tobytes())
            prog['color'].value = color
            ctx.line_width = lw
            self._line_vao.render(0x0001)  # GL_LINES

            # ── arrowhead cone ──
            h = self.CONE_HEIGHT * scale
            r = self.CONE_RADIUS * scale
            model = self._cone_model(shaft_end, direction, h, r)
            mvp = model @ vp
            prog['mvp'].write(mvp.astype(np.float32).tobytes())
            prog['color'].value = color
            self._cone_vao.render(moderngl.TRIANGLES)

        ctx.line_width = 1.0
        ctx.enable(0x0B71)   # GL_DEPTH_TEST

    # ── Cone model matrix ───────────────────────────────────────────
    @staticmethod
    def _cone_model(base_pos, direction, height, radius) -> np.ndarray:
        """4×4 model matrix:  unit-cone → scaled/rotated/translated."""
        d = direction / (np.linalg.norm(direction) + 1e-9)
        # Build orthonormal basis with d = local Y
        if abs(d[1]) < 0.99:
            up = np.array([0, 1, 0], dtype=np.float32)
        else:
            up = np.array([1, 0, 0], dtype=np.float32)
        right = np.cross(up, d)
        right /= np.linalg.norm(right) + 1e-9
        up2 = np.cross(d, right)

        # R maps:  local-X → right*radius,  local-Y → d*height,  local-Z → up2*radius
        R = np.eye(4, dtype=np.float32)
        R[0, :3] = right * radius
        R[1, :3] = d     * height
        R[2, :3] = up2   * radius

        T = np.eye(4, dtype=np.float32)
        T[3, :3] = base_pos

        return R @ T

    # ── Screen-space scale helper ───────────────────────────────────
    def _screen_scale(self, world_pos, camera, window) -> float:
        """Return a world-space scale factor so that gizmo ≈ SCREEN_SIZE_PX pixels."""
        sp0 = window.project_point(tuple(world_pos))
        if sp0 is None:
            return 1.0
        right = world_pos + np.array([1, 0, 0], dtype=np.float32)
        sp1 = window.project_point(tuple(right))
        if sp1 is None:
            return 1.0
        pixels_per_unit = max(math.hypot(sp1[0] - sp0[0], sp1[1] - sp0[1]), 1.0)
        return self.SCREEN_SIZE_PX / pixels_per_unit

    # ── Hit testing ─────────────────────────────────────────────────
    def hit_test(self, mx: int, my: int, window,
                 objects: List['GameObject']) -> int:
        """Return AXIS_X / Y / Z if (mx, my) is close to an arrow, else AXIS_NONE."""
        if not objects:
            return AXIS_NONE

        camera = window.active_camera_override or (
            window._current_scene.camera if window._current_scene else window.camera
        )
        if not camera:
            return AXIS_NONE

        center = np.mean(
            [np.array(o.transform.world_position, dtype=np.float32) for o in objects],
            axis=0,
        )
        scale = self._screen_scale(center, camera, window)

        origin_scr = window.project_point(tuple(center))
        if origin_scr is None:
            return AXIS_NONE

        best_axis = AXIS_NONE
        best_dist = self.HIT_RADIUS_PX

        for axis_id in (AXIS_X, AXIS_Y, AXIS_Z):
            direction = _DIRECTIONS[axis_id]
            # Full arrow tip = shaft end + cone height (the visible tip)
            tip_world = center + direction * (self.SHAFT_LENGTH + self.CONE_HEIGHT) * scale
            tip_scr = window.project_point(tuple(tip_world))
            if tip_scr is None:
                continue
            # Test against the whole arrow from center to cone tip
            dist = _point_to_segment_dist(
                mx, my,
                origin_scr[0], origin_scr[1],
                tip_scr[0], tip_scr[1],
            )
            if dist < best_dist:
                best_dist = dist
                best_axis = axis_id

        return best_axis

    # ── Drag logic ──────────────────────────────────────────────────
    def begin_drag(self, axis: int, mx: int, my: int,
                   objects: List['GameObject']) -> None:
        """Start dragging *objects* along *axis*."""
        self.active_axis = axis
        self._dragging = True
        self._drag_start_mouse = (mx, my)
        self._drag_objects = list(objects)
        self._drag_start_positions = [
            np.array(o.transform.position, dtype=np.float32) for o in objects
        ]

    def update_drag(self, mx: int, my: int, window) -> None:
        """Move dragged objects based on new mouse position."""
        if not self._dragging or self.active_axis == AXIS_NONE:
            return

        camera = window.active_camera_override or (
            window._current_scene.camera if window._current_scene else window.camera
        )
        if not camera:
            return

        direction = _DIRECTIONS[self.active_axis]
        ref_pos = np.array(self._drag_start_positions[0], dtype=np.float32)

        # Project axis into screen space
        sp0 = window.project_point(tuple(ref_pos))
        sp1 = window.project_point(tuple(ref_pos + direction))
        if sp0 is None or sp1 is None:
            return

        axis_scr = np.array([sp1[0] - sp0[0], sp1[1] - sp0[1]], dtype=np.float64)
        axis_len_sq = float(np.dot(axis_scr, axis_scr))
        if axis_len_sq < 1.0:
            return

        # Mouse delta projected onto the screen-space axis direction
        mouse_delta = np.array(
            [mx - self._drag_start_mouse[0], my - self._drag_start_mouse[1]],
            dtype=np.float64,
        )
        t = float(np.dot(mouse_delta, axis_scr)) / axis_len_sq

        # t == 1 ⇒ the mouse moved exactly the screen-length of 1 world unit
        # along the axis, so world_offset = direction * t
        world_offset = direction * t

        for obj, start_pos in zip(self._drag_objects, self._drag_start_positions):
            new_pos = start_pos + world_offset
            obj.transform.position = tuple(new_pos)

    def end_drag(self) -> None:
        """Finish the current drag."""
        self._dragging = False
        self.active_axis = AXIS_NONE
        self._drag_objects = []
        self._drag_start_positions = []

    @property
    def is_dragging(self) -> bool:
        return self._dragging


class TranslateGizmo2D:
    """2D-only translate gizmo – X (red) and Y (green) arrows only."""

    SHAFT_LENGTH   = 1.0
    CONE_HEIGHT    = 0.22
    CONE_RADIUS    = 0.07
    HIT_RADIUS_PX  = 18
    SCREEN_SIZE_PX = 120

    _2D_AXES = (AXIS_X, AXIS_Y)

    def __init__(self):
        self.active_axis: int = AXIS_NONE
        self.hovered_axis: int = AXIS_NONE
        self._dragging: bool = False
        self._drag_start_mouse: Optional[Tuple[int, int]] = None
        self._drag_objects: List['GameObject'] = []
        self._drag_start_positions: List[np.ndarray] = []
        self._cone_vbo = None
        self._cone_vao = None
        self._line_vbo = None
        self._line_vao = None
        self._gpu_ready = False

    def _ensure_gpu(self, ctx, program) -> None:
        if self._gpu_ready:
            return
        self._cone_vbo = ctx.buffer(_CONE_VERTS.tobytes())
        self._cone_vao = ctx.vertex_array(program, [(self._cone_vbo, '3f', 'in_position')])
        self._line_vbo = ctx.buffer(reserve=6 * 4)
        self._line_vao = ctx.vertex_array(program, [(self._line_vbo, '3f', 'in_position')])
        self._gpu_ready = True

    def draw(self, window, objects: List['GameObject']) -> None:
        if not objects:
            return
        import moderngl
        ctx = window._ctx
        prog = window._collider_program
        self._ensure_gpu(ctx, prog)

        camera = window.active_camera_override or (
            window._current_scene.camera if window._current_scene else window.camera
        )
        if not camera:
            return

        center = np.mean(
            [np.array(o.transform.world_position, dtype=np.float32) for o in objects],
            axis=0,
        )
        scale = self._screen_scale(center, camera, window)

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()
        # Build 4x4 VP for 2D (view is 3x3, proj is 4x4)
        view4 = np.eye(4, dtype=np.float32)
        if view.shape == (3, 3):
            view4[0, 0] = view[0, 0]; view4[0, 1] = view[0, 1]; view4[0, 3] = view[0, 2]
            view4[1, 0] = view[1, 0]; view4[1, 1] = view[1, 1]; view4[1, 3] = view[1, 2]
        else:
            view4 = view.astype(np.float32)
        vp = view4 @ proj.astype(np.float32)

        ctx.disable(0x0B71)  # GL_DEPTH_TEST

        for axis_id in self._2D_AXES:
            direction = _DIRECTIONS[axis_id]
            is_hot = (axis_id == self.active_axis) or (axis_id == self.hovered_axis)
            color = _COLORS[axis_id][1] if is_hot else _COLORS[axis_id][0]
            lw = 3.0 if is_hot else 2.0

            shaft_end = center + direction * self.SHAFT_LENGTH * scale

            pts = np.array([center, shaft_end], dtype=np.float32)
            self._line_vbo.write(pts.tobytes())
            identity = np.eye(4, dtype=np.float32)
            prog['mvp'].write((identity @ vp).astype(np.float32).tobytes())
            prog['color'].value = color
            ctx.line_width = lw
            self._line_vao.render(0x0001)  # GL_LINES

            h = self.CONE_HEIGHT * scale
            r = self.CONE_RADIUS * scale
            model = TranslateGizmo._cone_model(shaft_end, direction, h, r)
            mvp = model @ vp
            prog['mvp'].write(mvp.astype(np.float32).tobytes())
            prog['color'].value = color
            self._cone_vao.render(moderngl.TRIANGLES)

        ctx.line_width = 1.0
        ctx.enable(0x0B71)

    def _screen_scale(self, world_pos, camera, window) -> float:
        sp0 = window.project_point(tuple(world_pos))
        if sp0 is None:
            return 1.0
        right = world_pos + np.array([1, 0, 0], dtype=np.float32)
        sp1 = window.project_point(tuple(right))
        if sp1 is None:
            return 1.0
        pixels_per_unit = max(math.hypot(sp1[0] - sp0[0], sp1[1] - sp0[1]), 1.0)
        return self.SCREEN_SIZE_PX / pixels_per_unit

    def hit_test(self, mx: int, my: int, window,
                 objects: List['GameObject']) -> int:
        if not objects:
            return AXIS_NONE
        camera = window.active_camera_override or (
            window._current_scene.camera if window._current_scene else window.camera
        )
        if not camera:
            return AXIS_NONE

        center = np.mean(
            [np.array(o.transform.world_position, dtype=np.float32) for o in objects],
            axis=0,
        )
        scale = self._screen_scale(center, camera, window)
        origin_scr = window.project_point(tuple(center))
        if origin_scr is None:
            return AXIS_NONE

        best_axis = AXIS_NONE
        best_dist = self.HIT_RADIUS_PX

        for axis_id in self._2D_AXES:
            direction = _DIRECTIONS[axis_id]
            tip_world = center + direction * (self.SHAFT_LENGTH + self.CONE_HEIGHT) * scale
            tip_scr = window.project_point(tuple(tip_world))
            if tip_scr is None:
                continue
            dist = _point_to_segment_dist(
                mx, my, origin_scr[0], origin_scr[1], tip_scr[0], tip_scr[1],
            )
            if dist < best_dist:
                best_dist = dist
                best_axis = axis_id

        return best_axis

    def begin_drag(self, axis: int, mx: int, my: int,
                   objects: List['GameObject']) -> None:
        self.active_axis = axis
        self._dragging = True
        self._drag_start_mouse = (mx, my)
        self._drag_objects = list(objects)
        self._drag_start_positions = [
            np.array(o.transform.position, dtype=np.float32) for o in objects
        ]

    def update_drag(self, mx: int, my: int, window) -> None:
        if not self._dragging or self.active_axis == AXIS_NONE:
            return
        camera = window.active_camera_override or (
            window._current_scene.camera if window._current_scene else window.camera
        )
        if not camera:
            return

        direction = _DIRECTIONS[self.active_axis]
        ref_pos = np.array(self._drag_start_positions[0], dtype=np.float32)

        sp0 = window.project_point(tuple(ref_pos))
        sp1 = window.project_point(tuple(ref_pos + direction))
        if sp0 is None or sp1 is None:
            return

        axis_scr = np.array([sp1[0] - sp0[0], sp1[1] - sp0[1]], dtype=np.float64)
        axis_len_sq = float(np.dot(axis_scr, axis_scr))
        if axis_len_sq < 1.0:
            return

        mouse_delta = np.array(
            [mx - self._drag_start_mouse[0], my - self._drag_start_mouse[1]],
            dtype=np.float64,
        )
        t = float(np.dot(mouse_delta, axis_scr)) / axis_len_sq
        world_offset = direction * t

        for obj, start_pos in zip(self._drag_objects, self._drag_start_positions):
            new_pos = start_pos + world_offset
            obj.transform.position = tuple(new_pos)

    def end_drag(self) -> None:
        self._dragging = False
        self.active_axis = AXIS_NONE
        self._drag_objects = []
        self._drag_start_positions = []

    @property
    def is_dragging(self) -> bool:
        return self._dragging


# ── Utility ─────────────────────────────────────────────────────────
def _point_to_segment_dist(px, py, ax, ay, bx, by) -> float:
    """Shortest distance from point (px, py) to line segment (a → b)."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-6:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)
