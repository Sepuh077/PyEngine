from typing import Optional, TYPE_CHECKING, List
import numpy as np

from engine.component import Component
from engine.types import Vector3, Vector3Like
from engine.types.quaternion import Quaternion

try:
    from engine.cython import CYTHON_ENABLED
    if not CYTHON_ENABLED:
        raise ImportError("Cython disabled via PYENGINE_PURE_PYTHON=1")
    from engine.cython.cy_transform import compute_world_transform_fast as _cy_compute_world
    _USE_CYTHON = True
except (ImportError, ModuleNotFoundError):
    _USE_CYTHON = False

if TYPE_CHECKING:
    from .gameobject import GameObject


class _Vector3Proxy:
    """
    Proxy object returned by transform.position / scale_xyz etc.

    Allows in-place component changes like:
        t.position.x = 5
        t.position.x += 1
        t.position[0] = 10
    to actually update the transform (calling update_prev + mark_dirty + rb notify).
    It otherwise behaves like a Vector3 for reading and math.
    """

    def __init__(self, read_func, write_func):
        # read_func() -> something convertible to Vector3 (current value)
        # write_func( tuple_or_list_or_vec ) -> sets it (and does side effects)
        self._read = read_func
        self._write = write_func

    def _current(self) -> Vector3:
        v = self._read()
        if isinstance(v, Vector3):
            return Vector3(v)  # copy for safety in delegation
        return Vector3(v)

    # --- Component access (the main feature) ---
    @property
    def x(self) -> float:
        return float(self._current().x)

    @x.setter
    def x(self, value: float):
        cur = self._current()
        self._write((float(value), cur.y, cur.z))

    @property
    def y(self) -> float:
        return float(self._current().y)

    @y.setter
    def y(self, value: float):
        cur = self._current()
        self._write((cur.x, float(value), cur.z))

    @property
    def z(self) -> float:
        return float(self._current().z)

    @z.setter
    def z(self, value: float):
        cur = self._current()
        self._write((cur.x, cur.y, float(value)))

    # --- Indexing support e.g. pos[0] += 1 ---
    def __getitem__(self, index):
        return self._current()[index]

    def __setitem__(self, index, value):
        cur = list(self._current().to_tuple())
        cur[index] = float(value)
        if len(cur) == 2:
            cur.append(0.0)
        self._write(tuple(cur[:3]))

    # --- Delegation for read-only / value ops ---
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self._current(), name)

    # Common operators (return real Vector3 so assignment works as before)
    def __add__(self, other):
        return self._current() + other

    def __radd__(self, other):
        return other + self._current()

    def __sub__(self, other):
        return self._current() - other

    def __rsub__(self, other):
        return other - self._current()

    def __mul__(self, other):
        return self._current() * other

    def __rmul__(self, other):
        return other * self._current()

    def __truediv__(self, other):
        return self._current() / other

    def __rtruediv__(self, other):
        return other / self._current()

    def __neg__(self):
        return -self._current()

    # Convenience
    def __repr__(self):
        return repr(self._current())

    def __str__(self):
        return str(self._current())

    def __len__(self):
        return 3

    def __iter__(self):
        return iter(self._current())

    def to_tuple(self):
        return self._current().to_tuple()

    def to_list(self):
        return self._current().to_list()

    def to_numpy(self, dtype=None):
        return self._current().to_numpy(dtype=dtype)


class Transform(Component):
    """Component storing position, rotation, and scale.
    
    Rotation is stored internally as a Quaternion to avoid gimbal lock.
    Euler-angle getters/setters are kept for backward compatibility and
    editor integration. The cached ``_local_rotation`` / ``_world_rotation``
    numpy arrays (radians, XYZ intrinsic) are always kept in sync.
    
    Supports parent-child relationships where children's transforms are
    relative to parent.  World position, rotation, and scale are computed
    from parent + local values.
    """

    def __init__(self):
        super().__init__()
        # Local transform values (relative to parent)
        self._local_position = Vector3.zero()
        self._local_quaternion = Quaternion.identity()
        self._local_rotation = np.zeros(3, dtype=np.float32)   # cached euler (rad)
        self._local_scale = Vector3.one()
        
        # Cached world transform values
        self._world_position = Vector3.zero()
        self._world_quaternion = Quaternion.identity()
        self._world_rotation = np.zeros(3, dtype=np.float32)   # cached euler (rad)
        self._world_scale = Vector3.one()

        self._transform_dirty = True
        self._world_dirty = True
        self._cached_model = None
        self._cached_rotation = None
        self._prev_position = Vector3(self._local_position)
        
        # Parent-child relationships
        self._parent: Optional['Transform'] = None
        self._children: List['Transform'] = []

    def _mark_dirty(self):
        self._transform_dirty = True
        self._world_dirty = True
        # Mark all children as dirty too
        for child in self._children:
            child._mark_dirty()
        if self.game_object:
            for comp in self.game_object.components:
                if hasattr(comp, '_transform_dirty') and comp is not self:
                    comp._transform_dirty = True

    def _update_prev_position(self):
        self._prev_position = Vector3(self._local_position)
    
    # =========================================================================
    # Parent-child relationship methods
    # =========================================================================
    
    @property
    def parent(self) -> Optional['Transform']:
        """Get the parent transform."""
        return self._parent
    
    @parent.setter
    def parent(self, value: Optional['Transform']):
        """Set the parent transform."""
        if self._parent is value:
            return
        
        # Remove from old parent
        if self._parent is not None:
            self._parent._children.remove(self)
        
        self._parent = value
        
        # Add to new parent
        if self._parent is not None:
            self._parent._children.append(self)
        
        self._mark_dirty()
    
    @property
    def children(self) -> List['Transform']:
        """Get list of child transforms."""
        return self._children.copy()
    
    def add_child(self, child: 'Transform'):
        """Add a child transform."""
        child.parent = self
    
    def remove_child(self, child: 'Transform'):
        """Remove a child transform."""
        if child in self._children:
            child.parent = None
    
    def detach_from_parent(self):
        """Detach this transform from its parent."""
        self.parent = None
    
    # =========================================================================
    # Local transform properties (relative to parent)
    # =========================================================================
    
    @property
    def local_position(self) -> Vector3:
        """Get local position (relative to parent)."""
        return _Vector3Proxy(
            read_func=lambda: self._local_position,
            write_func=self._set_local_position_vector,
        )
    
    @local_position.setter
    def local_position(self, value):
        self._set_local_position_vector(value)
    
    @property
    def local_rotation(self) -> tuple:
        """Get local rotation in degrees (relative to parent)."""
        return tuple(np.degrees(self._local_rotation))
    
    @local_rotation.setter
    def local_rotation(self, value):
        self._local_rotation = np.radians(value).astype(np.float32)
        self._local_quaternion = Quaternion.from_euler(*self._local_rotation)
        self._mark_dirty()
    
    @property
    def local_scale(self) -> Vector3:
        """Get local scale (relative to parent)."""
        return _Vector3Proxy(
            read_func=lambda: self._local_scale,
            write_func=self._set_local_scale_vector,
        )
    
    @local_scale.setter
    def local_scale(self, value):
        self._set_local_scale_vector(value)
    
    # =========================================================================
    # World transform properties (computed from parent + local)
    # =========================================================================
    
    def _compute_world_transform(self):
        """Compute world position, rotation, and scale from parent."""
        if not self._world_dirty:
            return
        if self._parent is None:
            self._world_position = Vector3(self._local_position)
            self._world_quaternion = Quaternion(self._local_quaternion)
            self._world_rotation = self._local_rotation.copy()
            self._world_scale = Vector3(self._local_scale)
        else:
            parent = self._parent
            parent._compute_world_transform()

            if _USE_CYTHON:
                lp = self._local_position
                lq = self._local_quaternion
                ls = self._local_scale
                pp = parent._world_position
                pq = parent._world_quaternion
                ps = parent._world_scale

                wp_t, wq_t, ws_t, euler = _cy_compute_world(
                    lp._x, lp._y, lp._z,
                    lq._w, lq._x, lq._y, lq._z,
                    ls._x, ls._y, ls._z,
                    pp._x, pp._y, pp._z,
                    pq._w, pq._x, pq._y, pq._z,
                    ps._x, ps._y, ps._z,
                )
                self._world_position = Vector3(wp_t[0], wp_t[1], wp_t[2])
                self._world_quaternion = Quaternion(wq_t[0], wq_t[1], wq_t[2], wq_t[3])
                self._world_scale = Vector3(ws_t[0], ws_t[1], ws_t[2])
                self._world_rotation = euler
            else:
                # World scale = parent scale * local scale
                self._world_scale = Vector3.scale(parent._world_scale, self._local_scale)

                # World rotation = parent quaternion * local quaternion (proper composition)
                self._world_quaternion = parent._world_quaternion * self._local_quaternion
                self._world_rotation = self._world_quaternion.to_euler_array()

                # Rotate local position by parent's world quaternion, scaled by parent scale
                R = parent._world_quaternion.to_rotation_matrix()
                scaled_local = self._local_position.to_numpy() * parent._world_scale.to_numpy()
                rotated_local = scaled_local @ R
                self._world_position = parent._world_position + rotated_local
        self._world_dirty = False
    
    @property
    def world_position(self) -> Vector3:
        """Get world position (computed from parent + local).

        Supports component-wise mutation (will convert back to local):
            t.world_position.x = 10
        """
        def _read():
            self._compute_world_transform()
            return self._world_position

        def _write(v):
            # Delegate to setter (handles conversion + notify)
            self.world_position = v

        return _Vector3Proxy(read_func=_read, write_func=_write)
    
    @world_position.setter
    def world_position(self, value):
        """Set world position (converts to local based on parent)."""
        world_pos = Vector3(value)
        
        if self._parent is None:
            self._local_position = world_pos
        else:
            parent = self._parent
            parent._compute_world_transform()
            
            # Inverse rotation via quaternion conjugate
            R_inv = parent._world_quaternion.conjugate.to_rotation_matrix()
            
            delta = (world_pos - parent._world_position).to_numpy()
            rotated_delta = delta @ R_inv
            self._local_position = Vector3(rotated_delta / parent._world_scale.to_numpy())
        
        self._mark_dirty()
    
    @property
    def world_rotation(self) -> tuple:
        """Get world rotation in degrees (computed from parent + local)."""
        self._compute_world_transform()
        return tuple(np.degrees(self._world_rotation))
    
    @world_rotation.setter
    def world_rotation(self, value):
        """Set world rotation (converts to local based on parent)."""
        world_rot = np.radians(value).astype(np.float32)
        world_q = Quaternion.from_euler(*world_rot)
        
        if self._parent is None:
            self._local_quaternion = world_q
            self._local_rotation = world_rot
        else:
            parent = self._parent
            parent._compute_world_transform()
            # local = parent_world^-1 * world
            self._local_quaternion = parent._world_quaternion.conjugate * world_q
            self._local_rotation = self._local_quaternion.to_euler_array()
        
        self._mark_dirty()
    
    @property
    def world_scale(self) -> Vector3:
        """Get world scale (computed from parent + local)."""
        def _read():
            self._compute_world_transform()
            return self._world_scale

        def _write(v):
            self.world_scale = v

        return _Vector3Proxy(read_func=_read, write_func=_write)

    @world_scale.setter
    def world_scale(self, value):
        """Set world scale (converts to local based on parent)."""
        if isinstance(value, (int, float)):
            world_scale = Vector3(value, value, value)
        else:
            world_scale = Vector3(value)

        if self._parent is None:
            self._local_scale = world_scale
        else:
            parent = self._parent
            parent._compute_world_transform()
            self._local_scale = Vector3(world_scale.to_numpy() / parent._world_scale.to_numpy())

        self._mark_dirty()
    
    # =========================================================================
    # Convenience properties (alias to local transform for backward compatibility)
    # =========================================================================
    
    def _set_local_position_vector(self, value):
        """Internal setter used by position/local_position and their proxies."""
        self._update_prev_position()
        self._local_position = Vector3(value)
        self._mark_dirty()

    def _set_local_scale_vector(self, value):
        """Internal setter used by scale_xyz / local_scale and their proxies."""
        if isinstance(value, (int, float)):
            self._local_scale = Vector3(value, value, value)
        else:
            self._local_scale = Vector3(value)
        self._mark_dirty()

    @property
    def position(self) -> Vector3:
        """Get local position (alias for local_position).

        Returns a proxy so that component-wise changes work:
            t.position.x = 5
            t.position.x += 1
            t.position[1] = 10
        """
        return _Vector3Proxy(
            read_func=lambda: self._local_position,
            write_func=self._set_local_position_vector,
        )

    @position.setter
    def position(self, value):
        self._set_local_position_vector(value)

    @property
    def x(self) -> float:
        return float(self._local_position.x)

    @x.setter
    def x(self, value: float):
        self.position = (value, self._local_position.y, self._local_position.z)

    @property
    def y(self) -> float:
        return float(self._local_position.y)

    @y.setter
    def y(self, value: float):
        self.position = (self._local_position.x, value, self._local_position.z)

    @property
    def z(self) -> float:
        return float(self._local_position.z)

    @z.setter
    def z(self, value: float):
        self.position = (self._local_position.x, self._local_position.y, value)

    def move(self, dx: float = 0, dy: float = 0, dz: float = 0):
        self._update_prev_position()
        self._local_position = self._local_position + Vector3(dx, dy, dz)
        self._mark_dirty()

    @property
    def rotation(self) -> tuple:
        return tuple(np.degrees(self._local_rotation))

    @rotation.setter
    def rotation(self, value):
        self._local_rotation = np.radians(value).astype(np.float32)
        self._local_quaternion = Quaternion.from_euler(*self._local_rotation)
        self._mark_dirty()

    @property
    def rotation_x(self) -> float:
        return float(np.degrees(self._local_rotation[0]))

    @rotation_x.setter
    def rotation_x(self, value: float):
        self._local_rotation[0] = np.radians(value)
        self._local_quaternion = Quaternion.from_euler(*self._local_rotation)
        self._mark_dirty()

    @property
    def rotation_y(self) -> float:
        return float(np.degrees(self._local_rotation[1]))

    @rotation_y.setter
    def rotation_y(self, value: float):
        self._local_rotation[1] = np.radians(value)
        self._local_quaternion = Quaternion.from_euler(*self._local_rotation)
        self._mark_dirty()

    @property
    def rotation_z(self) -> float:
        return float(np.degrees(self._local_rotation[2]))

    @rotation_z.setter
    def rotation_z(self, value: float):
        self._local_rotation[2] = np.radians(value)
        self._local_quaternion = Quaternion.from_euler(*self._local_rotation)
        self._mark_dirty()

    def rotate(self, dx: float = 0, dy: float = 0, dz: float = 0):
        delta_q = Quaternion.from_euler(
            float(np.radians(dx)),
            float(np.radians(dy)),
            float(np.radians(dz)),
        )
        self._local_quaternion = delta_q * self._local_quaternion
        self._local_rotation = self._local_quaternion.to_euler_array()
        self._mark_dirty()

    def set_rotation_quaternion(self, q: Quaternion):
        """Set rotation directly from a Quaternion (used by physics)."""
        self._local_quaternion = q.normalized
        self._local_rotation = self._local_quaternion.to_euler_array()
        self._mark_dirty()

    @property
    def scale(self) -> float:
        return float(self._local_scale.x)

    @scale.setter
    def scale(self, value):
        """Set uniform or per-axis scale.
        
        Accepts a scalar (for uniform scale) or a vector-like value (tuple, list,
        Vector3, ndarray) which will be used directly for (possibly non-uniform) scale.
        This provides robustness and backward compatibility with code that assigns
        vectors/tuples to the 'scale' convenience property.
        """
        if isinstance(value, (int, float)):
            s = float(value)
            self._local_scale = Vector3(s, s, s)
        else:
            self._local_scale = Vector3(value)
        self._mark_dirty()

    @property
    def scale_xyz(self) -> Vector3:
        """Get local scale as Vector3 (x,y,z).

        Supports component mutation via proxy:
            t.scale_xyz.x = 2.0
        """
        return _Vector3Proxy(
            read_func=lambda: self._local_scale,
            write_func=self._set_local_scale_vector,
        )

    @scale_xyz.setter
    def scale_xyz(self, value):
        self._set_local_scale_vector(value)

    def get_model_matrix(self) -> np.ndarray:
        if not self._transform_dirty:
            return self._cached_model
        
        # Compute world transform first
        self._compute_world_transform()

        # Rotation matrix directly from quaternion (avoids Euler gimbal lock)
        R = self._world_quaternion.to_rotation_matrix()
        self._cached_rotation = R

        s_x, s_y, s_z = self._world_scale.to_tuple()
        tx, ty, tz = self._world_position.to_tuple()
        S = np.array([[s_x, 0, 0, 0], [0, s_y, 0, 0], [0, 0, s_z, 0], [0, 0, 0, 1]], dtype=np.float32)
        R4 = np.eye(4, dtype=np.float32)
        R4[:3, :3] = R
        T = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [tx, ty, tz, 1]], dtype=np.float32)
        self._cached_model = S @ R4 @ T
        self._transform_dirty = False
        return self._cached_model

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get the 3x3 rotation matrix (world space)."""
        # Ensure cached rotation is up to date
        self.get_model_matrix() 
        return self._cached_rotation

    @property
    def forward(self) -> np.ndarray:
        """Get forward vector (world space, assuming -Z is forward)."""
        return -self.rotation_matrix[2, :]

    @property
    def backward(self) -> np.ndarray:
        """Get backward vector (world space, +Z)."""
        return self.rotation_matrix[2, :]

    @property
    def right(self) -> np.ndarray:
        """Get right vector (world space, +X)."""
        return self.rotation_matrix[0, :]

    @property
    def left(self) -> np.ndarray:
        """Get left vector (world space, -X)."""
        return -self.rotation_matrix[0, :]

    @property
    def up(self) -> np.ndarray:
        """Get up vector (world space, +Y)."""
        return self.rotation_matrix[1, :]

    @property
    def down(self) -> np.ndarray:
        """Get down vector (world space, -Y)."""
        return -self.rotation_matrix[1, :]

    def look_at(self, target: 'Vector3Like', world_up: 'Vector3Like' = (0, 1, 0)):
        """Look at a target position."""
        eye = self.world_position
        target = Vector3(target)
        world_up = Vector3(world_up)
        
        f = target - eye
        dist = f.magnitude
        if dist < 1e-6:
            return
        f = f.normalized
        
        r = Vector3.cross(f, world_up)
        if r.magnitude < 1e-6:
            r = Vector3.right()
        else:
            r = r.normalized
            
        u = Vector3.cross(r, f)
        
        # Rotation matrix [r, u, -f]  (rows = local basis in world space)
        R = np.vstack([r.to_numpy(), u.to_numpy(), (-f).to_numpy()])
        
        # Convert to quaternion (robust, avoids gimbal lock)
        q = Quaternion.from_rotation_matrix(R)
        
        if self._parent is None:
            self._local_quaternion = q
        else:
            self._parent._compute_world_transform()
            self._local_quaternion = self._parent._world_quaternion.conjugate * q
        
        self._local_rotation = self._local_quaternion.to_euler_array()
        self._mark_dirty()
