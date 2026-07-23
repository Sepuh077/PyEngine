# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated particle system update loops for both 3D and 2D.
"""

from libc.math cimport sqrt
import numpy as np
cimport numpy as cnp

cnp.import_array()


# =========================================================================
# 3D particle helpers (existing – use Vector3 objects on Particle instances)
# =========================================================================

def update_particles_fast(
    list particles,
    double dt,
    double gravity_y,
    double gravity_scale,
):
    """
    Batch-update 3D particles (age, lifetime, gravity on velocity, and basic position).

    This is the C-accelerated core. It mutates Particle and Vector3 objects
    in place to avoid Python object allocations in the hot loop.

    Parameters
    ----------
    particles : list of Particle objects
    dt, gravity_y, gravity_scale : simulation params

    Returns
    -------
    list of indices of particles that just expired.
    """
    cdef double grav_y = gravity_y * gravity_scale
    cdef double grav_y_dt = grav_y * dt

    expired = []
    cdef int i
    cdef int n = len(particles)
    cdef double vx, vy, vz
    cdef double px, py, pz

    for i in range(n):
        p = particles[i]
        if not p.active:
            continue

        p.age += dt
        if p.age >= p.life:
            expired.append(i)
            continue

        # Gravity on velocity (in-place, avoids Vector3 alloc)
        vel = p.velocity
        vx = vel._x
        vy = vel._y + grav_y_dt
        vz = vel._z
        vel._x = vx
        vel._y = vy
        vel._z = vz

        # Basic position integration (in-place on local_position).
        pos = p.local_position
        pos._x = pos._x + vx * dt
        pos._y = pos._y + vy * dt
        pos._z = pos._z + vz * dt

    return expired


def update_particles_full(
    list particles,
    double dt,
    double gravity_y,
    double gravity_scale,
    bint has_velocity_curve,
    bint has_size_curve,
    bint has_color_curve,
):
    """
    Extended 3D particle update that also computes life_ratio for active particles.

    Returns (expired_indices, active_indices_with_ratios).
    active_indices_with_ratios is a list of (index, life_ratio) tuples.
    """
    cdef double grav_y_val = gravity_y * gravity_scale
    cdef double grav_y_dt = grav_y_val * dt

    cdef list expired = []
    cdef list active_ratios = []
    cdef int i
    cdef int n = len(particles)
    cdef double vx, vy, vz
    cdef double age, life, ratio

    cdef bint need_ratio = has_velocity_curve or has_size_curve or has_color_curve

    for i in range(n):
        p = particles[i]
        if not p.active:
            continue

        p.age += dt
        age = p.age
        life = p.life

        if age >= life:
            expired.append(i)
            continue

        # Gravity on velocity (in-place)
        vel = p.velocity
        vx = vel._x
        vy = vel._y + grav_y_dt
        vz = vel._z
        vel._x = vx
        vel._y = vy
        vel._z = vz

        # Position integration
        pos = p.local_position
        pos._x = pos._x + vx * dt
        pos._y = pos._y + vy * dt
        pos._z = pos._z + vz * dt

        if need_ratio:
            if life > 1e-6:
                ratio = age / life
            else:
                ratio = 1.0
            active_ratios.append((i, ratio))
        else:
            active_ratios.append((i, 0.0))

    return expired, active_ratios


# =========================================================================
# 3D lightweight particles (Particle3DLight scalar slots: px/py/pz/vx/vy/vz)
# =========================================================================

def update_particles_3d_light_fast(
    list particles,
    double dt,
    double gravity_y,
):
    """
    Batch-update lightweight 3D particles (age, gravity on Y, position).

    Particle3DLight has scalar slots: px, py, pz, vx, vy, vz, age, life, active.

    Returns list of indices whose lifetime has expired.
    """
    cdef double grav_y_dt = gravity_y * dt
    cdef list expired = []
    cdef int i
    cdef int n = len(particles)
    cdef double vx, vy, vz

    for i in range(n):
        p = particles[i]
        if not p.active:
            continue

        p.age += dt
        if p.age >= p.life:
            expired.append(i)
            continue

        vy = p.vy + grav_y_dt
        vx = p.vx
        vz = p.vz
        p.vy = vy

        p.px = p.px + vx * dt
        p.py = p.py + vy * dt
        p.pz = p.pz + vz * dt

    return expired


def update_particles_3d_light_full(
    list particles,
    double dt,
    double gravity_y,
    bint has_velocity_curve,
    bint has_size_curve,
    bint has_color_curve,
):
    """
    Extended lightweight 3D update with life-ratio for Python-side curves.

    Returns (expired_indices, active_ratios) where active_ratios is
    a list of (index, life_ratio).
    """
    cdef double grav_y_dt = gravity_y * dt
    cdef list expired = []
    cdef list active_ratios = []
    cdef int i
    cdef int n = len(particles)
    cdef double vx, vy, vz
    cdef double age, life, ratio
    cdef bint need_ratio = has_velocity_curve or has_size_curve or has_color_curve

    for i in range(n):
        p = particles[i]
        if not p.active:
            continue

        p.age += dt
        age = p.age
        life = p.life

        if age >= life:
            expired.append(i)
            continue

        vy = p.vy + grav_y_dt
        vx = p.vx
        vz = p.vz
        p.vy = vy

        p.px = p.px + vx * dt
        p.py = p.py + vy * dt
        p.pz = p.pz + vz * dt

        if need_ratio:
            if life > 1e-6:
                ratio = age / life
            else:
                ratio = 1.0
            active_ratios.append((i, ratio))
        else:
            active_ratios.append((i, 0.0))

    return expired, active_ratios


def pack_particles_3d_render_data(
    list particles,
    double ox,
    double oy,
    double oz,
):
    """
    Build a contiguous float32 array of (px, py, pz, size, r, g, b, a)
    for all active lightweight 3D particles.  Faster than Python loop + list.
    """
    cdef int i, n, count, j
    cdef object p
    n = len(particles)
    count = 0
    for i in range(n):
        if particles[i].active:
            count += 1

    if count == 0:
        return np.empty((0, 8), dtype=np.float32)

    cdef cnp.ndarray[cnp.float32_t, ndim=2] data = np.empty((count, 8), dtype=np.float32)
    j = 0
    for i in range(n):
        p = particles[i]
        if not p.active:
            continue
        data[j, 0] = <cnp.float32_t>(p.px + ox)
        data[j, 1] = <cnp.float32_t>(p.py + oy)
        data[j, 2] = <cnp.float32_t>(p.pz + oz)
        data[j, 3] = <cnp.float32_t>p.size
        data[j, 4] = <cnp.float32_t>p.r
        data[j, 5] = <cnp.float32_t>p.g
        data[j, 6] = <cnp.float32_t>p.b
        data[j, 7] = <cnp.float32_t>p.a
        j += 1
    return data


# =========================================================================
# 2D particle helpers (lightweight Particle2D with scalar px/py/vx/vy slots)
# =========================================================================

def update_particles_2d_fast(
    list particles,
    double dt,
    double gravity_y,
):
    """
    Batch-update 2D lightweight particles (age, gravity, position integration).

    Particle2D has scalar slots: px, py, vx, vy, age, life, active.

    Returns list of indices whose lifetime has expired.
    """
    cdef double grav_y_dt = gravity_y * dt
    cdef list expired = []
    cdef int i
    cdef int n = len(particles)
    cdef double vx, vy

    for i in range(n):
        p = particles[i]
        if not p.active:
            continue

        p.age += dt
        if p.age >= p.life:
            expired.append(i)
            continue

        # Gravity on velocity (Y axis only in 2D)
        vy = p.vy + grav_y_dt
        vx = p.vx
        p.vy = vy

        # Position integration
        p.px = p.px + vx * dt
        p.py = p.py + vy * dt

    return expired


def update_particles_2d_full(
    list particles,
    double dt,
    double gravity_y,
    bint has_velocity_curve,
    bint has_size_curve,
    bint has_color_curve,
):
    """
    Extended 2D particle update with life-ratio calculation.

    Returns (expired_indices, active_ratios).
    active_ratios is a list of (index, life_ratio) for particles that need
    Python-side curve evaluation (velocity / size / color over lifetime).
    """
    cdef double grav_y_dt = gravity_y * dt
    cdef list expired = []
    cdef list active_ratios = []
    cdef int i
    cdef int n = len(particles)
    cdef double vx, vy
    cdef double age, life, ratio
    cdef bint need_ratio = has_velocity_curve or has_size_curve or has_color_curve

    for i in range(n):
        p = particles[i]
        if not p.active:
            continue

        p.age += dt
        age = p.age
        life = p.life

        if age >= life:
            expired.append(i)
            continue

        # Gravity on velocity
        vy = p.vy + grav_y_dt
        vx = p.vx
        p.vy = vy

        # Position integration
        p.px = p.px + vx * dt
        p.py = p.py + vy * dt

        if need_ratio:
            if life > 1e-6:
                ratio = age / life
            else:
                ratio = 1.0
            active_ratios.append((i, ratio))
        else:
            active_ratios.append((i, 0.0))

    return expired, active_ratios
