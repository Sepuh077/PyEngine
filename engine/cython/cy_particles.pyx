# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated particle system update loop.
"""

from libc.math cimport sqrt
import numpy as np
cimport numpy as cnp

cnp.import_array()


def update_particles_fast(
    list particles,
    double dt,
    double gravity_y,
    double gravity_scale,
):
    """
    Batch-update particles (age, lifetime, gravity on velocity, and basic position).

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
    Extended particle update that also computes life_ratio for active particles.

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
