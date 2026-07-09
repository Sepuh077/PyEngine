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

    When rebuilt after improvements, this also advances .local_position
    using the post-gravity velocity (basic integration). The caller can
    then decide whether to apply curves and sync to transforms.

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

    for i in range(n):
        p = particles[i]
        if not p.active:
            continue

        p.age += dt
        if p.age >= p.life:
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

        # Basic position integration (in-place on local_position).
        # This is the improvement: more work done without creating Vector3 temporaries.
        # Caller may override velocity via curves afterwards (affects next frame)
        # or re-compute position if needed.
        pos = p.local_position
        pos._x = pos._x + vx * dt
        pos._y = pos._y + vy * dt
        pos._z = pos._z + vz * dt

    return expired
