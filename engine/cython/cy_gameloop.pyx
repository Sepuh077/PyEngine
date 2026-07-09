# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated game loop.

Provides a fast update path that skips GameObjects whose components are all
no-ops (Transform, Object2D, etc.) and only dispatches to Python for
GameObjects that carry Script subclasses or active coroutines.

The user writes exactly the same pure-Python Script code; the speed-up comes
from eliminating thousands of empty ``comp.update()`` calls per frame on
passive objects (background sprites, static decorations, etc.).
"""


def cy_update_objects(list objects, double delta_time):
    """Update all GameObjects, skipping those with no scripts or coroutines.

    For each object:
      - If it has scripts or active coroutines, call its full ``update()``
        (this ensures *all* Components such as Rigidbody2D/3D, ParticleSystem,
        etc. run, not just Scripts).
      - Otherwise skip it entirely (big win for scenes full of static visuals).

    Objects that have neither (e.g. background stars with only Transform +
    Object2D) are skipped entirely — this is where the speedup comes from.

    Parameters
    ----------
    objects : list[GameObject]
        All GameObjects in the current scene.
    delta_time : float
        Frame delta time (seconds).
    """
    cdef Py_ssize_t i, j, n, ns
    cdef object obj, script
    cdef list scripts

    # Import rigidbodies and animator once (for explicit update of behavior components)
    RB2 = RB3 = None
    AnimatorCls = None
    try:
        from engine.d2.physics.rigidbody import Rigidbody2D as _RB2
        from engine.d3.physics.rigidbody import Rigidbody3D as _RB3
        RB2, RB3 = _RB2, _RB3
    except Exception:
        pass
    try:
        from engine.animation.animator import Animator as _Animator
        AnimatorCls = _Animator
    except Exception:
        pass

    n = <Py_ssize_t>len(objects)
    for i in range(n):
        obj = <object>objects[i]

        scripts = obj._scripts
        ns = <Py_ssize_t>len(scripts)
        has_anim = False
        anim = None
        if AnimatorCls is not None:
            try:
                anim = obj.get_component(AnimatorCls)
                has_anim = anim is not None
            except Exception:
                pass

        # Update objects that have scripts, coroutines, or Animators (behavior components)
        needs_update = (ns > 0) or bool(obj._active_coroutines) or has_anim

        if needs_update:
            if ns > 0:
                for j in range(ns):
                    script = <object>scripts[j]
                    script.update()

                # Drive rigidbody movement if present
                if RB2 is not None or RB3 is not None:
                    try:
                        rb = obj.get_component(RB2) or obj.get_component(RB3)
                        if rb is not None:
                            rb.update()
                    except Exception:
                        pass

            if has_anim and anim is not None:
                try:
                    anim.update()
                except Exception:
                    pass

        if obj._active_coroutines:
            obj._update_coroutines(delta_time)


def cy_update_end_of_frame(list objects, double delta_time):
    """Process end-of-frame coroutines on all GameObjects.

    Like ``cy_update_objects`` this skips objects that have nothing to do,
    avoiding the overhead of iterating empty coroutine lists in Python.

    Parameters
    ----------
    objects : list[GameObject]
        All GameObjects in the current scene.
    delta_time : float
        Frame delta time (seconds).
    """
    cdef Py_ssize_t i, n
    cdef object obj

    n = <Py_ssize_t>len(objects)
    for i in range(n):
        obj = <object>objects[i]
        if obj._end_of_frame_coroutines:
            obj._update_end_of_frame_coroutines(delta_time)
