# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated game loop.

Provides a fast update path that skips GameObjects whose components are all
no-ops (Transform, Object2D, etc.) and only dispatches to Python for
GameObjects that carry Script subclasses or active coroutines.

When used together with Scene._updatables (maintained via the Cython
EntityContainer in cy_entities), the loop only ever sees the small subset
of objects that actually need work.

The user writes exactly the same pure-Python Script code; the speed-up comes
from eliminating thousands of empty ``comp.update()`` calls per frame on
passive objects (background sprites, static decorations, etc.).
"""


def cy_update_objects(list objects, double delta_time):
    """Update all GameObjects, skipping those with no scripts, coroutines, animators, rigidbodies or particle systems.

    For each object:
      - If it has scripts/active coroutines/animators/rigidbodies/ParticleSystem, run the relevant
        updates (scripts + explicit Rigidbody/Animator/ParticleSystem updates).
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

    # Import ParticleSystem for explicit update (like we do for RB/Animator)
    PS = None
    try:
        from engine.d3.particle import ParticleSystem as _PS
        PS = _PS
    except Exception:
        pass

    n = <Py_ssize_t>len(objects)
    for i in range(n):
        obj = <object>objects[i]

        scripts = obj._scripts
        ns = <Py_ssize_t>len(scripts)

        # Fast path: use cached direct references populated by GameObject.add_component
        # instead of calling the Python get_component() which does linear search.
        anim = getattr(obj, '_animator', None)
        has_anim = anim is not None

        rb = getattr(obj, '_rigidbody', None)
        has_rb = rb is not None

        has_ps = False
        ps = getattr(obj, '_particle_system', None)
        if ps is not None:
            has_ps = True
        elif PS is not None:
            # Fallback to get_component (for objects added before cache was introduced)
            try:
                ps = obj.get_component(PS)
                has_ps = ps is not None
            except Exception:
                pass

        # Update objects that have scripts, coroutines, Animators, Rigidbodies or ParticleSystems
        # (Rigidbody/ParticleSystem updates are required even without scripts)
        needs_update = (ns > 0) or bool(getattr(obj, '_active_coroutines', None)) or has_anim or has_rb or has_ps

        if needs_update:
            if ns > 0:
                for j in range(ns):
                    script = <object>scripts[j]
                    # Ensure awake/start are called before first update (mirrors pure Python start_components)
                    if not getattr(script, '_awoken', False):
                        script.awake()
                        script._awoken = True
                    if not getattr(script, '_started', False):
                        script.start()
                        script._started = True
                    script.update()

            if has_rb and rb is not None:
                if not getattr(rb, '_awoken', False):
                    rb.awake()
                    rb._awoken = True
                if not getattr(rb, '_started', False):
                    rb.start()
                    rb._started = True
                try:
                    rb.update()
                except Exception:
                    pass

            if has_anim and anim is not None:
                if not getattr(anim, '_awoken', False):
                    anim.awake()
                    anim._awoken = True
                if not getattr(anim, '_started', False):
                    anim.start()
                    anim._started = True
                try:
                    anim.update()
                except Exception:
                    pass

            if has_ps and ps is not None:
                if not getattr(ps, '_awoken', False):
                    ps.awake()
                    ps._awoken = True
                if not getattr(ps, '_started', False):
                    ps.start()
                    ps._started = True
                try:
                    ps.update()
                except Exception:
                    pass

        if getattr(obj, '_active_coroutines', None):
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
        if getattr(obj, '_end_of_frame_coroutines', None):
            obj._update_end_of_frame_coroutines(delta_time)
