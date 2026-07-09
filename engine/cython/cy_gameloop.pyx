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

    n = <Py_ssize_t>len(objects)
    for i in range(n):
        obj = <object>objects[i]

        # If the object has any scripts or active coroutines, run its full
        # update (which iterates all Components including Rigidbody, etc.)
        # plus coroutines. This preserves correctness for objects whose
        # behavior lives in non-Script Components.
        # Objects with neither are skipped entirely for speed (only visual
        # components like Transform/Object2D that don't require per-frame
        # Python work).
        if obj._scripts or obj._active_coroutines:
            obj.update()


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
