# cython: boundscheck=False, wraparound=False, cdivision=True, nonecheck=False
"""
Cython-accelerated entity / component container for very large numbers of objects.

The goal is to make scenes with tens of thousands of GameObjects (most of them
static / purely visual) cheap to simulate.

Key ideas:
- Scene.objects remains the single source of truth (for editor, serialization, queries, rendering).
- We maintain a much smaller "updatables" list containing only GameObjects that
  have active behavior (Scripts, Rigidbodies, Animators, or live coroutines).
- Cython helpers provide fast collection / filtering when a full scan/rebuild is needed.
- Cached hot component references on GameObject (_rigidbody, _animator) avoid
  expensive get_component linear searches inside per-frame loops.
- User code, editor, and public APIs are completely unaffected.

The container can be used internally by the game loop, physics broadphase
prep, etc. in the future.
"""

from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF

cdef class EntityContainer:
    """
    A lightweight Cython container that tracks full object list + the
    much smaller subset of objects that require per-frame simulation.

    This is an internal optimization.  All high-level code continues to
    use normal GameObject / Scene APIs.
    """

    cdef public list all_objects
    cdef public list updatables
    cdef set _updatable_set   # fast membership using id(obj)

    def __cinit__(self):
        self.all_objects = []
        self.updatables = []
        self._updatable_set = set()

    cpdef void add(self, object obj):
        """Add a GameObject to the container (idempotent for updatables)."""
        if obj is None:
            return
        # We don't force-add to all_objects here; Scene manages the master list.
        # This container mainly cares about fast updatable tracking.
        self._ensure_updatable(obj)

    cpdef void remove(self, object obj):
        """Remove a GameObject from updatable tracking."""
        if obj is None:
            return
        cdef long oid = <long>id(obj)
        if oid in self._updatable_set:
            self._updatable_set.remove(oid)
            # Remove from list (order does not matter much)
            try:
                self.updatables.remove(obj)
            except ValueError:
                pass

    cpdef void _ensure_updatable(self, object obj):
        """Register obj as needing simulation if not already."""
        if obj is None:
            return
        cdef long oid = <long>id(obj)
        if oid not in self._updatable_set:
            self._updatable_set.add(oid)
            self.updatables.append(obj)

    cpdef list get_updatables(self):
        """Return the current list of objects that need per-frame work."""
        return self.updatables

    cpdef Py_ssize_t num_updatables(self):
        return <Py_ssize_t>len(self.updatables)

    cpdef void clear(self):
        self.updatables.clear()
        self._updatable_set.clear()

    cpdef void rebuild_updatables(self, list objects):
        """
        Fast Cython scan over the full objects list to (re)build the
        updatables list. Useful after bulk operations or loading.
        """
        cdef Py_ssize_t i, n
        cdef object obj
        cdef list new_up = []
        cdef set seen = set()

        self.updatables.clear()
        self._updatable_set.clear()

        n = <Py_ssize_t>len(objects)
        for i in range(n):
            obj = <object>objects[i]
            if self._object_needs_update(obj):
                oid = <long>id(obj)
                if oid not in seen:
                    seen.add(oid)
                    new_up.append(obj)
                    self._updatable_set.add(oid)

        self.updatables = new_up

    cdef inline bint _object_needs_update(self, object obj):
        """
        Heuristic implemented in Cython for speed.
        Mirrors (and can be the source of truth for) the logic in cy_gameloop.
        """
        cdef list scripts
        cdef bint has_coro = False
        cdef bint has_rb = False
        cdef bint has_anim = False

        try:
            scripts = obj._scripts
            if scripts and len(scripts) > 0:
                return True
        except Exception:
            pass

        try:
            if obj._active_coroutines:
                return True
            if obj._end_of_frame_coroutines:
                return True
        except Exception:
            pass

        # Cached direct refs (set by GameObject when adding those components)
        try:
            if obj._rigidbody is not None:
                return True
        except Exception:
            pass

        try:
            if obj._animator is not None:
                return True
        except Exception:
            pass

        try:
            if obj._particle_system is not None:
                return True
        except Exception:
            pass

        return False

    cpdef list collect_updatables(self, list objects):
        """
        Pure scan that returns a new list of objects needing update.
        Does not mutate internal state.
        """
        cdef Py_ssize_t i, n
        cdef object obj
        cdef list result = []
        cdef set seen = set()

        n = <Py_ssize_t>len(objects)
        for i in range(n):
            obj = <object>objects[i]
            if self._object_needs_update(obj):
                oid = <long>id(obj)
                if oid not in seen:
                    seen.add(oid)
                    result.append(obj)
        return result


# -------------------------------------------------------------------------
# Convenience functions (usable from pure Python and other Cython modules)
# -------------------------------------------------------------------------

cpdef list get_updatables(list objects):
    """Quick one-shot collection of objects that require simulation."""
    cdef EntityContainer tmp = EntityContainer()
    return tmp.collect_updatables(objects)


cpdef bint object_needs_update(object obj):
    """Fast check whether a single GameObject has active behavior."""
    cdef EntityContainer tmp = EntityContainer()
    return tmp._object_needs_update(obj)
