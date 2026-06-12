"""
Animator state-machine that drives animation clips.

The module provides a Unity-style animator controller built from four
parts:

* ``AnimatorParameter`` -- named values (bool triggers, floats, etc.)
  that feed transition conditions.
* ``AnimationStateTransition`` -- a conditional edge between two states.
* ``AnimatorState`` -- wraps an ``AnimationClip`` and holds outgoing
  transitions.
* ``Animator`` -- the ``Component`` that owns states/parameters and
  ticks the active state each frame.

Typical usage::

    idle_clip  = AnimationClip(...)
    walk_clip  = AnimationClip(...)

    idle_state = AnimatorState("idle", idle_clip)
    walk_state = AnimatorState("walk", walk_clip)

    animator = Animator()
    animator.register_state(idle_state, is_initial=True)
    animator.register_state(walk_state)
    animator.register_parameter("is_walking", is_trigger=False, value=False)

    idle_state.add_transition(
        walk_state,
        lambda: animator._parameters["is_walking"].value is True,
    )

    # at runtime:
    animator.set_parameter("is_walking", True)   # triggers transition
"""

from enum import IntEnum
from typing import Any, Callable, overload, List, Dict, Union
from dataclasses import dataclass

from engine import Component, Time
from .clip import AnimationClip


class AnimatorUpdateMode(IntEnum):
    """Controls which time value the animator uses each tick.

    Attributes:
        SCALED: Use ``Time.delta_time`` (affected by ``Time.scale``).
        UNSCALED: Use ``Time.unscaled_delta_time`` (real wall-clock delta).
    """
    SCALED = 0
    UNSCALED = 1


class AnimationStateTransition:
    """A conditional edge from one animator state to another.

    The transition fires when *condition* returns ``True``.

    Args:
        to: Target ``AnimatorState`` to transition into.
        condition: Zero-argument callable returning ``bool``.
    """

    def __init__(self, to: "AnimatorState", condition: Callable[[], bool]):
        self.to = to
        self.condition = condition

    def check(self):
        """Return ``True`` if the transition condition is met."""
        return self.condition()


class AnimatorState:
    """A named node in the animator state-machine graph.

    Each state owns an ``AnimationClip`` and a list of outgoing
    ``AnimationStateTransition`` objects.

    Args:
        name: Unique identifier for this state inside its ``Animator``.
        clip: The animation clip to play while this state is active.
    """

    def __init__(self, name: str, clip: AnimationClip):
        self.name = name
        self.clip = clip
        self._transitions: List[AnimationStateTransition] = []

    def add_transition(self, to: "AnimatorState", condition: Callable[[], bool]):
        """Register a conditional transition to another state.

        Non-callable *condition* values are silently ignored.

        Args:
            to: The destination state.
            condition: Zero-argument callable returning ``bool``.
        """
        if not callable(condition):
            return
        self._transitions.append(
            AnimationStateTransition(to, condition)
        )

    def check(self) -> Union["AnimatorState", None]:
        """Evaluate all outgoing transitions and return the first matching target.

        Returns:
            The target ``AnimatorState`` of the first transition whose
            condition is ``True``, or ``None`` if no transition fires.
        """
        for transition in self._transitions:
            if transition.check():
                return transition.to

    def update(self, dt):
        """Tick the underlying clip by *dt* seconds."""
        self.clip.update(dt)


@dataclass
class AnimatorParameter:
    """A named value stored in the ``Animator`` for transition conditions.

    Attributes:
        name: Parameter identifier used in ``set_parameter`` /
            ``register_parameter``.
        is_trigger: If ``True`` the parameter is a one-shot trigger
            (not yet auto-reset -- callers must clear it manually).
        value: Current value of the parameter.
    """
    name: str
    is_trigger: bool
    value: Any


class Animator(Component):
    """Component that runs a finite state-machine of animation clips.

    Register states and parameters, wire up transitions via
    ``AnimatorState.add_transition``, then call ``start()`` to begin
    playback from the initial state.  Each ``update()`` tick advances
    the active clip and re-evaluates transitions when a parameter
    changes.

    Args:
        update_mode: Whether to use scaled or unscaled delta time.
    """

    def __init__(self, update_mode: AnimatorUpdateMode = AnimatorUpdateMode.UNSCALED):
        super().__init__()
        self.update_mode: AnimatorUpdateMode = update_mode
        self._states: Dict[str, AnimatorState] = {}
        self._parameters: Dict[str, AnimatorParameter] = {}
        self._initial_state: AnimatorState = None
        self._current_state: AnimatorState = None

    @property
    def current_state(self):
        """The currently active ``AnimatorState``, or ``None``."""
        return self._current_state

    def start(self):
        """Begin playback from the registered initial state.

        Raises:
            ValueError: If no initial state has been registered.
        """
        super().start()
        if self._initial_state is None:
            raise ValueError(f"Initial state should not be empty in the Animator.")
        self._set_state(self._initial_state)

    @overload
    def register_parameter(self, name: str, is_trigger: bool = True, value: Any = None) -> None: ...

    @overload
    def register_parameter(self, parameter: AnimatorParameter) -> None: ...

    def register_parameter(self, name_or_param: str | AnimatorParameter, is_trigger: bool = True, value: Any = None):
        """Register a parameter by name or from an ``AnimatorParameter`` instance.

        Args:
            name_or_param: Either the parameter name (str) or a
                pre-built ``AnimatorParameter``.
            is_trigger: Only used when *name_or_param* is a string.
            value: Initial value; only used when *name_or_param* is a
                string.
        """
        if isinstance(name_or_param, AnimatorParameter):
            parameter = name_or_param
        else:
            parameter = AnimatorParameter(name_or_param, is_trigger, value)

        self._parameters[parameter.name] = parameter

    def register_state(self, state: AnimatorState, is_initial: bool = False):
        """Add a state to the animator.

        Args:
            state: The ``AnimatorState`` to register.
            is_initial: If ``True`` this state becomes the entry point
                when ``start()`` is called.
        """
        self._states[state.name] = state
        if is_initial:
            self._initial_state = state

    def _set_state(self, state: AnimatorState):
        """Switch to *state* and restart its clip."""
        self._current_state = state
        state.clip.start()

    def set_parameter(self, name: str, value: Any = None):
        """Update a parameter and potentially trigger a state transition.

        If the parameter change causes the current state's transition
        condition to fire, the animator immediately switches to the
        target state.

        Args:
            name: Name of a previously registered parameter.
            value: New value to assign.

        Raises:
            AttributeError: If the transition targets a state that was
                never registered.
        """
        if name not in self._parameters:
            return
        self._parameters[name].value = value
        new_state = self._current_state.check()
        if new_state:
            if new_state.name not in self._states:
                raise AttributeError(f"The state {new_state.name} is not registered in the Animator")
            self._set_state(new_state)

    def update(self):
        """Advance the current state's clip by the appropriate delta time."""
        self._current_state.update(Time.delta_time if self.update_mode == AnimatorUpdateMode.SCALED else Time.unscaled_delta_time)
