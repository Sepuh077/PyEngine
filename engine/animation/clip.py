"""
Animation clip and keyframe primitives.

An ``AnimationClip`` drives a sequence of ``KeyFrame`` objects forward in
time.  Each keyframe can carry arbitrary start/end event callbacks and
property bindings so that component attributes are set automatically
when the frame becomes active.

Typical usage::

    kf0 = KeyFrame(step=0, start_events=lambda: print("begin"))
    kf1 = KeyFrame(step=1)
    kf1.bind_property(my_transform, "position", Vector3(1, 2, 3))
    kf2 = KeyFrame(step=3, end_events=lambda: print("done"))

    clip = AnimationClip(keyframes=[kf0, kf1, kf2], is_loop=False)
    clip.start()
    # call clip.update(dt) each frame
"""

from typing import List, Callable, Any, Union

from engine import Component


class KeyFrame:
    """A single frame in an animation timeline.

    Each keyframe sits at a discrete *step* index inside an
    ``AnimationClip``.  When the clip reaches this step the
    ``start_events`` fire; when the clip moves **away** from this step
    the ``end_events`` fire.

    Args:
        step: Zero-based position in the clip timeline.
        start_events: Callback(s) invoked when this frame becomes active.
            Accepts a single callable, a list of callables, or ``None``.
        end_events: Callback(s) invoked when this frame is deactivated.
            Same flexible input as *start_events*.
    """

    def __init__(self, step: int = 0, start_events: Union[List[Callable], Callable, None] = None, end_events: Union[List[Callable], Callable, None] = None):
        self.step = step
        self.start_events = self._normalize_events(start_events)
        self.end_events = self._normalize_events(end_events)

    def _normalize_events(self, events: Union[List[Callable], Callable, None]):
        """Coerce *events* into a list of callables."""
        if events is None:
            return []
        if callable(events):
            return [events]
        return events

    def bind_property(self, component: Component, parameter: str, value: Any):
        """Append a start-event that sets *parameter* on *component* to *value*.

        This is a convenience shortcut so that callers do not need to
        build ``setattr`` lambdas manually.

        Args:
            component: The component whose attribute will be set.
            parameter: Attribute name on *component*.
            value: The value to assign when this frame starts.
        """
        self.start_events.append(
            lambda: setattr(component, parameter, value)
        )

    def start(self):
        """Fire all start-event callbacks."""
        for event in self.start_events:
            event()

    def finish(self):
        """Fire all end-event callbacks."""
        for event in self.end_events:
            event()


class AnimationClip:
    """A linear sequence of keyframes played over time.

    The clip maintains an internal timer and advances through its
    keyframes at the rate given by *frame_update_time*.  Gaps between
    user-supplied keyframe steps are filled with empty ``KeyFrame``
    placeholders so that every step index has a frame.

    Args:
        keyframes: Initial keyframes.  Duplicates (same step) are
            deduplicated and the list is sorted automatically.
        is_loop: If ``True`` the clip wraps back to step 0 after the
            last keyframe instead of finishing.
        frame_update_time: Minimum elapsed time (seconds) before the
            clip advances to the next step.
    """

    def __init__(self, keyframes: List[KeyFrame] = None, is_loop: bool = False, frame_update_time: float = 0.05):
        self.is_loop: bool = is_loop
        self.frame_update_time: float = frame_update_time
        self.keyframes = keyframes or []
        self._timer: float = 0
        self._current_frame: KeyFrame = None
        self._is_finished: bool = False

    @property
    def keyframes(self):
        """The full list of keyframes (including auto-generated gap fillers)."""
        return self._keyframes

    @keyframes.setter
    def keyframes(self, keyframes: List[KeyFrame] = None):
        """Sort, deduplicate, and gap-fill the supplied keyframes.

        After this setter the internal list is guaranteed to have one
        ``KeyFrame`` per step from 0 to ``max(step)``.
        """
        if keyframes:
            keyframes = sorted(keyframes, key=lambda k: k.step)
            for i in range(len(keyframes) - 1, 0, -1):
                if keyframes[i].step == keyframes[i - 1].step:
                    keyframes.pop(i)

            new_keyframes = [None] * (keyframes[-1].step + 1)
            for k in keyframes:
                new_keyframes[k.step] = k
        else:
            new_keyframes = [None]

        for i in range(len(new_keyframes)):
            if new_keyframes[i] is None:
                new_keyframes[i] = KeyFrame(step=i)
        self._keyframes = new_keyframes

    @property
    def current_frame(self):
        """The keyframe that is currently active, or ``None``."""
        return self._current_frame

    @current_frame.setter
    def current_frame(self, frame: KeyFrame):
        """Transition to *frame*, finishing the previous one if needed."""
        if self._current_frame:
            if frame.step == self.current_frame.step:
                return
            self.current_frame.finish()
        self._current_frame = frame
        self._current_frame.start()

    def start(self):
        """Reset the clip to step 0 and begin playback."""
        self._timer = 0
        self._current_frame = None
        self._is_finished = False
        self._set_keyframe(0)

    def _set_keyframe(self, index):
        """Activate the keyframe at *index*, handling loop/finish logic."""
        self._timer = 0
        if index >= len(self.keyframes):
            if self.is_loop:
                index = 0
            else:
                if self._current_frame:
                    self._current_frame.finish()
                self._is_finished = True
        if index < len(self.keyframes):
            self.current_frame = self.keyframes[index]

    def _update_frame(self):
        """Advance to the next keyframe if one is currently active."""
        if self._current_frame is None:
            return
        self._set_keyframe(self._current_frame.step + 1)

    def update(self, dt: float):
        """Tick the clip by *dt* seconds, advancing frames as needed.

        Args:
            dt: Elapsed time since the last update (seconds).
        """
        if not self._is_finished:
            self._timer += dt
            if self._timer >= self.frame_update_time:
                self._update_frame()
