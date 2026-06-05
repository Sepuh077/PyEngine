"""
Audio system for Engine3D.

Provides AudioClip, AudioListener, and AudioSource components for
2D and 3D positional audio using pygame.mixer.
"""
import math
from typing import Optional, Dict, TYPE_CHECKING

import pygame
import pygame.mixer

from engine3d.component import Component, InspectorField
from engine3d.types import Vector3

if TYPE_CHECKING:
    from engine3d.gameobject import GameObject


_mixer_available: bool = False


def _ensure_mixer() -> None:
    """Initialize pygame.mixer if it hasn't been initialized yet."""
    global _mixer_available
    if _mixer_available:
        return
    if pygame.mixer.get_init():
        _mixer_available = True
        return
    try:
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
        _mixer_available = True
    except pygame.error:
        # Headless / CI – try the SDL dummy driver
        import os
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            _mixer_available = True
        except pygame.error:
            _mixer_available = False


# ---------------------------------------------------------------------------
# AudioClip
# ---------------------------------------------------------------------------

class AudioClip:
    """
    A loaded audio asset (WAV, OGG, MP3, etc.).

    AudioClip is a *resource*, not a Component.  You load it once and assign
    it to one or more AudioSource components.

    Example::

        clip = AudioClip("sounds/explosion.wav")
        source.clip = clip
        source.play()
    """

    _cache: Dict[str, "AudioClip"] = {}

    def __init__(self, file_path: str):
        _ensure_mixer()
        self.file_path: str = file_path
        self._sound: Optional[pygame.mixer.Sound] = None
        if _mixer_available:
            self._sound = pygame.mixer.Sound(file_path)

    # -- properties ----------------------------------------------------------

    @property
    def duration(self) -> float:
        """Duration of the clip in seconds."""
        if self._sound is None:
            return 0.0
        return self._sound.get_length()

    # -- factory -------------------------------------------------------------

    @classmethod
    def load(cls, file_path: str) -> "AudioClip":
        """Load a clip (cached – same path returns the same object)."""
        if file_path in cls._cache:
            return cls._cache[file_path]
        clip = cls(file_path)
        cls._cache[file_path] = clip
        return clip

    @classmethod
    def clear_cache(cls) -> None:
        cls._cache.clear()

    def __repr__(self) -> str:
        return f"AudioClip({self.file_path!r})"


# ---------------------------------------------------------------------------
# AudioListener
# ---------------------------------------------------------------------------

class AudioListener(Component):
    """
    Represents the "ears" in the scene.

    Attach to a camera (typically the main camera).  Only **one**
    AudioListener should be active per scene – extra listeners are ignored.

    The listener's world position (from its GameObject's Transform) is used
    by every AudioSource with ``spatial_blend > 0`` to calculate distance
    attenuation and stereo panning.
    """

    volume = InspectorField(
        float, default=1.0,
        min_value=0.0, max_value=1.0, step=0.01,
        tooltip="Master volume multiplier for all audio",
    )

    def __init__(self, volume: float = 1.0):
        super().__init__()
        self.volume = volume

    @property
    def world_position(self) -> Vector3:
        if self.game_object:
            return self.game_object.transform.position
        return Vector3.zero()

    @property
    def forward(self):
        if self.game_object:
            return self.game_object.transform.forward
        return Vector3(0, 0, -1)

    @property
    def right(self):
        if self.game_object:
            return self.game_object.transform.right
        return Vector3(1, 0, 0)


# ---------------------------------------------------------------------------
# AudioSource
# ---------------------------------------------------------------------------

class AudioSource(Component):
    """
    Plays an :class:`AudioClip` from the position of its GameObject.

    Inspector-visible fields let you tweak volume, pitch, looping,
    and a 2D/3D spatial blend directly in the editor.

    Example::

        source = AudioSource(clip=my_clip, play_on_awake=True)
        go.add_component(source)
    """

    # -- inspector fields ----------------------------------------------------

    volume = InspectorField(
        float, default=1.0,
        min_value=0.0, max_value=1.0, step=0.01,
        tooltip="Playback volume (0 = silent, 1 = full)",
    )
    pitch = InspectorField(
        float, default=1.0,
        min_value=0.1, max_value=3.0, step=0.01,
        tooltip="Playback speed / pitch multiplier",
    )
    loop = InspectorField(
        bool, default=False,
        tooltip="Loop the clip continuously",
    )
    play_on_awake = InspectorField(
        bool, default=False,
        tooltip="Start playing as soon as the component is attached",
    )
    mute = InspectorField(
        bool, default=False,
        tooltip="Mute this source (keeps playing silently)",
    )
    spatial_blend = InspectorField(
        float, default=0.0,
        min_value=0.0, max_value=1.0, step=0.01,
        tooltip="0 = fully 2D, 1 = fully 3D (distance-attenuated)",
    )
    min_distance = InspectorField(
        float, default=1.0,
        min_value=0.0, max_value=500.0, step=0.5,
        tooltip="Distance at which 3D volume starts to attenuate",
    )
    max_distance = InspectorField(
        float, default=50.0,
        min_value=0.0, max_value=1000.0, step=1.0,
        tooltip="Distance beyond which the source is silent",
    )

    def __init__(
        self,
        clip: Optional[AudioClip] = None,
        *,
        volume: float = 1.0,
        pitch: float = 1.0,
        loop: bool = False,
        play_on_awake: bool = False,
        mute: bool = False,
        spatial_blend: float = 0.0,
        min_distance: float = 1.0,
        max_distance: float = 50.0,
    ):
        super().__init__()
        _ensure_mixer()

        self.clip: Optional[AudioClip] = clip

        # Inspector-backed fields
        self.volume = volume
        self.pitch = pitch
        self.loop = loop
        self.play_on_awake = play_on_awake
        self.mute = mute
        self.spatial_blend = spatial_blend
        self.min_distance = min_distance
        self.max_distance = max_distance

        # Internal state
        self._channel: Optional[pygame.mixer.Channel] = None

    # -- lifecycle -----------------------------------------------------------

    def on_attach(self):
        if self.play_on_awake and self.clip:
            self.play()

    def update(self):
        if self._channel is None or not self._channel.get_busy():
            return
        self._apply_spatial()

    # -- playback API --------------------------------------------------------

    def play(self, clip: Optional[AudioClip] = None) -> None:
        """Play the assigned clip (or *clip* if given)."""
        if clip is not None:
            self.clip = clip
        if self.clip is None or self.clip._sound is None:
            return
        _ensure_mixer()
        if not _mixer_available:
            return
        loops = -1 if self.loop else 0
        self._channel = self.clip._sound.play(loops=loops)
        if self._channel is None:
            return
        self._apply_volume()
        self._apply_spatial()

    def stop(self) -> None:
        if self._channel:
            self._channel.stop()
            self._channel = None

    def pause(self) -> None:
        if self._channel:
            self._channel.pause()

    def unpause(self) -> None:
        if self._channel:
            self._channel.unpause()

    @property
    def is_playing(self) -> bool:
        return self._channel is not None and self._channel.get_busy()

    # -- internal helpers ----------------------------------------------------

    def _find_listener(self) -> Optional[AudioListener]:
        scene = self.scene
        if scene is None:
            return None
        for obj in scene.objects:
            listener = obj.get_component(AudioListener)
            if listener is not None:
                return listener
        return None

    def _apply_volume(self) -> None:
        if self._channel is None:
            return
        if self.mute:
            self._channel.set_volume(0.0)
            return
        self._channel.set_volume(max(0.0, min(1.0, self.volume)))

    def _apply_spatial(self) -> None:
        """Apply 3D distance attenuation and stereo panning."""
        if self._channel is None:
            return
        if self.spatial_blend <= 0.0:
            self._apply_volume()
            return

        listener = self._find_listener()
        if listener is None:
            self._apply_volume()
            return

        src_pos = self.game_object.transform.position if self.game_object else Vector3.zero()
        ear_pos = listener.world_position

        offset = src_pos - ear_pos
        dist = offset.magnitude

        # Distance attenuation (inverse-distance clamped)
        if dist <= self.min_distance:
            attenuation = 1.0
        elif dist >= self.max_distance:
            attenuation = 0.0
        else:
            attenuation = 1.0 - (dist - self.min_distance) / (self.max_distance - self.min_distance)

        # Blend between 2D (flat) and 3D (attenuated)
        blend = self.spatial_blend
        final_vol = self.volume * ((1.0 - blend) + blend * attenuation)
        final_vol *= listener.volume

        if self.mute:
            final_vol = 0.0

        # Stereo panning via per-channel volume
        right_vec = listener.right
        try:
            dot = float(offset.x * right_vec[0] + offset.y * right_vec[1] + offset.z * right_vec[2])
        except (TypeError, IndexError):
            dot = 0.0
        pan = max(-1.0, min(1.0, dot / max(dist, 0.001))) * blend

        left = max(0.0, min(1.0, final_vol * (1.0 - pan)))
        right = max(0.0, min(1.0, final_vol * (1.0 + pan)))
        self._channel.set_volume(left, right)
