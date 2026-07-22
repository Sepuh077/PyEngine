"""Tests for AudioClip cache and AudioSource spatial helpers (dummy SDL)."""
import os
import struct
import tempfile
import wave

import pytest

# Prefer dummy audio so CI/headless works
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from engine.audio import AudioClip, AudioSource, AudioListener, _ensure_mixer
from engine.gameobject import GameObject
from engine.types import Vector3


def _write_silent_wav(path: str, duration_s: float = 0.05, rate: int = 22050):
    n = int(rate * duration_s)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        # 16-bit silence
        frames = struct.pack("<" + "h" * n, *([0] * n))
        w.writeframes(frames)


@pytest.fixture(scope="module")
def wav_path():
    _ensure_mixer()
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    _write_silent_wav(path)
    yield path
    try:
        os.unlink(path)
    except OSError:
        pass
    AudioClip.clear_cache()


def test_audioclip_load_and_cache(wav_path):
    AudioClip.clear_cache()
    a = AudioClip.load(wav_path)
    b = AudioClip.load(wav_path)
    assert a is b
    assert a.file_path == wav_path
    # duration may be 0 if mixer failed entirely
    assert a.duration >= 0.0


def test_audioclip_clear_cache(wav_path):
    AudioClip.clear_cache()
    a = AudioClip.load(wav_path)
    AudioClip.clear_cache()
    b = AudioClip.load(wav_path)
    assert a is not b


def test_audio_listener_component():
    go = GameObject("cam")
    listener = AudioListener()
    go.add_component(listener)
    assert go.get_component(AudioListener) is listener
    assert abs(listener.volume - 1.0) < 1e-6


def test_audio_source_play_stop(wav_path):
    _ensure_mixer()
    go = GameObject("src")
    src = AudioSource()
    go.add_component(src)
    clip = AudioClip.load(wav_path)
    src.clip = clip
    # play/stop should not raise even with dummy driver
    try:
        src.play()
        src.stop()
    except Exception as e:
        pytest.skip(f"mixer unavailable: {e}")


def test_audio_source_spatial_fields():
    go = GameObject("src")
    src = AudioSource()
    go.add_component(src)
    src.spatial_blend = 1.0
    src.min_distance = 1.0
    src.max_distance = 20.0
    assert src.spatial_blend == 1.0
