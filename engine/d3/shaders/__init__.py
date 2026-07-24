"""GLSL sources for the default 3D pipeline.

Shaders live as ``.vert`` / ``.frag`` files next to this module and are loaded
once at import time. ``Window3D`` reads these constants instead of embedding
multi-kilobyte strings.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_shader(name: str) -> str:
    """Load a shader file by name (e.g. ``'forward.vert'``)."""
    path = _DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Shader not found: {path}")
    return path.read_text(encoding="utf-8")


# Eager names used by Window3D
VERTEX_SHADER = load_shader("forward.vert")
VERTEX_SHADER_INSTANCED = load_shader("forward_instanced.vert")
FRAGMENT_SHADER = load_shader("forward.frag")
COLLIDER_VERTEX_SHADER = load_shader("collider.vert")
COLLIDER_FRAGMENT_SHADER = load_shader("collider.frag")
SHADOW_VERTEX_SHADER = load_shader("shadow.vert")
SHADOW_VERTEX_SHADER_INSTANCED = load_shader("shadow_instanced.vert")
SHADOW_FRAGMENT_SHADER = load_shader("shadow.frag")
PARTICLE_VERTEX_SHADER = load_shader("particle.vert")
PARTICLE_FRAGMENT_SHADER = load_shader("particle.frag")
FULLSCREEN_VERTEX_SHADER = load_shader("fullscreen.vert")
BLOOM_EXTRACT_FRAGMENT_SHADER = load_shader("bloom_extract.frag")
BLOOM_BLUR_FRAGMENT_SHADER = load_shader("bloom_blur.frag")
SSAO_FRAGMENT_SHADER = load_shader("ssao.frag")
TONEMAP_FRAGMENT_SHADER = load_shader("tonemap.frag")

__all__ = [
    "load_shader",
    "VERTEX_SHADER",
    "VERTEX_SHADER_INSTANCED",
    "FRAGMENT_SHADER",
    "COLLIDER_VERTEX_SHADER",
    "COLLIDER_FRAGMENT_SHADER",
    "SHADOW_VERTEX_SHADER",
    "SHADOW_VERTEX_SHADER_INSTANCED",
    "SHADOW_FRAGMENT_SHADER",
    "PARTICLE_VERTEX_SHADER",
    "PARTICLE_FRAGMENT_SHADER",
    "FULLSCREEN_VERTEX_SHADER",
    "BLOOM_EXTRACT_FRAGMENT_SHADER",
    "BLOOM_BLUR_FRAGMENT_SHADER",
    "SSAO_FRAGMENT_SHADER",
    "TONEMAP_FRAGMENT_SHADER",
]
