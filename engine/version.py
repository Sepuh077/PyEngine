"""Single source of truth for the PyEngine package version."""

__version__ = "0.1.0"


def version_string() -> str:
    """Human-readable version including Cython acceleration status."""
    try:
        from engine.cython import CYTHON_ENABLED, _LOADED, _FAILED
        if CYTHON_ENABLED:
            extra = f"cython={len(_LOADED)} modules"
        elif _FAILED:
            extra = f"pure-python ({len(_FAILED)} extension(s) unavailable)"
        else:
            extra = "pure-python"
    except Exception:
        extra = "unknown"
    return f"{__version__} ({extra})"
