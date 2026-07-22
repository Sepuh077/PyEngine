"""Central logging helpers for PyEngine.

Silent ``except Exception: pass`` blocks should log at DEBUG (or WARNING for
unexpected failures) via ``get_logger()`` so developers can enable diagnostics
with::

    import logging
    logging.basicConfig(level=logging.DEBUG)
"""
from __future__ import annotations

import logging
from typing import Optional

_LOGGER_NAME = "pyengine"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the ``pyengine`` namespace."""
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)


def log_exception(logger: logging.Logger, msg: str, *args, level: int = logging.DEBUG) -> None:
    """Log the active exception with traceback at the given level."""
    logger.log(level, msg, *args, exc_info=True)
