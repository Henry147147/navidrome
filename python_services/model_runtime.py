"""
Shared runtime helpers for GPU-backed model execution.

This module enforces a single active model run at a time and provides a
configurable idle timeout used by model wrappers to offload/unload safely.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from threading import RLock
from typing import Iterator, Optional


DEFAULT_MODEL_IDLE_TIMEOUT_SECONDS = int(
    os.getenv("NAVIDROME_MODEL_IDLE_TIMEOUT_SECONDS", "360")
)

_MODEL_RUN_LOCK = RLock()


@contextmanager
def exclusive_model_access(
    owner: str, logger: Optional[logging.Logger] = None
) -> Iterator[None]:
    """Ensure only one model is running at a time (re-entrant for same thread)."""
    if logger:
        logger.debug("Waiting for exclusive model access: %s", owner)
    with _MODEL_RUN_LOCK:
        if logger:
            logger.debug("Acquired exclusive model access: %s", owner)
        yield
