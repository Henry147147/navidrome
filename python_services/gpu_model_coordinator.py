"""
Simple coordinator to ensure only one heavyweight model occupies the GPU at a time.

Each model registers an offload callback that moves its weights to CPU. Whenever a
model claims GPU ownership, the coordinator offloads the previously active model
before letting the new claimant proceed. This keeps VRAM usage bounded across
multiple upload cycles.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Optional


class GPUModelCoordinator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: Optional[str] = None
        self._offloaders: Dict[str, Callable[[], None]] = {}

    def register(self, owner: str, offload_fn: Callable[[], None]) -> None:
        """
        Register an owner with its offload callback.
        """
        if not owner:
            return
        with self._lock:
            self._offloaders[owner] = offload_fn

    def claim(self, owner: str, logger: Optional[logging.Logger] = None) -> None:
        """
        Claim exclusive GPU access for `owner`. If another owner is active, its
        offload callback is invoked to free VRAM before granting the claim.
        """
        if not owner:
            return

        offload_prev: Optional[Callable[[], None]] = None
        with self._lock:
            if self._current and self._current != owner:
                offload_prev = self._offloaders.get(self._current)
            self._current = owner

        if offload_prev:
            try:
                offload_prev()
            except Exception:  # pragma: no cover - defensive
                if logger:
                    logger.exception(
                        "Failed to offload previous GPU owner %s", self._current
                    )

    def current_owner(self) -> Optional[str]:
        with self._lock:
            return self._current


GPU_COORDINATOR = GPUModelCoordinator()
