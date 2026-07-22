"""Abstract base class for routing policies."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from slm_router.types import ModelSpec, RouteDecision

if TYPE_CHECKING:
    from slm_router.trace import RunTrace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defensive joblib import
# ---------------------------------------------------------------------------
try:
    import joblib as _joblib  # type: ignore
    _JOBLIB_AVAILABLE = True
except Exception:  # noqa: BLE001
    _joblib = None  # type: ignore
    _JOBLIB_AVAILABLE = False


class RoutingPolicy(ABC):
    """Abstract interface for all routing policies.

    A policy receives a context vector and a list of candidate models, returns
    a :class:`~slm_router.types.RouteDecision`, and can be updated from
    observed traces.

    Subclasses must implement :meth:`select` and :meth:`update`.
    """

    policy_version: str = "base-v0"

    @abstractmethod
    def select(
        self,
        ctx: np.ndarray,
        candidates: list[ModelSpec],
    ) -> RouteDecision:
        """Choose a model for the given context.

        Parameters
        ----------
        ctx:
            Context feature vector (difficulty feats + any other signals).
        candidates:
            Ordered list of candidate :class:`~slm_router.types.ModelSpec`.

        Returns
        -------
        RouteDecision
        """

    @abstractmethod
    def update(self, trace: RunTrace) -> None:  # type: ignore[override]
        """Update policy parameters from an observed trace.

        Parameters
        ----------
        trace:
            Completed :class:`~slm_router.trace.RunTrace` with reward signal.
        """

    # ------------------------------------------------------------------
    # Persistence helpers (optional override)
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise policy state to *path* via joblib.

        Subclasses may override to customise serialisation.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not _JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is required for policy save/load.")
        _joblib.dump(self.__dict__, path)
        logger.info("%s saved to %s.", type(self).__name__, path)

    def load(self, path: str | Path) -> None:
        """Restore policy state from *path*.

        Subclasses may override for custom deserialisation.
        """
        if not _JOBLIB_AVAILABLE:
            raise RuntimeError("joblib is required for policy save/load.")
        data = _joblib.load(path)
        self.__dict__.update(data)
        logger.info("%s loaded from %s.", type(self).__name__, path)
