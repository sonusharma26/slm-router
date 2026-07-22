"""ml_core: machine-learning components for the SLM router."""
from __future__ import annotations

from slm_router.ml_core.features import Embedder, DifficultyFeaturizer
from slm_router.ml_core.difficulty import DifficultyEstimator
from slm_router.ml_core.confidence import (
    TemperatureScaler,
    ConformalCalibrator,
    EnsembleUncertainty,
    ConfidencePredictor,
)
from slm_router.ml_core.routing import (
    RoutingPolicy,
    LinUCBPolicy,
    ThompsonSamplingPolicy,
    OfflineRLPolicy,
    CascadePolicy,
    compute_reward,
)

__all__ = [
    "Embedder",
    "DifficultyFeaturizer",
    "DifficultyEstimator",
    "TemperatureScaler",
    "ConformalCalibrator",
    "EnsembleUncertainty",
    "ConfidencePredictor",
    "RoutingPolicy",
    "LinUCBPolicy",
    "ThompsonSamplingPolicy",
    "OfflineRLPolicy",
    "CascadePolicy",
    "compute_reward",
]
