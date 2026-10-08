"""Pooled evidence snapshots and configuration for learned, calibrated routing."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise

import numpy as np

from inference_control.routing.calibration import DEFAULT_FRACTIONS
from inference_control.routing.predictor import PredictorConfig, QualityPredictor

POOLED_METHOD = "pooled-ltt"

POOLED_ASSUMPTIONS = (
    "Quality scores are bounded in [0,1].",
    "Calibration requests are exchangeable with future traffic (same request mix and endpoint behavior).",
    "Each calibration request was scored on every endpoint the policy allowed; which requests were scored is unrelated to their quality.",
    "Forecasts and the threshold grid were fixed from training and validation evidence only; calibration outcomes are used only by the threshold test.",
    "Guarantee: with probability at least 1 - quality_risk, the mean quality over all requests certified by the selected rule (every routed request under the utility rule; requests clearing the threshold under the threshold rule) is at least minimum_quality (Hoeffding-Bentkus fixed-sequence test). It is a population guarantee, not a per-answer guarantee.",
    "Requests that are not certified are served by the uncertified fallback (when the policy allows it) and are excluded from the guarantee.",
    "Latency samples are IID per endpoint; p95 uses a binomial tolerance bound.",
)


@dataclass(frozen=True)
class PooledConfig:
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    thresholds: tuple[float, ...] | None = None
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS
    # Uncertified fallback leaves the endpoint with the best training mean only when another
    # forecast beats it by more than this margin (tuned on validation; 0 = plain argmax).
    fallback_margin: float = 0.
    # "utility": route each request to argmax(forecast - lambda * cost) with lambda calibrated so
    # that mean quality over ALL routed traffic meets the target. "threshold": certify the first
    # endpoint (in selection order) whose forecast clears a calibrated threshold.
    rule: str = "utility"

    def __post_init__(self):
        if self.rule not in {"utility", "threshold"}:
            raise ValueError("rule must be 'utility' or 'threshold'")
        if not 0 <= self.fallback_margin <= 1:
            raise ValueError("fallback_margin must lie in [0, 1]")
        if self.thresholds is not None:
            values = tuple(float(t) for t in self.thresholds)
            if not values or any(not math.isfinite(t) for t in values):
                raise ValueError("explicit thresholds must be finite")
            if any(b >= a for a, b in pairwise(values)):
                raise ValueError("explicit thresholds must be strictly decreasing (strict to loose)")
            object.__setattr__(self, "thresholds", values)
        if not self.fractions or any(not 0 < f <= 1 for f in self.fractions):
            raise ValueError("certified fractions must lie in (0, 1]")

    def identity(self) -> dict:
        value = {"predictor": self.predictor.identity(),
                 "thresholds": list(self.thresholds) if self.thresholds is not None else None,
                 "fractions": list(self.fractions)}
        if self.fallback_margin:
            value["fallback_margin"] = self.fallback_margin
        value["rule"] = self.rule
        return value

    @classmethod
    def from_identity(cls, value: dict | None) -> PooledConfig:
        if not value:
            return cls()
        thresholds = value.get("thresholds")
        return cls(predictor=PredictorConfig.from_identity(value.get("predictor")),
                   thresholds=tuple(thresholds) if thresholds is not None else None,
                   fractions=tuple(value.get("fractions", DEFAULT_FRACTIONS)),
                   fallback_margin=float(value.get("fallback_margin", 0.)),
                   rule=value.get("rule", "threshold"))  # maps saved before the utility rule


@dataclass(frozen=True)
class EndpointEvidence:
    training_outcomes: int
    latency_calibration: tuple[float, ...]
    latency_mean: float | None
    latency_max: float | None
    failures: int
    failure_observations: int
    output_tokens: float | None
    certification_eligible: bool


@dataclass
class PooledEvidence:
    """Training and calibration matrices for one endpoint set at one evidence state."""
    endpoint_ids: tuple[str, ...]
    feature_version: str
    fitted_at: datetime
    next_change: datetime | None
    train_requests: list
    train_quality: np.ndarray
    train_cost: np.ndarray
    calibration_requests: list
    calibration_quality: np.ndarray
    endpoints: dict[str, EndpointEvidence]
    measured_at: datetime | None
    trusted_until: datetime | None
    training_hash: str
    calibration_hash: str
    predictor: QualityPredictor | None
    identity: str
    _training_forecasts: tuple | None = None
    _forecast_cache: dict = field(default_factory=dict)
    _training_means: dict | None = None

    def training_mean(self, endpoint_id: str) -> float | None:
        if self._training_means is None:
            means = {}
            for index, endpoint in enumerate(self.endpoint_ids):
                column = self.train_quality[:, index] if len(self.train_requests) else np.zeros(0)
                observed = column[~np.isnan(column)]
                means[endpoint] = float(math.fsum(observed.tolist()) / len(observed)) if len(observed) else None
            self._training_means = means
        return self._training_means.get(endpoint_id)

    def usable(self, at: datetime) -> bool:
        return self.fitted_at <= at and (self.next_change is None or at < self.next_change)

    def column(self, endpoint_id: str) -> int:
        return self.endpoint_ids.index(endpoint_id)

    def forecast(self, requests) -> tuple[np.ndarray, np.ndarray]:
        """Quality and cost forecasts (requests x endpoints); NaN when no predictor exists."""
        n, m = len(requests), len(self.endpoint_ids)
        if self.predictor is None or not n:
            return np.full((n, m), np.nan), np.full((n, m), np.nan)
        if n == 1:
            request = requests[0]
            cached = self._forecast_cache.get(request.request_id)
            if cached is not None and cached[0] is request:
                return cached[1], cached[2]
        features = [r.query_features for r in requests]
        quality, cost = self.predictor.predict(features, [r.task_hint for r in requests])
        if n == 1:
            if len(self._forecast_cache) >= 4096:
                self._forecast_cache.clear()
            self._forecast_cache[requests[0].request_id] = (requests[0], quality, cost)
        return quality, cost

    def training_forecasts(self) -> tuple[np.ndarray, np.ndarray]:
        """Leave-one-out training forecasts, used as an outcome-independent threshold reference."""
        if self._training_forecasts is None:
            if self.predictor is None:
                n, m = len(self.train_requests), len(self.endpoint_ids)
                self._training_forecasts = (np.full((n, m), np.nan), np.full((n, m), np.nan))
            else:
                self._training_forecasts = self.predictor.predict_training()
        return self._training_forecasts
