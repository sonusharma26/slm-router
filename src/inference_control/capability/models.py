"""Endpoint/plan-aware capability estimates and calibration lineage (V2-300..302)."""

from __future__ import annotations
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class CalibrationLineage:
    calibration_id: str
    dataset_hash: str
    artifact_hash: str
    feature_version: str
    fitted_at: str


@dataclass(frozen=True)
class DistributionEstimate:
    mean: float
    lower: float
    upper: float
    sample_size: int
    lineage: CalibrationLineage


@dataclass(frozen=True)
class Observation:
    target_id: str
    plan_type: str
    features: tuple[float, ...]
    quality: float
    cost: float
    latency_ms: float


class EmpiricalCapabilityModel:
    """Small, auditable target-aware estimator with explicit missing coverage."""

    def __init__(self, lineage: CalibrationLineage):
        self.lineage = lineage
        self._rows: dict[tuple[str, str], list[Observation]] = {}

    def fit(self, observations: list[Observation]) -> None:
        self._rows = {}
        for row in observations:
            self._rows.setdefault((row.target_id, row.plan_type), []).append(row)

    def estimate(self, target_id: str, plan_type: str, metric: str) -> DistributionEstimate | None:
        rows = self._rows.get((target_id, plan_type), [])
        if not rows:
            return None
        values = sorted(float(getattr(r, metric)) for r in rows)
        n = len(values)
        radius = min(0.5, 1 / (n**0.5))
        center = mean(values)
        return DistributionEstimate(
            center, max(0, center - radius), center + radius, n, self.lineage
        )


class QuantileCapabilityModel(EmpiricalCapabilityModel):
    def estimate_quantiles(
        self, target_id: str, plan_type: str, metric: str, low: float = 0.05, high: float = 0.95
    ) -> DistributionEstimate | None:
        rows = self._rows.get((target_id, plan_type), [])
        if not rows:
            return None
        values = sorted(float(getattr(r, metric)) for r in rows)

        def pick(q: float) -> float:
            return values[min(len(values) - 1, int(q * (len(values) - 1)))]

        return DistributionEstimate(mean(values), pick(low), pick(high), len(values), self.lineage)
