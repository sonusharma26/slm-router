"""Minimum-evidence drift detection and impact mapping (V2-600..603)."""

from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum


class DriftType(StrEnum):
    PRICE = "price"
    CONFIGURATION = "configuration"
    AVAILABILITY = "availability"
    WORKLOAD = "workload"
    LATENCY = "latency"
    QUALITY = "quality"
    EVALUATOR = "evaluator"


@dataclass(frozen=True)
class DriftEvent:
    drift_type: DriftType
    target_id: str
    actionable: bool
    evidence_count: int
    magnitude: float
    affected_estimates: tuple[str, ...]
    invalidates_certificates: bool


IMPACTS = {
    DriftType.PRICE: (("cost",), True),
    DriftType.CONFIGURATION: (("quality", "cost", "latency"), True),
    DriftType.AVAILABILITY: (("availability",), True),
    DriftType.WORKLOAD: (("quality", "cost", "latency"), True),
    DriftType.LATENCY: (("latency",), True),
    DriftType.QUALITY: (("quality",), True),
    DriftType.EVALUATOR: (("quality",), True),
}


def deterministic_change(
    kind: DriftType, target_id: str, before: object, after: object
) -> DriftEvent:
    if kind not in {DriftType.PRICE, DriftType.CONFIGURATION, DriftType.AVAILABILITY}:
        raise ValueError("statistical drift requires monitor")
    estimates, invalidates = IMPACTS[kind]
    return DriftEvent(
        kind,
        target_id,
        before != after,
        1,
        float(before != after),
        estimates,
        invalidates and before != after,
    )


class StatisticalMonitor:
    def __init__(self, kind: DriftType, threshold: float, minimum_evidence: int):
        self.kind = kind
        self.threshold = threshold
        self.minimum_evidence = minimum_evidence

    def evaluate(self, target_id: str, reference: list[float], current: list[float]) -> DriftEvent:
        n = min(len(reference), len(current))
        magnitude = (
            abs(sum(reference) / len(reference) - sum(current) / len(current))
            if reference and current
            else 0
        )
        actionable = n >= self.minimum_evidence and magnitude >= self.threshold
        estimates, invalidates = IMPACTS[self.kind]
        return DriftEvent(
            self.kind, target_id, actionable, n, magnitude, estimates, invalidates and actionable
        )


def diagnostic_targets(event: DriftEvent) -> tuple[tuple[str, str], ...]:
    """Return bounded probe targets only for actionable statistical drift."""
    if not event.actionable:
        return ()
    return tuple((event.target_id, metric) for metric in event.affected_estimates)
