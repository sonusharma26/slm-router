"""Direct full-information replay; no OPE is used (V2-205)."""

from __future__ import annotations
from dataclasses import dataclass
from inference_control.learning.baselines import ArmEstimate, Policy


@dataclass(frozen=True)
class MatrixRow:
    request_id: str
    endpoint_id: str
    quality: float
    cost: float
    latency_ms: float


@dataclass(frozen=True)
class ReplayResult:
    n: int
    mean_quality: float
    mean_cost: float
    mean_latency_ms: float
    selections: tuple[str, ...]


def replay_full_information(rows: list[MatrixRow], policy: Policy) -> ReplayResult:
    grouped: dict[str, list[MatrixRow]] = {}
    for row in rows:
        grouped.setdefault(row.request_id, []).append(row)
    totals = [0.0, 0.0, 0.0]
    selections = []
    for request_id in sorted(grouped):
        candidates = grouped[request_id]
        arms = [ArmEstimate(r.endpoint_id, r.quality, r.cost) for r in candidates]
        selected, _ = policy.choose(request_id, arms)
        actual = next(r for r in candidates if r.endpoint_id == selected)
        selections.append(selected)
        totals[0] += actual.quality
        totals[1] += actual.cost
        totals[2] += actual.latency_ms
    n = len(grouped)
    if not n:
        raise ValueError("full-information matrix is empty")
    return ReplayResult(n, *(v / n for v in totals), tuple(selections))
