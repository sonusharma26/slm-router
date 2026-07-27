"""OPE estimators with mandatory support/overlap/ESS diagnostics (V2-503/V2-504)."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BanditObservation:
    reward: float
    behavior_propensity: float
    target_probability: float
    reward_model: float = 0


@dataclass(frozen=True)
class OPEDiagnostics:
    n: int
    effective_sample_size: float
    minimum_propensity: float
    overlap: float
    passed: bool
    reason: str | None


@dataclass(frozen=True)
class OPEResult:
    estimator: str
    value: float
    diagnostics: OPEDiagnostics


def _weights(rows):
    return [r.target_probability / r.behavior_propensity for r in rows]


def diagnose(
    rows: list[BanditObservation], min_ess: float = 10, min_overlap: float = 0.9
) -> OPEDiagnostics:
    if not rows or any(r.behavior_propensity <= 0 for r in rows):
        return OPEDiagnostics(len(rows), 0, 0, 0, False, "NONZERO_SUPPORT_REQUIRED")
    w = _weights(rows)
    ess = sum(w) ** 2 / max(sum(x * x for x in w), 1e-12)
    overlap = sum(r.target_probability > 0 and r.behavior_propensity > 0 for r in rows) / len(rows)
    passed = ess >= min_ess and overlap >= min_overlap
    return OPEDiagnostics(
        len(rows),
        ess,
        min(r.behavior_propensity for r in rows),
        overlap,
        passed,
        None if passed else "INSUFFICIENT_OVERLAP_OR_ESS",
    )


def evaluate_ope(
    rows: list[BanditObservation],
    estimator: str,
    min_ess: float = 10,
    min_overlap: float = 0.9,
    switch_threshold: float = 10,
) -> OPEResult:
    diagnostics = diagnose(rows, min_ess, min_overlap)
    if not diagnostics.passed:
        raise ValueError(diagnostics.reason)
    weights = _weights(rows)
    if estimator == "ips":
        value = sum(w * r.reward for w, r in zip(weights, rows)) / len(rows)
    elif estimator == "snips":
        value = sum(w * r.reward for w, r in zip(weights, rows)) / sum(weights)
    elif estimator == "dr":
        value = sum(
            r.reward_model + w * (r.reward - r.reward_model) for w, r in zip(weights, rows)
        ) / len(rows)
    elif estimator == "switch":
        value = sum(
            (
                r.reward_model
                if w > switch_threshold
                else r.reward_model + w * (r.reward - r.reward_model)
            )
            for w, r in zip(weights, rows)
        ) / len(rows)
    else:
        raise ValueError("unknown OPE estimator")
    return OPEResult(estimator, value, diagnostics)
