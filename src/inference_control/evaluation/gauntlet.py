"""Dataset-independent Dynamic-Pool Gauntlet definitions (V2-702..704)."""

from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from inference_control.contracts import PolicySpec


@dataclass(frozen=True)
class GauntletScenario:
    scenario_id: str
    version: str
    change_type: str
    at_step: int
    parameters: dict[str, object]


SCENARIOS = (
    GauntletScenario("price-change", "1", "price", 100, {"multiplier": 2}),
    GauntletScenario("model-replacement", "1", "configuration", 100, {"revision": "replacement"}),
    GauntletScenario("endpoint-loss", "1", "availability", 100, {"available": False}),
    GauntletScenario("latency-shift", "1", "latency", 100, {"multiplier": 3}),
    GauntletScenario("quality-regression", "1", "quality", 100, {"delta": -0.2}),
    GauntletScenario("workload-shift", "1", "workload", 100, {"slice": "shifted"}),
    GauntletScenario("reward-delay", "1", "outcome", 100, {"delay_steps": 50}),
    GauntletScenario("evaluator-corruption", "1", "evaluator", 100, {"bias": 0.2}),
)


def policy_profiles() -> dict[str, PolicySpec]:
    common = dict(
        version="1",
        minimum_quality=0.8,
        quality_risk=0.05,
        cost_risk=0.05,
        latency_risk=0.05,
        max_calls=2,
    )
    return {
        "cost-capped": PolicySpec(
            policy_id="cost-capped",
            max_expected_spend=0.01,
            max_absolute_spend=0.02,
            deadline_ms=5000,
            **common,
        ),
        "latency-critical": PolicySpec(
            policy_id="latency-critical",
            max_expected_spend=0.1,
            max_absolute_spend=0.2,
            deadline_ms=300,
            **common,
        ),
        "privacy-restricted": PolicySpec(
            policy_id="privacy-restricted",
            max_expected_spend=0.1,
            max_absolute_spend=0.2,
            deadline_ms=5000,
            data_boundary="local",
            allowed_providers=frozenset({"local"}),
            **common,
        ),
    }


@dataclass(frozen=True)
class Preregistration:
    primary_hypotheses: tuple[str, ...]
    metrics: tuple[str, ...]
    exclusions: tuple[str, ...]
    thresholds: dict[str, float]
    frozen_hash: str

    @classmethod
    def freeze(
        cls,
        hypotheses: tuple[str, ...],
        metrics: tuple[str, ...],
        exclusions: tuple[str, ...],
        thresholds: dict[str, float],
    ):
        payload = {
            "hypotheses": hypotheses,
            "metrics": metrics,
            "exclusions": exclusions,
            "thresholds": thresholds,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return cls(hypotheses, metrics, exclusions, thresholds, digest)
