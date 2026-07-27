from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FaultScenario:
    name: str
    target: str
    mutation: dict[str, object]
    expected_control: str


FAULT_SCENARIOS = (
    FaultScenario(
        "silent_quality_regression", "endpoint", {"quality": -0.2}, "invalidate_and_probe"
    ),
    FaultScenario("price_change", "price", {"multiplier": 2}, "replan"),
    FaultScenario("endpoint_loss", "endpoint", {"available": False}, "remove_infeasible"),
    FaultScenario("latency_spike", "endpoint", {"latency_multiplier": 3}, "invalidate_and_probe"),
    FaultScenario("bad_evaluator", "evaluator", {"bias": 0.3}, "disable_reward_stream"),
    FaultScenario("missing_labels", "outcomes", {"drop_rate": 1}, "block_promotion"),
    FaultScenario("corrupted_policy_artifact", "policy", {"hash_valid": False}, "reject_artifact"),
)
