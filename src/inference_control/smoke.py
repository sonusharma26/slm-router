"""Deterministic, no-network V2 smoke path (V2-004)."""

from inference_control.contracts import EndpointSnapshot, PolicySpec, Prediction, RequestContext
from inference_control.planning import CapabilityEstimate, Planner


def main() -> None:
    endpoint = EndpointSnapshot(
        endpoint_id="local",
        provider="local",
        upstream_model="fixture",
        revision="sha256:fixture",
        region="local",
        context_window=4096,
        config_hash="fixture",
        price_version="free",
        input_price_per_million=0,
        output_price_per_million=0,
        governance=frozenset({"local"}),
    )
    estimate = CapabilityEstimate(
        "local",
        Prediction(mean=0.95, lower=0.9, upper=0.99),
        Prediction(mean=0, lower=0, upper=0),
        Prediction(mean=10, lower=8, upper=12),
    )
    request = RequestContext(
        request_id="smoke",
        application_id="smoke",
        tenant_policy_id="smoke",
        input_tokens=10,
        max_output_tokens=10,
    )
    policy = PolicySpec(
        policy_id="smoke",
        version="1",
        minimum_quality=0.8,
        max_expected_spend=0,
        max_absolute_spend=0,
        deadline_ms=100,
        data_boundary="local",
    )
    decision = Planner([endpoint], [estimate]).decide(request, policy, seed=7)
    assert decision.selected_plan.plan_type == "direct" and decision.eligible_endpoints == (
        "local",
    )
    print("v2 smoke: PASS (network calls: 0)")


if __name__ == "__main__":
    main()
