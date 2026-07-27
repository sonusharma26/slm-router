from pydantic import ValidationError
import pytest
from inference_control.contracts import (
    EndpointSnapshot,
    OutcomeRecord,
    PolicySpec,
    Prediction,
    RequestContext,
)
from inference_control.ledger import SQLiteLedger
from inference_control.planning import CapabilityEstimate, Planner
from inference_control.simulation import DynamicModelPool, EndpointBehavior


def endpoint(**updates):
    data = dict(
        endpoint_id="local",
        provider="local",
        upstream_model="m",
        revision="r1",
        region="local",
        context_window=1000,
        config_hash="h",
        price_version="p1",
        input_price_per_million=1,
        output_price_per_million=2,
        governance=frozenset({"local"}),
    )
    data.update(updates)
    return EndpointSnapshot(**data)


def policy(**updates):
    data = dict(
        policy_id="p",
        version="1",
        minimum_quality=0.8,
        max_expected_spend=0.01,
        max_absolute_spend=0.01,
        deadline_ms=100,
        data_boundary="local",
    )
    data.update(updates)
    return PolicySpec(**data)


def estimate(endpoint_id="local", quality=0.9, cost=0.001, latency=50):
    return CapabilityEstimate(
        endpoint_id,
        Prediction(mean=quality, lower=quality - 0.05, upper=quality + 0.05),
        Prediction(mean=cost, lower=cost, upper=cost),
        Prediction(mean=latency, lower=latency, upper=latency),
    )


def request():
    return RequestContext(
        request_id="q",
        application_id="a",
        tenant_policy_id="p",
        input_tokens=10,
        max_output_tokens=10,
    )


def test_contracts_are_frozen_and_proxy_cannot_promote():
    ep = endpoint()
    with pytest.raises(ValidationError):
        ep.revision = "r2"
    with pytest.raises(ValidationError):
        OutcomeRecord(
            decision_id="d",
            quality={"ok": 1},
            evaluator_type="proxy",
            evaluator_version="1",
            source_artifact="x",
            label_confidence=0.5,
            uncertainty=0.5,
            causal_scope="plan",
            training_eligible=True,
            promotion_eligible=True,
        )


def test_infeasible_privacy_and_budget_abstain():
    decision = Planner([endpoint()], [estimate()]).decide(request(), policy(data_boundary="eu"), 7)
    assert decision.selected_plan.plan_type == "abstain"
    assert "DATA_BOUNDARY_VIOLATION" in decision.rejected_alternatives[0].reason_codes
    tiny = Planner([endpoint(input_price_per_million=1_000_000)], [estimate()]).decide(
        request(), policy(), 7
    )
    assert "ABSOLUTE_SPEND_BOUND" in tiny.rejected_alternatives[0].reason_codes


def test_decision_replay_is_deterministic_except_metadata():
    planner = Planner([endpoint()], [estimate()], "map-1")
    a = planner.decide(request(), policy(), 7)
    b = planner.decide(request(), policy(), 7)
    assert a.selected_plan.steps == b.selected_plan.steps
    assert a.eligible_endpoints == b.eligible_endpoints
    assert a.rejected_alternatives == b.rejected_alternatives


def test_ledger_is_append_only_idempotent_and_hash_chained(tmp_path):
    ledger = SQLiteLedger(tmp_path / "events.db")
    a = ledger.append("decision", {"id": "d"}, actor="test", idempotency_key="one")
    b = ledger.append("decision", {"id": "changed"}, actor="test", idempotency_key="one")
    ledger.append("execution", {"id": "e"}, actor="test", idempotency_key="two")
    assert a == b and len(ledger.events()) == 2 and ledger.verify()
    with pytest.raises(Exception):
        ledger.connection.execute("UPDATE events SET payload='{}' WHERE sequence=1")
        ledger.connection.commit()


def test_simulator_repeats_and_supports_drift():
    pool = DynamicModelPool([EndpointBehavior("a", "r1", 1, 1, 10)], seed=2)
    assert pool.call("a", "q") == pool.call("a", "q")
    pool.inject("a", quality=0, revision="r2")
    changed = pool.call("a", "q")
    assert changed.revision == "r2" and changed.quality == 0
