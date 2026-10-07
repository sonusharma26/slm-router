from datetime import UTC, datetime, timedelta

import pytest

from inference_control.contracts import Prediction
from inference_control.drift.recovery import DriftRecovery
from inference_control.ledger import SQLiteLedger
from inference_control.planning import CapabilityEstimate, Planner
from inference_control.probes.active import ActiveMeasurementLoop, DailyProbeBudget, ProbeTask
from tests.test_adaptive_release import ep, evidence, policy, request


@pytest.mark.parametrize("objective", ["cost", "quality", "latency"])
def test_probe_impact_respects_serving_winner_and_tie_breaks(objective):
    at = datetime.now(UTC)
    endpoints = [ep("cheap", 1), ep("best", 10)]
    cmap = evidence(endpoints, n=200, at=at - timedelta(hours=13))
    for sample_id, row in tuple(cmap.rows.items()):
        cmap.rows[sample_id] = row.model_copy(update={
            "quality": .78 if row.target_id == "cheap" else 1.,
            "latency_ms": 30 if row.target_id == "cheap" else 10,
        })
    planner = Planner(endpoints, capability_map=cmap)
    p = policy(minimum_quality=.65, objective=objective, deadline_ms=1000,
               minimum_evidence_samples=8, require_certificate=False)
    served = planner.decide(request(), p, at=at)
    assert served.selected_plan.steps[0].endpoint_id == ("cheap" if objective == "cost" else "best")
    loop = ActiveMeasurementLoop(planner, DailyProbeBudget(SQLiteLedger(":memory:"), max_dollars=1, max_calls=2), recovery=DriftRecovery(planner))
    certificates = planner.certificates.export()
    ranked = {choice.task.endpoint_id: choice for choice in loop.rank(
        [ProbeTask("cheap-probe", "cheap", request()), ProbeTask("best-probe", "best", request())], p, at=at
    )}
    assert planner.certificates.export() == certificates
    if objective == "quality":
        assert ranked["cheap"].could_change_decision is False
    elif objective == "latency":
        assert ranked["best"].could_change_decision is True
    else:
        assert ranked["best"].could_change_decision is False


def test_probe_impact_never_bypasses_incomplete_search_or_expected_spend():
    endpoints = [ep("a"), ep("b")]
    cmap = evidence(endpoints, n=100, at=datetime.now(UTC) - timedelta(hours=13))
    planner = Planner(endpoints, capability_map=cmap)
    loop = ActiveMeasurementLoop(planner, DailyProbeBudget(SQLiteLedger(":memory:"), max_dollars=1, max_calls=2), recovery=DriftRecovery(planner))
    task = ProbeTask("probe", "a", request())
    truncated = policy(max_plan_candidates=1)
    assert planner.decide(request(), truncated).planning_complete is False
    assert loop.rank([task], truncated)[0].could_change_decision is False
    too_expensive = policy(max_expected_spend=.00001, require_certificate=False)
    assert planner.decide(request(), too_expensive).selected_plan.plan_type == "abstain"
    assert loop.rank([task], too_expensive)[0].could_change_decision is False


@pytest.mark.parametrize("objective", ["cost", "quality", "latency"])
def test_probe_projections_use_serving_secondary_tie_breaks(objective, monkeypatch):
    endpoints = [ep("a"), ep("b")]
    a = CapabilityEstimate("a", Prediction(mean=.9, lower=.9, upper=.9),
                           Prediction(mean=.0001, lower=0, upper=.0001),
                           Prediction(mean=20, lower=20, upper=20))
    b = CapabilityEstimate("b", Prediction(mean=.7, lower=.7, upper=.9),
                           Prediction(mean=.0001 if objective == "cost" else .00008, lower=0, upper=.0001),
                           Prediction(mean=30, lower=0 if objective == "cost" else 20, upper=40))
    planner = Planner(endpoints, capability_map=evidence(endpoints))
    monkeypatch.setattr(planner, "estimate_plan", lambda request, policy, plan, at, comparisons:
                        {"a": a, "b": b}[plan.evidence_key])
    p = policy(minimum_quality=.65, deadline_ms=30, objective=objective, require_certificate=False)
    assert planner.decide(request(), p).selected_plan.steps[0].endpoint_id == "a"
    loop = ActiveMeasurementLoop(planner, DailyProbeBudget(SQLiteLedger(":memory:"), max_dollars=1, max_calls=1), recovery=DriftRecovery(planner))
    assert loop.rank([ProbeTask("tie", "b", request())], p)[0].could_change_decision is True


def test_probe_projection_preserves_failure_gate_in_uncertified_fallback(monkeypatch):
    endpoints = [ep("a"), ep("b")]
    a = CapabilityEstimate("a", Prediction(mean=.8, lower=.8, upper=.8),
                           Prediction(mean=.0001, lower=0, upper=.0001),
                           Prediction(mean=20, lower=0, upper=20), Prediction(mean=0, lower=0, upper=.05))
    b = CapabilityEstimate("b", Prediction(mean=.7, lower=.7, upper=1),
                           Prediction(mean=.00008, lower=0, upper=.0001),
                           Prediction(mean=20, lower=0, upper=20), Prediction(mean=.4, lower=.3, upper=.5))
    planner = Planner(endpoints, capability_map=evidence(endpoints))
    monkeypatch.setattr(planner, "estimate_plan", lambda request, policy, plan, at, comparisons:
                        {"a": a, "b": b}[plan.evidence_key])
    p = policy(minimum_quality=.99, objective="quality", require_certificate=False,
               maximum_failure_probability=.1, infeasible_behavior="least_shortfall_uncertified")
    assert planner.decide(request(), p).selected_plan.steps[0].endpoint_id == "a"
    loop = ActiveMeasurementLoop(planner, DailyProbeBudget(SQLiteLedger(":memory:"), max_dollars=1, max_calls=1), recovery=DriftRecovery(planner))
    assert loop.rank([ProbeTask("unsafe", "b", request())], p)[0].could_change_decision is False
