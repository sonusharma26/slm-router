import pytest

from inference_control.benchmarks.dynamic import run_sparse
from inference_control.drift.recovery import DriftRecovery
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner
from inference_control.probes.active import ActiveMeasurementLoop, DailyProbeBudget, ProbeTask
from tests.test_adaptive_release import ep, evidence, policy, request


def test_sparse_refresh_includes_frozen_control_and_total_spend():
    report = run_sparse(steps=8, seed=42, fractions=(0., 1.0), initial_requests=400)
    assert len(report["results"]) == 2
    rows = {row["strategy"]: row for row in report["results"]}

    assert set(rows) == {"frozen", "exhaustive"}
    frozen = rows["frozen"]
    assert frozen["budget_fraction"] == 0
    assert frozen["actual_probe_calls"] == 0
    assert frozen["actual_probe_cost"] == 0
    assert frozen["known_probe_cost_coverage"] == 1
    for row in rows.values():
        assert row["total_serving_cost"] == pytest.approx(row["mean_serving_cost"] * 8)
        assert row["total_known_cost"] == pytest.approx(
            row["total_serving_cost"] + row["actual_probe_cost"]
        )
        assert row["total_reserved_cost"] == pytest.approx(
            row["total_serving_cost"] + row["reserved_probe_cost"]
        )
        assert row["mean_total_known_cost_per_request"] == pytest.approx(
            row["total_known_cost"] / 8
        )


def test_probe_with_unknown_cost_keeps_reservation_and_reports_partial_spend():
    cmap = evidence()
    planner = Planner([ep()], capability_map=cmap)
    ledger = SQLiteLedger(":memory:")
    loop = ActiveMeasurementLoop(planner, DailyProbeBudget(ledger, max_dollars=1, max_calls=1),
                                 recovery=DriftRecovery(planner))
    template = next(iter(cmap.rows.values()))
    row = template.model_copy(update={"sample_id": "unknown-cost", "request": request(),
                                     "metrics": frozenset({"quality", "failure", "latency"})})
    result = loop.run([ProbeTask("unknown-cost", "a", request())], policy(), lambda task: row,
                      strategy="exhaustive")
    assert result["completed"] == ["unknown-cost"]
    assert result["budget"]["dollars_reserved"] == pytest.approx(.000192)
    assert result["budget"]["realized_known_cost"] == 0
    assert result["budget"]["known_cost_coverage"] == 0
