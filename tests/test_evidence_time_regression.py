from dataclasses import replace
from datetime import UTC, datetime, timedelta
from importlib import import_module

import pytest

from inference_control.api.app import ControlPlane, DecideRequest
from inference_control.contracts import ExecutionRecord
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner
from inference_control.util import digest
from tests.test_adaptive_release import ep, evidence, outcome, policy, request


class Clock(datetime):
    at = datetime(2026, 10, 7, 10, tzinfo=UTC)

    @classmethod
    def now(cls, tz=None):
        return cls.at if tz else cls.at.replace(tzinfo=None)


@pytest.fixture
def clock(monkeypatch):
    Clock.at = datetime(2026, 10, 7, 10, tzinfo=UTC)
    for module in (
        "inference_control.ledger.sqlite",
        "inference_control.planning.planner",
        "inference_control.api.app",
    ):
        monkeypatch.setattr(import_module(module), "datetime", Clock)
    return Clock


def calibration_request():
    request_id = next(f"delayed:{i}" for i in range(100)
                      if int(digest(f"delayed:{i}")[:8], 16) % 5 == 0)
    return request(request_id=request_id)


def assert_no_fresh_calibration(control, at):
    estimate = control.planner.capability_map.estimate(
        request(), "a", (ep(),), at=at, min_samples=10)
    assert estimate.lineage.calibration_size == 0
    assert "CALIBRATION_COVERAGE" in estimate.lineage.reasons
    decision = control.planner.decide(request(), policy(minimum_quality=.65), at=at)
    assert decision.selected_plan.plan_type == "abstain"
    assert decision.certificate_status == "uncertified"


def test_label_after_drift_keeps_claim_time_and_cannot_refill_recovery(tmp_path, clock):
    path = tmp_path / "delayed.db"
    started_at = clock.at

    class DelayedExecution:
        adapter = None

        def execute(self, decision, **kwargs):
            clock.at = started_at + timedelta(seconds=2)
            control.drift.invalidate("a", ("quality",), slice_id="code/simple",
                                     reason="fixture_shift", at=clock.at)
            clock.at += timedelta(seconds=1)
            return ExecutionRecord(decision_id=decision.decision_id, state="completed",
                                   created_at=clock.at, attempted_calls=1,
                                   accounting_complete=False, total_latency_ms=20)

    control = ControlPlane(Planner([ep()], capability_map=evidence(at=started_at - timedelta(seconds=1))),
                           SQLiteLedger(path), DelayedExecution())
    control.drift.set_reference("a", "quality", "code/simple", [1.] * control.drift.window)
    decision = control.decide(DecideRequest(request=calibration_request(), policy=policy(minimum_quality=.65)))
    assert decision.certificate_status == "current"
    execution = control.execute(decision.decision_id)
    claim_event = control.ledger.events("execution_started")[0]
    assert datetime.fromisoformat(claim_event.occurred_at) < execution.created_at
    clock.at = started_at + timedelta(seconds=20)
    fresh = next(row for row in control.planner.capability_map.rows.values() if row.split == "train")
    control.planner.capability_map.add(fresh.model_copy(update={
        "sample_id": "fresh-training", "request": request(request_id="fresh-training"),
        "observed_at": clock.at}))
    qualified = control.add_outcome(outcome(decision, execution_id=execution.execution_id, evaluated_at=clock.at))
    row = control.planner.capability_map.rows[f"outcome:{qualified.outcome_id}"]
    assert_no_fresh_calibration(control, clock.at)
    assert row.observed_at == datetime.fromisoformat(claim_event.occurred_at)
    assert control.state.claims[decision.decision_id]["started_at"] == row.observed_at
    assert row.metrics == frozenset({"quality", "failure"})
    assert qualified.training_eligible and qualified.promotion_eligible
    assert not control.drift.recent[("a", "quality", "code/simple")]
    assert control.planner.certificates.current(decision.certificate_id, clock.at) is None
    control.ledger.close()
    restored = ControlPlane(Planner([]), SQLiteLedger(path))
    assert_no_fresh_calibration(restored, clock.at)
    assert restored.planner.capability_map.rows[row.sample_id] == row
    assert restored.state.outcomes[qualified.outcome_id] == qualified
    assert not restored.drift.recent[("a", "quality", "code/simple")]
    assert restored.ledger.verify()
    restored.ledger.close()


def test_restart_corrects_checkpoint_time_and_revokes_dependent_certificate(tmp_path, clock):
    path = tmp_path / "legacy-time.db"
    control = ControlPlane(Planner([ep()], capability_map=evidence(at=clock.at - timedelta(seconds=1))),
                           SQLiteLedger(path))
    decision = control.decide(DecideRequest(request=calibration_request(), policy=policy(minimum_quality=.65)))
    claim = control.state.write("execution_started", {"decision_id": decision.decision_id, "claim": "durable"})
    started_at = datetime.fromisoformat(claim.occurred_at)
    clock.at += timedelta(seconds=2)
    control.drift.invalidate("a", ("quality",), slice_id="code/simple", reason="fixture_shift", at=clock.at)
    clock.at += timedelta(seconds=10)
    execution = ExecutionRecord(decision_id=decision.decision_id, state="completed", attempted_calls=1,
                                created_at=clock.at, total_latency_ms=20)
    control.state.write("execution", execution.model_dump(mode="json"))
    qualified = outcome(decision, execution_id=execution.execution_id, evaluated_at=clock.at)
    control.state.write("outcome", qualified.model_dump(mode="json"))
    template = next(row for row in control.planner.capability_map.rows.values() if row.split == "train")
    control.planner.capability_map.add(template.model_copy(update={
        "sample_id": "fresh-training", "request": request(request_id="fresh-training"), "observed_at": clock.at}))
    legacy_row = template.model_copy(update={
        "sample_id": f"outcome:{qualified.outcome_id}", "request": decision.request,
        "split": "calibration", "observed_at": qualified.evaluated_at, "outcome_id": qualified.outcome_id})
    control.planner.capability_map.add(legacy_row)
    original_certificate = control.planner.certificates._versions[decision.certificate_id][-1]
    dependent = replace(original_certificate, certificate_id="legacy-dependent", valid_from=clock.at,
                        capability_map_version=control.planner.map_version,
                        invalidated_reason=None, drift_state="clear")
    control.planner.certificates.issue(dependent)
    control.state.store_planner(control.planner)
    old_version = control.planner.map_version
    events = control.ledger.events()
    control.ledger.close()
    restored = ControlPlane(Planner([]), SQLiteLedger(path))
    corrected = restored.planner.capability_map.rows[legacy_row.sample_id]
    assert corrected.observed_at == started_at
    assert corrected.model_dump(exclude={"observed_at"}) == legacy_row.model_dump(exclude={"observed_at"})
    assert restored.planner.map_version != old_version
    assert restored.state.latest_map == restored.planner.map_version
    assert restored.planner.certificates.current(dependent.certificate_id, clock.at) is None
    assert_no_fresh_calibration(restored, clock.at)
    assert restored.ledger.events()[:len(events)] == events
    assert restored.state.outcomes[qualified.outcome_id] == qualified
    assert restored.ledger.verify()
    count = len(restored.ledger.events())
    version = restored.planner.map_version
    restored.ledger.close()
    again = ControlPlane(Planner([]), SQLiteLedger(path))
    assert again.planner.map_version == version
    assert len(again.ledger.events()) == count
    again.ledger.close()


def test_missing_durable_claim_skips_learning_and_quarantines_checkpoint(tmp_path, clock):
    path = tmp_path / "no-claim.db"
    control = ControlPlane(Planner([ep()], capability_map=evidence(at=clock.at - timedelta(seconds=1))),
                           SQLiteLedger(path))
    decision = control.decide(DecideRequest(request=calibration_request(), policy=policy(minimum_quality=.65)))
    execution = ExecutionRecord(decision_id=decision.decision_id, state="completed", attempted_calls=1,
                                created_at=clock.at, total_latency_ms=20)
    control.state.write("execution", execution.model_dump(mode="json"))
    qualified = control.add_outcome(outcome(decision, execution_id=execution.execution_id, evaluated_at=clock.at))
    sample_id = f"outcome:{qualified.outcome_id}"
    assert qualified.training_eligible and qualified.promotion_eligible
    assert sample_id not in control.planner.capability_map.rows
    template = next(row for row in control.planner.capability_map.rows.values() if row.split == "calibration")
    legacy_row = template.model_copy(update={
        "sample_id": sample_id, "request": decision.request, "outcome_id": qualified.outcome_id,
        "observed_at": qualified.evaluated_at})
    control.planner.capability_map.add(legacy_row)
    control.state.store_planner(control.planner)
    version = control.planner.map_version
    events = control.ledger.events()
    control.ledger.close()
    restored = ControlPlane(Planner([]), SQLiteLedger(path))
    quarantined = restored.planner.capability_map.rows[sample_id]
    assert quarantined.disputed and not quarantined.trusted
    assert quarantined.metrics == legacy_row.metrics and quarantined.split == legacy_row.split
    assert restored.planner.map_version != version
    assert restored.planner.certificates.current(decision.certificate_id, clock.at) is None
    assert restored.state.outcomes[qualified.outcome_id] == qualified
    assert restored.ledger.events()[:len(events)] == events
    assert restored.ledger.verify()
    restored.ledger.close()
