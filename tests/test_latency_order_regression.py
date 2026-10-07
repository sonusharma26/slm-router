from datetime import datetime, timezone

import pytest

from inference_control.api.app import ControlPlane, DecideRequest
from inference_control.contracts import ExecutionRecord
from inference_control.drift.recovery import DriftRecovery
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner
from tests.test_adaptive_release import ep, evidence, policy, request


class MeasuredExecution:
    adapter = None

    def __init__(self, latency):
        self.latency = latency

    def execute(self, decision, **kwargs):
        return ExecutionRecord(decision_id=decision.decision_id, attempted_calls=1,
            state="failed" if self.latency > 50 else "completed", total_latency_ms=self.latency,
            accounting_complete=False)


@pytest.mark.parametrize("first,second,should_trip", [(80, 20, False), (20, 80, True)])
def test_restart_preserves_durable_terminal_order_after_interrupted_monitor(tmp_path, monkeypatch, first, second, should_trip):
    path = tmp_path / "order.db"
    control = ControlPlane(Planner([ep()], capability_map=evidence()), SQLiteLedger(path), MeasuredExecution(first))
    p = policy(deadline_ms=50)
    original = control.drift.observe_execution

    def interrupted(*args, **kwargs):
        raise RuntimeError("monitor interrupted")

    a = control.decide(DecideRequest(request=request(request_id="a"), policy=p))
    monkeypatch.setattr(control.drift, "observe_execution", interrupted)
    with pytest.raises(RuntimeError, match="monitor interrupted"):
        control.execute(a.decision_id)
    monkeypatch.setattr(control.drift, "observe_execution", original)
    control.executor = MeasuredExecution(second)
    b = control.decide(DecideRequest(request=request(request_id="b"), policy=p))
    control.execute(b.decision_id)
    control.ledger.close()

    restored = ControlPlane(Planner([]), SQLiteLedger(path), MeasuredExecution(80))
    c = restored.decide(DecideRequest(request=request(request_id="c"), policy=p))
    restored.execute(c.decision_id)
    assert bool(restored.drift.pending) is should_trip
    assert len(restored.state.latency_receipts) == 3
    restored.ledger.close()


def test_recovery_requires_a_certificate_for_the_affected_slice():
    planner = Planner([ep()], capability_map=evidence())
    recovery = DriftRecovery(planner)
    event = recovery.invalidate("a", ("latency",), slice_id="other", reason="affected other slice")
    with pytest.raises(ValueError, match="slice"):
        recovery.acknowledge_recovered(event["event_id"], request(), policy())
    assert event["event_id"] in recovery.pending


def legacy_history(path, *, fully_receipted=False):
    endpoints = [ep(), ep("b", 2.)]
    cmap = evidence(endpoints)
    cmap.add_many([row.model_copy(update={"sample_id":"other:"+row.sample_id,
        "request":row.request.model_copy(update={"traffic_slices":frozenset({"other"})})})
        for row in tuple(cmap.rows.values()) if row.target_id == "a"])
    control = ControlPlane(Planner(endpoints, capability_map=cmap), SQLiteLedger(path))
    p = policy(deadline_ms=50, allowed_endpoints=frozenset({"a"}))
    decisions = [control.decide(DecideRequest(request=request(request_id=name), policy=p)) for name in ("a", "b")]
    observations = []
    for decision, duration in zip(decisions, (20, 80)):
        claim = control.state.write("execution_started", {"decision_id":decision.decision_id,"claim":decision.decision_id},
            key=f"execute-once:{decision.decision_id}")
        record = MeasuredExecution(duration).execute(decision)
        control.state.write("execution", record.model_dump(mode="json"), key=f"execution-final:{decision.decision_id}")
        observations.append(control.drift._execution_observations(decision, record, p,
            observed_at=datetime.fromisoformat(claim.occurred_at))[0][0])
    # Write the old monitor's suffix directly, bypassing the chronological drain.
    control.drift.observe_latency(observations[1])
    if fully_receipted: control.drift.observe_latency(observations[0])
    control.state.store_planner(control.planner)
    return control, decisions, p


@pytest.mark.parametrize("fully_receipted", [False, True])
def test_legacy_order_repair_is_once_and_preserves_other_targets_and_quality(tmp_path, fully_receipted):
    path = tmp_path / "legacy.db"
    control, decisions, p = legacy_history(path, fully_receipted=fully_receipted)
    quality_rows = {key:row for key,row in control.planner.capability_map.rows.items()}
    control.ledger.close()
    restored = ControlPlane(Planner([]), SQLiteLedger(path))
    assert len(restored.ledger.events("latency_order_repair")) == 1
    assert len(restored.state.latency_receipts) == 2
    assert {event["reason"] for event in restored.drift.pending.values()} == {"latency_history_out_of_order"}
    assert all(counter["streak"] == 0 for counter in restored.drift.operational.values())
    assert not restored.drift.recent[("a", "latency", "code/simple")]
    assert restored.planner.capability_map.rows == quality_rows
    now = datetime.now(timezone.utc)
    assert all(restored.planner.capability_map._valid(row, "quality", now) for row in quality_rows.values())
    assert restored.planner.certificates.current(decisions[1].certificate_id) is None
    assert restored.decide(DecideRequest(request=request(traffic_slices=frozenset({"other"})), policy=p)).certificate_status == "current"
    other_policy = policy(version="2", deadline_ms=50, allowed_endpoints=frozenset({"b"}))
    assert restored.decide(DecideRequest(request=request(), policy=other_policy)).certificate_status == "current"
    sequence = restored.state.sequence
    restored.ledger.close()
    replayed = ControlPlane(Planner([]), SQLiteLedger(path))
    assert replayed.state.sequence == sequence
    assert len(replayed.ledger.events("latency_order_repair")) == 1
    assert len(replayed.state.latency_receipts) == 2
    replayed.ledger.close()


def test_legacy_order_repair_rolls_back_effects_and_receipts_together(tmp_path, monkeypatch):
    control, decisions, _ = legacy_history(tmp_path / "rollback.db")
    before = (control.state.sequence, control.planner.map_version, dict(control.drift.operational))
    original = control.ledger.append_batch

    def interrupted(entries, **kwargs):
        if any(kind == "latency_order_repair" for kind, _, _ in entries):
            raise RuntimeError("repair interrupted")
        return original(entries, **kwargs)

    monkeypatch.setattr(control.ledger, "append_batch", interrupted)
    with pytest.raises(RuntimeError, match="repair interrupted"):
        control.drift.reconcile_latency()
    assert (control.state.sequence, control.planner.map_version, control.drift.operational) == before
    assert len(control.state.latency_receipts) == 1
    assert not control.state.drifts and not control.drift.pending
    assert control.planner.certificates.current(decisions[1].certificate_id)
    assert not control.ledger.events("latency_order_repair")
    monkeypatch.setattr(control.ledger, "append_batch", original)
    control.drift.reconcile_latency()
    assert len(control.state.latency_receipts) == 2
    assert len(control.ledger.events("latency_order_repair")) == 1
    sequence = control.state.sequence
    control.drift.reconcile_latency()
    assert control.state.sequence == sequence
    control.ledger.close()
