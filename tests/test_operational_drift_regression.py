from datetime import UTC, datetime, timedelta

import httpx
import pytest

from inference_control.adapters.providers import OpenAICompatibleAdapter, ProviderTimeout
from inference_control.adapters.runtime import RuntimeProviderAdapter
from inference_control.api.app import ControlPlane, DecideRequest
from inference_control.capability.conditional import EvidenceObservation
from inference_control.contracts import Call, ExecutionRecord
from inference_control.drift.recovery import DriftRecovery, EndpointLatency
from inference_control.execution import Executor
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner
from inference_control.probes.active import ActiveMeasurementLoop, DailyProbeBudget, ProbeTask
from inference_control.util import digest
from tests.test_adaptive_release import ep, evidence, policy, request


class RecordedExecution:
    adapter = None

    def __init__(self, accounting_complete):
        self.accounting_complete = accounting_complete

    def execute(self, decision, **kwargs):
        return ExecutionRecord(decision_id=decision.decision_id, state="failed", attempted_calls=1,
                               total_latency_ms=80, accounting_complete=self.accounting_complete,
                               constraint_violations=("REALIZED_DEADLINE_EXCEEDED",))


@pytest.mark.parametrize("accounting_complete", [False, True])
@pytest.mark.parametrize("slices", [frozenset({"code/simple"}), frozenset()])
def test_repeated_deadline_failures_revoke_certificates_independently_of_billing(accounting_complete, slices):
    cmap = evidence()
    if not slices:
        from inference_control.capability.conditional import ConditionalCapabilityMap
        empty = ConditionalCapabilityMap(k=256,min_samples=10)
        empty.add_many([row.model_copy(update={"request":row.request.model_copy(update={"traffic_slices":slices})})
                        for row in cmap.rows.values()])
        cmap = empty
    planner = Planner([ep()], capability_map=cmap)
    control = ControlPlane(planner, SQLiteLedger(":memory:"), RecordedExecution(accounting_complete))
    p = policy(deadline_ms=50)
    decisions = []
    for i in range(2):
        decision = control.decide(DecideRequest(request=request(request_id=f"overrun-{i}",traffic_slices=slices), policy=p))
        assert decision.certificate_status == "current"
        decisions.append(decision)
        control.execute(decision.decision_id)
        if i == 0:
            assert not control.drift.pending
    assert len(control.drift.pending) == 1
    assert planner.certificates.current(decisions[0].certificate_id) is None
    assert planner.certificates.current(decisions[1].certificate_id) is None
    assert control.decide(DecideRequest(request=request(traffic_slices=slices), policy=p)).selected_plan.plan_type == "abstain"


def measured_row(sample_id="probe", latency=80):
    return EvidenceObservation(sample_id=sample_id,request=request(request_id=sample_id),target_id="a",
        endpoint_ids=("a",),endpoint_revisions=(ep().capability_revision,),quality=.99,cost=.0001,
        latency_ms=latency,split="train",evaluator_type="deterministic",evaluator_version="unit-v1")


def loop_for(control, max_calls=4):
    return ActiveMeasurementLoop(control.planner,DailyProbeBudget(control.ledger,max_dollars=1,max_calls=max_calls),
                                 recovery=control.drift)


@pytest.mark.parametrize("terminal", ["execution", "probe"])
def test_interrupted_monitor_reconciles_once_and_shares_serving_probe_counter(tmp_path, monkeypatch, terminal):
    path = tmp_path / "restart.db"
    control = ControlPlane(Planner([ep()],capability_map=evidence()),SQLiteLedger(path),RecordedExecution(False))
    p = policy(deadline_ms=50)
    decision = control.decide(DecideRequest(request=request(),policy=p))
    def interrupted(*args, **kwargs): raise RuntimeError("monitor interrupted")
    if terminal == "execution":
        monkeypatch.setattr(control.drift,"observe_execution",interrupted)
        with pytest.raises(RuntimeError,match="monitor interrupted"): control.execute(decision.decision_id)
    else:
        control.execute(decision.decision_id)
        monkeypatch.setattr(control.drift,"observe_probe",interrupted)
        row = measured_row()
        with pytest.raises(RuntimeError,match="monitor interrupted"):
            loop_for(control).run([ProbeTask(row.sample_id,"a",row.request)],p,lambda task:row,strategy="exhaustive")
    control.ledger.close()
    restored = ControlPlane(Planner([ep()],capability_map=evidence()),SQLiteLedger(path),RecordedExecution(False))
    if terminal == "execution":
        assert not restored.drift.pending
        row = measured_row()
        loop_for(restored).run([ProbeTask(row.sample_id,"a",row.request)],p,lambda task:row,strategy="exhaustive")
    assert len(restored.drift.pending) == 1
    assert len(restored.state.latency_receipts) == 2
    assert len(restored.ledger.events("drift")) == 1
    assert restored.planner.certificates.current(decision.certificate_id) is None
    sequence = restored.state.sequence
    restored.ledger.close()
    replayed = ControlPlane(Planner([ep()],capability_map=evidence()),SQLiteLedger(path))
    assert replayed.state.sequence == sequence
    assert len(replayed.state.latency_receipts) == 2
    assert len(replayed.ledger.events("drift")) == 1
    assert replayed.decide(DecideRequest(request=request(),policy=p)).selected_plan.plan_type == "abstain"
    replayed.ledger.close()


def test_actual_transport_timeouts_trip_while_generic_errors_stay_neutral_and_reservations_remain():
    def timeout(req): raise httpx.ReadTimeout("offline timeout",request=req)
    provider = OpenAICompatibleAdapter("https://offline.invalid","fixture",httpx.Client(transport=httpx.MockTransport(timeout)))
    with pytest.raises(ProviderTimeout) as failure: provider.call("a",[],{})
    assert isinstance(failure.value.__cause__,httpx.ReadTimeout)
    planner = Planner([ep()],capability_map=evidence())
    bridge = RuntimeProviderAdapter(planner.endpoints,{"a":provider})
    control = ControlPlane(planner,SQLiteLedger(":memory:"),Executor(bridge))
    p = policy(deadline_ms=50)
    first = control.decide(DecideRequest(request=request(),policy=p))
    record = control.execute(first.decision_id,messages=[{"role":"user","content":"hi"}])
    assert record.timed_out_endpoint_ids == ("a",)
    assert not record.accounting_complete and record.reserved_spend == first.selected_plan.max_spend
    assert not control.drift.recent[("a","latency","code/simple")]
    class LocalGuard:
        def call(self,endpoint_id,config): raise TimeoutError("local timeout, not a transport outcome")
    control.executor = Executor(LocalGuard())
    neutral = control.decide(DecideRequest(request=request(request_id="neutral"),policy=p))
    neutral_record = control.execute(neutral.decision_id)
    assert neutral_record.timed_out_endpoint_ids == () and not control.drift.pending
    calls = [ProbeTask("generic","a",request(request_id="generic")),ProbeTask("timeout","a",request(request_id="timeout"))]
    def measure(task):
        if task.probe_id == "generic": raise TimeoutError("local guard, not transport")
        raise ProviderTimeout("normalized transport timeout")
    version = planner.map_version
    result = loop_for(control).run(calls,p,measure,strategy="exhaustive")
    assert len(control.drift.pending) == 1
    assert result["completed"] == [] and set(result["failed"]) == {"generic","timeout"}
    assert result["budget"]["calls_reserved"] == 2 and result["budget"]["realized_known_cost"] == 0
    assert result["budget"]["dollars_reserved"] > 0 and len(planner.capability_map.rows) == 200
    assert planner.map_version != version
    provider.client.close()


def test_probe_spanning_invalidation_keeps_measurement_start_and_cannot_recertify():
    control = ControlPlane(Planner([ep()],capability_map=evidence()),SQLiteLedger(":memory:"))
    p = policy(deadline_ms=50)
    sampled = datetime.now(UTC) - timedelta(seconds=1)
    task = ProbeTask("spanning","a",request(request_id="spanning"))
    def measure(task):
        control.drift.invalidate("a",("quality","latency"),reason="changed during probe",slice_id="code/simple")
        return measured_row(task.probe_id,latency=20)
    loop_for(control).run([task],p,measure,strategy="exhaustive",at=sampled)
    row = control.planner.capability_map.rows[task.probe_id]
    assert row.observed_at == sampled
    assert not control.planner.capability_map._valid(row,"latency",datetime.now(UTC))
    assert not control.drift.recent[("a","latency","code/simple")]
    assert control.decide(DecideRequest(request=request(),policy=p)).selected_plan.plan_type == "abstain"


def test_healthy_measurement_resets_counter_and_pre_cutoff_samples_cannot_refill_recovery():
    planner = Planner([ep()],capability_map=evidence())
    recovery = DriftRecovery(planner,window=4)
    p = policy(deadline_ms=50)
    sampled = datetime.now(UTC)
    def observe(source, duration, at):
        return recovery.observe_latency(EndpointLatency(source_id=source,endpoint_id="a",slice_id="code/simple",
            capability_revision=ep().capability_revision,policy_hash=digest(p),deadline_ms=50,
            observed_at=at,received_at=at,duration_ms=duration))
    observe("bad-1",80,sampled)
    observe("healthy",20,sampled+timedelta(seconds=1))
    observe("bad-2",80,sampled+timedelta(seconds=2))
    assert not recovery.pending
    observe("bad-3",80,sampled+timedelta(seconds=3))
    assert len(recovery.pending) == 1
    assert not recovery.recent[("a","latency","code/simple")]
    assert observe("late",20,sampled) == ()
    recovery.observe("a","quality","code/simple",.99,at=sampled+timedelta(seconds=4),observed_at=sampled)
    recovery.invalidate("a",("quality",),reason="quality cut",slice_id="code/simple",at=sampled+timedelta(seconds=4))
    recovery.observe("a","quality","code/simple",.99,at=sampled+timedelta(seconds=5),observed_at=sampled)
    assert not recovery.recent[("a","quality","code/simple")]
    assert observe("late-again",20,sampled) == ()
    assert not recovery.recent[("a","latency","code/simple")]


@pytest.mark.parametrize("kind", ["cascade", "verify_escalate", "parallel"])
def test_compound_failures_revoke_dependencies_without_fabricating_endpoint_statistics_and_fresh_evidence_recovers(kind):
    endpoints = [ep(),ep("b",2.)]
    p = policy(deadline_ms=50,permitted_plan_types=frozenset({kind}),max_calls=2,verifier="check",selector="first")
    planner = Planner(endpoints,capability_map=evidence(endpoints))
    plan = planner.generate(request(),p)[0][0]
    ids = tuple(step.endpoint_id for step in plan.steps if isinstance(step,Call))
    revisions = tuple(planner.endpoints[e].capability_revision for e in ids)
    joint = [row.model_copy(update={"sample_id":"joint:"+row.sample_id,"target_id":plan.evidence_key,
             "plan_type":kind,"endpoint_ids":ids,"endpoint_revisions":revisions})
             for row in evidence().rows.values()]
    planner.capability_map.add_many(joint)
    control = ControlPlane(planner,SQLiteLedger(":memory:"),RecordedExecution(False))
    decisions = []
    class PartialTimeoutExecution:
        adapter = None
        def execute(self,decision,**kwargs):
            return ExecutionRecord(decision_id=decision.decision_id,state="failed",attempted_calls=2,
                total_latency_ms=10,timed_out_endpoint_ids=(ids[-1],),provider_errors=("normalized timeout",),
                accounting_complete=False,reserved_spend=decision.selected_plan.max_spend)
    for i in range(2):
        if i == 1: control.executor = PartialTimeoutExecution()
        decision = control.decide(DecideRequest(request=request(request_id=f"compound-{i}"),policy=p))
        assert decision.selected_plan.plan_type == kind and decision.certificate_status == "current"
        decisions.append(decision)
        control.execute(decision.decision_id)
    assert len(control.drift.pending) == 2
    assert not any(values for key,values in control.drift.recent.items() if key[1] == "latency")
    assert all(event["details"]["scope"] == "compound" for event in control.drift.pending.values())
    assert all(planner.certificates.current(decision.certificate_id) is None for decision in decisions)
    assert control.decide(DecideRequest(request=request(),policy=p)).selected_plan.plan_type == "abstain"
    fresh = datetime.now(UTC)
    planner.capability_map.add_many([row.model_copy(update={"sample_id":"fresh:"+row.sample_id,"observed_at":fresh,
        "request":row.request.model_copy(update={"request_id":"fresh:"+row.request.request_id})}) for row in joint])
    for event_id in tuple(control.drift.pending):
        assert control.drift.acknowledge_recovered(event_id,request(),p).certificate_status == "current"
    assert not control.drift.pending


def test_monitor_batch_failure_leaves_no_receipt_or_partial_invalidation_and_retries_once(monkeypatch):
    control = ControlPlane(Planner([ep()],capability_map=evidence()),SQLiteLedger(":memory:"),RecordedExecution(False))
    p = policy(deadline_ms=50)
    first = control.decide(DecideRequest(request=request(request_id="first"),policy=p))
    control.execute(first.decision_id)
    second = control.decide(DecideRequest(request=request(request_id="second"),policy=p))
    version = control.planner.map_version
    original = control.ledger.append_batch
    def fail_monitor(entries, **kwargs):
        if any(kind == "latency_receipt" for kind,_,_ in entries): raise RuntimeError("batch interrupted")
        return original(entries,**kwargs)
    monkeypatch.setattr(control.ledger,"append_batch",fail_monitor)
    with pytest.raises(RuntimeError,match="batch interrupted"): control.execute(second.decision_id)
    assert len(control.state.latency_receipts) == 1 and not control.state.drifts and not control.drift.pending
    assert control.planner.map_version == version and control.planner.certificates.current(second.certificate_id)
    monkeypatch.setattr(control.ledger,"append_batch",original)
    record = control.state.execution_by_decision[second.decision_id]
    claim = next(e for e in control.ledger.events("execution_started") if e.payload["decision_id"] == second.decision_id)
    control.drift.observe_execution(second,record,p,observed_at=datetime.fromisoformat(claim.occurred_at))
    sequence = control.state.sequence
    assert len(control.drift.pending) == 1 and len(control.state.latency_receipts) == 2
    control.drift.observe_execution(second,record,p,observed_at=datetime.fromisoformat(claim.occurred_at))
    assert control.state.sequence == sequence
