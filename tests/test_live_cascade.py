"""Live answer-agreement cascade: planner calibration, executor, contracts, answer evidence, control plane."""
import json
import math
import threading
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from inference_control.api.app import ControlPlane, DecideRequest
from inference_control.benchmarks.routers import SLMRouter, evaluate_live_cascade, realize_selection
from inference_control.benchmarks.runner import default_policy, evaluate_router, split_data
from inference_control.benchmarks.zoo import make_zoo, zoo_answers
from inference_control.capability.conditional import ConditionalCapabilityMap, EvidenceObservation, answer_key
from inference_control.contracts import (
    Call,
    DecisionRecord,
    EndpointSnapshot,
    ExecutionPlan,
    OutcomeRecord,
    PolicySpec,
    RequestContext,
    Verify,
)
from inference_control.execution import Executor, RuntimeResult
from inference_control.execution.agreement import answer_agreement, extract_final_answer
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner, agree_escalate
from inference_control.policies.compiler import compile_policy
from inference_control.routing import cascade, live_cascade
from inference_control.routing.pooled import PooledConfig

AT = datetime(2026, 1, 1, tzinfo=UTC)
AGREE = "answer_agreement"


# ----- fixtures ---------------------------------------------------------------------------------
def cascade_policy(base, **updates):
    return PolicySpec.model_validate({**base.model_dump(), "permitted_plan_types": ["direct", "agree_escalate"],
                                      "max_calls": 3, "verifier": AGREE, **updates})


SMALL_POOL = ("small-8b", "mid-32b", "large-70b", "frontier", "frontier-reason", "math-7b")


def fitted(n, seed, at, *, with_answers=True, rule="utility", pool=None):
    data = make_zoo(n, seed=seed)
    answers = zoo_answers(data)
    base = default_policy(minimum_quality=.8, missing_latency=True)
    endpoints = [e for e in data.endpoints if pool is None or e.endpoint_id in pool]
    router = SLMRouter(endpoints, min_samples=8, at=at, pooled=PooledConfig(rule=rule))
    scored = lambda queries, rows: (queries, [r for r in rows if pool is None or r.endpoint_id in pool])
    router.fit(*scored(*split_data(data, {"train", "calibration"})), answers=answers if with_answers else None)
    router.tune(*scored(*split_data(data, {"validation"})), base)
    return SimpleNamespace(data=data, answers=answers, router=router, base=base, endpoints=endpoints,
                           feature_version=data.queries[0].request.feature_version,
                           truth={(r.request_id, r.endpoint_id): r for r in data.outcomes})


@pytest.fixture(scope="module")
def zoo():
    return fitted(2400, 5, AT)


@pytest.fixture(scope="module")
def zoo_without_answers(zoo):
    """The same evidence and tuned predictor, without any stored answers."""
    router = SLMRouter(zoo.data.endpoints, min_samples=8, at=AT, pooled=zoo.router.map.pooled_config)
    router.fit(*split_data(zoo.data, {"train", "calibration"}))
    return SimpleNamespace(**{**vars(zoo), "router": router})


SCENARIOS = {
    "open": {"minimum_quality": .8},
    "two-calls": {"minimum_quality": .8, "max_calls": 2},
    "expected-spend": {"minimum_quality": .7, "max_expected_spend": .03},
    "deadline": {"minimum_quality": .7, "deadline_ms": 1500.},
}


def realized(zoo, decision):
    request_id = decision.request.request_id
    calls = [s.endpoint_id for s in decision.selected_plan.steps if isinstance(s, Call)]
    metadata = {}
    if decision.selected_plan.plan_type == "agree_escalate":
        metadata = {"cascade": {"first": calls[0], "checker": calls[1], "final": calls[-1]}}
    selection = SimpleNamespace(endpoint_id=calls[0], plan_type=decision.selected_plan.plan_type, metadata=metadata)
    return realize_selection(selection, lambda e: zoo.truth[(request_id, e)], zoo.answers)


# ----- planner: serving equals the calibration replay -------------------------------------------
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_serving_decisions_match_the_calibration_replay(zoo, scenario):
    planner = zoo.router.planner
    policy = cascade_policy(zoo.base, version=scenario, **SCENARIOS[scenario])
    evidence, live = planner.pooled_calibration(policy, at=AT, feature_version=zoo.feature_version)
    assert live.kind == "cascade" and live.config is not None and live.threshold is None
    requests = evidence.calibration_requests
    quality, cost = evidence.forecast(requests)
    reference = planner._cascade_reference(evidence, requests, policy, quality, cost)
    gate = planner._compound_gate(policy, np.array(live.disagreement))
    first, checker, final, check, raw = live_cascade.gated_plan(reference, live.config, gate)
    gated_off = int((raw & ~check).sum())
    assert check.any()
    assert gated_off == 0 if scenario == "open" else gated_off > 0 if scenario == "two-calls" else gated_off >= 0
    qualities, costs, escalations = [], [], 0
    for i, request in enumerate(requests):
        decision = planner.decide(request, policy, at=AT)
        assert decision.fallback == policy.infeasible_behavior.value and decision.certificate_status == "current"
        assert decision.estimated_quality.lower == policy.minimum_quality
        ids = reference.ids[i]
        calls = [s.endpoint_id for s in decision.selected_plan.steps if isinstance(s, Call)]
        note = decision.plan_metadata["cascade"]
        if check[i]:
            expected = [ids[first[i]], ids[checker[i]]] + ([ids[final[i]]] if final[i] not in (first[i], checker[i]) else [])
            assert decision.selected_plan.plan_type == "agree_escalate" and calls == expected
            assert decision.selected_plan.max_calls == len(expected)
            assert decision.selected_plan.steps[2] == Verify(verifier=AGREE, acceptance_rule="agree")
            assert note["check_planned"] and note["checker"] == ids[checker[i]] and note["final"] == ids[final[i]]
        else:
            assert decision.selected_plan.plan_type == "direct" and calls == [ids[first[i]]]
            assert not note["check_planned"]
        assert note["first"] == ids[first[i]] and note["served_plan_type"] == decision.selected_plan.plan_type
        assert note["config"] == {"lam": live.config.lam, "accept": live.config.accept}
        # Every plan respects the policy's hard limits.
        assert decision.selected_plan.max_calls <= policy.max_calls
        assert decision.selected_plan.max_spend <= policy.max_absolute_spend
        assert decision.estimated_cost.mean <= policy.max_expected_spend + 1e-12
        assert decision.estimated_latency.upper <= policy.deadline_ms
        q, c, escalated = realized(zoo, decision)
        qualities.append(q)
        costs.append(c)
        escalations += escalated
    # Serving reproduces the calibrated population: same mean quality and cost as the tested step.
    step = next(s for s in live.calibration.steps if s.config == live.config)
    assert len(qualities) == step.requests == live.calibration_requests
    assert math.isclose(sum(qualities) / len(qualities), step.mean_quality, abs_tol=1e-12)
    assert math.isclose(sum(costs) / len(costs), step.mean_cost, rel_tol=1e-9)
    assert escalations > 0


def test_a_check_fails_a_gate_only_when_the_compound_plan_is_infeasible(zoo):
    planner = zoo.router.planner
    wide = cascade_policy(zoo.base, version="wide", minimum_quality=.8)
    tight = cascade_policy(zoo.base, version="tight", minimum_quality=.8, max_calls=2)
    evidence, live = planner.pooled_calibration(wide, at=AT, feature_version=zoo.feature_version)
    requests = evidence.calibration_requests[:120]
    quality, cost = evidence.forecast(requests)
    disagree = np.array(live.disagreement)
    config = cascade.CascadeConfig(lam=live.config.lam, accept=2.)  # always wants to check
    outcomes = {}
    for name, policy in (("wide", wide), ("tight", tight)):
        reference = planner._cascade_reference(evidence, requests, policy, quality, cost)
        gate = planner._compound_gate(policy, disagree)
        _, _, final, check, raw = live_cascade.gated_plan(reference, config, gate)
        outcomes[name] = (raw, check, reference, final)
    raw, check, reference, final = outcomes["tight"]
    assert raw.any() and (raw & ~check).any()  # a 2-call limit rejects every escalation to a third endpoint
    wide_check = outcomes["wide"][1]
    assert wide_check[raw].all() and (wide_check.sum() > check.sum())
    # With two calls allowed, only requests whose final endpoint is the checker may still check.
    checker = cascade.plan(reference.forecasts, reference.costs, config)[1]
    for i in np.flatnonzero(check):
        assert final[i] == checker[i]


def test_gated_replay_equals_the_offline_cascade_when_no_gate_binds():
    rng = np.random.default_rng(3)
    n, m = 600, 4
    p = np.array([.5, .65, .8, .93])
    Q = (rng.random((n, m)) < p).astype(float)
    shared = rng.random((n, m)) < .5
    answers = [["ok" if Q[i, j] else ("common" if shared[i, j] else f"other{j}") for j in range(m)] for i in range(n)]
    F = np.clip(p + rng.normal(0, .08, (n, m)), 0, 1)
    C = np.broadcast_to([1., 2., 5., 20.], (n, m)).copy()
    A = cascade.agreement_matrix(answers)
    reference = live_cascade.CascadeReference([None] * n, [[0, 1, 2, 3]] * n, F, C, np.zeros((n, m), int), C, [None] * n,
                                              Q, C, A, np.ones(n, bool))
    open_gate = lambda *args: True
    configs = cascade.default_configs((0., .02, .05, .15), accepts=(-1., .8, 2.))
    assert live_cascade.order_gated(configs, reference, open_gate) == cascade.order_configs(configs, F, C, Q, C, A)
    ordered = cascade.order_configs(configs, F, C, Q, C, A)
    ours = live_cascade.calibrate_gated(ordered, reference, open_gate, target=.8, risk=.1, minimum_requests=20)
    theirs = cascade.calibrate(ordered, F, C, Q, C, A, target=.8, risk=.1, minimum_requests=20)
    assert ours.identity() == theirs.identity() and ours.selected is not None
    # A gate that blocks checks serves those requests by the first endpoint alone.
    closed = live_cascade.gated_run(reference, cascade.CascadeConfig(.05, 2.), lambda *args: False)
    plain = cascade.run(F, C, Q, C, A, cascade.CascadeConfig(.05, -1.))
    assert np.array_equal(closed[0], plain[0]) and np.array_equal(closed[1], plain[1]) and not closed[3].any()


def test_gated_calibration_controls_false_certification():
    rng = np.random.default_rng(8)
    p = np.array([.55, .7, .9])
    def draw(n):
        Q = (rng.random((n, 3)) < p).astype(float)
        common = rng.random((n, 3)) < .5
        answers = [["ok" if Q[i, j] else ("common" if common[i, j] else f"other{j}") for j in range(3)] for i in range(n)]
        F = np.clip(p + rng.normal(0, .05, (n, 3)), 0, 1)
        C = np.broadcast_to([1., 2., 10.], (n, 3)).copy()
        return live_cascade.CascadeReference([None] * n, [[0, 1, 2]] * n, F, C, np.zeros((n, 3), int), C, [None] * n,
                                             Q, C, cascade.agreement_matrix(answers), np.ones(n, bool))
    # The gate depends on forecasts only, as a spend or deadline gate does: no check when first looks confident.
    gate = lambda ref, i, f, k, z: bool(ref.forecasts[i, f] < .62)
    configs = cascade.default_configs((0., .02, .05, .1, .3), accepts=(-1., .6, 2.))
    population = draw(150_000)
    truth = {cfg: float(np.nanmean(live_cascade.gated_run(population, cfg, gate)[0])) for cfg in configs}
    ordered = live_cascade.order_gated(configs, draw(3000), gate)
    target, risk, trials, false = .78, .1, 150, 0
    assert any(v < target for v in truth.values()) and any(v >= target for v in truth.values())
    for _ in range(trials):
        result = live_cascade.calibrate_gated(ordered, draw(400), gate, target=target, risk=risk, minimum_requests=20)
        if result.selected is not None and truth[result.selected] < target:
            false += 1
    assert false / trials <= risk + 3 * math.sqrt(risk * (1 - risk) / trials)


def test_certified_realized_quality_over_routed_test_traffic(zoo):
    policy = cascade_policy(zoo.base, version="realized", minimum_quality=.85)
    rows = evaluate_live_cascade(zoo.router, zoo.data, policy, zoo.answers)
    assert rows and not any(r["abstained"] for r in rows)
    assert all(r["metadata"]["quality_certified"] for r in rows)
    quality = sum(r["quality"] for r in rows) / len(rows)
    assert quality >= policy.minimum_quality - .04
    assert any(r["checked"] for r in rows) and any(r["escalated"] for r in rows) and not all(r["checked"] for r in rows)
    plain = evaluate_router(zoo.router, zoo.data, zoo.base.model_copy(update={"minimum_quality": .85, "version": "plain"}))
    assert sum(r["quality"] for r in plain) / len(plain) >= policy.minimum_quality - .04


def test_certificate_names_the_cascade_method_and_validates(zoo):
    planner = zoo.router.planner
    policy = cascade_policy(zoo.base, version="certificate", minimum_quality=.8, require_certificate=True,
                            infeasible_behavior="abstain")
    seen = set()
    for query in [q for q in zoo.data.queries if q.split == "test"][:80]:
        decision = planner.decide(query.request, policy, at=AT)
        assert decision.certificate_id and decision.selected_plan.plan_type != "abstain"
        certificate = planner.certificates.current(decision.certificate_id, AT)
        assert certificate.estimator_version == "pooled-ltt/hb-fixed-sequence-cascade-v1/binomial-p95-v3"
        assert any("agreement" in a or "Cascade" in a for a in certificate.assumptions)
        assert certificate.plan_key == decision.selected_plan.evidence_key
        assert certificate.cost_upper == decision.selected_plan.max_spend
        assert planner.certificates.validate_decision(decision, policy, planner.endpoints, AT) == ()
        seen.add(decision.selected_plan.plan_type)
    assert seen == {"direct", "agree_escalate"}


def test_threshold_rule_also_serves_the_cascade():
    threshold = fitted(1400, 5, AT, rule="threshold", pool=SMALL_POOL)
    policy = cascade_policy(threshold.base, version="threshold", minimum_quality=.8)
    planner = threshold.router.planner
    _, live = planner.pooled_calibration(policy, at=AT, feature_version=threshold.feature_version)
    assert live.kind == "cascade" and live.config is not None
    kinds = {planner.decide(q.request, policy, at=AT).selected_plan.plan_type
             for q in threshold.data.queries if q.split == "test"}
    assert kinds == {"direct", "agree_escalate"}


# ----- planner: behaviour is unchanged unless the cascade is permitted and answers exist --------
def test_behaviour_is_unchanged_when_agree_escalate_is_not_permitted(zoo, zoo_without_answers):
    plain = zoo.base.model_copy(update={"minimum_quality": .8, "version": "unchanged"})
    queries = [q for q in zoo.data.queries if q.split == "test"][:40]
    a, b = zoo.router.planner, zoo_without_answers.router.planner
    _, cal_a = a.pooled_calibration(plain, at=AT, feature_version=zoo.feature_version)
    _, cal_b = b.pooled_calibration(plain, at=AT, feature_version=zoo.feature_version)
    assert cal_a.identity() == cal_b.identity() and cal_a.risk == plain.quality_risk
    for query in queries:
        da, db = a.decide(query.request, plain, at=AT), b.decide(query.request, plain, at=AT)
        assert da.selected_plan == db.selected_plan and da.estimated_quality == db.estimated_quality
        assert da.certificate_status == db.certificate_status and da.fallback == db.fallback
        assert da.plan_metadata is None and "plan_metadata" not in da.model_dump(mode="json")


def test_cascade_is_not_attempted_without_answers_or_without_a_second_call(zoo, zoo_without_answers):
    plain = zoo.base.model_copy(update={"minimum_quality": .8, "version": "base"})
    expected = zoo_without_answers.router.planner.pooled_calibration(plain, at=AT, feature_version=zoo.feature_version)[1]
    # No stored answers: permitting agree_escalate changes nothing, and the full risk is spent on the direct test.
    permitted = cascade_policy(zoo.base, version="no-answers", minimum_quality=.8)
    got = zoo_without_answers.router.planner.pooled_calibration(permitted, at=AT, feature_version=zoo.feature_version)[1]
    assert got.identity() == expected.identity() and got.risk == permitted.quality_risk
    # One call is not enough for a check: the cascade is not attempted either.
    single = cascade_policy(zoo.base, version="single", minimum_quality=.8, max_calls=1)
    got = zoo.router.planner.pooled_calibration(single, at=AT, feature_version=zoo.feature_version)[1]
    assert got.identity() == expected.identity() and got.risk == single.quality_risk
    queries = [q for q in zoo_without_answers.data.queries if q.split == "test"][:25]
    for query in queries:
        da = zoo_without_answers.router.planner.decide(query.request, permitted, at=AT)
        db = zoo_without_answers.router.planner.decide(query.request, plain, at=AT)
        assert da.selected_plan.steps == db.selected_plan.steps and da.plan_metadata is None


def test_a_failed_cascade_splits_the_risk_with_the_direct_fallback(zoo):
    # Quality 0.99 is out of reach for every configuration: the cascade fails and so does the fallback.
    hard = cascade_policy(zoo.base, version="hard", minimum_quality=.99)
    _, fallback = zoo.router.planner.pooled_calibration(hard, at=AT, feature_version=zoo.feature_version)
    assert getattr(fallback, "kind", None) == "utility" and fallback.selected is None
    assert fallback.risk == hard.quality_risk / 2
    kinds = set()
    for query in [q for q in zoo.data.queries if q.split == "test"][:30]:
        decision = zoo.router.planner.decide(query.request, hard, at=AT)
        assert decision.certificate_status == "uncertified" and decision.plan_metadata is None
        kinds.add(decision.selected_plan.plan_type)
    assert kinds == {"direct"}


def test_the_compound_plan_is_never_enumerated_outside_pooled_routing():
    endpoint = lambda name, rate: EndpointSnapshot(endpoint_id=name, provider="local", upstream_model=name,
        revision="r1", region="local", context_window=8192, config_hash="h", price_version="p1",
        input_price_per_million=rate, output_price_per_million=rate, governance=frozenset({"local"}))
    policy = PolicySpec(policy_id="p", version="1", minimum_quality=.8, max_expected_spend=.01, max_absolute_spend=.02,
                        deadline_ms=1000, max_calls=3, verifier=AGREE,
                        permitted_plan_types=frozenset({"direct", "agree_escalate"}))
    plans, _, _ = Planner([endpoint("a", 1.), endpoint("b", 2.)]).generate(
        RequestContext(application_id="app", tenant_policy_id="p", max_output_tokens=8), policy)
    assert {p.plan_type for p in plans} == {"direct"}


# ----- executor ---------------------------------------------------------------------------------
class Scripted:
    """Adapter: endpoint -> reply text or exception, optional per-call delay; records the calls."""
    def __init__(self, replies, delay=0., barrier=None):
        self.replies, self.delay, self.barrier, self.calls = replies, delay, barrier, []
        self.lock = threading.Lock()

    def call(self, endpoint_id, generation_config):
        with self.lock:
            self.calls.append((endpoint_id, generation_config))
        if self.barrier is not None and endpoint_id in ("a", "b"):
            self.barrier.wait(timeout=3)
        if self.delay:
            time.sleep(self.delay)
        reply = self.replies[endpoint_id]
        if isinstance(reply, Exception):
            raise reply
        return RuntimeResult(f"hash-{endpoint_id}", f"id-{endpoint_id}", "r1", 10, 5, .001, None,
                             endpoint_id=endpoint_id, message={"role": "assistant", "content": reply})


def plan_decision(final="c", spends=(.01, .01, .01)):
    calls = ["a", "b"] + ([final] if final not in (None, "a", "b") else [])
    plan = agree_escalate("a", "b", AGREE, final, list(spends[:len(calls)]), max_output_tokens=64)
    decision = DecisionRecord(request=RequestContext(application_id="app", tenant_policy_id="p", max_output_tokens=64),
                              policy_id="p", policy_version="1", capability_map_version="m", eligible_endpoints=("a", "b", "c"),
                              selected_plan=plan, fallback="abstain", replay_seed=0)
    return decision, plan


def policy_for(**updates):
    return PolicySpec(**{"policy_id": "p", "version": "1", "minimum_quality": .8, "max_expected_spend": .05,
                         "max_absolute_spend": .1, "deadline_ms": 10000, "max_calls": 3, "verifier": AGREE,
                         "permitted_plan_types": frozenset({"direct", "agree_escalate"}), **updates})


def run(adapter, decision, *, policy=None, agreements=None, **kwargs):
    agreements = {AGREE: answer_agreement} if agreements is None else agreements
    return Executor(adapter, agreements=agreements).execute(decision, policy=policy or policy_for(), **kwargs)


def test_agreement_serves_the_first_answer_after_both_calls():
    decision, _ = plan_decision()
    adapter, sunk = Scripted({"a": "Reasoning...\nFinal answer: 42", "b": "so \\boxed{42}", "c": "Final answer: 41"}), []
    record = run(adapter, decision, result_sink=sunk.append)
    assert record.state == "completed" and record.selected_endpoint_id == "a" and record.attempted_calls == 2
    assert sorted(e for e, _ in adapter.calls) == ["a", "b"] and sunk[0].endpoint_id == "a"
    assert record.realized_spend == pytest.approx(.002) and record.reserved_spend == pytest.approx(.03)
    assert record.fallback_transitions == ("agreed:a,b",) and record.accounting_complete


def test_disagreement_escalates_to_the_final_endpoint():
    decision, plan = plan_decision()
    adapter = Scripted({"a": "Final answer: 42", "b": "Final answer: 7", "c": "Final answer: 42"})
    record = run(adapter, decision)
    assert record.state == "completed" and record.selected_endpoint_id == "c" and record.attempted_calls == 3
    assert [e for e, _ in adapter.calls][-1] == "c" and plan.max_calls == 3
    assert record.realized_spend == pytest.approx(.003) and record.provider_request_ids == ("id-a", "id-b", "id-c")
    assert record.fallback_transitions == ("escalated:c",)


def test_disagreement_serves_the_checker_when_it_is_the_final_endpoint():
    decision, plan = plan_decision(final="b", spends=(.01, .01))
    assert plan.max_calls == 2 and len(plan.steps) == 3
    adapter = Scripted({"a": "Final answer: 1", "b": "Final answer: 2"})
    record = run(adapter, decision)
    assert record.selected_endpoint_id == "b" and record.attempted_calls == 2 and len(adapter.calls) == 2


@pytest.mark.parametrize("failing, final, state, selected, attempts", [
    ("b", "c", "completed", "c", 3),      # checker error: disagreement path, escalate
    ("b", "b", "failed", None, 2),        # checker is the final endpoint and it failed: fail closed
    ("a", "c", "completed", "c", 3),      # first error: cannot check agreement, escalate
    ("a", "b", "completed", "b", 2),      # first error, checker is final: serve it
])
def test_failed_calls_fail_closed(failing, final, state, selected, attempts):
    decision, _ = plan_decision(final=final)
    replies = {"a": "Final answer: 5", "b": "Final answer: 5", "c": "Final answer: 5"}
    replies[failing] = TimeoutError("billed timeout")
    record = run(Scripted(replies), decision)
    assert record.state == state and record.selected_endpoint_id == selected and record.attempted_calls == attempts
    assert not record.accounting_complete and any(e.startswith(failing) for e in record.provider_errors)
    if selected is None:
        assert record.output_reference is None


def test_both_checked_calls_failing_still_escalates_and_a_failed_escalation_serves_nothing():
    decision, _ = plan_decision()
    escalated = run(Scripted({"a": OSError(), "b": OSError(), "c": "Final answer: 1"}), decision)
    assert escalated.state == "completed" and escalated.selected_endpoint_id == "c" and escalated.attempted_calls == 3
    refused = run(Scripted({"a": "Final answer: 1", "b": "Final answer: 2", "c": OSError()}), decision)
    assert refused.state == "failed" and refused.output_reference is None and refused.attempted_calls == 3


def test_agreement_function_errors_and_empty_answers_fail_closed():
    decision, _ = plan_decision()
    def broken(first, checker):
        raise RuntimeError("boom")
    record = run(Scripted({"a": "Final answer: 1", "b": "Final answer: 1", "c": "Final answer: 1"}), decision,
                 agreements={AGREE: broken})
    assert record.selected_endpoint_id == "c" and "agreement_failed_closed" in record.fallback_transitions
    empty = run(Scripted({"a": "", "b": "", "c": "Final answer: 3"}), decision)
    assert empty.selected_endpoint_id == "c"  # nothing comparable never counts as agreement


def test_unknown_agreement_is_rejected_before_any_spend():
    decision, _ = plan_decision()
    adapter = Scripted({"a": "x", "b": "x", "c": "x"})
    with pytest.raises(ValueError, match="agreement"):
        Executor(adapter).execute(decision, policy=policy_for())
    # A verifier of the same name does not stand in for an agreement function.
    with pytest.raises(ValueError, match="agreement"):
        Executor(adapter, verifiers={AGREE: lambda r, rule: True}).execute(decision, policy=policy_for())
    assert not adapter.calls


def test_plan_type_must_be_permitted_and_within_the_call_limit():
    decision, _ = plan_decision()
    adapter = Scripted({"a": "x", "b": "x", "c": "x"})
    with pytest.raises(ValueError, match="policy"):
        run(adapter, decision, policy=policy_for(permitted_plan_types=frozenset({"direct"})))
    with pytest.raises(ValueError, match="policy"):
        run(adapter, decision, policy=policy_for(max_calls=2))
    assert not adapter.calls


def test_first_and_checker_run_concurrently_and_the_final_call_waits():
    decision, _ = plan_decision()
    barrier = threading.Barrier(2)
    adapter = Scripted({"a": "Final answer: 1", "b": "Final answer: 2", "c": "Final answer: 2"}, barrier=barrier)
    record = run(adapter, decision)  # a sequential executor would time out at the barrier
    assert record.state == "completed" and record.selected_endpoint_id == "c"
    assert [e for e, _ in adapter.calls][2] == "c" and not barrier.broken


def test_deadline_guard_stops_the_escalation_call():
    decision, _ = plan_decision()
    adapter = Scripted({"a": "Final answer: 1", "b": "Final answer: 2", "c": "Final answer: 2"}, delay=.08)
    record = run(adapter, decision, policy=policy_for(deadline_ms=50))
    assert record.state == "failed" and record.attempted_calls == 2
    assert "PRECALL_GUARD_REJECTED" in record.constraint_violations
    assert "c" not in [e for e, _ in adapter.calls]


def test_output_and_timeout_bounds_apply_to_every_call():
    decision, _ = plan_decision()
    adapter = Scripted({"a": "Final answer: 1", "b": "Final answer: 2", "c": "Final answer: 2"})
    run(adapter, decision)
    assert len(adapter.calls) == 3
    for _, config in adapter.calls:
        assert config["max_completion_tokens"] == 64 and 0 < config["timeout"] <= 10


# ----- default agreement function ---------------------------------------------------------------
@pytest.mark.parametrize("text, answer", [
    ("The result is \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),
    ("\\boxed{a{b}c} and later \\boxed{9}", "9"),
    ("work\nFinal Answer: **42**\nthanks", "42"),
    ("work\nanswer: B\nmore text", "B"),
    ("first\n\nlast line  \n", "last line"),
    ("   \n  ", None),
    (None, None),
])
def test_final_answer_extraction(text, answer):
    assert extract_final_answer(text) == answer


def test_answer_agreement_compares_normalized_answers():
    def result(content):
        return RuntimeResult("r", "p", "v", 1, 1, 0., None, endpoint_id="a", message={"role": "assistant", "content": content})
    assert answer_agreement(result("Final answer: 42."), result("so \\boxed{ 42 }"))
    assert answer_agreement(result("Answer:  The Cat"), result("...\nthe cat"))
    assert not answer_agreement(result("Final answer: 42"), result("Final answer: 43"))
    assert not answer_agreement(result(""), result(""))
    tools = RuntimeResult("r", "p", "v", 1, 1, 0., None, message={"role": "assistant", "content": None, "tool_calls": [{}]})
    assert not answer_agreement(tools, tools) and not answer_agreement(result("x"), RuntimeResult("r", "p", "v", 1, 1, 0., None))


def test_service_registers_the_default_agreement_by_a_stable_name():
    import inspect

    from inference_control import configuration
    assert "answer_agreement" in inspect.getsource(configuration.load_service)
    assert configuration.answer_agreement is answer_agreement


# ----- contracts and compiler -------------------------------------------------------------------
def document(**plans):
    return {"minimum_quality": .9, "cost": {"expected_max": .01, "absolute_max": .03},
            "latency": {"p95_max_ms": 2500}, "plans": plans}


def test_compiler_validates_agree_escalate():
    policy = compile_policy(document(allowed=["direct", "agree_escalate"], max_calls=3, verifier=AGREE))
    assert policy.permitted_plan_types == {"direct", "agree_escalate"} and policy.verifier == AGREE
    with pytest.raises(ValueError, match="answer-agreement"):
        compile_policy(document(allowed=["direct", "agree_escalate"], max_calls=3))
    with pytest.raises(ValueError, match="two calls"):
        compile_policy(document(allowed=["direct", "agree_escalate"], max_calls=1, verifier=AGREE))
    with pytest.raises(ValueError, match="direct"):
        compile_policy(document(allowed=["agree_escalate"], max_calls=3, verifier=AGREE))
    with pytest.raises(ValueError):
        compile_policy(document(allowed=["agree_escalate", "direct", "teleport"], max_calls=3, verifier=AGREE))
    # Existing plan types are unchanged.
    assert compile_policy(document()).permitted_plan_types == {"direct"}


def test_plan_shapes_are_validated():
    plan = agree_escalate("a", "b", AGREE, "c", [.01, .01, .01])
    assert plan.plan_type == "agree_escalate" and plan.max_calls == 3 and plan.max_spend == pytest.approx(.03)
    assert [type(s) for s in plan.steps] == [Call, Call, Verify, Call]
    assert agree_escalate("a", "b", AGREE, "b", [.01, .01]).max_calls == 2
    assert agree_escalate("a", "b", AGREE, None, [.01, .01]).max_calls == 2
    assert plan.evidence_key.startswith("plan:") and plan.evidence_key != agree_escalate("a", "b", AGREE, "d", [.01] * 3).evidence_key
    steps = plan.steps
    bad = [steps[:2], (steps[0], steps[2], steps[1]), steps[:3] + (steps[3], steps[3]),
           (steps[0], steps[0], steps[2]), (steps[0], steps[1], Verify(verifier=AGREE, acceptance_rule="accept")),
           (steps[0], steps[1], steps[2], steps[0])]
    for shape in bad:
        with pytest.raises(ValueError):
            ExecutionPlan(plan_type="agree_escalate", steps=shape, max_calls=3, max_spend=.03)
    with pytest.raises(ValueError):
        ExecutionPlan(plan_type="agree_escalate", steps=steps, max_calls=2, max_spend=.03)
    with pytest.raises(ValueError):
        PolicySpec(policy_id="p", version="1", minimum_quality=.5, max_expected_spend=0, max_absolute_spend=0,
                   deadline_ms=1, permitted_plan_types=frozenset({"nonsense"}))
    PolicySpec(policy_id="p", version="1", minimum_quality=.5, max_expected_spend=0, max_absolute_spend=0,
               deadline_ms=1, permitted_plan_types=frozenset({"agree_escalate"}))


# ----- answers as map evidence ------------------------------------------------------------------
def row(i, endpoint="a", split="train", **updates):
    request = RequestContext(request_id=f"r{i}", application_id="app", tenant_policy_id="p", query_features=(float(i), 1.),
                             feature_version="q-v1")
    data = {"sample_id": f"s{i}:{endpoint}", "request": request, "target_id": endpoint, "endpoint_ids": (endpoint,),
            "endpoint_revisions": ("rev",), "quality": 1., "cost": .01, "latency_ms": 10., "split": split,
            "evaluator_type": "deterministic", "evaluator_version": "v1", "observed_at": AT}
    return EvidenceObservation(**{**data, **updates})


def small_map(n=4):
    cmap = ConditionalCapabilityMap(k=8, min_samples=2)
    cmap.add_many([row(i, e, "train" if i % 2 else "calibration") for i in range(n) for e in ("a", "b")])
    return cmap


def test_maps_without_answers_keep_their_exports_and_hashes():
    from inference_control.util import digest
    cmap = small_map()
    version, evidence_version, export = cmap.version, cmap.evidence_version, json.dumps(cmap.export(), sort_keys=True)
    cmap.add_answers({})
    assert (cmap.version, cmap.evidence_version) == (version, evidence_version)
    assert json.dumps(cmap.export(), sort_keys=True) == export and "answers" not in cmap.export()
    # The identity formula predates answers; recomputing it by hand gives the same hashes.
    rows = {k: cmap._row_hash(cmap.rows[k]) for k in sorted(cmap.rows)}
    assert evidence_version == digest({"rows": rows, "invalidations": []})
    identity = {"format": "content-addressed-capability-v3", "k": 8, "min_samples": 2, "max_age_seconds": 86400,
                "quality_method": "pooled-ltt", "quality_predictors": (), "rows": rows, "invalidations": [],
                "pooled": cmap.pooled_config.identity()}
    assert version == "map:" + digest(identity)
    restored = ConditionalCapabilityMap.restore(json.loads(export))
    assert restored.version == version and "answers" not in restored.export()


def test_answer_store_round_trips_changes_versions_and_never_keeps_text():
    cmap = small_map()
    before = (cmap.version, cmap.evidence_version)
    cmap.add_answers({"s0:a": "The Answer is 42.", "s0:b": None, "s1:a": "forty-two"})
    assert cmap.version != before[0] and cmap.evidence_version != before[1]
    exported = cmap.export()
    assert exported["answers"]["s0:a"] == answer_key("the answer is 42") and exported["answers"]["s0:b"] is None
    assert len(exported["answers"]["s1:a"]) == 16 and "forty" not in json.dumps(exported)
    restored = ConditionalCapabilityMap.restore(json.loads(json.dumps(exported)))
    assert restored.version == cmap.version and restored.evidence_version == cmap.evidence_version
    assert restored.answers == cmap.answers and restored.export()["answers"] == exported["answers"]
    cmap.add_answers({"s0:a": "the answer is 42"})  # same normalized answer: no conflict
    with pytest.raises(ValueError, match="immutable"):
        cmap.add_answers({"s0:a": "43"})
    with pytest.raises(ValueError):
        cmap.add_answers({"s9:a": "not a key"}, hashed=True)
    assert answer_key("  A  b. ") == answer_key("a b") and answer_key("") is None and answer_key(None) is None


def test_pooled_evidence_exposes_answer_matrices():
    cmap = ConditionalCapabilityMap(k=8, min_samples=2)
    cmap.add_many([row(i, e, "train" if i < 3 else "calibration") for i in range(5) for e in ("a", "b")])
    endpoints = tuple(EndpointSnapshot(endpoint_id=e, provider="x", upstream_model=e, revision="r", region="x",
        context_window=10, config_hash="h", price_version="p", input_price_per_million=1., output_price_per_million=1.)
        for e in ("a", "b"))
    ids = tuple(e.endpoint_id for e in endpoints)
    cmap.add_many([row(i, e, "train" if i < 3 else "calibration", endpoint_revisions=(endpoints[ids.index(e)].capability_revision,),
                       sample_id=f"t{i}:{e}") for i in range(5) for e in ids])
    plain = cmap.pooled_evidence(endpoints, "q-v1", AT + timedelta(seconds=1), 86400.)
    assert plain.train_answers is None and plain.calibration_answered is None and not plain.has_answers
    cmap.add_answers({f"t{i}:a": f"ans{i}" for i in range(5)} | {"t0:b": None, "t1:b": "ans1"})
    evidence = cmap.pooled_evidence(endpoints, "q-v1", AT + timedelta(seconds=1), 86400.)
    assert evidence.has_answers and evidence.answers_hash and evidence.identity != plain.identity
    assert evidence.train_answers.shape == evidence.train_quality.shape
    assert evidence.train_answered[:, 0].all() and evidence.train_answered[:2, 1].all() and not evidence.train_answered[2, 1]
    row0 = evidence.train_requests.index(next(r for r in evidence.train_requests if r.request_id == "r0"))
    assert evidence.train_answers[row0, 0] == answer_key("ans0") and evidence.train_answers[row0, 1] is None
    assert evidence.calibration_cost.shape == evidence.calibration_quality.shape


# ----- control plane: decide -> execute -> learn -> drift ---------------------------------------
class ZooAdapter:
    """Replies with the zoo's stored answers; ``for_request`` binds the request being served."""
    def __init__(self, zoo):
        self.zoo, self.request_id, self.calls = zoo, None, []

    def for_request(self, request, messages, response_config=None):
        bound = ZooAdapter(self.zoo)
        bound.request_id, bound.calls = request.request_id, self.calls
        return bound

    def call(self, endpoint_id, generation_config):
        self.calls.append((self.request_id, endpoint_id))
        truth = self.zoo.truth[(self.request_id, endpoint_id)]
        answer = self.zoo.answers[(self.request_id, endpoint_id)]
        return RuntimeResult(f"hash:{endpoint_id}", f"id:{self.request_id}:{endpoint_id}", "zoo-r1", truth.input_tokens,
                             truth.output_tokens, truth.cost, None, endpoint_id=endpoint_id,
                             message={"role": "assistant", "content": f"Final answer: {answer}"})


def outcome_for(decision, execution, quality):
    return OutcomeRecord(decision_id=decision.decision_id, execution_id=execution.execution_id, quality={"success": quality},
                         evaluator_type="human", evaluator_version="human-v1", source_artifact="sha256:fixture",
                         label_confidence=1., uncertainty=0., causal_scope="plan", training_eligible=True,
                         promotion_eligible=True)


def test_control_plane_decide_execute_learn_and_replay(tmp_path):
    now = datetime.now(UTC)
    fixture = fitted(1400, 5, now - timedelta(minutes=5), pool=SMALL_POOL)
    # The map travels through JSON like a stored capability map; answers must survive it.
    cmap = ConditionalCapabilityMap.restore(json.loads(json.dumps(fixture.router.map.export())))
    assert cmap.answers == fixture.router.map.answers and cmap.version == fixture.router.map.version
    path = tmp_path / "plane.db"
    control = ControlPlane(Planner(fixture.endpoints, capability_map=cmap), SQLiteLedger(path),
                           Executor(ZooAdapter(fixture), agreements={AGREE: answer_agreement}))
    policy = cascade_policy(fixture.base, version="plane-1", minimum_quality=.8, require_certificate=True,
                            infeasible_behavior="abstain")
    served = {}
    for query in [q for q in fixture.data.queries if q.split == "test"]:
        decision = control.decide(DecideRequest(request=query.request, policy=policy))
        kind = decision.selected_plan.plan_type
        if kind in served:
            continue
        assert kind in ("direct", "agree_escalate") and decision.certificate_status == "current"
        assert decision.plan_metadata["cascade"]["check_planned"] == (kind == "agree_escalate")
        record = control.execute(decision.decision_id, messages=[{"role": "user", "content": "q"}])
        calls = [s.endpoint_id for s in decision.selected_plan.steps if isinstance(s, Call)]
        assert record.state == "completed" and not record.constraint_violations and record.selected_endpoint_id in calls
        assert record.attempted_calls == 1 if kind == "direct" else 2 <= record.attempted_calls <= len(calls)
        quality, cost, _ = realized(fixture, decision)
        assert record.realized_spend == pytest.approx(cost)
        served[kind] = (decision, record, quality, calls)
        if len(served) == 2:
            break
    assert set(served) == {"direct", "agree_escalate"}
    for kind, (decision, record, quality, calls) in served.items():
        outcome = control.add_outcome(outcome_for(decision, record, quality))
        learned = control.planner.capability_map.rows[f"outcome:{outcome.outcome_id}"]
        assert learned.plan_type == kind and learned.endpoint_ids == tuple(calls)
    # Compound outcomes are stored but never enter the pooled training or calibration matrices.
    evidence = control.planner._pooled_evidence(policy, datetime.now(UTC), fixture.feature_version)
    pooled = {r.request_id for r in evidence.train_requests + evidence.calibration_requests}
    assert served["agree_escalate"][0].request.request_id not in pooled
    # Latency of both executions reached drift recovery (a compound observation for the checked one).
    receipts = [e for e in control.ledger.events() if e.event_type == "latency_receipt"]
    assert len(receipts) == 2 and control.ledger.verify()
    # Restart: the stored map (with answers) rebuilds the planner, and the recorded decision replays.
    compound = served["agree_escalate"][0]
    control.ledger.close()
    restored = ControlPlane(Planner([]), SQLiteLedger(path))
    assert restored.decisions[compound.decision_id].plan_metadata == compound.plan_metadata
    assert restored.planner.capability_map.answers == cmap.answers
    replay = restored.replay(compound.decision_id)
    assert replay["matched"] and replay["plan"]["plan_type"] == "agree_escalate"


def test_the_service_api_serializes_cascade_decisions():
    from fastapi.testclient import TestClient

    from inference_control.api.app import create_app
    now = datetime.now(UTC)
    fixture = fitted(1400, 5, now - timedelta(minutes=5), pool=SMALL_POOL)
    control = ControlPlane(fixture.router.planner, SQLiteLedger(":memory:"), Executor(ZooAdapter(fixture), agreements={AGREE: answer_agreement}))
    policy = cascade_policy(fixture.base, version="api-1", minimum_quality=.8)
    client = TestClient(create_app(control))
    found = {}
    queries_request = next(q.request for q in fixture.data.queries if q.split == "test")
    for query in [q for q in fixture.data.queries if q.split == "test"]:
        body = client.post("/v2/decide", json={"request": query.request.model_dump(mode="json"),
                                               "policy": policy.model_dump(mode="json")})
        assert body.status_code == 200
        found.setdefault(body.json()["selected_plan"]["plan_type"], body.json())
        if len(found) == 2:
            break
    assert set(found) == {"direct", "agree_escalate"}
    compound = found["agree_escalate"]
    assert compound["plan_metadata"]["cascade"]["check_planned"] is True
    executed = client.post("/v2/execute", json={"decision_id": compound["decision_id"],
                                                 "messages": [{"role": "user", "content": "q"}]})
    assert executed.status_code == 200 and executed.json()["state"] == "completed"
    plain = fixture.base.model_copy(update={"minimum_quality": .8, "version": "api-plain"})
    body = client.post("/v2/decide", json={"request": queries_request.model_dump(mode="json"),
                                           "policy": plain.model_dump(mode="json")}).json()
    assert body["selected_plan"]["plan_type"] == "direct" and "plan_metadata" not in body
