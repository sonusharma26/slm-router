"""Pooled learned routing: calibration validity, serving/calibration agreement, fallback, wiring."""
import importlib.util
import math
import random
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from inference_control.benchmarks.compare import aiq, hull_cost, hull_quality, run_compare, upper_hull
from inference_control.benchmarks.routers import SLMRouter
from inference_control.benchmarks.runner import default_policy, evaluate_router, split_data
from inference_control.benchmarks.zoo import make_zoo
from inference_control.capability.conditional import ConditionalCapabilityMap
from inference_control.contracts import Call, InfeasibleBehavior, PolicySpec
from inference_control.policies.compiler import compile_policy
from inference_control.routing.calibration import (
    anchor_start,
    binomial_cdf,
    calibrate_threshold,
    first_certified,
    hb_pvalue,
    threshold_grid,
)
from inference_control.routing.pooled import PooledConfig
from inference_control.routing.predictor import PredictorConfig, QualityPredictor, brier

AT = datetime(2026, 1, 1, tzinfo=UTC)


# ----- calibration mathematics -----------------------------------------------------------------
def test_binomial_cdf_matches_direct_sum():
    for n, p in ((10, .3), (37, .81), (5, .5)):
        for k in range(-1, n + 1):
            direct = sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(max(0, k) + 1)) if k >= 0 else 0.
            assert binomial_cdf(k, n, p) == pytest.approx(min(1., direct), abs=1e-12)


def test_hb_pvalue_is_valid_at_the_null_boundary():
    rng = random.Random(3)
    for target, n in ((.7, 40), (.9, 120)):
        rejections = 0
        trials = 3000
        for _ in range(trials):
            total = sum(rng.random() < target for _ in range(n))
            rejections += hb_pvalue(total, n, target) <= .05
        assert rejections / trials <= .05 + 3 * math.sqrt(.05 * .95 / trials)


def test_hb_pvalue_edges():
    assert hb_pvalue(0, 0, .7) == 1.
    assert hb_pvalue(5, 5, 0.) == 0.
    assert hb_pvalue(5, 5, 1.) == 1.
    assert hb_pvalue(100, 100, .7) < 1e-10


def test_first_certified_matches_brute_force():
    rng = np.random.default_rng(1)
    forecasts = rng.random((50, 6))
    forecasts[rng.random((50, 6)) < .2] = -np.inf
    thresholds = (.9, .7, .4, .1)
    positions = first_certified(forecasts, thresholds)
    for g, t in enumerate(thresholds):
        for i in range(50):
            expected = next((j for j in range(6) if forecasts[i, j] >= t), 6)
            assert positions[g, i] == expected


def _draw(rng, n):
    """Requests with two endpoints; true quality falls with the forecast gap at loose thresholds."""
    f = rng.random((n, 2))
    q = np.empty((n, 2))
    q[:, 0] = (rng.random(n) < np.clip(f[:, 0] - .05, 0, 1)).astype(float)
    q[:, 1] = (rng.random(n) < np.clip(f[:, 1] + .02, 0, 1)).astype(float)
    return f, q


def test_fixed_sequence_controls_false_certification():
    rng = np.random.default_rng(7)
    target, risk = .75, .1
    thresholds = tuple(np.round(np.linspace(.98, .3, 35), 4))
    big_f, big_q = _draw(rng, 400_000)
    positions = first_certified(big_f, thresholds)
    truth = {}
    for g, t in enumerate(thresholds):
        chosen = positions[g]
        certified = chosen < 2
        truth[t] = big_q[np.arange(len(big_q))[certified], chosen[certified]].mean()
    assert any(v < target for v in truth.values()) and any(v >= target for v in truth.values())
    trials, false = 400, 0
    for _ in range(trials):
        f, q = _draw(rng, 600)
        result = calibrate_threshold(f, q, thresholds, target=target, risk=risk, minimum_certified=20)
        if result.threshold is not None and truth[result.threshold] < target:
            false += 1
    assert false / trials <= risk + 3 * math.sqrt(risk * (1 - risk) / trials)


def test_threshold_grid_continues_below_full_certification():
    best = [.95, .9, .8, .7]
    every = [.95, .9, .8, .7, .6, .5, .4, .3]
    grid = threshold_grid(best, (.5, 1.), every, tail_steps=4)
    assert all(b < a for a, b in pairwise(grid))
    assert grid[:2] == (.9, .7) and min(grid) < .7


def test_anchor_prefers_thresholds_with_enough_expected_samples():
    rng = np.random.default_rng(2)
    forecasts = rng.random((500, 1))
    values = (rng.random((500, 1)) < forecasts).astype(float)
    thresholds = tuple(np.round(np.linspace(.99, .2, 30), 3))
    start = anchor_start(forecasts, values, thresholds, calibration_size=300, target=.7, risk=.05, minimum_certified=8)
    expected = (forecasts[:, 0] >= thresholds[start]).mean() * 300
    assert expected >= 40 and start > 0


# ----- predictor -------------------------------------------------------------------------------
def test_predictor_beats_constant_and_leave_one_out_excludes_self():
    data = make_zoo(800, seed=3)
    endpoints = [e.endpoint_id for e in data.endpoints]
    def matrix(split):
        queries = [q for q in data.queries if q.split == split]
        index = {q.request.request_id: i for i, q in enumerate(queries)}
        Q = np.full((len(queries), len(endpoints)), np.nan)
        for r in data.outcomes:
            if r.request_id in index:
                Q[index[r.request_id], endpoints.index(r.endpoint_id)] = r.quality
        return [q.request.query_features for q in queries], Q
    Xt, Qt = matrix("train")
    Xv, Qv = matrix("validation")
    model = QualityPredictor(PredictorConfig(neighbors=16, clusters=8)).fit(Xt, Qt)
    predicted, _ = model.predict(Xv)
    constant = np.broadcast_to(np.nanmean(Qt, axis=0), Qv.shape)
    assert brier(predicted, Qv) < brier(constant, Qv)
    one = QualityPredictor(PredictorConfig(neighbors=1, neighbor_prior=1e-6, clusters=4)).fit(Xt, Qt)
    in_sample, _ = one.predict(Xt[:20])
    loo, _ = one.predict(Xt[:20], exclude=np.arange(20))
    assert np.allclose(in_sample, np.clip(Qt[:20], 0, 1), atol=1e-4)
    assert not np.allclose(loo, in_sample)


def test_forecasts_are_identical_alone_or_in_a_batch():
    data = make_zoo(600, seed=4)
    train = [q for q in data.queries if q.split == "train"]
    endpoints = [e.endpoint_id for e in data.endpoints]
    index = {q.request.request_id: i for i, q in enumerate(train)}
    Q = np.full((len(train), len(endpoints)), np.nan)
    for r in data.outcomes:
        if r.request_id in index:
            Q[index[r.request_id], endpoints.index(r.endpoint_id)] = r.quality
    model = QualityPredictor(PredictorConfig(neighbors=24, clusters=8)).fit([q.request.query_features for q in train], Q)
    others = [q.request.query_features for q in data.queries if q.split == "test"][:40]
    batch, _ = model.predict(others)
    single = np.vstack([model.predict([x])[0] for x in others])
    assert np.array_equal(batch, single)


# ----- planner integration ---------------------------------------------------------------------
@pytest.fixture(scope="module", params=["utility", "threshold"])
def tuned(request):
    data = make_zoo(2400, seed=5)
    policy = default_policy(minimum_quality=.65, missing_latency=True)
    router = SLMRouter(data.endpoints, min_samples=8, at=AT, pooled=PooledConfig(rule=request.param))
    router.fit(*split_data(data, {"train", "calibration"}))
    router.tune(*split_data(data, {"validation"}), policy)
    assert router.map.pooled_config.rule == request.param
    return data, router, policy


def test_serving_decisions_match_the_calibration_replay(tuned):
    data, router, policy = tuned
    planner = router.planner
    feature_version = data.queries[0].request.feature_version
    evidence, calibration = planner.pooled_calibration(policy, at=AT, feature_version=feature_version)
    utility = router.map.pooled_config.rule == "utility"
    assert (calibration.parameter if utility else calibration.threshold) is not None
    requests = evidence.calibration_requests[:60]
    quality, cost = evidence.forecast(requests)
    for i, request in enumerate(requests):
        rule = planner._pooled_rule(evidence, request, policy, quality[i], cost[i])
        if utility:
            scores = [f - calibration.parameter * c for _, f, c in rule]
            expected = rule[scores.index(max(scores))][0] if rule else None
        else:
            expected = next((e for e, f, _ in rule if f >= calibration.threshold), None)
        decision = planner.decide(request, policy, at=AT)
        chosen = next(s.endpoint_id for s in decision.selected_plan.steps if isinstance(s, Call))
        if expected is not None:
            assert chosen == expected and decision.fallback == policy.infeasible_behavior.value
        else:
            assert decision.fallback == "best_predicted_uncertified"


def test_certified_test_traffic_meets_target_and_certificates_validate(tuned):
    data, router, policy = tuned
    rows = evaluate_router(router, data, policy)
    certified = [r for r in rows if r["metadata"]["quality_certified"]]
    assert certified and not any(r["abstained"] for r in rows)
    assert sum(r["quality"] for r in certified) / len(certified) >= policy.minimum_quality - .05
    query = next(q for q in data.queries if q.split == "test")
    decision = router.planner.decide(query.request, policy, at=AT)
    if decision.certificate_id:
        certificate = router.planner.certificates.current(decision.certificate_id, AT)
        assert certificate.estimator_version.startswith("pooled-ltt/")
        assert router.planner.certificates.validate_decision(decision, policy, router.planner.endpoints, AT) == ()


def test_fallback_margin_keeps_the_best_training_endpoint(tuned):
    data, router, policy = tuned
    strict = policy.model_copy(update={"minimum_quality": 1., "version": "max-quality"})
    config = router.map.pooled_config
    router.map.configure_pooled(PooledConfig(predictor=config.predictor, thresholds=config.thresholds,
                                             fallback_margin=1., rule=config.rule))
    try:
        evidence, _ = router.planner.pooled_calibration(strict, at=AT, feature_version=data.queries[0].request.feature_version)
        anchor = max(evidence.endpoint_ids, key=lambda e: (evidence.training_mean(e), e))
        for query in [q for q in data.queries if q.split == "test"][:15]:
            decision = router.planner.decide(query.request, strict, at=AT)
            assert decision.fallback == "best_predicted_uncertified"
            assert decision.selected_plan.steps[0].endpoint_id == anchor
    finally:
        router.map.configure_pooled(config)


def test_utility_rule_certifies_all_routed_traffic(tuned):
    data, router, policy = tuned
    if router.map.pooled_config.rule != "utility":
        pytest.skip("utility rule only")
    rows = evaluate_router(router, data, policy)
    assert all(r["metadata"]["quality_certified"] for r in rows)
    assert sum(r["quality"] for r in rows) / len(rows) >= policy.minimum_quality - .05


def test_rule_sequence_calibration_controls_false_certification():
    from inference_control.routing.calibration import calibrate_sequence
    rng = np.random.default_rng(11)
    means = np.linspace(.95, .55, 25)  # rules ordered safest first; true mean quality falls
    target, risk, trials, false = .75, .1, 500, 0
    for _ in range(trials):
        values = [(rng.random(400) < mu).astype(float) for mu in means]
        result = calibrate_sequence(values, tuple(range(len(means))), target=target, risk=risk, minimum_requests=10)
        if result.selected is not None and means[result.selected] < target:
            false += 1
    assert false / trials <= risk + 3 * math.sqrt(risk * (1 - risk) / trials)


def test_pooled_map_config_roundtrip_changes_identity():
    cmap = ConditionalCapabilityMap()
    assert cmap.quality_method == "pooled-ltt"
    before = cmap.version
    cmap.configure_pooled(PooledConfig(predictor=PredictorConfig(neighbors=8), thresholds=(.9, .8), fallback_margin=.05))
    assert cmap.version != before
    restored = ConditionalCapabilityMap.restore(cmap.export())
    assert restored.pooled_config == cmap.pooled_config and restored.version == cmap.version
    legacy = ConditionalCapabilityMap.restore({k: v for k, v in cmap.export().items() if k not in {"quality_method", "pooled"}})
    assert legacy.quality_method == "local-mean-hoeffding"


def test_best_predicted_fallback_cannot_bypass_required_certificates():
    with pytest.raises(ValueError):
        PolicySpec(policy_id="p", version="1", minimum_quality=.5, max_expected_spend=1, max_absolute_spend=1,
                   deadline_ms=1000, require_certificate=True, infeasible_behavior=InfeasibleBehavior.BEST_PREDICTED)
    base = {"minimum_quality": .5, "cost": {"expected_max": 1, "absolute_max": 1}, "latency": {"p95_max_ms": 1000}}
    assert compile_policy(base).infeasible_behavior == InfeasibleBehavior.BEST_PREDICTED
    certified = dict(base, evidence={"require_certificate": True})
    assert compile_policy(certified).infeasible_behavior == InfeasibleBehavior.ABSTAIN
    with pytest.raises(ValueError):
        compile_policy(dict(certified, on_infeasible="best_predicted_fallback"))


# ----- comparison harness ----------------------------------------------------------------------
def test_hull_and_aiq():
    hull = upper_hull([(1, .5), (2, .55), (3, .9), (4, .8), (5, .95), (.5, .2)])
    assert hull == [(.5, .2), (1., .5), (3., .9), (5., .95)]
    assert hull_quality(hull, 2) == pytest.approx(.7) and hull_cost(hull, .9) == 3.
    assert hull_quality(hull, .1) is None and hull_cost(hull, .99) is None
    assert aiq(hull, .5, 5) == pytest.approx(.7611, abs=1e-4)


def test_compare_runs_end_to_end_on_a_small_fixture():
    report = run_compare(make_zoo(400, seed=9), seeds=(1,), targets=(.6, 1.), knn_targets=(.5,), thresholds=(.5,),
                         local_targets=())
    summary = report["aggregate"]
    assert {"slm-router", "knn", "best_single", "oracle (reference)"} <= set(summary)
    assert report["claim_gate"]["blockers"][0] == "SYNTHETIC_DATA_IS_NOT_EXTERNAL_BENCHMARK_EVIDENCE"
    assert len(report["pooled_guarantee"]["checks"]) == 2


# ----- LLMRouterBench conversion ---------------------------------------------------------------
def _script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_llmrouterbench.py"
    spec = importlib.util.spec_from_file_location("run_llmrouterbench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_llmrouterbench_conversion_keeps_their_test_split_and_fills_missing_scores():
    script = _script()
    def record(dataset, prompt, model, score, cost=.01):
        return SimpleNamespace(dataset_id=dataset, prompt=prompt, origin_query=prompt, model_name=model, score=score,
                               cost=cost, prompt_tokens=10, completion_tokens=5, prediction=model.upper())
    train = [record("math", f"what is {i} plus {i}", m, float(i % 2)) for i in range(40) for m in ("a", "b")]
    test = [record("code", "write a sort", "a", 1.), record("code", "write a sort", "b", 0.),
            record("code", "fix the bug", "a", 1.)]  # model b missing on one test prompt
    version, featurize = script.featurizer(None)
    data, slice_of, answers = script.build_dataset(train, test, seed=1, featurize=featurize, feature_version=version,
                                                   task_hints=False, name="unit")
    tests = [q for q in data.queries if q.split == "test"]
    assert len(tests) == 2 and all(q.request.task_hint is None for q in data.queries)
    assert {q.split for q in data.queries if q.text.startswith("what")} <= {"train", "calibration", "validation"}
    assert data.provenance["filled_missing_scores"] == 1
    assert set(slice_of.values()) == {"math", "code"}
    sort_id = next(q.request.request_id for q in tests if q.text == "write a sort")
    assert answers[(sort_id, "a")] == "A" and answers[(sort_id, "b")] == "B"
    fix_id = next(q.request.request_id for q in tests if q.text == "fix the bug")
    assert answers[(fix_id, "b")] is None


# ----- answer-agreement cascades ---------------------------------------------------------------
def test_answer_normalization_and_agreement():
    from inference_control.routing.cascade import agreement_matrix, normalize_answer
    assert normalize_answer("  The  Answer. ") == "the answer" and normalize_answer("") is None
    agree = agreement_matrix([["42", " 42.", None], ["a", "b", "a"]])
    assert agree[0, 0, 1] and not agree[0, 0, 2] and not agree[0, 2, 2]
    assert agree[1, 0, 2] and not agree[1, 0, 1]


def test_cascade_mechanics_by_hand():
    from inference_control.routing.cascade import CascadeConfig, agreement_matrix, run
    # models ordered cheap, mid, strong; forecasts make 'cheap' the first choice at this penalty
    F = np.array([[.70, .80, .95]] * 3)
    C = np.array([[1., 2., 10.]] * 3)
    Q = np.array([[1., 1., 1.], [0., 0., 1.], [0., 1., 1.]])
    answers = [["x", "x", "x"], ["wrong", "wrong", "x"], ["wrong", "x", "x"]]
    agree = agreement_matrix(answers)
    q, c, escalated = run(F, C, Q, C, agree, CascadeConfig(lam=.2, accept=2.))
    # agree -> cheap answer (cost 1+2); shared mistake -> wrong but cheap; disagree -> escalate (1+2+10)
    assert q.tolist() == [1., 0., 1.] and c.tolist() == [3., 3., 13.] and escalated.tolist() == [False, False, True]
    q, c, _ = run(F, C, Q, C, agree, CascadeConfig(lam=.2, accept=-1.))
    assert c.tolist() == [1., 1., 1.]  # never checks: plain price-penalty rule
    q, c, _ = run(F, C, Q, C, agree, CascadeConfig(lam=0., accept=2.))
    assert c.tolist() == [10., 10., 10.]  # quality-first choice needs no check


def test_cascade_calibration_controls_false_certification():
    from inference_control.routing import cascade
    rng = np.random.default_rng(5)
    p = np.array([.55, .7, .9])
    def draw(n):
        Q = (rng.random((n, 3)) < p).astype(float)
        common = rng.random((n, 3)) < .5  # shared mistakes make wrong answers agree
        answers = [["ok" if Q[i, j] else ("common" if common[i, j] else f"other{j}") for j in range(3)] for i in range(n)]
        F = np.clip(p + rng.normal(0, .05, (n, 3)), 0, 1)
        return F, np.broadcast_to([1., 2., 10.], (n, 3)).copy(), Q, np.broadcast_to([1., 2., 10.], (n, 3)).copy(), \
            cascade.agreement_matrix(answers)
    configs = cascade.default_configs((0., .02, .05, .1, .3), accepts=(-1., .6, 2.))
    population = draw(150_000)
    truth = {cfg: float(np.nanmean(cascade.run(*population, cfg)[0])) for cfg in configs}
    ordered = cascade.order_configs(configs, *draw(3000))
    target, risk, trials, false = .78, .1, 200, 0
    assert any(v < target for v in truth.values()) and any(v >= target for v in truth.values())
    for _ in range(trials):
        result = cascade.calibrate(ordered, *draw(400), target=target, risk=risk, minimum_requests=20)
        if result.selected is not None and truth[result.selected] < target:
            false += 1
    assert false / trials <= risk + 3 * math.sqrt(risk * (1 - risk) / trials)


def test_compare_reports_cascade_with_answers():
    from inference_control.benchmarks.zoo import zoo_answers
    data = make_zoo(500, seed=9)
    report = run_compare(data, seeds=(1,), targets=(.6, 1.), knn_targets=(.5,), thresholds=(.5,), local_targets=(),
                         answers=zoo_answers(data))
    assert "slm-cascade" in report["aggregate"]
    routers = {c["router"] for c in report["pooled_guarantee"]["checks"]}
    assert routers == {"slm-router", "slm-cascade"}


# ----- majority votes --------------------------------------------------------------------------
def _vote_case(answers, quality, config, *, forecasts=(.5, .6, .8, .9), costs=(1., 2., 4., 10.), realized=None):
    """One request over endpoints 0..3 (cheap to strong); returns the cascade outcome."""
    from inference_control.routing.cascade import agreement_matrix, outcome
    F, C = np.array([forecasts], dtype=float), np.array([costs], dtype=float)
    R = C if realized is None else np.array([realized], dtype=float)
    return outcome(F, C, np.array([quality], dtype=float), R, agreement_matrix([answers]), config)


def _cfg(lam, k, q, *, sequential=False, accept=2.):
    from inference_control.routing.cascade import CascadeConfig
    return CascadeConfig(lam, accept, k, q, sequential)


Q4 = [0., 1., .75, .25]  # distinct qualities show which voter's answer is served


def test_vote_serves_the_plurality_answer_with_its_best_ranked_voters_quality():
    # lam=.05 ranks endpoints 2, 1, 0, 3 (scores .60, .50, .45, .40); final = 3
    for sequential in (False, True):
        r = _vote_case(["a", "b", "a", "z"], Q4, _cfg(.05, 3, 2, sequential=sequential))
        # voters 2 ('a'), 1 ('b'), 0 ('a'): 'a' has two votes; its best-ranked voter is endpoint 2
        assert r.quality[0] == .75 and r.cost[0] == 4 + 2 + 1 and r.calls[0] == 3
        assert r.consensus[0] and not r.escalated[0] and r.checked[0]
    # the outvoted answer's quality (endpoint 1) and the final model's are never used
    r = _vote_case(["x", "x", "y", "y"], Q4, _cfg(.05, 3, 2))
    assert r.quality[0] == 1. and r.consensus[0]  # 'x' from endpoints 1 and 0; best ranked is endpoint 1


def test_parallel_votes_pay_every_voter_and_sequential_votes_stop_at_quorum():
    answers = ["q", "x", "x", "d"]  # voters 2 ('x'), 1 ('x'), 0 ('q')
    parallel = _vote_case(answers, Q4, _cfg(.05, 3, 2))
    sequential = _vote_case(answers, Q4, _cfg(.05, 3, 2, sequential=True))
    assert parallel.cost[0] == 7 and parallel.calls[0] == 3
    assert sequential.cost[0] == 6 and sequential.calls[0] == 2  # endpoints 2 and 1 agree: stop
    assert parallel.quality[0] == sequential.quality[0] == .75


def test_vote_quorum_miss_escalates_to_final():
    # all three voters differ -> escalate; final (endpoint 3) is not a voter, so it is called as well
    for sequential in (False, True):
        r = _vote_case(["a", "b", "c", "d"], Q4, _cfg(.05, 3, 2, sequential=sequential))
        assert r.quality[0] == .25 and r.cost[0] == 4 + 2 + 1 + 10 and r.calls[0] == 4
        assert r.escalated[0] and not r.consensus[0]
    # quorum 3: two matching answers are not enough; sequential stops once three can no longer agree
    r = _vote_case(["x", "x", "y", "d"], Q4, _cfg(.05, 3, 3))
    assert r.escalated[0] and r.cost[0] == 4 + 2 + 1 + 10 and r.calls[0] == 4
    r = _vote_case(["x", "x", "y", "d"], Q4, _cfg(.05, 3, 3, sequential=True))
    assert r.escalated[0] and r.cost[0] == 4 + 2 + 10 and r.calls[0] == 3
    r = _vote_case(["x", "x", "x", "d"], Q4, _cfg(.05, 3, 3))
    assert r.consensus[0] and r.quality[0] == .75 and r.cost[0] == 7


def test_vote_without_comparable_answers_escalates_instead_of_guessing():
    for answers in ([None, "", None, None], [None, None, "a", None], ["  ", "...", None, "z"]):
        for sequential in (False, True):
            r = _vote_case(answers, Q4, _cfg(.05, 3, 2, sequential=sequential))
            assert r.escalated[0] and not r.consensus[0] and r.quality[0] == .25  # one lone answer is no quorum
    r = _vote_case([None, "", None, None], Q4, _cfg(.05, 3, 2))
    assert r.cost[0] == 7 + 10 and r.calls[0] == 4
    r = _vote_case([None, "", None, None], Q4, _cfg(.05, 3, 2, sequential=True))
    assert r.cost[0] == 4 + 2 + 10 and r.calls[0] == 3  # two missing answers leave no way to reach two votes


def test_vote_when_final_is_already_a_voter_adds_no_call():
    # lam=0 ranks by forecast: voters 3, 2, 1 and final = first = 3; the vote is still held
    r = _vote_case(["x", "c", "b", "a"], Q4, _cfg(0., 3, 2))
    assert r.escalated[0] and r.quality[0] == .25 and r.cost[0] == 10 + 4 + 2 and r.calls[0] == 3
    r = _vote_case(["x", "b", "b", "a"], Q4, _cfg(0., 3, 2))  # endpoints 2 and 1 outvote the best model
    assert r.consensus[0] and r.quality[0] == .75 and r.cost[0] == 16
    r = _vote_case(["x", "c", "b", "a"], Q4, _cfg(0., 3, 3, sequential=True))
    assert r.escalated[0] and r.quality[0] == .25 and r.cost[0] == 10 + 4 and r.calls[0] == 2  # final already called
    # lam=.04 ranks 2, 1, 3, 0: final is the third voter; stopping after two calls means calling it now
    r = _vote_case(["x", "b", "a", "d"], Q4, _cfg(.04, 3, 3, sequential=True))
    assert r.escalated[0] and r.cost[0] == 4 + 2 + 10 and r.calls[0] == 3 and r.quality[0] == .25
    r = _vote_case(["x", "b", "a", "d"], Q4, _cfg(.04, 3, 3))
    assert r.cost[0] == 4 + 2 + 10 and r.calls[0] == 3  # parallel: final was one of the three calls


def test_vote_tie_breaking_is_deterministic():
    answers = ["b", "b", "a", "a"]  # voters 2 ('a'), 1 ('b'), 0 ('b'), 3 ('a'): two answers with two votes each
    parallel = _vote_case(answers, Q4, _cfg(.05, 4, 2))
    assert parallel.quality[0] == .75 and parallel.cost[0] == 17 and parallel.calls[0] == 4  # best-ranked voter's answer
    sequential = _vote_case(answers, Q4, _cfg(.05, 4, 2, sequential=True))
    assert sequential.quality[0] == 1. and sequential.cost[0] == 7 and sequential.calls[0] == 3  # first to reach quorum
    # equal scores go to the earlier position: endpoints 0 and 1 vote, endpoint 2 is never called
    r = _vote_case(["a", "a", "a", "a"], [1., 0., 0., 0.], _cfg(.1, 2, 2), forecasts=(.7, .7, .7, -np.inf),
                   costs=(1., 1., 1., 1.), realized=(1., 2., 4., 8.))
    assert r.quality[0] == 1. and r.cost[0] == 3 and r.calls[0] == 2


def test_vote_is_skipped_when_unneeded_or_impossible():
    r = _vote_case(["a", "a", "a", "a"], Q4, _cfg(.05, 3, 2, accept=.8))  # first's forecast .8 >= accept
    assert not r.checked[0] and r.calls[0] == 1 and r.cost[0] == 4 and r.quality[0] == .75
    assert _vote_case(["a", "a", "a", "a"], Q4, _cfg(.05, 3, 2, accept=.81)).checked[0]
    r = _vote_case(["a", "a", "a", "a"], Q4, _cfg(.05, 3, 2), forecasts=(-np.inf, -np.inf, .8, -np.inf))
    assert not r.checked[0] and r.calls[0] == 1 and r.quality[0] == .75  # a single allowed endpoint cannot vote
    # two allowed endpoints cannot reach a quorum of three: escalate to the final one, already called
    r = _vote_case(["a", "a", "a", "a"], Q4, _cfg(.05, 3, 3), forecasts=(-np.inf, -np.inf, .8, .9))
    assert r.escalated[0] and r.quality[0] == .25 and r.cost[0] == 4 + 10 and r.calls[0] == 2
    r = _vote_case(["a"] * 4, Q4, _cfg(.05, 3, 2), forecasts=(-np.inf,) * 4)
    assert np.isnan(r.quality[0]) and np.isnan(r.cost[0]) and r.calls[0] == 0


def test_vote_diagnostics_count_rates_and_calls():
    from inference_control.routing.cascade import agreement_matrix, diagnostics, outcome
    answers = [["a", "b", "a", "z"], ["q", "x", "x", "d"], ["a", "b", "c", "d"], [None, None, None, None]]
    n = len(answers)
    F, C = np.tile([.5, .6, .8, .9], (n, 1)), np.tile([1., 2., 4., 10.], (n, 1))
    Q = np.tile(Q4, (n, 1))
    config = _cfg(.05, 3, 2, sequential=True)
    result = outcome(F, C, Q, C, agreement_matrix(answers), config)
    assert result.calls.tolist() == [3, 2, 4, 3] and result.cost.tolist() == [7, 6, 17, 16]
    d = diagnostics(result, config)
    assert (d["kind"], d["voters"], d["quorum"]) == ("vote", 3, 2)
    assert d["check_rate"] == 1. and d["vote_rate"] == .5 and d["escalation_rate"] == .5 and d["mean_calls"] == 3.
    plain = diagnostics(outcome(F, C, Q, C, agreement_matrix(answers), _cfg(.05, 0, 0, accept=-1.)), _cfg(.05, 0, 0, accept=-1.))
    assert plain["kind"] == "plain" and plain["mean_calls"] == 1. and plain["vote_rate"] == 0.


def test_two_voter_vote_is_the_agreement_cascade_where_final_differs_from_first():
    from inference_control.routing import cascade
    rng = np.random.default_rng(3)
    n, m = 600, 5
    F = rng.random((n, m)).round(2)
    F[rng.random((n, m)) < .1] = -np.inf
    C = rng.random((n, m)).round(2) + .1
    Q = (rng.random((n, m)) < .6).astype(float)
    answers = [[None if rng.random() < .1 else rng.choice(["a", "b"]) for _ in range(m)] for _ in range(n)]
    agree = cascade.agreement_matrix(answers)
    for lam in (.1, .5):
        first, _, final, _ = cascade.plan(F, C, cascade.CascadeConfig(lam, 2.))
        rows = (first >= 0) & (first != final)
        a = cascade.outcome(F, C, Q, C, agree, cascade.CascadeConfig(lam, 2.))
        parallel = cascade.outcome(F, C, Q, C, agree, cascade.CascadeConfig(lam, 2., 2, 2))
        for x, y in ((a.quality, parallel.quality), (a.cost, parallel.cost), (a.escalated, parallel.escalated),
                     (a.calls, parallel.calls)):
            assert np.array_equal(x[rows], y[rows], equal_nan=True)
        # sequential calling only differs by skipping the second call once the first answer is not comparable
        sequential = cascade.outcome(F, C, Q, C, agree, cascade.CascadeConfig(lam, 2., 2, 2, True))
        comparable = rows & agree[np.arange(n), np.maximum(first, 0), np.maximum(first, 0)]
        for x, y in ((a.quality, sequential.quality), (a.cost, sequential.cost), (a.calls, sequential.calls)):
            assert np.array_equal(x[comparable], y[comparable], equal_nan=True)
        assert np.array_equal(a.quality[rows], sequential.quality[rows]) and np.all(sequential.cost[rows] <= a.cost[rows] + 1e-12)
        assert (sequential.calls[rows] < a.calls[rows]).any()


def test_cascade_config_is_backward_compatible_and_validates_votes():
    from inference_control.routing import cascade
    config = cascade.CascadeConfig(.2, .8)
    assert (config.voters, config.quorum, config.sequential, config.kind) == (0, 0, False, "agree")
    assert cascade.CascadeConfig(.2, -1.).kind == "plain" and cascade.config_dict(config) == {"lam": .2, "accept": .8}
    assert cascade.CascadeConfig(.2, 2., 3).quorum == 2 and cascade.CascadeConfig(.2, 2., 5).quorum == 3  # strict majority
    for bad in ({"voters": 1}, {"voters": 3, "quorum": 1}, {"voters": 3, "quorum": 4}, {"voters": 3, "accept": -1.},
                {"voters": 0, "quorum": 2}, {"voters": 0, "sequential": True}):
        with pytest.raises(ValueError):
            cascade.CascadeConfig(**{"lam": .1, "accept": 2., **bad})
    base = cascade.default_configs((0., .1, .3))
    assert len(base) == 15 and all(c.voters == 0 for c in base)
    assert base == tuple(cascade.CascadeConfig(lam, a) for lam in (0., .1, .3) for a in (-1., .9, .8, .7, 2.))
    grid = tuple(np.linspace(0, 3, 30))
    family = cascade.default_configs(grid, voters=(3, 5))
    votes = [c for c in family if c.voters]
    assert family[:150] == cascade.default_configs(grid) and len(family) == 150 + 32
    assert {(c.voters, c.quorum) for c in votes} == {(3, 2), (3, 3), (5, 3), (5, 4)}
    assert len({c.lam for c in votes}) == 8 and all(c.sequential and c.accept == 2. for c in votes)
    assert cascade.vote_lambdas((0., 1., 2.), 8) == (0., 1., 2.) and cascade.vote_lambdas(range(30), 4)[0] == 0
    assert cascade.vote_lambdas(range(30), 4)[-1] == 29
    assert {(c.voters, c.quorum) for c in cascade.vote_configs((0., 1.), (2, 3), quorums=(2, 3, 4))} == {(2, 2), (3, 2), (3, 3)}


def test_mixed_family_calibration_controls_false_certification():
    from inference_control.routing import cascade
    rng = np.random.default_rng(7)
    p = np.array([.5, .65, .75, .9])
    cost = np.broadcast_to([1., 2., 4., 10.], (1, 4))
    def draw(n):
        Q = (rng.random((n, 4)) < p).astype(float)
        common = rng.random((n, 4)) < .4  # shared mistakes make some wrong answers agree
        pick = rng.integers(0, 3, (n, 4))
        answers = [["ok" if Q[i, j] else ("common" if common[i, j] else f"other{pick[i, j]}") for j in range(4)]
                   for i in range(n)]
        F = np.clip(p + rng.normal(0, .07, (n, 4)), 0, 1)
        c = np.repeat(cost, n, axis=0)
        return F, c, Q, c.copy(), cascade.agreement_matrix(answers)
    configs = cascade.default_configs((0., .02, .05, .1, .3), accepts=(-1., .6, 2.), voters=(3,),
                                      vote_accepts=(2., .6), vote_sequential=(False, True))
    assert {c.kind for c in configs} == {"plain", "agree", "vote"} and len(configs) == 15 + 40
    population = draw(100_000)
    truth = {cfg: float(np.nanmean(cascade.run(*population, cfg)[0])) for cfg in configs}
    ordered = cascade.order_configs(configs, *draw(3000))
    target, risk, trials, false, selected = .86, .1, 250, 0, {}
    assert any(v < target for v in truth.values()) and any(v >= target for v in truth.values())
    assert any(v < target for c, v in truth.items() if c.voters) and any(v >= target for c, v in truth.items() if c.voters)
    for _ in range(trials):
        result = cascade.calibrate(ordered, *draw(150), target=target, risk=risk, minimum_requests=20)
        if result.selected is not None:
            selected[result.selected.kind] = selected.get(result.selected.kind, 0) + 1
            false += truth[result.selected] < target
    assert false / trials <= risk + 3 * math.sqrt(risk * (1 - risk) / trials)
    assert selected.get("vote", 0) > 0 and selected.get("agree", 0) > 0  # the mixed family is really used
    identity = cascade.calibrate(ordered, *draw(150), target=target, risk=risk, minimum_requests=20).identity()
    assert identity["steps"] and all(set(s["config"]) == ({"lam", "accept", "voters", "quorum", "sequential"} if s["config"].get("voters")
                                                          else {"lam", "accept"}) for s in identity["steps"])


def test_compare_reports_vote_fields_and_can_disable_votes(tmp_path):
    from inference_control.benchmarks.compare import write_compare_report
    from inference_control.benchmarks.zoo import zoo_answers
    data = make_zoo(500, seed=9)
    answers = zoo_answers(data)
    options = {"seeds": (1,), "targets": (.6, .9, 1.), "knn_targets": (.5,), "thresholds": (.5,), "local_targets": ()}
    report = run_compare(data, answers=answers, **options)
    points = report["splits"][0]["runs"]["slm-cascade"]
    assert len(points) == 3
    for point in points:
        assert point["kind"] in {"plain", "agree", "vote"} and point["mean_calls"] >= 1.
        assert (point["kind"] == "vote") == (point["voters"] > 0) and point["quorum"] >= (2 if point["voters"] else 0)
        assert {"lam", "accept", "check_rate", "escalation_rate", "vote_rate", "consensus_rate", "calibrated"} <= set(point)
        assert point["kind"] != "vote" or (point["check_rate"] > 0 and point["vote_rate"] <= point["check_rate"])
        assert point["kind"] != "plain" or point["mean_calls"] == 1.
    checks = [c for c in report["pooled_guarantee"]["checks"] if c["router"] == "slm-cascade"]
    assert all({"kind", "voters", "quorum", "mean_calls", "price_penalty", "accept"} <= set(c) for c in checks)
    path = write_compare_report(report, tmp_path)
    markdown = (path.parent / "report.md").read_text(encoding="utf-8")
    assert "## Cascade operating points" in markdown
    agreement_only = run_compare(data, answers=answers, cascade_voters=(), **options)
    assert {p["kind"] for p in agreement_only["splits"][0]["runs"]["slm-cascade"]} <= {"plain", "agree"}


def test_zoo_answers_default_and_correlation_knob():
    from inference_control.benchmarks.zoo import zoo_answers
    data = make_zoo(400, seed=2)
    default = zoo_answers(data)
    assert default == zoo_answers(data, common_share=.5, alternatives=3)
    def wrong_pair_agreement(answers):
        agree = pairs = 0
        for q in data.queries:
            wrong = [answers[(q.request.request_id, e.endpoint_id)] for e in data.endpoints]
            wrong = [a for a in wrong if a != "reference" and not a.startswith("free-form")]
            for i, a in enumerate(wrong):
                for b in wrong[i + 1:]:
                    pairs += 1
                    agree += a == b
        return agree / pairs
    base, correlated, independent = (wrong_pair_agreement(zoo_answers(data, **kw)) for kw in (
        {}, {"common_share": .9}, {"common_share": 0., "alternatives": 20}))
    assert base == pytest.approx(1 / 3, abs=.03) and correlated == pytest.approx(.81 + .01 / 3, abs=.03)
    assert independent == pytest.approx(1 / 20, abs=.02)
    with pytest.raises(ValueError):
        zoo_answers(data, common_share=1.)


def test_llmrouterbench_votes_compare_normalized_stored_predictions():
    from inference_control.routing.cascade import agreement_matrix
    script = _script()
    def record(prediction):
        return SimpleNamespace(prediction=prediction)
    stored = [script.stored_answer(r) for r in (record(" 42."), record("42"), record(float("nan")), record(None), None, record(""))]
    assert stored[2:5] == [None, None, None]
    agree = agreement_matrix([stored])[0]
    assert agree[0, 1] and not agree[0, 2] and not agree[2, 3] and not agree[5, 5]  # NaN, None and blank never agree
    from inference_control.routing.cascade import CascadeConfig, outcome
    F, C = np.array([[.9, .8, .7, .6]]), np.array([[1., 1., 1., 1.]])
    result = outcome(F, C, np.array([[0., 1., 1., 1.]]), C, agreement_matrix([[stored[0], stored[1], stored[2], stored[3]]]),
                     CascadeConfig(0., 2., 3, 2))
    assert result.consensus[0] and result.quality[0] == 0.  # '42' twice: served with the best-ranked voter's score
