from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient
from inference_control.adapters import (
    AdapterResponse,
    MockAdapter,
    OpenAICompatibleAdapter,
)
from inference_control.api import ControlPlane, create_app
from inference_control.capability import (
    CalibrationLineage,
    EmpiricalCapabilityModel,
    Observation,
    QuantileCapabilityModel,
)
from inference_control.contracts import (
    EndpointSnapshot,
    OutcomeRecord,
    PolicySpec,
    Prediction,
    RequestContext,
)
from inference_control.drift import (
    DriftType,
    StatisticalMonitor,
    deterministic_change,
    FAULT_SCENARIOS,
)
from inference_control.evaluation import (
    MatrixRow,
    Preregistration,
    SCENARIOS,
    policy_profiles,
    replay_full_information,
)
from inference_control.execution import Executor, RuntimeResult
from inference_control.features import Feature, FeaturePipeline, FeatureStage
from inference_control.learning import (
    ArmEstimate,
    BanditObservation,
    CandidatePolicyArtifact,
    CheapestEligible,
    EvaluationReport,
    RandomPolicy,
    evaluate_ope,
)
from inference_control.ledger import SQLiteLedger
from inference_control.lifecycle import PolicyLifecycle, PolicyState, SequentialStopRule
from inference_control.outcomes import OutcomeStore
from inference_control.planning import (
    CapabilityEstimate,
    Planner,
    cascade,
    parallel,
    verify_escalate,
)
from inference_control.policies import (
    CalibrationRow,
    CertificateStore,
    RiskCertificate,
    calibration_report,
)
from inference_control.probes import (
    AcquisitionTerms,
    Probe,
    ProbeCandidate,
    ProbeCatalog,
    ProbeScheduler,
)
from inference_control.registry import EndpointRegistry, PriceSnapshot


def ep(endpoint_id="local", revision="r1"):
    return EndpointSnapshot(
        endpoint_id=endpoint_id,
        provider="local",
        upstream_model="m",
        revision=revision,
        region="local",
        context_window=1000,
        config_hash="h",
        price_version="p",
        input_price_per_million=1,
        output_price_per_million=2,
        governance=frozenset({"local"}),
    )


def pol():
    return PolicySpec(
        policy_id="p",
        version="1",
        minimum_quality=0.8,
        max_expected_spend=0.01,
        max_absolute_spend=0.02,
        deadline_ms=100,
        data_boundary="local",
        training_outcome_sources=frozenset({"deterministic", "human"}),
        promotion_outcome_sources=frozenset({"deterministic", "human"}),
    )


def req():
    return RequestContext(
        request_id="q",
        application_id="a",
        tenant_policy_id="p",
        input_tokens=10,
        max_output_tokens=10,
    )


def planner():
    return Planner(
        [ep()],
        [
            CapabilityEstimate(
                "local",
                Prediction(mean=0.9, lower=0.85, upper=0.95),
                Prediction(mean=0.001, lower=0, upper=0.002),
                Prediction(mean=50, lower=40, upper=60),
            )
        ],
    )


def test_registry_snapshots_are_immutable():
    registry = EndpointRegistry()
    registry.register(ep())
    registry.register(ep())
    with pytest.raises(ValueError):
        registry.register(ep().model_copy(update={"context_window": 2000}))
    price = PriceSnapshot("p", "local", 1, 2, "now")
    registry.register_price(price)
    with pytest.raises(ValueError):
        registry.register_price(PriceSnapshot("p", "local", 2, 2, "now"))


def test_adapters_are_injectable_without_network():
    response = AdapterResponse("ok", "id", "r", 1, 1, 0, 2, {})
    assert MockAdapter(lambda *_: response).call("e", [], {}).text == "ok"
    assert OpenAICompatibleAdapter.__name__


def test_bounded_compound_plan_validation():
    assert cascade(["a", "b"], [1, 2]).max_spend == 3
    assert verify_escalate("a", "rule", "b", (1, 2)).max_calls == 2
    assert parallel(["a", "b"], "deterministic", [1, 2]).plan_type == "parallel"


def test_feature_stages_cannot_leak():
    query = Feature("length", "1", FeatureStage.QUERY, lambda d: float(d["n"]))
    post = Feature("accepted", "1", FeatureStage.POST_RESPONSE, lambda d: 1)
    assert FeaturePipeline(FeatureStage.QUERY, [query]).transform({"n": 2}) == (2,)
    with pytest.raises(ValueError):
        FeaturePipeline(FeatureStage.QUERY, [post])


def test_baselines_and_full_information_direct_replay():
    rows = [
        MatrixRow("q1", "cheap", 0.8, 0.1, 10),
        MatrixRow("q1", "safe", 1, 0.5, 20),
        MatrixRow("q2", "cheap", 0.7, 0.1, 10),
        MatrixRow("q2", "safe", 1, 0.5, 20),
    ]
    result = replay_full_information(rows, CheapestEligible())
    assert result.mean_quality == pytest.approx(0.75) and result.selections == ("cheap", "cheap")
    arms = [ArmEstimate("a", 1, 1), ArmEstimate("b", 1, 1)]
    assert RandomPolicy(3).choose("q", arms) == RandomPolicy(3).choose("q", arms)


def test_capability_models_are_target_aware_and_missing_is_visible():
    lineage = CalibrationLineage("c", "data", "artifact", "f", "now")
    rows = [Observation("e", "direct", (), q, 0.1, 10 + q) for q in (0.7, 0.9, 1)]
    model = EmpiricalCapabilityModel(lineage)
    model.fit(rows)
    assert model.estimate("e", "direct", "quality").sample_size == 3
    assert model.estimate("missing", "direct", "quality") is None
    quantile = QuantileCapabilityModel(lineage)
    quantile.fit(rows)
    assert quantile.estimate_quantiles("e", "direct", "latency_ms").upper >= 10.9


def test_probe_catalog_and_scheduler_are_explainable_and_bounded():
    catalog = ProbeCatalog()
    probe = Probe("p", "held_out", "hash", "math", True, 0.2)
    catalog.add(probe)
    terms = AcquisitionTerms(1, 0.5, 0.5, 0.2, 0.1, 0.4, 0.2)
    candidates = [
        ProbeCandidate("e", probe, terms),
        ProbeCandidate(
            "f",
            Probe("p2", "canary", "h2", "code", True, 0.9),
            AcquisitionTerms(1, 0.5, 0.5, 0.2, 0.1, 0.4, 0.9),
        ),
    ]
    selected = ProbeScheduler(0.5, 7).schedule(candidates)
    assert (
        len(selected) == 1
        and selected[0].explanation == terms
        and sum(x.expected_cost for x in selected) <= 0.5
    )
    assert catalog.stable_canaries() and catalog.held_out()


def test_certificates_expire_invalidate_and_calibration_cannot_hide_slice():
    now = datetime.now(timezone.utc)
    cert = RiskCertificate(
        "c",
        "p1",
        "e1",
        ("r",),
        "d",
        "a",
        0.05,
        0.05,
        0.05,
        ("math",),
        now - timedelta(seconds=1),
        now + timedelta(hours=1),
        ("exchangeable",),
    )
    store = CertificateStore()
    store.issue(cert)
    assert store.current("c")
    store.invalidate("c", "quality drift")
    assert store.current("c") is None
    report = calibration_report(
        [CalibrationRow("math", 0.9, 0.85, "calibration")], {"math", "code"}, 0.1
    )
    assert {r.slice_id: r.passed for r in report} == {"code": False, "math": True}
    with pytest.raises(ValueError):
        calibration_report([CalibrationRow("math", 0.9, 0.9, "train")], {"math"}, 0.1)


def test_outcome_eligibility_delayed_linkage_and_dispute():
    outcome = OutcomeRecord(
        decision_id="d",
        quality={"success": 1},
        evaluator_type="human",
        evaluator_version="1",
        source_artifact="label",
        label_confidence=1,
        uncertainty=0,
        causal_scope="task",
        training_eligible=False,
        promotion_eligible=False,
    )
    store = OutcomeStore()
    qualified = store.add(outcome, pol())
    assert qualified.promotion_eligible and store.active("d")
    disputed = store.dispute(qualified.outcome_id)
    assert not disputed.training_eligible and disputed.disputed


def test_ope_refuses_bad_support_and_computes_all_estimators():
    bad = [BanditObservation(1, 1, 0.1)]
    with pytest.raises(ValueError):
        evaluate_ope(bad, "ips", min_ess=2)
    rows = [BanditObservation(float(i % 2), 0.5, 0.5, 0.4) for i in range(20)]
    for estimator in ("ips", "snips", "dr", "switch"):
        assert evaluate_ope(rows, estimator).diagnostics.passed


def test_artifact_lineage_and_lifecycle_rollback():
    report = EvaluationReport("bandit", "quality", 0.8, 0.9, 0.81, True, True, True, True)
    a = CandidatePolicyArtifact.build("p2", "data", "features", "code", "config", report)
    assert a == CandidatePolicyArtifact.build("p2", "data", "features", "code", "config", report)
    life = PolicyLifecycle()
    for version in ("p1", "p2"):
        life.create(version)
        for i, state in enumerate(
            (
                PolicyState.OFFLINE_VALIDATED,
                PolicyState.SHADOW,
                PolicyState.CANARY,
                PolicyState.ACTIVE,
            )
        ):
            life.transition(
                version, state, actor="op", reason="gate", idempotency_key=f"{version}-{i}"
            )
    life.transition(
        "p2", PolicyState.ROLLED_BACK, actor="monitor", reason="stop", idempotency_key="rollback"
    )
    assert life.active == "p1" and SequentialStopRule(3, 0.1, 10).stop(3, 10)


def test_drift_minimum_evidence_mapping_and_faults():
    assert deterministic_change(DriftType.PRICE, "e", 1, 2).invalidates_certificates
    early = StatisticalMonitor(DriftType.QUALITY, 0.1, 5).evaluate("e", [1, 1], [0, 0])
    assert not early.actionable
    event = StatisticalMonitor(DriftType.QUALITY, 0.1, 2).evaluate("e", [1, 1], [0, 0])
    assert event.actionable and event.affected_estimates == ("quality",)
    assert len(FAULT_SCENARIOS) == 7


def test_api_persists_decision_before_execution_and_audits(tmp_path):
    class Adapter:
        def call(self, endpoint_id, generation_config):
            return RuntimeResult("ref", "provider", "r1", 10, 10, 0, 1)

    ledger = SQLiteLedger(tmp_path / "api.db")
    control = ControlPlane(planner(), ledger, Executor(Adapter()))
    client = TestClient(create_app(control))
    decision = client.post(
        "/v2/decide",
        json={
            "request": req().model_dump(mode="json"),
            "policy": pol().model_dump(mode="json"),
            "seed": 2,
        },
    )
    assert decision.status_code == 200
    decision_id = decision.json()["decision_id"]
    executed = client.post("/v2/execute", json={"decision_id": decision_id})
    assert executed.status_code == 200
    kinds = [e.event_type for e in ledger.events()]
    assert kinds.index("decision") < kinds.index("execution_started") < kinds.index("execution")
    assert {"endpoint_snapshot", "policy", "static_estimates"} <= set(kinds)
    assert ledger.verify()
    assert client.get(f"/v2/decisions/{decision_id}").status_code == 200


def test_gauntlet_is_versioned_and_profiles_and_preregistration_are_frozen():
    assert len(SCENARIOS) == 8 and set(policy_profiles()) == {
        "cost-capped",
        "latency-critical",
        "privacy-restricted",
    }
    registration = Preregistration.freeze(
        ("adaptation beats frozen",), ("violations",), (), {"alpha": 0.05}
    )
    assert len(registration.frozen_hash) == 64


def test_outcome_adapters_and_traffic_assignment_are_safe():
    from inference_control.lifecycle import TrafficController, TrafficMode
    from inference_control.outcomes import CalibratedJudgeOutcomeAdapter, ProxyOutcomeAdapter

    with pytest.raises(ValueError):
        CalibratedJudgeOutcomeAdapter("j1", 0.5, 0.8)
    proxy = ProxyOutcomeAdapter("proxy1").record("d", {"score": 0.2}, "event", confidence=0.3)
    assert not proxy.training_eligible and not proxy.promotion_eligible
    controller = TrafficController()
    a = controller.assign("q", "p2", TrafficMode.CANARY, 0.1)
    b = controller.assign("q", "p2", TrafficMode.CANARY, 0.1)
    assert a == b and a.propensity == 0.1
