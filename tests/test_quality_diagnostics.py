import math
from datetime import UTC, datetime

import pytest

from inference_control.benchmarks.data import BenchmarkDataset, Query, Truth
from inference_control.benchmarks.diagnostics import diagnose_quality, required_calibration_samples
from inference_control.benchmarks.routers import SLMRouter
from inference_control.benchmarks.runner import default_policy, run_static, split_data, write_report
from inference_control.capability.conditional import ConditionalCapabilityMap, EvidenceObservation
from inference_control.contracts import EndpointSnapshot, RequestContext


def dataset():
    endpoints = [EndpointSnapshot(endpoint_id=name, provider="fixture", upstream_model=name,
        revision="r1", region="local", context_window=8192, config_hash="h", price_version="p1",
        input_price_per_million=1, output_price_per_million=1) for name in ("a", "b")]
    queries, outcomes = [], []
    for split, count in (("train", 10), ("calibration", 5), ("test", 2)):
        for i in range(count):
            req = RequestContext(request_id=f"{split}-{i}", application_id="app", tenant_policy_id="p",
                input_tokens=128, max_output_tokens=64, query_features=(i/count,), feature_version="fixed")
            queries.append(Query(request=req, text="fixture", split=split, group_id=req.request_id))
            for ep in endpoints:
                outcomes.append(Truth(request_id=req.request_id, endpoint_id=ep.endpoint_id,
                    quality=.9 if split == "train" else 1., cost=.0001, latency_ms=20,
                    input_tokens=128, output_tokens=32, evaluator_type="deterministic"))
    return BenchmarkDataset(name="sparse-fixture", evidence_kind="synthetic", queries=queries,
                            outcomes=outcomes, endpoints=endpoints)


def test_sparse_static_report_explains_every_rejected_candidate_and_writes_diagnosis(tmp_path):
    report = run_static(dataset(), resamples=4, sweep=False)
    diagnostic = report["quality_diagnostics"]
    assert report["results"]["slm-router"]["abstention_rate"] == 1.
    assert diagnostic["rejection_counts"] == {"QUALITY_RISK": 4}
    for row in diagnostic["records"]:
        assert row["bound"]["effective_risk"] == .025
        assert row["quality_lower"] == pytest.approx(math.exp(-math.log(80)/5))
        assert row["sample_size_blocks_target_even_with_perfect_scores"]
        assert row["required_samples_at_perfect_mean"] == 11
        assert row["bound"]["training_mean_clamp"] == 0
    write_report(report, tmp_path)
    assert "Quality bound diagnosis" in (tmp_path/"report.md").read_text()


def test_heldout_scores_change_only_evaluator_errors_and_do_not_mutate_fitted_map():
    data = dataset()
    router = SLMRouter(data.endpoints, min_samples=8, at=datetime.now(UTC))
    router.fit(*split_data(data, {"train", "calibration"}))
    before = router.map.export(), router.map.version
    original = diagnose_quality(router, data, default_policy())
    changed = data.model_copy(update={"outcomes": [
        r.model_copy(update={"quality": 0.}) if r.request_id.startswith("test-") else r
        for r in data.outcomes]})
    other = diagnose_quality(router, changed, default_policy())
    assert original["records"] == other["records"]
    for endpoint_id in original["endpoints"]:
        a, b = original["endpoints"][endpoint_id], other["endpoints"][endpoint_id]
        assert a["calibration_prediction_error"] == b["calibration_prediction_error"]
        assert a["test_prediction_error"] != b["test_prediction_error"]
    assert before == (router.map.export(), router.map.version)


def test_diagnosis_uses_eligible_search_family_instead_of_entire_pool():
    data = dataset()
    router = SLMRouter(data.endpoints, min_samples=8)
    router.fit(*split_data(data, {"train", "calibration"}))
    policy = default_policy().model_copy(update={"allowed_endpoints": frozenset({"a"})})
    diagnostic = diagnose_quality(router, data, policy)
    assert len(diagnostic["records"]) == 2
    assert all(r["comparisons"] == 1 and r["bound"]["effective_risk"] == .05 for r in diagnostic["records"])


def test_training_clamp_missing_calibration_and_k_limit_are_separate_from_sampling_penalty():
    data = dataset()
    ep = data.endpoints[0]
    at = datetime.now(UTC)
    cmap = ConditionalCapabilityMap(k=128, min_samples=1, quality_method="local-mean-hoeffding")
    for split, count, quality in (("train", 1, .1), ("calibration", 100, 1.)):
        for i in range(count):
            cmap.add(EvidenceObservation(sample_id=f"{split}-{i}",
                request=data.queries[0].request.model_copy(update={"request_id": f"{split}-{i}"}),
                target_id=ep.endpoint_id, endpoint_ids=(ep.endpoint_id,), endpoint_revisions=(ep.capability_revision,),
                quality=quality, cost=.0001, latency_ms=20, split=split,
                evaluator_type="deterministic", evaluator_version="fixture", observed_at=at))
    estimate = cmap.estimate(data.queries[0].request, ep.endpoint_id, (ep,), at=at)
    assert estimate.quality.lower == .1
    assert estimate.lineage.quality_diagnostics.training_mean_clamp > .7
    no_calibration = ConditionalCapabilityMap(min_samples=1)
    no_calibration.add_many([r for r in cmap.rows.values() if r.split == "train"])
    missing = no_calibration.estimate(data.queries[0].request, ep.endpoint_id, (ep,), at=at)
    assert missing.quality.lower == 0
    assert missing.lineage.quality_diagnostics.hoeffding_radius is None
    assert required_calibration_samples(.7, .65, .025) == 877
    assert required_calibration_samples(.65, .65, .025) is None
    assert required_calibration_samples(0., 0., .025) == 0
