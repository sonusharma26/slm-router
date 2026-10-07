import math
from datetime import UTC, datetime
from decimal import Decimal, localcontext

import pytest

from inference_control.benchmarks.routers import Baseline, SLMRouter
from inference_control.benchmarks.runner import default_policy, evaluate_router, split_data
from inference_control.capability.conditional import ConditionalCapabilityMap, quality_interval
from tests.test_quality_diagnostics import dataset


def test_paired_kl_keeps_total_risk_and_conservative_boundaries():
    with localcontext() as context:
        context.prec = 60
        lower, upper = quality_interval(1., 1, .025)
        assert Decimal(lower) < Decimal("0.0125")
        assert upper == 1.
        for observed in (0., .1, .5, .65, 1.):
            for n in (2, 12, 100):
                lower, upper = quality_interval(observed, n, .025)
                q = Decimal(observed)
                threshold = (Decimal(2)/Decimal.from_float(.025)).ln()/n
                for bound in (lower, upper):
                    p = Decimal(bound)
                    if p in (0, 1):
                        continue
                    relative_entropy = Decimal(0)
                    if q:
                        relative_entropy += q*(q/p).ln()
                    if q < 1:
                        relative_entropy += (1-q)*((1-q)/(1-p)).ln()
                    assert relative_entropy >= threshold
    for n in (2, 12, 30):
        intervals = [quality_interval(k/n, n, .025) for k in range(n+1)]
        for mu in (.01, .1, .3, .65, .9, .99):
            failure = sum(math.comb(n, k)*mu**k*(1-mu)**(n-k)
                          for k, (lower, upper) in enumerate(intervals) if not lower <= mu <= upper)
            assert failure <= .025
    assert quality_interval(1., 12, .025)[0] > .65
    assert quality_interval(1., 12, .025, "local-mean-hoeffding")[0] < .65


def test_estimator_config_roundtrip_and_legacy_restore_do_not_reinterpret_bounds():
    data = dataset()
    at = datetime.now(UTC)
    router = SLMRouter(data.endpoints, min_samples=1, at=at,
                       quality_predictors={"a": "cohort-mean"})
    router.fit(*split_data(data, {"train", "calibration"}))
    ep, query = data.endpoints[0], data.queries[-1]
    original = router.map.estimate(query.request, ep.endpoint_id, (ep,), at=at)
    restored = ConditionalCapabilityMap.restore(router.map.export())
    assert restored.version == router.map.version
    assert restored.estimate(query.request, ep.endpoint_id, (ep,), at=at) == original
    checkpoint = router.map.export()
    checkpoint.pop("quality_method")
    checkpoint.pop("quality_predictors")
    legacy = ConditionalCapabilityMap.restore(checkpoint)
    assert legacy.quality_method == "local-mean-hoeffding"
    assert legacy.version != router.map.version
    assert legacy.estimate(query.request, ep.endpoint_id, (ep,), at=at).quality.lower == pytest.approx(
        1-math.sqrt(math.log(40)/10))


def test_low_raw_forecast_does_not_cap_kl_and_heldout_labels_do_not_select_routes():
    data = dataset()
    queries, rows = split_data(data, {"train", "calibration"})
    rows = [r.model_copy(update={"quality": .1}) if r.request_id.startswith("train-") else r for r in rows]
    at = datetime.now(UTC)
    router = SLMRouter(data.endpoints, min_samples=1, at=at)
    router.fit(queries, rows)
    query, ep = data.queries[-1], data.endpoints[0]
    estimate = router.map.estimate(query.request, ep.endpoint_id, (ep,), at=at)
    assert estimate.quality.lower > .1
    assert estimate.quality.mean == estimate.quality.lower
    assert estimate.lineage.quality_diagnostics.raw_prediction == .1
    assert estimate.lineage.quality_diagnostics.training_mean_clamp == 0
    before = router.map.version
    original = evaluate_router(router, data, default_policy())
    changed = data.model_copy(update={"outcomes": [
        r.model_copy(update={"quality": 0.}) if r.request_id.startswith("test-") else r for r in data.outcomes]})
    after = evaluate_router(router, changed, default_policy())
    assert router.map.version == before
    assert [r["metadata"] for r in after] == [r["metadata"] for r in original]
    assert [r["endpoint_id"] for r in after] == [r["endpoint_id"] for r in original]


def test_archived_responses_over_output_cap_are_reported_as_violations():
    data = dataset()
    changed = data.model_copy(update={"outcomes": [
        r.model_copy(update={"output_tokens":65}) if r.request_id.startswith("test-") else r for r in data.outcomes]})
    router = Baseline("cheapest")
    router.fit(*split_data(changed, {"train"}))
    rows = evaluate_router(router, changed, default_policy())
    assert len(rows) == 2
    assert all(row["output_budget_mismatch"] and row["constraint_violation"] for row in rows)
