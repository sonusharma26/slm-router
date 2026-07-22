"""Offline tests for evaluation metrics (ECE, regret, Pareto) on synthetic data."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from slm_router.eval.metrics import pareto as pf
from slm_router.eval.metrics import routing as rt
from slm_router.eval.metrics.calibration import expected_calibration_error
from slm_router.trace import RunTrace


def _t(item, model, score, cost):
    return RunTrace(
        run_id=f"{item}-{model}", item_id=item, dataset="d", task_type="mcq",
        model=model, model_tier="SLM", score=score, cost_usd=cost,
        run_kind="oracle", config_hash="h",
    )


def test_ece_perfect_calibration_is_zero():
    conf = [0.0, 0.0, 1.0, 1.0]
    correct = [False, False, True, True]
    assert expected_calibration_error(conf, correct, n_bins=2) == pytest.approx(0.0, abs=1e-9)


def test_regret_and_oracle():
    matrix = {
        "q1": {"slm": _t("q1", "slm", 0.0, 0.001), "llm": _t("q1", "llm", 1.0, 0.05)},
        "q2": {"slm": _t("q2", "slm", 1.0, 0.001), "llm": _t("q2", "llm", 1.0, 0.05)},
    }
    oracle = rt.oracle_score(matrix)
    assert oracle == pytest.approx(1.0)  # best per item
    singles = rt.single_model_scores(matrix)
    assert singles["slm"] == pytest.approx(0.5)
    assert rt.regret(oracle, singles["slm"]) == pytest.approx(0.5)


def test_pareto_frontier_dominance():
    pts = [
        pf.CQPoint("cheap_bad", 0.001, 0.5),
        pf.CQPoint("mid", 0.01, 0.7),
        pf.CQPoint("expensive_good", 0.05, 0.9),
        pf.CQPoint("dominated", 0.02, 0.6),  # worse than mid on both axes-ish
    ]
    front = {p.label for p in pf.pareto_frontier(pts)}
    assert "dominated" not in front
    assert "expensive_good" in front
