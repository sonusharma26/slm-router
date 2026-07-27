"""CLI test for `slm train-confidence`: fits ConfidencePredictor on oracle
traces and saves a confidence.joblib artifact the serving path can load."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
typer_testing = pytest.importorskip("typer.testing")

from slm_router.eval.datasets.schema import EvalItem, TaskType
from slm_router.trace.models import RunTrace


def _trace(item_id: str, model: str, score: float, run_id: str) -> RunTrace:
    return RunTrace(
        run_id=run_id,
        item_id=item_id,
        dataset="ds",
        task_type="qa",
        model=model,
        model_tier="SLM",
        score=score,
        run_kind="oracle",
        config_hash="deadbeef",
    )


def _synthetic_traces(n_items=40, models=("m0", "m1")):
    rng = np.random.default_rng(0)
    traces = []
    for i in range(n_items):
        for model in models:
            score = float(rng.random())
            traces.append(_trace(f"item{i}", model, score, f"{model}-item{i}"))
    return traces


class _FakeStore:
    def __init__(self, traces):
        self._traces = traces

    def query(self, **filters):
        run_kind = filters.get("run_kind")
        if run_kind is not None:
            return [t for t in self._traces if t.run_kind == run_kind]
        return list(self._traces)


def test_train_confidence_command_fits_and_saves(tmp_path, monkeypatch):
    from slm_router import cli
    from slm_router.config import AppConfig

    cfg = AppConfig.from_yaml("configs/config.yaml")
    cfg.candidate_models = ["m0", "m1"]
    cfg.datasets = ["ds"]
    cfg.paths.models_dir = str(tmp_path)

    traces = _synthetic_traces()
    store = _FakeStore(traces)

    monkeypatch.setattr(
        cli, "_bootstrap", lambda config: (None, cfg, None, store, None)
    )
    monkeypatch.setattr(
        "slm_router.eval.workflows.build_oracle.config_hash",
        lambda _cfg: "deadbeef",
    )

    items = [
        EvalItem(
            item_id=f"item{i}",
            query=f"question {i}",
            dataset="ds",
            task_type=TaskType.OPEN_ENDED,
        )
        for i in range(40)
    ]
    monkeypatch.setattr(
        "slm_router.eval.datasets.load_dataset_items",
        lambda name, split, limit, seed: items,
    )

    runner = typer_testing.CliRunner()
    result = runner.invoke(cli.app, ["train-confidence", "--quality-threshold", "0.5"])

    assert result.exit_code == 0, result.output
    assert "n_examples=" in result.output

    saved_path = tmp_path / "confidence.joblib"
    assert saved_path.exists()

    from slm_router.ml_core.confidence import ConfidencePredictor

    restored = ConfidencePredictor()
    restored.load(saved_path)
    feats = np.zeros(9, dtype=np.float32)
    out = restored.predict_prior(feats)
    assert 0.0 <= out.p_correct <= 1.0
