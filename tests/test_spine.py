"""Smoke tests for the shared spine: config, registry cost math, and trace store.

These run fully offline (no API, no datasets, no heavy ML libs).
"""

from __future__ import annotations

from slm_router.config import AppConfig
from slm_router.models.registry import ModelRegistry
from slm_router.trace import RunTrace, TraceStore
from slm_router.types import Usage


def test_config_loads():
    cfg = AppConfig.from_yaml("configs/config.yaml")
    assert cfg.reward.lambda_cost > 0
    assert "gsm8k" in cfg.datasets
    assert cfg.candidate_models


def test_registry_cost_math():
    reg = ModelRegistry.from_yaml("configs/models.yaml")
    spec = reg.get("openai/gpt-4o")
    assert spec.tier == "frontier"
    # 1M in + 1M out at $2.50/$10.00 -> $12.50
    cost = reg.cost("openai/gpt-4o", Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000))
    assert abs(cost.total_usd - 12.50) < 1e-6


def test_registry_by_tier_sorted():
    reg = ModelRegistry.from_yaml("configs/models.yaml")
    slms = reg.by_tier("SLM")
    assert all(s.tier == "SLM" for s in slms)


def test_trace_store_roundtrip_and_idempotency(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    t = RunTrace(
        run_id="r1", item_id="mmlu:0", dataset="mmlu", task_type="mcq",
        model="phi", model_tier="SLM", score=1.0, correct=True,
        cost_usd=0.001, run_kind="oracle", config_hash="h1",
    )
    store.insert_run(t)
    assert store.has_run("mmlu:0", "phi", "h1", "oracle")
    # re-insert same logical cell -> still one row (UNIQUE constraint)
    store.insert_run(t)
    assert len(store.query(run_kind="oracle")) == 1


def test_oracle_matrix_and_best_model(tmp_path):
    store = TraceStore(tmp_path / "t.db")
    store.insert_many([
        RunTrace(run_id="a", item_id="q1", dataset="d", task_type="mcq",
                 model="slm", model_tier="SLM", score=0.0, cost_usd=0.001,
                 run_kind="oracle", config_hash="h"),
        RunTrace(run_id="b", item_id="q1", dataset="d", task_type="mcq",
                 model="llm", model_tier="frontier", score=1.0, cost_usd=0.05,
                 run_kind="oracle", config_hash="h"),
    ])
    matrix = store.get_oracle_matrix()
    assert set(matrix["q1"]) == {"slm", "llm"}
    assert store.best_model_per_item("score")["q1"] == "llm"
    # cost-aware: slm has score 0 so llm still wins; flip with a tie
    assert store.best_model_per_item("score_per_cost")["q1"] == "llm"
