"""Trace record schema + SQLite DDL. The trace store is the metrics source of truth."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RunKind = Literal["oracle", "router_eval", "live"]


class RunTrace(BaseModel):
    """One (item, model) run. NULL routing fields for oracle runs."""

    run_id: str
    item_id: str
    dataset: str
    task_type: str
    model: str
    model_tier: str

    # generation
    raw_output: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    # scoring
    score: float = 0.0
    correct: bool | None = None
    judge_score: float | None = None
    human_score: float | None = None

    # routing (filled for router_eval / live)
    route_decision: str | None = None
    confidence: float | None = None
    difficulty: float | None = None

    # provenance
    run_kind: RunKind = "oracle"
    config_hash: str = ""
    created_at: str = ""
    error: str | None = None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    item_id          TEXT NOT NULL,
    dataset          TEXT NOT NULL,
    task_type        TEXT NOT NULL,
    model            TEXT NOT NULL,
    model_tier       TEXT NOT NULL,
    raw_output       TEXT,
    prompt_tokens    INTEGER,
    completion_tokens INTEGER,
    cost_usd         REAL,
    latency_ms       REAL,
    score            REAL,
    correct          INTEGER,
    judge_score      REAL,
    human_score      REAL,
    route_decision   TEXT,
    confidence       REAL,
    difficulty       REAL,
    run_kind         TEXT NOT NULL,
    config_hash      TEXT,
    created_at       TEXT,
    error            TEXT,
    UNIQUE(item_id, model, config_hash, run_kind)
);
CREATE INDEX IF NOT EXISTS idx_runs_item_model ON runs(item_id, model);
CREATE INDEX IF NOT EXISTS idx_runs_kind_dataset ON runs(run_kind, dataset);
CREATE INDEX IF NOT EXISTS idx_runs_item ON runs(item_id);
"""

# Columns in insertion order (matches RunTrace field order above).
COLUMNS: list[str] = [
    "run_id", "item_id", "dataset", "task_type", "model", "model_tier",
    "raw_output", "prompt_tokens", "completion_tokens", "cost_usd", "latency_ms",
    "score", "correct", "judge_score", "human_score",
    "route_decision", "confidence", "difficulty",
    "run_kind", "config_hash", "created_at", "error",
]
