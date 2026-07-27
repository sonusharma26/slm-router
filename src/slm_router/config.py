"""YAML-backed research configuration (reward weights, dataset selection, paths)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class RewardConfig(BaseModel):
    correctness_weight: float = 1.0
    lambda_cost: float = 50.0
    beta_latency: float = 0.0001


class PathsConfig(BaseModel):
    data_dir: str = "data"
    results_dir: str = "results"
    traces_dir: str = "data/traces"
    models_dir: str = "models"


class AppConfig(BaseModel):
    reward: RewardConfig = RewardConfig()
    paths: PathsConfig = PathsConfig()
    datasets: list[str] = ["mmlu", "gsm8k"]
    dataset_limit: int = 50
    dataset_seed: int = 13
    router_holdout_fraction: float = 0.2
    minimum_holdout_items: int = 20
    slm_quality_tolerance: float = 0.05
    candidate_models: list[str] = []
    judge_model: str = "openai/gpt-4o"
    judge_temperature: float = 0.0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
