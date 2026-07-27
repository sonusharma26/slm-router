"""Stage-separated feature pipelines (V2-203)."""

from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Callable


class FeatureStage(StrEnum):
    QUERY = "query"
    POST_RESPONSE = "post_response"


@dataclass(frozen=True)
class Feature:
    name: str
    version: str
    stage: FeatureStage
    compute: Callable[[dict[str, object]], float]


class FeaturePipeline:
    def __init__(self, stage: FeatureStage, features: list[Feature]):
        if any(f.stage != stage for f in features):
            raise ValueError("feature availability stage violation")
        self.stage = stage
        self.features = tuple(features)

    def transform(self, values: dict[str, object]) -> tuple[float, ...]:
        return tuple(f.compute(values) for f in self.features)
