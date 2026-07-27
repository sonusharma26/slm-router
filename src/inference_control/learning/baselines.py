"""Auditable simple routing baselines (V2-204)."""

from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ArmEstimate:
    endpoint_id: str
    quality: float
    cost: float
    eligible: bool = True
    features: tuple[float, ...] = ()


class Policy(Protocol):
    def choose(
        self, request_id: str, arms: list[ArmEstimate], features: tuple[float, ...] = ()
    ) -> tuple[str, float]: ...


class AlwaysEndpoint:
    def __init__(self, endpoint_id: str):
        self.endpoint_id = endpoint_id

    def choose(
        self, request_id: str, arms: list[ArmEstimate], features: tuple[float, ...] = ()
    ) -> tuple[str, float]:
        if not any(a.endpoint_id == self.endpoint_id and a.eligible for a in arms):
            raise ValueError("fixed endpoint ineligible")
        return self.endpoint_id, 1.0


class CheapestEligible:
    def choose(
        self, request_id: str, arms: list[ArmEstimate], features: tuple[float, ...] = ()
    ) -> tuple[str, float]:
        return min(
            (a for a in arms if a.eligible), key=lambda a: (a.cost, a.endpoint_id)
        ).endpoint_id, 1.0


class StaticBest:
    def choose(
        self, request_id: str, arms: list[ArmEstimate], features: tuple[float, ...] = ()
    ) -> tuple[str, float]:
        return max(
            (a for a in arms if a.eligible), key=lambda a: (a.quality, -a.cost, a.endpoint_id)
        ).endpoint_id, 1.0


class RandomPolicy:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def choose(
        self, request_id: str, arms: list[ArmEstimate], features: tuple[float, ...] = ()
    ) -> tuple[str, float]:
        candidates = sorted(a.endpoint_id for a in arms if a.eligible)
        index = int(hashlib.sha256(f"{self.seed}:{request_id}".encode()).hexdigest(), 16) % len(
            candidates
        )
        return candidates[index], 1 / len(candidates)


class KNNPolicy:
    def choose(
        self, request_id: str, arms: list[ArmEstimate], features: tuple[float, ...] = ()
    ) -> tuple[str, float]:
        def distance(a):
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a.features, features)))

        chosen = min(
            (a for a in arms if a.eligible), key=lambda a: (distance(a), -a.quality, a.endpoint_id)
        )
        return chosen.endpoint_id, 1.0


class LogisticPolicy:
    def __init__(
        self, weights: dict[str, tuple[float, ...]], intercepts: dict[str, float] | None = None
    ):
        self.weights = weights
        self.intercepts = intercepts or {}

    def choose(
        self, request_id: str, arms: list[ArmEstimate], features: tuple[float, ...] = ()
    ) -> tuple[str, float]:
        def score(a: ArmEstimate) -> float:
            return self.intercepts.get(a.endpoint_id, 0) + sum(
                w * x for w, x in zip(self.weights[a.endpoint_id], features)
            )

        chosen = max(
            (a for a in arms if a.eligible), key=lambda a: (score(a), -a.cost, a.endpoint_id)
        )
        return chosen.endpoint_id, 1.0


class TunedThreshold:
    def __init__(self, cheap: str, safe: str, threshold: float):
        self.cheap = cheap
        self.safe = safe
        self.threshold = threshold

    def choose(
        self, request_id: str, arms: list[ArmEstimate], features: tuple[float, ...] = ()
    ) -> tuple[str, float]:
        lookup = {a.endpoint_id: a for a in arms if a.eligible}
        chosen = (
            self.cheap
            if self.cheap in lookup and lookup[self.cheap].quality >= self.threshold
            else self.safe
        )
        if chosen not in lookup:
            raise ValueError("threshold endpoints ineligible")
        return chosen, 1.0
