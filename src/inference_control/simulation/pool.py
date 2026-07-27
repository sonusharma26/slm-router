"""Deterministic changing model-pool simulator (V2-104)."""

from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EndpointBehavior:
    endpoint_id: str
    revision: str
    quality: float
    cost: float
    latency_ms: float
    availability: float = 1.0


@dataclass(frozen=True)
class SimulatedResult:
    endpoint_id: str
    revision: str
    success: bool
    quality: float
    cost: float
    latency_ms: float


class DynamicModelPool:
    def __init__(self, endpoints: list[EndpointBehavior], seed: int = 0):
        self._endpoints = {e.endpoint_id: e for e in endpoints}
        self.seed = seed
        self.epoch = 0

    def snapshot(self) -> tuple[EndpointBehavior, ...]:
        return tuple(sorted(self._endpoints.values(), key=lambda e: e.endpoint_id))

    def inject(self, endpoint_id: str, **changes: float | str) -> None:
        self._endpoints[endpoint_id] = replace(self._endpoints[endpoint_id], **changes)
        self.epoch += 1

    def call(self, endpoint_id: str, request_id: str) -> SimulatedResult:
        e = self._endpoints[endpoint_id]
        material = f"{self.seed}:{self.epoch}:{request_id}:{endpoint_id}:{e.revision}"
        rng = random.Random(int(hashlib.sha256(material.encode()).hexdigest(), 16))
        success = rng.random() < e.availability
        jitter = 0.9 + 0.2 * rng.random()
        quality = float(rng.random() < e.quality) if success else 0.0
        return SimulatedResult(
            e.endpoint_id,
            e.revision,
            success,
            quality,
            e.cost if success else 0,
            e.latency_ms * jitter,
        )
