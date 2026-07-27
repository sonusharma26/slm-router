from __future__ import annotations
import hashlib
from dataclasses import dataclass
from enum import StrEnum


class TrafficMode(StrEnum):
    SHADOW = "shadow"
    CANARY = "canary"
    ACTIVE = "active"


@dataclass(frozen=True)
class TrafficAssignment:
    request_id: str
    policy_version: str
    mode: TrafficMode
    selected: bool
    propensity: float


class TrafficController:
    def assign(
        self, request_id: str, policy_version: str, mode: TrafficMode, fraction: float
    ) -> TrafficAssignment:
        if not 0 <= fraction <= 1:
            raise ValueError("traffic fraction must be in [0,1]")
        bucket = int(
            hashlib.sha256(f"{request_id}:{policy_version}:{mode}".encode()).hexdigest(), 16
        ) / ((1 << 256) - 1)
        return TrafficAssignment(request_id, policy_version, mode, bucket < fraction, fraction)
