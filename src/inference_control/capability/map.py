from __future__ import annotations
from dataclasses import dataclass
from inference_control.capability.models import DistributionEstimate


@dataclass(frozen=True)
class CapabilityEntry:
    target_id: str
    plan_type: str
    quality: DistributionEstimate | None
    cost: DistributionEstimate | None
    latency: DistributionEstimate | None
    updated_at: str


class LivingCapabilityMap:
    def __init__(self, version: str):
        self.version = version
        self._entries: dict[tuple[str, str], CapabilityEntry] = {}

    def update(self, entry: CapabilityEntry) -> None:
        self._entries[(entry.target_id, entry.plan_type)] = entry

    def get(self, target_id: str, plan_type: str) -> CapabilityEntry | None:
        return self._entries.get((target_id, plan_type))

    def missing(self, targets: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple(t for t in targets if t not in self._entries)
