from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    probe_id: str
    source: str
    artifact_hash: str
    traffic_slice: str
    stable: bool
    expected_cost: float


class ProbeCatalog:
    def __init__(self):
        self._probes: dict[str, Probe] = {}

    def add(self, probe: Probe) -> None:
        prior = self._probes.get(probe.probe_id)
        if prior and prior != probe:
            raise ValueError("immutable probe conflict")
        self._probes[probe.probe_id] = probe

    def stable_canaries(self) -> tuple[Probe, ...]:
        return tuple(
            sorted((p for p in self._probes.values() if p.stable), key=lambda p: p.probe_id)
        )

    def held_out(self) -> tuple[Probe, ...]:
        return tuple(
            sorted(
                (p for p in self._probes.values() if p.source == "held_out"),
                key=lambda p: p.probe_id,
            )
        )
