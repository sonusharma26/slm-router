"""Explainable, hard-budget active probe scheduling (V2-304/V2-305)."""

from __future__ import annotations
import hashlib
from dataclasses import dataclass
from inference_control.probes.catalog import Probe


@dataclass(frozen=True)
class AcquisitionTerms:
    traffic_mass: float
    uncertainty: float
    boundary_sensitivity: float
    staleness: float
    drift: float
    value_of_information: float
    cost: float

    @property
    def score(self) -> float:
        return (
            self.traffic_mass
            * (
                self.uncertainty
                + self.boundary_sensitivity
                + self.staleness
                + self.drift
                + self.value_of_information
            )
            / max(self.cost, 1e-12)
        )


@dataclass(frozen=True)
class ProbeCandidate:
    endpoint_id: str
    probe: Probe
    terms: AcquisitionTerms


@dataclass(frozen=True)
class ScheduledProbe:
    endpoint_id: str
    probe_id: str
    expected_cost: float
    score: float
    explanation: AcquisitionTerms


class ProbeScheduler:
    def __init__(self, budget: float, seed: int = 0):
        self.budget = budget
        self.seed = seed

    def schedule(
        self, candidates: list[ProbeCandidate], strategy: str = "impact", minimum_per_slice: int = 0
    ) -> tuple[ScheduledProbe, ...]:
        if strategy not in {"impact", "uncertainty", "random"}:
            raise ValueError("unknown acquisition strategy")

        def rank(c: ProbeCandidate):
            if strategy == "impact":
                primary = c.terms.score
            elif strategy == "uncertainty":
                primary = c.terms.uncertainty / max(c.terms.cost, 1e-12)
            else:
                primary = int(
                    hashlib.sha256(
                        f"{self.seed}:{c.endpoint_id}:{c.probe.probe_id}".encode()
                    ).hexdigest(),
                    16,
                )
            return (-primary, c.endpoint_id, c.probe.probe_id)

        ordered = sorted(candidates, key=rank)
        chosen = []
        spent = 0.0
        slice_counts: dict[str, int] = {}
        # Quotas are prioritized, while every selection still respects the cap.
        if minimum_per_slice:
            ordered = sorted(
                ordered,
                key=lambda c: (
                    slice_counts.get(c.probe.traffic_slice, 0) >= minimum_per_slice,
                    rank(c),
                ),
            )
        for c in ordered:
            if spent + c.terms.cost > self.budget + 1e-12:
                continue
            chosen.append(
                ScheduledProbe(
                    c.endpoint_id, c.probe.probe_id, c.terms.cost, c.terms.score, c.terms
                )
            )
            spent += c.terms.cost
            slice_counts[c.probe.traffic_slice] = slice_counts.get(c.probe.traffic_slice, 0) + 1
        assert sum(c.expected_cost for c in chosen) <= self.budget + 1e-12
        return tuple(chosen)
