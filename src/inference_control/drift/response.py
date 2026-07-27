"""Drift-to-certificate/probe response orchestration (V2-602/V2-603)."""

from __future__ import annotations
from dataclasses import dataclass
from inference_control.drift.monitors import DriftEvent
from inference_control.policies import CertificateStore
from inference_control.probes import ProbeCandidate, ProbeScheduler, ScheduledProbe


@dataclass(frozen=True)
class DriftResponse:
    event: DriftEvent
    invalidated_certificates: tuple[str, ...]
    diagnostic_probes: tuple[ScheduledProbe, ...]


class DriftResponseController:
    def __init__(self, certificates: CertificateStore, scheduler: ProbeScheduler):
        self.certificates = certificates
        self.scheduler = scheduler

    def respond(
        self, event: DriftEvent, certificate_ids: list[str], candidates: list[ProbeCandidate]
    ) -> DriftResponse:
        if not event.actionable:
            return DriftResponse(event, (), ())
        invalidated = []
        if event.invalidates_certificates:
            for certificate_id in certificate_ids:
                if self.certificates.current(certificate_id):
                    self.certificates.invalidate(
                        certificate_id, f"{event.drift_type}:{event.target_id}"
                    )
                    invalidated.append(certificate_id)
        probes = self.scheduler.schedule(candidates, strategy="impact")
        return DriftResponse(event, tuple(invalidated), probes)
