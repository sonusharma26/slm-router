"""Versioned expiring risk certificates and isolated calibration (V2-402..406)."""

from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timezone


@dataclass(frozen=True)
class RiskCertificate:
    certificate_id: str
    policy_version: str
    estimator_version: str
    endpoint_revisions: tuple[str, ...]
    calibration_dataset_hash: str
    artifact_hash: str
    quality_risk: float
    cost_risk: float
    latency_risk: float
    traffic_slices: tuple[str, ...]
    valid_from: datetime
    expires_at: datetime
    assumptions: tuple[str, ...]
    drift_state: str = "clear"
    invalidated_reason: str | None = None

    def current(self, at: datetime | None = None) -> bool:
        now = at or datetime.now(timezone.utc)
        return (
            self.valid_from <= now < self.expires_at
            and self.drift_state == "clear"
            and self.invalidated_reason is None
        )


class CertificateStore:
    def __init__(self):
        self._versions: dict[str, list[RiskCertificate]] = {}

    def issue(self, certificate: RiskCertificate) -> None:
        if not certificate.assumptions:
            raise ValueError("certificate assumptions required")
        self._versions.setdefault(certificate.certificate_id, []).append(certificate)

    def current(self, certificate_id: str, at: datetime | None = None) -> RiskCertificate | None:
        versions = self._versions.get(certificate_id, [])
        if not versions:
            return None
        latest = versions[-1]
        return latest if latest.current(at) else None

    def invalidate(
        self, certificate_id: str, reason: str, drift_state: str = "actionable"
    ) -> RiskCertificate:
        current = self._versions[certificate_id][-1]
        stale = replace(current, invalidated_reason=reason, drift_state=drift_state)
        self._versions[certificate_id].append(stale)
        return stale


@dataclass(frozen=True)
class CalibrationRow:
    slice_id: str
    predicted: float
    observed: float
    split: str


@dataclass(frozen=True)
class SliceCalibration:
    slice_id: str
    n: int
    mean_predicted: float
    mean_observed: float
    absolute_gap: float
    passed: bool


def calibration_report(
    rows: list[CalibrationRow], required_slices: set[str], max_gap: float
) -> tuple[SliceCalibration, ...]:
    if any(r.split != "calibration" for r in rows):
        raise ValueError("calibration report accepts isolated calibration rows only")
    output = []
    for slice_id in sorted(required_slices):
        selected = [r for r in rows if r.slice_id == slice_id]
        if not selected:
            output.append(SliceCalibration(slice_id, 0, 0, 0, 1, False))
            continue
        pred = sum(r.predicted for r in selected) / len(selected)
        obs = sum(r.observed for r in selected) / len(selected)
        gap = abs(pred - obs)
        output.append(SliceCalibration(slice_id, len(selected), pred, obs, gap, gap <= max_gap))
    return tuple(output)
