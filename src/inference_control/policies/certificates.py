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
    policy_hash: str | None = None
    capability_map_version: str | None = None
    endpoint_ids: tuple[str, ...] = ()
    endpoint_snapshot_ids: tuple[str, ...] = ()
    plan_key: str | None = None
    feature_version: str | None = None
    traffic_stratum: str | None = None
    quality_lower: float | None = None
    cost_upper: float | None = None
    latency_upper_ms: float | None = None
    sample_size: int = 0
    calibration_size: int = 0
    request_hash: str | None = None

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
        if certificate.valid_from.tzinfo is None or certificate.expires_at.tzinfo is None:
            raise ValueError("certificate timestamps must be timezone-aware")
        if certificate.expires_at <= certificate.valid_from:
            raise ValueError("certificate expiration must follow issue time")
        if not certificate.assumptions:
            raise ValueError("certificate assumptions required")
        prior = self._versions.get(certificate.certificate_id)
        if prior and prior[-1] == certificate:
            return
        if prior:
            raise ValueError("immutable certificate_id conflict")
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


    def invalidate_dependencies(self, endpoint_id: str, metrics: tuple[str, ...],
                                slice_id: str | None = None, reason: str = "drift",
                                at: datetime | None = None) -> tuple[str, ...]:
        invalidated = []
        for key, versions in list(self._versions.items()):
            cert = versions[-1]
            if endpoint_id not in cert.endpoint_ids or cert.invalidated_reason or (at is not None and cert.valid_from > at):
                continue
            if slice_id and slice_id not in (cert.traffic_slices or ("default",)):
                continue
            if set(metrics) & {"quality", "cost", "latency", "failure", "availability"}:
                self.invalidate(key, reason)
                invalidated.append(key)
        return tuple(invalidated)

    def validate_decision(self, decision, policy, endpoints, at: datetime | None = None) -> tuple[str, ...]:
        from inference_control.util import digest
        from inference_control.capability.conditional import stratum
        from inference_control.contracts import Call
        cert = self.current(decision.certificate_id or "", at)
        if not cert:
            return ("CERTIFICATE_MISSING_EXPIRED_OR_INVALIDATED",)
        reasons = []
        ids = tuple(s.endpoint_id for s in decision.selected_plan.steps if isinstance(s, Call))
        snapshots = tuple(endpoints[e].snapshot_id for e in ids if e in endpoints)
        if cert.policy_hash != digest(policy): reasons.append("CERTIFICATE_POLICY_MISMATCH")
        if cert.request_hash != digest(decision.request): reasons.append("CERTIFICATE_REQUEST_MISMATCH")
        if cert.endpoint_snapshot_ids != snapshots: reasons.append("CERTIFICATE_ENDPOINT_MISMATCH")
        if cert.plan_key != decision.selected_plan.evidence_key: reasons.append("CERTIFICATE_PLAN_MISMATCH")
        if cert.traffic_stratum != stratum(decision.request): reasons.append("CERTIFICATE_SLICE_MISMATCH")
        if cert.feature_version != decision.request.feature_version: reasons.append("CERTIFICATE_FEATURE_MISMATCH")
        if cert.capability_map_version != decision.capability_map_version: reasons.append("CERTIFICATE_MAP_MISMATCH")
        if cert.artifact_hash != decision.evidence_hash: reasons.append("CERTIFICATE_EVIDENCE_MISMATCH")
        if cert.quality_lower is None or cert.quality_lower < policy.minimum_quality:
            reasons.append("CERTIFICATE_QUALITY_MISMATCH")
        if cert.cost_upper is None or cert.cost_upper > policy.max_absolute_spend:
            reasons.append("CERTIFICATE_COST_MISMATCH")
        if cert.latency_upper_ms is None or cert.latency_upper_ms > policy.deadline_ms:
            reasons.append("CERTIFICATE_LATENCY_MISMATCH")
        if cert.quality_risk > policy.quality_risk or cert.latency_risk > policy.latency_risk:
            reasons.append("CERTIFICATE_RISK_MISMATCH")
        return tuple(reasons)

    def export(self) -> list[dict]:
        from inference_control.util import primitive
        return [primitive(versions[-1]) for _, versions in sorted(self._versions.items())]

    def restore(self, rows: list[dict]) -> None:
        for row in rows:
            values = dict(row)
            for field in ("valid_from", "expires_at"):
                values[field] = datetime.fromisoformat(values[field])
            for field in ("endpoint_revisions", "traffic_slices", "assumptions", "endpoint_ids", "endpoint_snapshot_ids"):
                values[field] = tuple(values.get(field, ()))
            cert = RiskCertificate(**values)
            self._versions[cert.certificate_id] = [cert]


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
