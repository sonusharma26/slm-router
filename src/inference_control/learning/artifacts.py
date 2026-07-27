from __future__ import annotations
import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EvaluationReport:
    regime: str
    primary_metric: str
    incumbent_value: float
    candidate_value: float
    lower_bound: float
    worst_slice_passed: bool
    overlap_passed: bool
    drift_clear: bool
    promotable: bool


@dataclass(frozen=True)
class CandidatePolicyArtifact:
    artifact_id: str
    policy_version: str
    data_hash: str
    feature_version: str
    code_version: str
    config_hash: str
    report: EvaluationReport

    @classmethod
    def build(
        cls,
        policy_version: str,
        data_hash: str,
        feature_version: str,
        code_version: str,
        config_hash: str,
        report: EvaluationReport,
    ):
        material = json.dumps(
            {
                "policy_version": policy_version,
                "data_hash": data_hash,
                "feature_version": feature_version,
                "code_version": code_version,
                "config_hash": config_hash,
                "report": asdict(report),
            },
            sort_keys=True,
        )
        return cls(
            hashlib.sha256(material.encode()).hexdigest(),
            policy_version,
            data_hash,
            feature_version,
            code_version,
            config_hash,
            report,
        )
