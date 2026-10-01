"""Bounded YAML/JSON policy compiler; unknown fields fail closed."""
from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from inference_control.contracts import PolicySpec, EndpointSnapshot, RequestContext, InfeasibleBehavior
from inference_control.policies.validator import eligibility_reasons
from inference_control.util import digest


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class Cost(Strict):
    expected_max: float = Field(ge=0)
    absolute_max: float = Field(ge=0)

class Latency(Strict):
    p95_max_ms: float = Field(gt=0)

class Privacy(Strict):
    providers: list[str] = Field(default_factory=list)
    endpoints: list[str] = Field(default_factory=list)
    boundary: str = "any"

class Plans(Strict):
    allowed: list[Literal["direct", "cascade", "verify_escalate", "parallel", "abstain"]] = ["direct"]
    max_calls: int = Field(default=1, ge=0, le=8)
    verifier: str | None = None
    selector: str | None = None
    max_candidates: int = Field(default=4096, ge=1, le=100000)

class Evidence(Strict):
    require_certificate: bool = True
    max_age_seconds: int = Field(default=86400, gt=0)
    min_samples: int = Field(default=30, ge=1)
    certificate_ttl_seconds: int = Field(default=3600, gt=0)
    quality_risk: float = Field(default=0.05, gt=0, lt=1)
    latency_risk: float = Field(default=0.05, gt=0, lt=1)

class PolicyDocument(Strict):
    policy_id: str = "default"
    version: str | None = None
    minimum_quality: float = Field(ge=0, le=1)
    cost: Cost
    latency: Latency
    privacy: Privacy = Field(default_factory=Privacy)
    plans: Plans = Field(default_factory=Plans)
    evidence: Evidence = Field(default_factory=Evidence)
    required_capabilities: list[str] = Field(default_factory=list)
    required_traffic_slices: list[str] = Field(default_factory=list)
    maximum_failure_probability: float = Field(default=1, ge=0, le=1)
    objective: Literal["cost", "quality", "latency"] = "cost"
    on_infeasible: Literal["abstain", "request_relaxation", "safe_fallback", "least_shortfall_uncertified"] = "abstain"
    safe_fallback_endpoint: str | None = None


def compile_policy(value: dict | str | Path) -> PolicySpec:
    if isinstance(value, (str, Path)):
        value = yaml.safe_load(Path(value).read_text(encoding="utf-8"))
    doc = PolicyDocument.model_validate(value)
    allowed = set(doc.plans.allowed)
    if not allowed:
        raise ValueError("at least one plan type is required")
    if allowed - {"abstain"} and doc.plans.max_calls == 0:
        raise ValueError("non-abstain plans require calls")
    if allowed & {"cascade", "parallel", "verify_escalate"} and doc.plans.max_calls < 2:
        raise ValueError("compound plans require at least two calls")
    if "verify_escalate" in allowed and not doc.plans.verifier:
        raise ValueError("verify_escalate requires an explicitly registered local verifier")
    if "parallel" in allowed and not doc.plans.selector:
        raise ValueError("parallel requires an explicitly registered local selector")
    if doc.evidence.require_certificate and doc.on_infeasible == "least_shortfall_uncertified":
        raise ValueError("required certificates cannot be bypassed by fallback")
    return PolicySpec(
        policy_id=doc.policy_id, version=doc.version or digest(doc)[:16],
        minimum_quality=doc.minimum_quality, quality_risk=doc.evidence.quality_risk,
        latency_risk=doc.evidence.latency_risk, max_expected_spend=doc.cost.expected_max,
        max_absolute_spend=doc.cost.absolute_max, deadline_ms=doc.latency.p95_max_ms,
        data_boundary=doc.privacy.boundary, allowed_providers=frozenset(doc.privacy.providers),
        allowed_endpoints=frozenset(doc.privacy.endpoints), max_calls=doc.plans.max_calls,
        permitted_plan_types=frozenset(allowed), verifier=doc.plans.verifier, selector=doc.plans.selector,
        max_plan_candidates=doc.plans.max_candidates, required_capabilities=frozenset(doc.required_capabilities),
        required_traffic_slices=frozenset(doc.required_traffic_slices),
        maximum_failure_probability=doc.maximum_failure_probability, objective=doc.objective,
        require_certificate=doc.evidence.require_certificate,
        minimum_evidence_samples=doc.evidence.min_samples,
        evidence_max_age_seconds=doc.evidence.max_age_seconds,
        certificate_ttl_seconds=doc.evidence.certificate_ttl_seconds,
        infeasible_behavior=InfeasibleBehavior(doc.on_infeasible), safe_fallback_endpoint=doc.safe_fallback_endpoint)


def validate_pool(policy: PolicySpec, endpoints: list[EndpointSnapshot], request: RequestContext | None = None) -> dict:
    request = request or RequestContext(application_id="validation", tenant_policy_id=policy.policy_id,
                                        max_output_tokens=1)
    eligible = [e.endpoint_id for e in endpoints if not eligibility_reasons(e, request, policy)]
    warnings = ["Statistical feasibility needs current request-conditioned calibration evidence."]
    if not eligible and policy.permitted_plan_types != frozenset({"abstain"}):
        raise ValueError("no endpoint satisfies policy eligibility")
    if policy.safe_fallback_endpoint and policy.safe_fallback_endpoint not in eligible:
        raise ValueError("safe fallback violates endpoint eligibility")
    return {"valid": True, "policy_hash": digest(policy), "eligible_endpoints": eligible, "warnings": warnings}
