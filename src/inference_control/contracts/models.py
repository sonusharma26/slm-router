from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


def now() -> datetime:
    return datetime.now(timezone.utc)


class FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, validate_default=True)
    schema_version: Literal["2.0"] = "2.0"

    @field_validator("*", mode="after", check_fields=False)
    @classmethod
    def freeze_mappings(cls, value):
        from inference_control.util import freeze
        return freeze(value) if isinstance(value, dict) else value


class EndpointSnapshot(FrozenContract):
    endpoint_id: str
    provider: str
    upstream_model: str
    revision: str
    region: str
    deployment: str = "default"
    modalities: frozenset[str] = frozenset({"text"})
    capabilities: frozenset[str] = frozenset()
    context_window: int = Field(gt=0)
    config_hash: str
    price_version: str
    input_price_per_million: float = Field(ge=0)
    output_price_per_million: float = Field(ge=0)
    available: bool = True
    healthy: bool = True
    governance: frozenset[str] = frozenset()
    inference_config: dict[str, object] = Field(default_factory=dict)
    # Must be an operator-verified upper bound on provider token overhead.
    input_token_overhead: int = Field(default=0, ge=0)

    @field_validator("inference_config")
    @classmethod
    def validate_inference_config(cls, value):
        allowed = {"temperature", "top_p", "frequency_penalty", "presence_penalty", "seed",
                   "stop", "reasoning_effort", "verbosity", "logprobs", "top_logprobs"}
        if set(value) - allowed:
            raise ValueError("inference_config contains unsupported fields; credentials/transport/retries/output budgets are not snapshot generation settings")
        from inference_control.util import freeze
        return freeze(value)

    @property
    def snapshot_id(self) -> str:
        from inference_control.util import digest
        return digest(self)

    @property
    def capability_revision(self) -> str:
        from inference_control.util import digest
        return digest({k: v for k, v in self.model_dump().items() if k not in {
            "price_version", "input_price_per_million", "output_price_per_million",
            "available", "healthy"}})


class RequestContext(FrozenContract):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    application_id: str
    tenant_policy_id: str
    modality: str = "text"
    input_tokens: int = Field(default=0, ge=0)
    max_output_tokens: int = Field(default=256, ge=0)
    feature_version: str = "query-v1"
    task_hint: str | None = None
    privacy_classification: str = "public"
    retention_policy: str = "no_raw_input"
    traffic_slices: frozenset[str] = frozenset()
    query_features: tuple[float, ...] = ()
    expected_output_tokens: int | None = Field(default=None, ge=0)
    required_capabilities: frozenset[str] = frozenset()
    tags: frozenset[str] = frozenset()
    structured_output_schema_hash: str | None = None
    tool_schema_hash: str | None = None

    @model_validator(mode="after")
    def request_bounds(self) -> RequestContext:
        import math
        if any(not math.isfinite(x) for x in self.query_features):
            raise ValueError("features must be finite")
        if self.expected_output_tokens is not None and self.expected_output_tokens > self.max_output_tokens:
            raise ValueError("expected output exceeds maximum output")
        return self


class InfeasibleBehavior(StrEnum):
    ABSTAIN = "abstain"
    RELAX = "request_relaxation"
    SAFE_FALLBACK = "safe_fallback"
    UNCERTIFIED = "least_shortfall_uncertified"


class PolicySpec(FrozenContract):
    policy_id: str
    version: str
    minimum_quality: float = Field(ge=0, le=1)
    quality_risk: float = Field(default=0.05, gt=0, lt=1)
    cost_risk: float = Field(default=0.05, gt=0, lt=1)
    latency_risk: float = Field(default=0.05, gt=0, lt=1)
    max_expected_spend: float = Field(ge=0)
    max_absolute_spend: float = Field(ge=0)
    deadline_ms: float = Field(gt=0)
    data_boundary: str = "any"
    allowed_providers: frozenset[str] = frozenset()
    allowed_endpoints: frozenset[str] = frozenset()
    max_calls: int = Field(default=1, ge=0)
    permitted_plan_types: frozenset[str] = frozenset({"direct", "abstain"})
    required_capabilities: frozenset[str] = frozenset()
    infeasible_behavior: InfeasibleBehavior = InfeasibleBehavior.ABSTAIN
    safe_fallback_endpoint: str | None = None
    exploration_budget: float = Field(default=0, ge=0, le=1)
    training_outcome_sources: frozenset[str] = frozenset({"deterministic", "application", "human"})
    promotion_outcome_sources: frozenset[str] = frozenset({"deterministic", "application", "human"})
    require_certificate: bool = False
    evidence_max_age_seconds: int = Field(default=86400, gt=0)
    minimum_evidence_samples: int = Field(default=30, ge=1)
    certificate_ttl_seconds: int = Field(default=3600, gt=0)
    maximum_failure_probability: float = Field(default=1.0, ge=0, le=1)
    verifier: str | None = None
    selector: str | None = None
    max_plan_candidates: int = Field(default=4096, ge=1, le=100000)
    objective: Literal["cost", "quality", "latency"] = "cost"
    required_traffic_slices: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def bounds(self) -> PolicySpec:
        allowed = {"direct", "cascade", "verify_escalate", "parallel", "abstain"}
        if not self.permitted_plan_types or not self.permitted_plan_types <= allowed:
            raise ValueError("unsupported or empty permitted_plan_types")
        if self.require_certificate and self.infeasible_behavior == InfeasibleBehavior.UNCERTIFIED:
            raise ValueError("certified policy cannot allow uncertified fallback")
        if self.max_expected_spend > self.max_absolute_spend:
            raise ValueError("max_expected_spend cannot exceed max_absolute_spend")
        if (
            self.infeasible_behavior == InfeasibleBehavior.SAFE_FALLBACK
            and not self.safe_fallback_endpoint
        ):
            raise ValueError("safe_fallback requires safe_fallback_endpoint")
        return self


class Call(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, validate_default=True)
    operator: Literal["call"] = "call"
    endpoint_id: str
    generation_config: dict[str, object] = Field(default_factory=dict)
    max_output_tokens: int = Field(default=256, ge=0)


class Verify(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, validate_default=True)
    operator: Literal["verify"] = "verify"
    verifier: str
    acceptance_rule: str


class Select(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, validate_default=True)
    operator: Literal["select"] = "select"
    selector: str


class Abstain(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, validate_default=True)
    operator: Literal["abstain"] = "abstain"
    reason: str


PlanStep = Annotated[Union[Call, Verify, Select, Abstain], Field(discriminator="operator")]


class ExecutionPlan(FrozenContract):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    plan_type: Literal["direct", "cascade", "verify_escalate", "parallel", "abstain"]
    steps: tuple[PlanStep, ...]
    max_calls: int = Field(ge=0)
    max_spend: float = Field(ge=0)

    @property
    def evidence_key(self) -> str:
        from inference_control.util import digest
        if self.plan_type == "direct":
            return self.steps[0].endpoint_id
        # Budget and random plan_id are not a statistical target's identity.
        return "plan:" + digest({"type": self.plan_type, "steps": [
            {"operator": s.operator, "endpoint_id": s.endpoint_id} if isinstance(s, Call)
            else s.model_dump() for s in self.steps]})

    @model_validator(mode="after")
    def shape(self) -> ExecutionPlan:
        if not self.steps:
            raise ValueError("plan cannot be empty")
        calls = sum(isinstance(step, Call) for step in self.steps)
        if calls > self.max_calls:
            raise ValueError("steps exceed declared max_calls")
        if self.plan_type == "direct" and (len(self.steps) != 1 or calls != 1):
            raise ValueError("direct plans contain exactly one call")
        if self.plan_type == "abstain" and not all(isinstance(s, Abstain) for s in self.steps):
            raise ValueError("abstain plans may not call endpoints")
        if self.plan_type == "cascade" and calls < 2:
            raise ValueError("cascade plans require at least two calls")
        if self.plan_type == "verify_escalate" and not any(
            isinstance(s, Verify) for s in self.steps
        ):
            raise ValueError("verify-escalate plans require a verifier")
        if self.plan_type == "parallel" and (
            calls < 2 or not any(isinstance(s, Select) for s in self.steps)
        ):
            raise ValueError("parallel plans require multiple calls and a selector")
        if self.plan_type == "verify_escalate" and not (
            len(self.steps) == 3 and isinstance(self.steps[0], Call)
            and isinstance(self.steps[1], Verify) and isinstance(self.steps[2], Call)
        ):
            raise ValueError("verify-escalate is exactly call, local verify, call")
        if self.plan_type == "cascade" and not all(isinstance(s, Call) for s in self.steps):
            raise ValueError("cascade contains only calls; escalation is on provider failure")
        if self.plan_type == "parallel" and not (
            isinstance(self.steps[-1], Select)
            and all(isinstance(s, Call) for s in self.steps[:-1])
        ):
            raise ValueError("parallel is calls followed by one local selector")
        return self


class Prediction(FrozenContract):
    mean: float
    lower: float
    upper: float

    @model_validator(mode="after")
    def ordered(self) -> Prediction:
        if not self.lower <= self.mean <= self.upper:
            raise ValueError("prediction requires lower <= mean <= upper")
        return self


class RejectedAlternative(FrozenContract):
    endpoint_id: str
    reason_codes: tuple[str, ...]
    plan_type: str = "direct"


class DecisionRecord(FrozenContract):
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=now)
    request: RequestContext
    policy_id: str
    policy_version: str
    capability_map_version: str
    eligible_endpoints: tuple[str, ...]
    rejected_alternatives: tuple[RejectedAlternative, ...] = ()
    estimated_quality: Prediction | None = None
    estimated_cost: Prediction | None = None
    estimated_latency: Prediction | None = None
    selected_plan: ExecutionPlan
    fallback: str
    certificate_id: str | None = None
    certificate_status: Literal["current", "stale", "uncertified"] = "uncertified"
    exploration_probability: float = 0
    action_propensity: float | None = None
    decision_latency_ms: float = Field(default=0, ge=0)
    replay_seed: int
    estimated_failure: Prediction | None = None
    endpoint_snapshot_ids: tuple[str, ...] = ()
    policy_hash: str | None = None
    evidence_hash: str | None = None
    decision_fingerprint: str | None = None
    planning_complete: bool = True


class ExecutionRecord(FrozenContract):
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    decision_id: str
    created_at: datetime = Field(default_factory=now)
    endpoint_revisions: tuple[str, ...] = ()
    provider_request_ids: tuple[str, ...] = ()
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    time_to_first_token_ms: float | None = None
    total_latency_ms: float = Field(default=0, ge=0)
    provider_errors: tuple[str, ...] = ()
    fallback_transitions: tuple[str, ...] = ()
    realized_spend: float = Field(default=0, ge=0)
    output_reference: str | None = None
    selected_endpoint_id: str | None = None
    attempted_calls: int = Field(default=0, ge=0)
    reserved_spend: float = Field(default=0, ge=0)
    accounting_complete: bool = True
    constraint_violations: tuple[str, ...] = ()
    state: Literal["completed", "cancelled", "failed", "abstained"]


class OutcomeRecord(FrozenContract):
    outcome_id: str = Field(default_factory=lambda: str(uuid4()))
    decision_id: str
    execution_id: str | None = None
    quality: dict[str, float]
    evaluator_type: Literal["deterministic", "application", "human", "calibrated_judge", "proxy"]
    evaluator_version: str
    source_artifact: str
    label_confidence: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0)
    evaluated_at: datetime = Field(default_factory=now)
    causal_scope: Literal["request", "response", "plan", "task"]
    training_eligible: bool
    promotion_eligible: bool
    supersedes: str | None = None
    disputed: bool = False

    @model_validator(mode="after")
    def proxy_cannot_promote(self) -> OutcomeRecord:
        if self.evaluated_at.tzinfo is None:
            raise ValueError("outcome timestamps must have a timezone")
        if any(not 0 <= score <= 1 for score in self.quality.values()):
            raise ValueError("quality scores must be normalized to [0,1]")
        if self.evaluator_type == "proxy" and self.promotion_eligible:
            raise ValueError("proxy outcomes cannot promote a policy")
        return self
