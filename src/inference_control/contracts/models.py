from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Literal, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now() -> datetime:
    return datetime.now(timezone.utc)


class FrozenContract(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["2.0"] = "2.0"


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

    @model_validator(mode="after")
    def bounds(self) -> PolicySpec:
        if self.max_expected_spend > self.max_absolute_spend:
            raise ValueError("max_expected_spend cannot exceed max_absolute_spend")
        if (
            self.infeasible_behavior == InfeasibleBehavior.SAFE_FALLBACK
            and not self.safe_fallback_endpoint
        ):
            raise ValueError("safe_fallback requires safe_fallback_endpoint")
        return self


class Call(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    operator: Literal["call"] = "call"
    endpoint_id: str
    generation_config: dict[str, object] = Field(default_factory=dict)
    max_output_tokens: int = Field(default=256, ge=0)


class Verify(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    operator: Literal["verify"] = "verify"
    verifier: str
    acceptance_rule: str


class Select(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    operator: Literal["select"] = "select"
    selector: str


class Abstain(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    operator: Literal["abstain"] = "abstain"
    reason: str


PlanStep = Annotated[Union[Call, Verify, Select, Abstain], Field(discriminator="operator")]


class ExecutionPlan(FrozenContract):
    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    plan_type: Literal["direct", "cascade", "verify_escalate", "parallel", "abstain"]
    steps: tuple[PlanStep, ...]
    max_calls: int = Field(ge=0)
    max_spend: float = Field(ge=0)

    @model_validator(mode="after")
    def shape(self) -> ExecutionPlan:
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
        return self


class Prediction(FrozenContract):
    mean: float
    lower: float
    upper: float


class RejectedAlternative(FrozenContract):
    endpoint_id: str
    reason_codes: tuple[str, ...]


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
        if self.evaluator_type == "proxy" and self.promotion_eligible:
            raise ValueError("proxy outcomes cannot promote a policy")
        return self
