"""V2 decision, execution, outcome, and audit API (V2-206)."""

from __future__ import annotations
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, model_validator
from inference_control.contracts import (
    DecisionRecord,
    ExecutionRecord,
    OutcomeRecord,
    PolicySpec,
    RequestContext,
)
from inference_control.execution import Executor
from inference_control.ledger import SQLiteLedger
from inference_control.planning import Planner
from inference_control.outcomes import OutcomeStore


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: RequestContext
    policy: PolicySpec
    seed: int = 0


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision_id: str | None = None
    request: RequestContext | None = None
    policy: PolicySpec | None = None
    seed: int = 0

    @model_validator(mode="after")
    def source(self) -> "ExecuteRequest":
        has_prior = self.decision_id is not None
        has_inline = self.request is not None and self.policy is not None
        if has_prior == has_inline:
            raise ValueError("provide either decision_id or both request and policy")
        return self


class ControlPlane:
    def __init__(self, planner: Planner, ledger: SQLiteLedger, executor: Executor | None = None):
        self.planner = planner
        self.ledger = ledger
        self.executor = executor
        self.decisions: dict[str, DecisionRecord] = {}
        self.executions: dict[str, ExecutionRecord] = {}
        self.outcomes = OutcomeStore()
        self.policies: dict[tuple[str, str], PolicySpec] = {}

    def decide(self, payload: DecideRequest) -> DecisionRecord:
        decision = self.planner.decide(payload.request, payload.policy, payload.seed)
        self.ledger.append(
            "decision",
            decision.model_dump(mode="json"),
            actor="api",
            idempotency_key=f"decision:{decision.decision_id}",
        )
        self.decisions[decision.decision_id] = decision
        self.policies[(payload.policy.policy_id, payload.policy.version)] = payload.policy
        return decision


def create_app(control: ControlPlane | None = None) -> FastAPI:
    api = FastAPI(title="Adaptive Inference Control Plane", version="2.0.0-alpha")

    @api.post("/v2/decide", response_model=DecisionRecord)
    def decide(payload: DecideRequest) -> DecisionRecord:
        if not control:
            raise HTTPException(503, "control plane not configured")
        return control.decide(payload)

    @api.post("/v2/execute", response_model=ExecutionRecord)
    def execute(payload: ExecuteRequest) -> ExecutionRecord:
        if not control or not control.executor:
            raise HTTPException(503, "executor not configured")
        if payload.decision_id:
            decision = control.decisions.get(payload.decision_id)
            if not decision:
                raise HTTPException(404, "decision not found")
        else:
            assert payload.request is not None and payload.policy is not None
            decision = control.decide(
                DecideRequest(request=payload.request, policy=payload.policy, seed=payload.seed)
            )
        # A decision event is guaranteed durable before the executor is invoked.
        record = control.executor.execute(decision)
        control.ledger.append(
            "execution",
            record.model_dump(mode="json"),
            actor="api",
            idempotency_key=f"execution:{record.execution_id}",
        )
        control.executions[record.execution_id] = record
        return record

    @api.post("/v2/outcomes", response_model=OutcomeRecord, status_code=201)
    def add_outcome(outcome: OutcomeRecord) -> OutcomeRecord:
        if not control:
            raise HTTPException(503, "control plane not configured")
        decision = control.decisions.get(outcome.decision_id)
        if not decision:
            raise HTTPException(404, "decision not found")
        policy = control.policies[(decision.policy_id, decision.policy_version)]
        qualified = control.outcomes.add(outcome, policy)
        control.ledger.append(
            "outcome",
            qualified.model_dump(mode="json"),
            actor="api",
            idempotency_key=f"outcome:{qualified.outcome_id}",
        )
        return qualified

    @api.get("/v2/decisions/{decision_id}", response_model=DecisionRecord)
    def get_decision(decision_id: str) -> DecisionRecord:
        if not control:
            raise HTTPException(503, "control plane not configured")
        try:
            return control.decisions[decision_id]
        except KeyError:
            raise HTTPException(404, "decision not found") from None

    return api


app = create_app()
