"""Validated runtime for the bounded V2 plan operators (V2-202)."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Protocol
from inference_control.contracts import (
    Abstain,
    Call,
    DecisionRecord,
    ExecutionRecord,
    Select,
    Verify,
)


@dataclass(frozen=True)
class RuntimeResult:
    output_reference: str
    provider_request_id: str
    revision: str
    input_tokens: int
    output_tokens: int
    spend: float
    time_to_first_token_ms: float


class Adapter(Protocol):
    def call(self, endpoint_id: str, generation_config: dict[str, object]) -> RuntimeResult: ...


class Executor:
    def __init__(
        self,
        adapter: Adapter,
        verifiers: dict[str, Callable[[RuntimeResult, str], bool]] | None = None,
        selectors: dict[str, Callable[[tuple[RuntimeResult, ...]], RuntimeResult]] | None = None,
    ):
        self.adapter = adapter
        self.verifiers = verifiers or {}
        self.selectors = selectors or {}

    def execute(self, decision: DecisionRecord) -> ExecutionRecord:
        plan = decision.selected_plan
        if isinstance(plan.steps[0], Abstain):
            return ExecutionRecord(decision_id=decision.decision_id, state="abstained")
        started = perf_counter()
        results: list[RuntimeResult] = []
        errors = []
        transitions = []
        pending_verification: Verify | None = None
        for step in plan.steps:
            if isinstance(step, Verify):
                pending_verification = step
                continue
            if isinstance(step, Select):
                continue
            if not isinstance(step, Call):
                continue
            if plan.plan_type == "cascade" and results:
                break
            if plan.plan_type == "verify_escalate" and results and pending_verification:
                verifier = self.verifiers.get(pending_verification.verifier)
                if verifier is None:
                    raise ValueError("unknown verifier")
                if verifier(results[-1], pending_verification.acceptance_rule):
                    break
                transitions.append(f"verify_rejected:{step.endpoint_id}")
            try:
                results.append(self.adapter.call(step.endpoint_id, step.generation_config))
            except Exception as exc:
                errors.append(f"{step.endpoint_id}:{type(exc).__name__}")
                if plan.plan_type not in {"cascade", "verify_escalate"}:
                    raise
                transitions.append(f"call_failed:{step.endpoint_id}")
            if len(results) + len(errors) > plan.max_calls:
                raise RuntimeError("runtime exceeded call bound")
        if not results:
            return ExecutionRecord(
                decision_id=decision.decision_id,
                total_latency_ms=(perf_counter() - started) * 1000,
                provider_errors=tuple(errors),
                fallback_transitions=tuple(transitions),
                state="failed",
            )
        if plan.plan_type == "parallel":
            select_step = next(s for s in plan.steps if isinstance(s, Select))
            selector = self.selectors.get(select_step.selector)
            chosen = (
                selector(tuple(results))
                if selector
                else min(results, key=lambda r: (r.output_reference, r.provider_request_id))
            )
        else:
            chosen = results[-1]
        spend = sum(r.spend for r in results)
        if spend > plan.max_spend + 1e-12:
            raise RuntimeError("runtime spend exceeded static bound")
        return ExecutionRecord(
            decision_id=decision.decision_id,
            endpoint_revisions=tuple(r.revision for r in results),
            provider_request_ids=tuple(r.provider_request_id for r in results),
            input_tokens=sum(r.input_tokens for r in results),
            output_tokens=sum(r.output_tokens for r in results),
            time_to_first_token_ms=min(r.time_to_first_token_ms for r in results),
            total_latency_ms=(perf_counter() - started) * 1000,
            provider_errors=tuple(errors),
            fallback_transitions=tuple(transitions),
            realized_spend=spend,
            output_reference=chosen.output_reference,
            state="completed",
        )
