"""Bounded execution. Local verification/selection only; every provider attempt counts."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import perf_counter
import math
from typing import Callable, Protocol
from inference_control.contracts import Abstain, Call, DecisionRecord, ExecutionRecord, Select, Verify


@dataclass(frozen=True)
class RuntimeResult:
    output_reference: str
    provider_request_id: str
    revision: str
    input_tokens: int
    output_tokens: int
    spend: float
    time_to_first_token_ms: float | None
    endpoint_id: str | None = None
    message: dict | None = None
    provider_latency_ms: float | None = None
    constraint_violations: tuple[str,...] = ()
    revision_observed: bool = False


class Adapter(Protocol):
    def call(self, endpoint_id: str, generation_config: dict) -> RuntimeResult: ...


class Executor:
    def __init__(self, adapter: Adapter, verifiers=None, selectors=None):
        self.adapter, self.verifiers, self.selectors = adapter, verifiers or {}, selectors or {}

    def execute(self, decision: DecisionRecord, *, policy=None, endpoints=None, certificates=None,
                messages=None, response_config=None, result_sink: Callable | None = None) -> ExecutionRecord:
        plan, started = decision.selected_plan, perf_counter()
        if plan.plan_type == "abstain":
            return ExecutionRecord(decision_id=decision.decision_id,state="abstained")
        calls = [s for s in plan.steps if isinstance(s,Call)]
        if len(calls) > plan.max_calls: raise ValueError("call bound violated")
        # Reject unknown local operations BEFORE an endpoint can spend money.
        for s in plan.steps:
            if isinstance(s,Verify) and s.verifier not in self.verifiers: raise ValueError("unknown local verifier")
            if isinstance(s,Select) and s.selector not in self.selectors: raise ValueError("unknown local selector")
        if policy is not None:
            if plan.plan_type not in policy.permitted_plan_types or plan.max_calls > policy.max_calls:
                raise ValueError("plan violates policy")
            if plan.max_spend > policy.max_absolute_spend+1e-12: raise ValueError("plan exceeds spend bound")
        deadline = policy.deadline_ms if policy is not None else None
        def guard():
            if policy is not None and endpoints is not None:
                from inference_control.policies.validator import eligibility_reasons
                ids = tuple(step.endpoint_id for step in calls)
                current = tuple(endpoints[e].snapshot_id for e in ids if e in endpoints)
                if current != decision.endpoint_snapshot_ids:
                    raise ValueError("endpoint snapshot changed before dispatch")
                if any(eligibility_reasons(endpoints[e],decision.request,policy) for e in ids):
                    raise ValueError("endpoint eligibility changed before dispatch")
            if policy is not None and policy.require_certificate:
                if certificates is None or endpoints is None:
                    raise ValueError("certified execution requires certificate store and endpoint snapshots")
                reasons = certificates.validate_decision(decision,policy,endpoints)
                if reasons: raise ValueError(";".join(reasons))
            if deadline is not None and (perf_counter()-started)*1000 >= deadline:
                raise TimeoutError("execution deadline exhausted")
        guard()
        adapter = self.adapter.for_request(decision.request,messages,response_config) if hasattr(self.adapter,"for_request") else self.adapter
        results, errors, transitions, violations = [], [], [], []
        attempts, accounted = 0, True
        chosen = None
        def invoke(step):
            guard()
            config = dict(step.generation_config)
            # Maximum output and timeout cannot be overridden by an inference config.
            config.pop("max_tokens",None)
            config["max_completion_tokens"] = step.max_output_tokens
            if deadline is not None:
                config["timeout"] = max(.001,(deadline-(perf_counter()-started)*1000)/1000)
            return adapter.call(step.endpoint_id, config)
        def accept(step, result):
            if not math.isfinite(result.spend) or result.spend < 0:
                raise ValueError("adapter returned invalid spend")
            results.append(result)
            violations.extend(result.constraint_violations)
        if plan.plan_type == "parallel":
            # Actually concurrent. All branches reserved before dispatch; no fictitious max latency from sequential execution.
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                futures = {pool.submit(invoke,s):s for s in calls}
                attempts = len(futures)
                for future in as_completed(futures):
                    step = futures[future]
                    try: accept(step,future.result())
                    except Exception as exc:
                        errors.append(f"{step.endpoint_id}:{type(exc).__name__}")
                        accounted = False
            if results:
                select = next(s for s in plan.steps if isinstance(s,Select))
                ordered = tuple(sorted(results,key=lambda r:(r.endpoint_id or "",r.provider_request_id)))
                chosen = self.selectors[select.selector](ordered)
                if chosen not in ordered: raise ValueError("selector returned an unexecuted response")
        else:
            for index, step in enumerate(calls):
                if attempts >= plan.max_calls: raise RuntimeError("call reservation exhausted")
                if results and plan.plan_type == "cascade": break
                if results and plan.plan_type == "verify_escalate":
                    verify = next(s for s in plan.steps if isinstance(s,Verify))
                    try: accepted = self.verifiers[verify.verifier](results[-1],verify.acceptance_rule)
                    except Exception:
                        accepted = False
                        transitions.append("verifier_failed_closed")
                    if accepted:
                        chosen = results[-1]
                        break
                    transitions.append(f"verify_rejected:{step.endpoint_id}")
                    chosen = None
                # Invalidate/expiry/deadline check occurs before charging another attempt.
                try: guard()
                except (ValueError,TimeoutError) as exc:
                    violations.append("PRECALL_GUARD_REJECTED")
                    break
                attempts += 1
                try:
                    result = invoke(step)
                    accept(step,result)
                    chosen = result
                except Exception as exc:
                    errors.append(f"{step.endpoint_id}:{type(exc).__name__}")
                    accounted = False  # The provider may have billed a timed-out/erroring request.
                    transitions.append(f"call_failed:{step.endpoint_id}")
                    chosen = None
                    if plan.plan_type == "direct": break
                if violations: break
        elapsed = (perf_counter()-started)*1000
        spend = sum(r.spend for r in results)
        if spend > plan.max_spend+1e-12: violations.append("REALIZED_SPEND_EXCEEDED")
        if deadline is not None and elapsed > deadline: violations.append("REALIZED_DEADLINE_EXCEEDED")
        if not accounted and policy is not None and policy.require_certificate:
            violations.append("ACCOUNTING_INCOMPLETE")
        if violations: chosen = None
        if chosen is not None and result_sink is not None: result_sink(chosen)
        ttft = [r.time_to_first_token_ms for r in results if r.time_to_first_token_ms is not None]
        return ExecutionRecord(
            decision_id=decision.decision_id, endpoint_revisions=tuple(r.revision for r in results),
            provider_request_ids=tuple(r.provider_request_id for r in results),
            input_tokens=sum(r.input_tokens for r in results), output_tokens=sum(r.output_tokens for r in results),
            time_to_first_token_ms=min(ttft) if ttft else None, total_latency_ms=elapsed,
            provider_errors=tuple(errors), fallback_transitions=tuple(transitions), realized_spend=spend,
            output_reference=chosen.output_reference if chosen else None,
            selected_endpoint_id=chosen.endpoint_id if chosen else None,
            attempted_calls=attempts, reserved_spend=plan.max_spend, accounting_complete=accounted,
            constraint_violations=tuple(sorted(set(violations))), state="completed" if chosen else "failed")
