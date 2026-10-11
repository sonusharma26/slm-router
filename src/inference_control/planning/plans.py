"""Bounded V2 plan constructors (V2-202)."""

from inference_control.contracts import Abstain, Call, ExecutionPlan, Select, Verify


def direct(endpoint: str, spend: float, max_output_tokens: int = 256) -> ExecutionPlan:
    return ExecutionPlan(
        plan_type="direct",
        steps=(Call(endpoint_id=endpoint, max_output_tokens=max_output_tokens),),
        max_calls=1,
        max_spend=spend,
    )


def cascade(endpoints: list[str], spend_bounds: list[float]) -> ExecutionPlan:
    if len(endpoints) != len(spend_bounds):
        raise ValueError("bounds must match endpoints")
    return ExecutionPlan(
        plan_type="cascade",
        steps=tuple(Call(endpoint_id=e) for e in endpoints),
        max_calls=len(endpoints),
        max_spend=sum(spend_bounds),
    )


def verify_escalate(
    cheap: str, verifier: str, frontier: str, spend_bounds: tuple[float, float]
) -> ExecutionPlan:
    return ExecutionPlan(
        plan_type="verify_escalate",
        steps=(
            Call(endpoint_id=cheap),
            Verify(verifier=verifier, acceptance_rule="accept"),
            Call(endpoint_id=frontier),
        ),
        max_calls=2,
        max_spend=sum(spend_bounds),
    )


def agree_escalate(
    first: str, checker: str, agreement: str, final: str | None, spend_bounds: list[float],
    max_output_tokens: int = 256,
) -> ExecutionPlan:
    """Run ``first`` and ``checker`` concurrently; serve ``first`` when they agree, otherwise
    serve ``checker`` (when it is the quality-first ``final``) or escalate to ``final``."""
    escalate = final is not None and final not in (first, checker)
    calls = [first, checker] + ([final] if escalate else [])
    if len(spend_bounds) != len(calls):
        raise ValueError("bounds must match endpoints")
    steps = [Call(endpoint_id=e, max_output_tokens=max_output_tokens) for e in calls[:2]]
    steps += [Verify(verifier=agreement, acceptance_rule="agree")]
    steps += [Call(endpoint_id=final, max_output_tokens=max_output_tokens)] if escalate else []
    return ExecutionPlan(plan_type="agree_escalate", steps=tuple(steps), max_calls=len(calls),
                         max_spend=sum(spend_bounds))


def parallel(endpoints: list[str], selector: str, spend_bounds: list[float]) -> ExecutionPlan:
    return ExecutionPlan(
        plan_type="parallel",
        steps=tuple(Call(endpoint_id=e) for e in endpoints) + (Select(selector=selector),),
        max_calls=len(endpoints),
        max_spend=sum(spend_bounds),
    )


def abstain(reason: str) -> ExecutionPlan:
    return ExecutionPlan(
        plan_type="abstain", steps=(Abstain(reason=reason),), max_calls=0, max_spend=0
    )
