"""Deterministic feasibility-first, lexicographic planner (V2-103/V2-206)."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from inference_control.contracts import (
    Abstain,
    Call,
    DecisionRecord,
    EndpointSnapshot,
    ExecutionPlan,
    InfeasibleBehavior,
    PolicySpec,
    Prediction,
    RejectedAlternative,
    RequestContext,
)
from inference_control.policies import eligibility_reasons


@dataclass(frozen=True)
class CapabilityEstimate:
    endpoint_id: str
    quality: Prediction
    cost: Prediction
    latency: Prediction


class Planner:
    def __init__(
        self,
        endpoints: list[EndpointSnapshot],
        estimates: list[CapabilityEstimate],
        map_version: str = "map-v1",
    ):
        self.endpoints = {e.endpoint_id: e for e in endpoints}
        self.estimates = {e.endpoint_id: e for e in estimates}
        self.map_version = map_version

    def decide(self, request: RequestContext, policy: PolicySpec, seed: int = 0) -> DecisionRecord:
        started = perf_counter()
        rejected = []
        feasible = []
        for endpoint_id in sorted(self.endpoints):
            endpoint = self.endpoints[endpoint_id]
            reasons = list(eligibility_reasons(endpoint, request, policy))
            estimate = self.estimates.get(endpoint_id)
            if not estimate:
                reasons.append("ESTIMATE_MISSING")
            else:
                absolute = (
                    request.input_tokens * endpoint.input_price_per_million
                    + request.max_output_tokens * endpoint.output_price_per_million
                ) / 1_000_000
                if absolute > policy.max_absolute_spend:
                    reasons.append("ABSOLUTE_SPEND_BOUND")
                if estimate.quality.lower < policy.minimum_quality:
                    reasons.append("QUALITY_RISK")
                if estimate.cost.upper > policy.max_expected_spend:
                    reasons.append("COST_RISK")
                if estimate.latency.upper > policy.deadline_ms:
                    reasons.append("LATENCY_RISK")
            if reasons:
                rejected.append(
                    RejectedAlternative(endpoint_id=endpoint_id, reason_codes=tuple(reasons))
                )
            else:
                feasible.append(estimate)
        chosen = (
            min(
                feasible,
                key=lambda e: (-e.quality.lower, e.cost.mean, e.latency.upper, e.endpoint_id),
            )
            if feasible
            else None
        )
        fallback_selected = False
        if not chosen and policy.infeasible_behavior == InfeasibleBehavior.SAFE_FALLBACK:
            fallback_id = policy.safe_fallback_endpoint
            endpoint = self.endpoints.get(fallback_id or "")
            estimate = self.estimates.get(fallback_id or "")
            if endpoint and estimate and not eligibility_reasons(endpoint, request, policy):
                absolute = (
                    request.input_tokens * endpoint.input_price_per_million
                    + request.max_output_tokens * endpoint.output_price_per_million
                ) / 1_000_000
                if absolute <= policy.max_absolute_spend:
                    chosen = estimate
                    fallback_selected = True
        if not chosen and policy.infeasible_behavior == InfeasibleBehavior.UNCERTIFIED:
            candidates = [
                estimate
                for endpoint_id, estimate in self.estimates.items()
                if endpoint_id in self.endpoints
                and not eligibility_reasons(self.endpoints[endpoint_id], request, policy)
            ]
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda e: (
                        max(0, policy.minimum_quality - e.quality.lower),
                        e.cost.mean,
                        e.latency.upper,
                        e.endpoint_id,
                    ),
                )
                fallback_selected = True
        if chosen:
            ep = self.endpoints[chosen.endpoint_id]
            bound = (
                request.input_tokens * ep.input_price_per_million
                + request.max_output_tokens * ep.output_price_per_million
            ) / 1_000_000
            plan = ExecutionPlan(
                plan_type="direct",
                steps=(
                    Call(endpoint_id=ep.endpoint_id, max_output_tokens=request.max_output_tokens),
                ),
                max_calls=1,
                max_spend=bound,
            )
            quality, cost, latency = chosen.quality, chosen.cost, chosen.latency
        else:
            plan = ExecutionPlan(
                plan_type="abstain",
                steps=(Abstain(reason="NO_FEASIBLE_PLAN"),),
                max_calls=0,
                max_spend=0,
            )
            quality = cost = latency = None
        return DecisionRecord(
            request=request,
            policy_id=policy.policy_id,
            policy_version=policy.version,
            capability_map_version=self.map_version,
            eligible_endpoints=tuple(e.endpoint_id for e in feasible),
            rejected_alternatives=tuple(rejected),
            estimated_quality=quality,
            estimated_cost=cost,
            estimated_latency=latency,
            selected_plan=plan,
            fallback=(
                f"selected:{chosen.endpoint_id}"
                if fallback_selected and chosen
                else policy.infeasible_behavior.value
            ),
            certificate_status="uncertified",
            replay_seed=seed,
            decision_latency_ms=(perf_counter() - started) * 1000,
        )
