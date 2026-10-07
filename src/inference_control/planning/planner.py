"""Bounded execution-plan search with hard feasibility and evidence enforcement."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import permutations, combinations
from time import perf_counter
from typing import Any
from inference_control.contracts import (
    Call, Verify, Select, DecisionRecord, EndpointSnapshot, ExecutionPlan,
    InfeasibleBehavior, PolicySpec, Prediction, RejectedAlternative, RequestContext,
)
from inference_control.policies import eligibility_reasons, CertificateStore, RiskCertificate
from inference_control.planning.plans import abstain
from inference_control.util import digest


@dataclass(frozen=True)
class CapabilityEstimate:
    endpoint_id: str
    quality: Prediction
    cost: Prediction
    latency: Prediction
    failure: Prediction | None = None
    lineage: Any = None


@dataclass(frozen=True)
class PlanningAssessment:
    candidates: tuple[tuple[ExecutionPlan, CapabilityEstimate | None], ...]
    feasible: tuple[tuple[ExecutionPlan, CapabilityEstimate], ...]
    rejected: tuple[RejectedAlternative, ...]
    complete: bool
    selection: tuple[ExecutionPlan, CapabilityEstimate] | None
    fallback: str


class Planner:
    def __init__(self, endpoints: list[EndpointSnapshot], estimates: list[CapabilityEstimate] | None = None,
                 map_version: str = "map-v1", *, capability_map=None,
                 certificates: CertificateStore | None = None):
        if len({e.endpoint_id for e in endpoints}) != len(endpoints):
            raise ValueError("one active snapshot per endpoint is required")
        self.endpoints = {e.endpoint_id: e for e in endpoints}
        self.estimates = {e.endpoint_id: e for e in (estimates or [])}
        self._map_version = map_version
        self.capability_map = capability_map
        self.certificates = certificates or CertificateStore()

    @property
    def map_version(self):
        return self.capability_map.version if self.capability_map is not None else self._map_version

    def absolute_cost(self, endpoint_id: str, request: RequestContext) -> float:
        ep = self.endpoints[endpoint_id]
        return ((request.input_tokens + ep.input_token_overhead) * ep.input_price_per_million
                + request.max_output_tokens * ep.output_price_per_million) / 1e6

    def generate(self, request: RequestContext, policy: PolicySpec) -> tuple[list[ExecutionPlan], list[RejectedAlternative], bool]:
        rejected, eligible, plans = [], [], []
        for endpoint_id, endpoint in sorted(self.endpoints.items()):
            reasons = list(eligibility_reasons(endpoint, request, policy))
            if self.absolute_cost(endpoint_id, request) > policy.max_absolute_spend + 1e-12:
                reasons.append("ABSOLUTE_SPEND_BOUND")
            if reasons:
                rejected.append(RejectedAlternative(endpoint_id=endpoint_id, reason_codes=tuple(reasons)))
            else:
                eligible.append(endpoint_id)
        def make(kind, ids, tail=None):
            steps = [Call(endpoint_id=e, max_output_tokens=request.max_output_tokens,
                          generation_config=self.endpoints[e].inference_config) for e in ids]
            if kind == "verify_escalate":
                steps.insert(1, Verify(verifier=policy.verifier, acceptance_rule="accept"))
            if kind == "parallel":
                steps.append(Select(selector=policy.selector))
            bound = sum(self.absolute_cost(e, request) for e in ids)
            identity = digest({"kind": kind, "steps": steps, "bound": bound})[:32]
            return ExecutionPlan(plan_id=identity, plan_type=kind, steps=tuple(steps),
                                 max_calls=len(ids), max_spend=bound)
        def candidates():
            if "direct" in policy.permitted_plan_types:
                for e in eligible:
                    yield make("direct", [e])
            if "cascade" in policy.permitted_plan_types:
                for pair in permutations(eligible, 2):
                    yield make("cascade", pair)
            if "verify_escalate" in policy.permitted_plan_types and policy.verifier:
                for pair in permutations(eligible, 2):
                    yield make("verify_escalate", pair)
            if "parallel" in policy.permitted_plan_types and policy.selector:
                for pair in combinations(eligible, 2):
                    yield make("parallel", pair)
        for plan in candidates():
            if len(plans) >= policy.max_plan_candidates:
                # Truncation is never silently treated as a globally optimal search.
                rejected.append(RejectedAlternative(endpoint_id="*", plan_type="search",
                                                     reason_codes=("PLAN_SEARCH_LIMIT",)))
                return plans, rejected, False
            plans.append(plan)
        return plans, rejected, True

    def estimate_plan(self, request, policy, plan, at, comparisons=1):
        if self.capability_map is None:
            return self.estimates.get(plan.evidence_key)
        endpoints = tuple(self.endpoints[s.endpoint_id] for s in plan.steps if isinstance(s, Call))
        return self.capability_map.estimate(
            request, plan.evidence_key, endpoints, at=at, quality_risk=policy.quality_risk,
            latency_risk=policy.latency_risk, min_samples=policy.minimum_evidence_samples,
            max_age_seconds=policy.evidence_max_age_seconds, comparisons=comparisons)

    def reasons(self, plan, estimate, policy):
        hard, statistical = [], []
        if plan.max_calls > policy.max_calls: hard.append("CALL_LIMIT")
        if plan.max_spend > policy.max_absolute_spend + 1e-12: hard.append("ABSOLUTE_SPEND_BOUND")
        if estimate is None:
            statistical.append("ESTIMATE_MISSING" if plan.plan_type == "direct" else "PLAN_EVIDENCE_MISSING")
        else:
            if estimate.quality.lower < policy.minimum_quality: statistical.append("QUALITY_RISK")
            if estimate.cost.mean > policy.max_expected_spend + 1e-12: hard.append("EXPECTED_SPEND_BOUND")
            if estimate.latency.upper > policy.deadline_ms: hard.append("LATENCY_RISK")
            if policy.maximum_failure_probability < 1 and (
                estimate.failure is None or estimate.failure.upper > policy.maximum_failure_probability
            ): statistical.append("FAILURE_RISK")
            if policy.require_certificate and (not estimate.lineage or not estimate.lineage.calibrated):
                statistical.append("CERTIFICATE_EVIDENCE_MISSING")
        return hard, statistical

    def selection_key(self, item, policy):
        plan, e = item
        fields = {"cost": (e.cost.mean, e.latency.upper, -e.quality.lower),
                  "quality": (-e.quality.lower, e.cost.mean, e.latency.upper),
                  "latency": (e.latency.upper, e.cost.mean, -e.quality.lower)}
        return (*fields[policy.objective], plan.max_calls, plan.evidence_key)

    def assess(self, request, policy, *, at=None):
        at = at or datetime.now(timezone.utc)
        plans, rejected, complete = self.generate(request, policy)
        comparisons = max(1, len(plans))
        candidates = tuple((p, self.estimate_plan(request, policy, p, at, comparisons)) for p in plans)
        return self.select(candidates, policy, complete=complete, rejected=rejected)

    def select(self, candidates, policy, *, complete, rejected=()):
        rejected = list(rejected)
        feasible, shortfall = [], []
        for plan, estimate in candidates:
            hard, statistical = self.reasons(plan, estimate, policy)
            if hard or statistical:
                rejected.append(RejectedAlternative(endpoint_id=plan.evidence_key, plan_type=plan.plan_type,
                                                     reason_codes=tuple(hard + statistical)))
            else:
                feasible.append((plan, estimate))
            if not hard and estimate:
                shortfall.append((plan, estimate))
        def objective(item):
            return self.selection_key(item, policy)
        chosen = min(feasible, key=objective) if feasible and complete else None
        fallback = policy.infeasible_behavior.value
        # A safe fallback still has to satisfy every hard/statistical/certificate gate.
        if chosen is None and complete and policy.infeasible_behavior == InfeasibleBehavior.SAFE_FALLBACK:
            choices = [(p,e) for p,e in feasible if p.plan_type == "direct"
                       and p.steps[0].endpoint_id == policy.safe_fallback_endpoint]
            chosen = min(choices, key=objective) if choices else None
        if chosen is None and complete and not policy.require_certificate and policy.infeasible_behavior == InfeasibleBehavior.UNCERTIFIED:
            # Explicit quality-risk relaxation only. Spend, deadline, privacy and call caps never relax.
            choices = [(p,e) for p,e in shortfall if policy.maximum_failure_probability >= 1 or
                       (e.failure and e.failure.upper <= policy.maximum_failure_probability)]
            chosen = min(choices, key=lambda x: (max(0, policy.minimum_quality-x[1].quality.lower), objective(x))) if choices else None
            if chosen: fallback = "least_shortfall_uncertified"
        return PlanningAssessment(tuple(candidates), tuple(feasible), tuple(rejected), complete, chosen, fallback)

    def decide(self, request: RequestContext, policy: PolicySpec, seed: int = 0,
               *, at: datetime | None = None) -> DecisionRecord:
        started, at = perf_counter(), at or datetime.now(timezone.utc)
        assessment = self.assess(request, policy, at=at)
        chosen, feasible = assessment.selection, assessment.feasible
        complete, fallback = assessment.complete, assessment.fallback
        rejected = list(assessment.rejected)
        plan, estimate = chosen if chosen else (abstain("NO_FEASIBLE_PLAN" if complete else "PLAN_SEARCH_LIMIT"), None)
        if not chosen:
            plan = plan.model_copy(update={"plan_id": digest({"request": request, "policy": policy, "reason": plan.steps})[:32]})
        ids = tuple(s.endpoint_id for s in plan.steps if isinstance(s, Call))
        snapshot_ids = tuple(self.endpoints[e].snapshot_id for e in ids)
        certificate = None
        lineage = estimate.lineage if estimate else None
        if lineage and lineage.calibrated and chosen in feasible:
            expires = min(at+timedelta(seconds=policy.certificate_ttl_seconds),
                          lineage.measured_at+timedelta(seconds=policy.evidence_max_age_seconds))
            if lineage.trusted_until is not None:
                expires = min(expires, lineage.trusted_until)
            if expires > at:
                cert_id = digest({"request":request,"policy": policy, "map": self.map_version, "plan": plan.evidence_key,
                                  "evidence": lineage.artifact_hash, "snapshots": snapshot_ids, "issued_at": at})
                certificate = RiskCertificate(
                    cert_id, policy.version,
                    f"{lineage.quality_diagnostics.method}/binomial-p95-v3" if lineage.quality_diagnostics else "local-knn-v1",
                    lineage.endpoint_revisions,
                    lineage.calibration_hash, lineage.artifact_hash, policy.quality_risk,
                    policy.cost_risk, policy.latency_risk, tuple(sorted(request.traffic_slices)),
                    at, expires, lineage.assumptions, policy_hash=digest(policy),
                    capability_map_version=self.map_version, endpoint_ids=ids,
                    endpoint_snapshot_ids=snapshot_ids, plan_key=plan.evidence_key,
                    feature_version=request.feature_version, traffic_stratum=lineage.traffic_stratum, request_hash=digest(request),
                    quality_lower=estimate.quality.lower, cost_upper=plan.max_spend,
                    latency_upper_ms=estimate.latency.upper, sample_size=lineage.sample_size,
                    calibration_size=lineage.calibration_size)
                self.certificates.issue(certificate)
        fingerprint = digest({"request": request, "policy": policy, "snapshots": snapshot_ids,
                              "map": self.map_version, "plan": plan, "seed": seed,
                              "certificate_status": "current" if certificate else "uncertified"})
        return DecisionRecord(
            created_at=at, request=request, policy_id=policy.policy_id, policy_version=policy.version,
            policy_hash=digest(policy), capability_map_version=self.map_version,
            eligible_endpoints=tuple(sorted({s.endpoint_id for p,_ in feasible for s in p.steps if isinstance(s,Call)})),
            rejected_alternatives=tuple(rejected), selected_plan=plan, fallback=fallback,
            estimated_quality=estimate.quality if estimate else None,
            estimated_cost=estimate.cost if estimate else None,
            estimated_latency=estimate.latency if estimate else None,
            estimated_failure=estimate.failure if estimate else None,
            certificate_id=certificate.certificate_id if certificate else None,
            certificate_status="current" if certificate else "uncertified",
            endpoint_snapshot_ids=snapshot_ids, evidence_hash=lineage.artifact_hash if lineage else None,
            decision_fingerprint=fingerprint, planning_complete=complete,
            replay_seed=seed, decision_latency_ms=(perf_counter()-started)*1000)
