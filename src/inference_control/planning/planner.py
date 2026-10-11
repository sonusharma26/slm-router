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
from inference_control.routing.pooled import CASCADE_ASSUMPTIONS
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
    notes: dict | None = None


POOLED = "pooled-ltt"


@dataclass(frozen=True)
class _PlanStub:
    """The plan fields Planner.reasons and selection_key read, for calibration replay."""
    evidence_key: str
    max_spend: float
    max_calls: int = 1
    plan_type: str = "direct"


@dataclass(frozen=True)
class _Value:
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class _PooledDiagnostics:
    method: str = POOLED


@dataclass(frozen=True)
class _PooledLineage:
    quality_diagnostics: _PooledDiagnostics = _PooledDiagnostics()
    calibrated: bool = True


@dataclass(frozen=True)
class _EstimateStub:
    quality: _Value
    cost: _Value
    latency: _Value
    failure: _Value
    lineage: Any = None


@dataclass(frozen=True)
class _CascadeRoute:
    """One request's cascade decision: positions (in selection order) and the notes recorded with it."""
    reference: Any
    disagree: Any
    first: int
    checker: int
    final: int
    check: bool
    direct_endpoint: str | None
    notes: dict


def _pooled_estimate(estimate) -> bool:
    lineage = getattr(estimate, "lineage", None)
    diagnostics = getattr(lineage, "quality_diagnostics", None)
    return getattr(diagnostics, "method", None) == POOLED


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
        self._pooled_calibrations: dict[tuple, Any] = {}

    @property
    def pooled(self) -> bool:
        return self.capability_map is not None and getattr(self.capability_map, "quality_method", None) == POOLED

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
        # Pooled estimates certify a population bound shared by all certified plans, so they
        # rank by their (projected) forecast; local estimates rank by their own lower bound.
        q = e.quality.mean if _pooled_estimate(e) else e.quality.lower
        fields = {"cost": (e.cost.mean, e.latency.upper, -q),
                  "quality": (-q, e.cost.mean, e.latency.upper),
                  "latency": (e.latency.upper, e.cost.mean, -q)}
        return (*fields[policy.objective], plan.max_calls, plan.evidence_key)

    def assess(self, request, policy, *, at=None):
        at = at or datetime.now(timezone.utc)
        plans, rejected, complete = self.generate(request, policy)
        if self.pooled:
            candidates, rejected, notes = self._pooled_candidates(request, policy, plans, rejected, at)
            return self.select(candidates, policy, complete=complete, rejected=rejected, notes=notes)
        comparisons = max(1, len(plans))
        candidates = tuple((p, self.estimate_plan(request, policy, p, at, comparisons)) for p in plans)
        return self.select(candidates, policy, complete=complete, rejected=rejected)

    # ----- pooled learned routing ---------------------------------------------------------
    def _pooled_evidence(self, policy, at, feature_version):
        cmap = self.capability_map
        endpoints = tuple(self.endpoints[e] for e in sorted(self.endpoints))
        max_age = min(cmap.max_age_seconds, policy.evidence_max_age_seconds)
        return cmap.pooled_evidence(endpoints, feature_version, at, max_age)

    def _plan_count(self, eligible: int, policy) -> int:
        """Number of candidates generate() would enumerate for this many eligible endpoints."""
        count = 0
        if "direct" in policy.permitted_plan_types: count += eligible
        pairs = eligible * (eligible - 1)
        if "cascade" in policy.permitted_plan_types: count += pairs
        if "verify_escalate" in policy.permitted_plan_types and policy.verifier: count += pairs
        if "parallel" in policy.permitted_plan_types and policy.selector: count += pairs // 2
        return count

    def _pooled_rule(self, evidence, request, policy, quality, cost, details: dict | None = None):
        """Allowed direct endpoints for one request and their selection order if certified.

        Replays generate() eligibility, the hard gates of reasons(), the failure gate and the
        selection key exactly as serving applies them; only the quality gate is left open.
        Returns (endpoint_id, forecast, expected cost) in selection order, or None when planning
        would be incomplete (the planner then never routes the request). ``details``, when given,
        receives each allowed endpoint's ``pooled_numbers`` (the compound-plan gates reuse them).
        """
        cmap = self.capability_map
        eligible = [e for e in evidence.endpoint_ids if e in self.endpoints
                    and not eligibility_reasons(self.endpoints[e], request, policy)
                    and self.absolute_cost(e, request) <= policy.max_absolute_spend + 1e-12]
        if self._plan_count(len(eligible), policy) > policy.max_plan_candidates:
            return None
        if "direct" not in policy.permitted_plan_types:
            return []
        minimum = max(cmap.min_samples, policy.minimum_evidence_samples)
        allowed = []
        for endpoint_id in eligible:
            column = evidence.column(endpoint_id)
            forecast = float(quality[column])
            if forecast != forecast:
                continue
            numbers = cmap.pooled_numbers(evidence, request, self.endpoints[endpoint_id], float(cost[column]),
                quality_risk=policy.quality_risk, latency_risk=policy.latency_risk,
                comparisons=len(eligible), minimum=minimum)
            projected = min(1., max(policy.minimum_quality, min(1., max(0., forecast))))
            stub = _EstimateStub(_Value(projected, policy.minimum_quality, 1.), _Value(*numbers["cost"]),
                                 _Value(*numbers["latency"]), _Value(*numbers["failure"]), _PooledLineage())
            plan = _PlanStub(endpoint_id, self.absolute_cost(endpoint_id, request))
            hard, _ = self.reasons(plan, stub, policy)
            if hard:
                continue
            if policy.maximum_failure_probability < 1 and stub.failure.upper > policy.maximum_failure_probability:
                continue
            if details is not None:
                details[endpoint_id] = numbers
            allowed.append((self.selection_key((plan, stub), policy), endpoint_id, forecast, numbers["cost"][0]))
        return [(endpoint_id, forecast, cost_mean) for _, endpoint_id, forecast, cost_mean in sorted(allowed)]

    def _pooled_reference(self, evidence, requests, policy, quality, cost, outcomes=None):
        """Ordered forecasts/outcomes of the routing rule on outcome-independent references."""
        import numpy as np
        m = len(evidence.endpoint_ids)
        ordered_f = np.full((len(requests), m), -np.inf)
        ordered_v = np.full((len(requests), m), np.nan)
        best, every = [], []
        for i, request in enumerate(requests):
            rule = self._pooled_rule(evidence, request, policy, quality[i], cost[i]) or []
            for position, (endpoint_id, forecast, _) in enumerate(rule):
                ordered_f[i, position] = forecast
                if outcomes is not None:
                    ordered_v[i, position] = outcomes[i, evidence.column(endpoint_id)]
            best.append(max((f for _, f, _ in rule), default=float("-inf")))
            every.extend(f for _, f, _ in rule)
        return ordered_f, ordered_v, best, every

    def pooled_threshold_grid(self, requests, policy, *, at=None, fractions=None) -> tuple[float, ...]:
        """Strict-to-loose thresholds from requests whose outcomes calibration never sees."""
        from inference_control.routing.calibration import threshold_grid, DEFAULT_FRACTIONS
        at = at or datetime.now(timezone.utc)
        if not requests:
            return ()
        evidence = self._pooled_evidence(policy, at, requests[0].feature_version)
        quality, cost = evidence.forecast(list(requests))
        _, _, best, every = self._pooled_reference(evidence, list(requests), policy, quality, cost)
        return threshold_grid(best, fractions or DEFAULT_FRACTIONS, every)

    def pooled_calibration(self, policy, *, at=None, feature_version: str | None = None):
        """Learn-then-test threshold for this policy over full-information calibration requests."""
        import numpy as np
        from inference_control.routing.calibration import anchor_start, calibrate_threshold, threshold_grid
        at = at or datetime.now(timezone.utc)
        cmap = self.capability_map
        evidence = self._pooled_evidence(policy, at, feature_version or "query-v1")
        key = (evidence.identity, self._digest_once(policy), self._snapshot_signature(),
               self._digest_once(cmap.pooled_config))
        cached = self._pooled_calibrations.get(key)
        if cached is not None:
            return evidence, cached[0]
        # Answer-agreement cascade (agree_escalate): tried first when the policy permits it and the
        # evidence carries answers. If it certifies a configuration the planner serves it; otherwise
        # the direct rule below is tested as the fallback. Two tests share the policy's risk, so the
        # served rule still errs with probability at most quality_risk (union bound).
        live = self._cascade_calibration(evidence, policy)
        if live is not None and live.config is not None:
            if len(self._pooled_calibrations) >= 64:
                self._pooled_calibrations.clear()
            self._pooled_calibrations[key] = (live, digest(live.identity()))
            return evidence, live
        risk = policy.quality_risk if live is None else policy.quality_risk / 2
        if cmap.pooled_config.rule == "utility":
            calibration = self._utility_calibration(evidence, policy, risk=risk)
            if len(self._pooled_calibrations) >= 64:
                self._pooled_calibrations.clear()
            self._pooled_calibrations[key] = (calibration, digest(calibration.identity()))
            return evidence, calibration
        requests = evidence.calibration_requests
        quality, cost = evidence.forecast(requests)
        m = len(evidence.endpoint_ids)
        forecasts, outcomes, kept = [], [], 0
        for i, request in enumerate(requests):
            rule = self._pooled_rule(evidence, request, policy, quality[i], cost[i])
            row_f, row_q = np.full(m, -np.inf), np.full(m, np.nan)
            if rule:
                values = [evidence.calibration_quality[i, evidence.column(e)] for e, _, _ in rule]
                if any(v != v for v in values):
                    continue  # not scored on every allowed endpoint: excluded from calibration
                for position, ((_, forecast, _), value) in enumerate(zip(rule, values)):
                    row_f[position], row_q[position] = forecast, value
            forecasts.append(row_f)
            outcomes.append(row_q)
            kept += 1
        # Training references (leave-one-out forecasts, training outcomes) are independent of
        # calibration outcomes; they build the default grid and choose where the sequence opens.
        train_q, train_c = evidence.training_forecasts()
        ref_f, ref_v, reference, every = self._pooled_reference(evidence, evidence.train_requests, policy,
                                                                train_q, train_c, evidence.train_quality)
        thresholds = cmap.pooled_config.thresholds
        if thresholds is None:
            thresholds = threshold_grid(reference, cmap.pooled_config.fractions, every)
        ordered_f = np.array(forecasts) if forecasts else np.zeros((0, m))
        ordered_q = np.array(outcomes) if outcomes else np.zeros((0, m))
        minimum = max(cmap.min_samples, policy.minimum_evidence_samples)
        start = anchor_start(ref_f, ref_v, thresholds, calibration_size=kept, target=policy.minimum_quality,
                             risk=risk, minimum_certified=minimum)
        calibration = calibrate_threshold(ordered_f, ordered_q, tuple(thresholds)[start:], target=policy.minimum_quality,
                                          risk=risk, minimum_certified=minimum)
        if len(self._pooled_calibrations) >= 64:
            self._pooled_calibrations.clear()
        self._pooled_calibrations[key] = (calibration, digest(calibration.identity()))
        return evidence, calibration

    def _rule_matrices(self, evidence, requests, policy, quality, cost, outcomes=None):
        """Forecast, expected-cost and outcome matrices in each request's selection order."""
        import numpy as np
        m = len(evidence.endpoint_ids)
        F = np.full((len(requests), m), -np.inf)
        C = np.full((len(requests), m), np.nan)
        V = np.full((len(requests), m), np.nan)
        complete = np.ones(len(requests), dtype=bool)
        for i, request in enumerate(requests):
            rule = self._pooled_rule(evidence, request, policy, quality[i], cost[i]) or []
            for position, (endpoint_id, forecast, cost_mean) in enumerate(rule):
                F[i, position], C[i, position] = forecast, cost_mean
                if outcomes is not None:
                    V[i, position] = outcomes[i, evidence.column(endpoint_id)]
                    complete[i] &= V[i, position] == V[i, position]
        return F, C, V, complete

    def _utility_calibration(self, evidence, policy, risk: float | None = None):
        """Learn-then-test over price penalties: lambda = 0 (quality first) upward to cheap."""
        from inference_control.routing.calibration import calibrate_sequence, utility_choice, utility_grid
        import numpy as np
        cmap = self.capability_map
        train_q, train_c = evidence.training_forecasts()
        ref_f, ref_cost, _, _ = self._rule_matrices(evidence, evidence.train_requests, policy, train_q, train_c)
        with np.errstate(invalid="ignore"):  # -inf forecasts of disallowed endpoints differ to nan
            grid = utility_grid(ref_f, ref_cost)
        requests = evidence.calibration_requests
        quality, cost = evidence.forecast(requests)
        F, C, V, complete = self._rule_matrices(evidence, requests, policy, quality, cost,
                                                evidence.calibration_quality)
        routable = np.isfinite(F).any(axis=1) & complete  # unscored allowed endpoints: excluded
        F, C, V = F[routable], C[routable], V[routable]
        rows = np.arange(len(F))
        values = []
        for lam in grid:
            choice = utility_choice(F, C, lam)
            values.append(V[rows, choice])
        minimum = max(cmap.min_samples, policy.minimum_evidence_samples)
        return calibrate_sequence(values, grid, target=policy.minimum_quality,
                                  risk=policy.quality_risk if risk is None else risk,
                                  minimum_requests=minimum, kind="utility")

    def _certified_endpoint(self, rule, calibration):
        """The endpoint the calibrated rule certifies for one request, or None."""
        if not rule or calibration is None:
            return None
        parameter = getattr(calibration, "parameter", None)
        if getattr(calibration, "kind", None) == "utility":
            if parameter is None:
                return None
            best = max(range(len(rule)), key=lambda k: (rule[k][1] - parameter * rule[k][2], -k))
            return rule[best][0]
        threshold = getattr(calibration, "threshold", None)
        if threshold is None:
            return None
        return next((endpoint for endpoint, forecast, _ in rule if forecast >= threshold), None)

    def _calibration_digest(self, calibration) -> str:
        for stored, value in self._pooled_calibrations.values():
            if stored is calibration:
                return value
        return digest(calibration.identity())

    def _digest_once(self, value) -> str:
        """digest() memoized per immutable object (policies and pooled configs are frozen)."""
        cache = self.__dict__.setdefault("_digest_cache", {})
        cached = cache.get(id(value))
        if cached is None or cached[0] is not value:
            if len(cache) >= 256:
                cache.clear()
            cached = (value, digest(value.identity() if hasattr(value, "identity") else value))
            cache[id(value)] = cached
        return cached[1]

    def _snapshot_signature(self) -> tuple:
        endpoints = tuple(self.endpoints[e] for e in sorted(self.endpoints))
        cache = self.__dict__.setdefault("_snapshot_cache", {})
        key = tuple(id(e) for e in endpoints)
        cached = cache.get(key)
        if cached is None or any(a is not b for a, b in zip(cached[0], endpoints)):
            if len(cache) >= 256:
                cache.clear()
            cached = (endpoints, tuple(sorted(e.snapshot_id for e in endpoints)))
            cache[key] = cached
        return cached[1]

    def _pooled_candidates(self, request, policy, plans, rejected, at):
        rejected = list(rejected)
        direct = [p for p in plans if p.plan_type == "direct"]
        for plan in plans:
            if plan.plan_type != "direct":
                rejected.append(RejectedAlternative(endpoint_id=plan.evidence_key, plan_type=plan.plan_type,
                                                    reason_codes=("POOLED_DIRECT_ONLY",)))
        if not direct:
            return (), rejected, None
        cmap = self.capability_map
        evidence, calibration = self.pooled_calibration(policy, at=at, feature_version=request.feature_version)
        quality, cost = evidence.forecast([request])
        base = digest({"method": POOLED, "evidence": evidence.identity,
                       "calibration": self._calibration_digest(calibration), "policy": self._digest_once(policy)})
        minimum = max(cmap.min_samples, policy.minimum_evidence_samples)
        route, assumptions = None, None
        if getattr(calibration, "kind", None) == "cascade":
            route = self._cascade_route(evidence, request, policy, quality, cost, calibration)
            chosen = route.direct_endpoint if route else None
            assumptions = CASCADE_ASSUMPTIONS
        else:
            rule = self._pooled_rule(evidence, request, policy, quality[0], cost[0])
            chosen = self._certified_endpoint(rule, calibration)
        candidates = []
        for plan in direct:
            endpoint = self.endpoints[plan.steps[0].endpoint_id]
            if endpoint.endpoint_id not in evidence.endpoint_ids:
                candidates.append((plan, None))
                continue
            column = evidence.column(endpoint.endpoint_id)
            numbers = cmap.pooled_numbers(evidence, request, endpoint, float(cost[0, column]),
                quality_risk=policy.quality_risk, latency_risk=policy.latency_risk,
                comparisons=len(direct), minimum=minimum)
            estimate = cmap.pooled_estimate(evidence, request, endpoint, float(quality[0, column]), numbers,
                at=at, quality_risk=policy.quality_risk, target=policy.minimum_quality,
                calibration=calibration, artifact_base=base, certified=endpoint.endpoint_id == chosen,
                assumptions=assumptions)
            candidates.append((plan, estimate))
        notes = None
        if route is not None:
            notes = dict(route.notes)
            if route.check:
                candidates.append(self._compound_candidate(evidence, request, policy, route, calibration, base, at))
        return tuple(candidates), rejected, notes

    # ----- live answer-agreement cascade ------------------------------------------------------
    def _cascade_enabled(self, policy) -> bool:
        """The policy opts in to agree_escalate and can run it: a registered agreement function,
        two or more calls, and direct plans for the requests that skip the check."""
        return ("agree_escalate" in policy.permitted_plan_types and "direct" in policy.permitted_plan_types
                and bool(policy.verifier) and policy.max_calls >= 2)

    def _cascade_reference(self, evidence, requests, policy, quality, cost, *, outcomes=None, realized=None,
                           answers=None, answered=None, fill_realized=False):
        """Selection-ordered forecast/cost/outcome/agreement matrices of the routing rule for ``requests``.

        Serving builds a one-row reference with this same function, so the allowed endpoints, their
        order, forecasts and costs match what calibration replayed. ``complete`` marks requests
        scored, with a recorded answer (and, unless ``fill_realized``, a realized cost), on every
        allowed endpoint.
        """
        import numpy as np
        from inference_control.routing import cascade
        from inference_control.routing.live_cascade import CascadeReference
        m, n = len(evidence.endpoint_ids), len(requests)
        F = np.full((n, m), -np.inf)
        C = np.full((n, m), np.nan)
        absolute = np.zeros((n, m))
        columns = np.full((n, m), -1, dtype=int)
        Q = np.full((n, m), np.nan) if outcomes is not None else None
        R = np.full((n, m), np.nan) if realized is not None else None
        rows = [[None] * m for _ in range(n)] if answers is not None else None
        complete = np.ones(n, dtype=bool)
        ids, numbers = [], []
        for i, request in enumerate(requests):
            details = {}
            rule = self._pooled_rule(evidence, request, policy, quality[i], cost[i], details=details) or []
            ids.append([endpoint_id for endpoint_id, _, _ in rule])
            numbers.append([details[endpoint_id] for endpoint_id, _, _ in rule])
            for position, (endpoint_id, forecast, cost_mean) in enumerate(rule):
                column = evidence.column(endpoint_id)
                F[i, position], C[i, position], columns[i, position] = forecast, cost_mean, column
                absolute[i, position] = self.absolute_cost(endpoint_id, request)
                if Q is not None:
                    Q[i, position] = outcomes[i, column]
                    complete[i] &= Q[i, position] == Q[i, position]
                if R is not None:
                    R[i, position] = realized[i, column]
                    if fill_realized and R[i, position] != R[i, position]:
                        R[i, position] = cost_mean
                    complete[i] &= R[i, position] == R[i, position]
                if rows is not None:
                    rows[i][position] = answers[i, column]
                    complete[i] &= bool(answered[i, column])
        agree = cascade.agreement_matrix(rows) if rows is not None else None
        return CascadeReference(list(requests), ids, F, C, columns, absolute, numbers, Q, R, agree, complete)

    def _training_disagreement(self, evidence, minimum: int):
        """P(two endpoints' answers differ) by evidence-column pair, from training answers only.

        Pairs with fewer than ``minimum`` jointly recorded training answers get 1.0 (always escalate),
        the conservative value for the expected-spend gate.
        """
        import numpy as np
        from inference_control.routing import cascade
        m = len(evidence.endpoint_ids)
        if evidence.train_answers is None or not len(evidence.train_answers):
            return np.ones((m, m))
        agree = cascade.agreement_matrix(evidence.train_answers.tolist())
        both = evidence.train_answered[:, :, None] & evidence.train_answered[:, None, :]
        count = both.sum(axis=0)
        agreed = (agree & both).sum(axis=0)
        return np.where(count >= minimum, 1. - agreed / np.maximum(count, 1), 1.)

    def _compound_values(self, reference, i, f, k, z, disagree):
        """Spend, cost, latency and failure predictions of one compound plan, from the rule's numbers.

        Members are first, checker and (when distinct) final, in call order. Expected cost adds the
        escalation call weighted by the training disagreement rate of the (first, checker) pair;
        the latency bound is max(first, checker) + final; failure is the worst member's.
        """
        members = [f, k] + ([z] if z not in (f, k) else [])
        extra = len(members) == 3
        numbers = [reference.numbers[i][p] for p in members]
        spend = sum(float(reference.absolute[i, p]) for p in members)
        p_differ = float(disagree[reference.columns[i, f], reference.columns[i, k]])
        expected = float(reference.costs[i, f]) + float(reference.costs[i, k])
        mean_latency = max(numbers[0]["latency"][0], numbers[1]["latency"][0])
        upper_latency = max(numbers[0]["latency"][2], numbers[1]["latency"][2])
        if extra:
            expected += p_differ * float(reference.costs[i, z])
            mean_latency += p_differ * numbers[2]["latency"][0]
            upper_latency += numbers[2]["latency"][2]
        failure = [n["failure"] for n in numbers]
        reasons = tuple(dict.fromkeys(r for n in numbers for r in n["reasons"]))
        return {"members": members, "spend": spend, "reasons": reasons,
                "cost": {"mean": expected, "lower": min(expected, 0.), "upper": max(spend, expected)},
                "latency": {"mean": mean_latency, "lower": 0., "upper": max(mean_latency, upper_latency)},
                "failure": {"mean": max(x[0] for x in failure), "lower": max(x[1] for x in failure),
                            "upper": max(x[2] for x in failure)}}

    def _compound_gate(self, policy, disagree):
        """gate(reference, i, first, checker, final): may the compound plan run for request i?

        Applies reasons() (call limit, spend bounds, expected spend, deadline) and the failure bound to
        the compound plan exactly as select() does when serving. Memoized per request and triple.
        """
        memo = {}
        def gate(reference, i, f, k, z):
            key = (id(reference), i, f, k, z)
            if key not in memo:
                values = self._compound_values(reference, i, f, k, z, disagree)
                stub = _EstimateStub(_Value(min(1., max(policy.minimum_quality, min(1., max(0., float(reference.forecasts[i, z]))))),
                                            policy.minimum_quality, 1.), _Value(**values["cost"]),
                                     _Value(**values["latency"]), _Value(**values["failure"]), _PooledLineage())
                plan = _PlanStub("agree_escalate", values["spend"], max_calls=len(values["members"]),
                                 plan_type="agree_escalate")
                hard, _ = self.reasons(plan, stub, policy)
                memo[key] = not hard and not (policy.maximum_failure_probability < 1
                                              and stub.failure.upper > policy.maximum_failure_probability)
            return memo[key]
        return gate

    def _cascade_calibration(self, evidence, policy):
        """Calibrate the answer-agreement cascade for this policy, or None when it is not attempted.

        Not attempted: the policy does not permit agree_escalate (or lacks an agreement function or a
        second call), the evidence carries no answers, or there are too few fully scored requests to
        test anything. Otherwise returns a LiveCascade whose ``config`` is None when no configuration
        passed. Configurations are ordered on training references (leave-one-out forecasts, training
        outcomes and answers) before any calibration outcome is read.
        """
        if not self._cascade_enabled(policy) or not evidence.has_answers:
            return None
        import numpy as np
        from inference_control.routing import cascade, live_cascade
        from inference_control.routing.calibration import utility_grid
        cmap = self.capability_map
        minimum = max(cmap.min_samples, policy.minimum_evidence_samples)
        disagree = self._training_disagreement(evidence, minimum)
        gate = self._compound_gate(policy, disagree)
        train_q, train_c = evidence.training_forecasts()
        train = self._cascade_reference(evidence, evidence.train_requests, policy, train_q, train_c,
            outcomes=evidence.train_quality, realized=evidence.train_cost, answers=evidence.train_answers,
            answered=evidence.train_answered, fill_realized=True)
        # The switching points come from every training request's forecasts and costs, answers or not.
        with np.errstate(invalid="ignore"):
            grid = utility_grid(train.forecasts, train.costs, steps=30)
        reference = train.subset(np.flatnonzero(train.complete & np.isfinite(train.forecasts).any(axis=1)))
        cal_q, cal_c = evidence.forecast(evidence.calibration_requests)
        calibration = self._cascade_reference(evidence, evidence.calibration_requests, policy, cal_q, cal_c,
            outcomes=evidence.calibration_quality, realized=evidence.calibration_cost,
            answers=evidence.calibration_answers, answered=evidence.calibration_answered)
        calibration = calibration.subset(np.flatnonzero(calibration.complete
                                                        & np.isfinite(calibration.forecasts).any(axis=1)))
        if len(reference) == 0 or len(calibration) < minimum:
            return None
        # Vote configurations (if the offline module grows them) are never served live.
        configs = tuple(c for c in cascade.default_configs(grid) if not getattr(c, "voters", None))
        ordered = live_cascade.order_gated(configs, reference, gate)
        result = live_cascade.calibrate_gated(ordered, calibration, gate, target=policy.minimum_quality,
                                              risk=policy.quality_risk / 2, minimum_requests=minimum)
        return live_cascade.LiveCascade(result, policy.verifier, evidence.endpoint_ids,
                                        tuple(tuple(float(x) for x in row) for row in disagree))

    def _cascade_route(self, evidence, request, policy, quality, cost, live):
        """The cascade's decision for one request, or None when it is not routable.

        Uses the same reference builder and gate as calibration; ``quality``/``cost`` are the
        request's forecasts (1 x endpoints).
        """
        import numpy as np
        from inference_control.routing.live_cascade import gated_plan
        disagree = np.array(live.disagreement)
        reference = self._cascade_reference(evidence, [request], policy, quality, cost)
        gate = self._compound_gate(policy, disagree)
        first, checker, final, check, raw = gated_plan(reference, live.config, gate)
        f, k, z = int(first[0]), int(checker[0]), int(final[0])
        if f < 0:
            return None
        ids = reference.ids[0]
        planned = bool(check[0])
        note = {"method": live.method, "config": {"lam": live.config.lam, "accept": live.config.accept},
                "agreement": live.agreement, "check_planned": planned, "first": ids[f],
                "check_gated_off": bool(raw[0] and not planned),
                "calibration_requests": live.calibration_requests, "calibration_mean_quality": live.certified_mean}
        if planned:
            note.update(checker=ids[k], final=ids[z])
        # With a check planned only the compound plan carries the certificate; the first endpoint's
        # direct plan is not certified on its own.
        return _CascadeRoute(reference, disagree, f, k, z, planned, None if planned else ids[f], {"cascade": note})

    def _compound_candidate(self, evidence, request, policy, route, live, base, at):
        """The certified agree_escalate plan and its estimate for a request whose check is planned."""
        reference, f, k, z = route.reference, route.first, route.checker, route.final
        values = self._compound_values(reference, 0, f, k, z, route.disagree)
        ids = [reference.ids[0][p] for p in values["members"]]
        steps = [Call(endpoint_id=e, max_output_tokens=request.max_output_tokens,
                      generation_config=self.endpoints[e].inference_config) for e in ids]
        steps.insert(2, Verify(verifier=policy.verifier, acceptance_rule="agree"))
        bound = sum(self.absolute_cost(e, request) for e in ids)
        identity = digest({"kind": "agree_escalate", "steps": steps, "bound": bound})[:32]
        plan = ExecutionPlan(plan_id=identity, plan_type="agree_escalate", steps=tuple(steps),
                             max_calls=len(ids), max_spend=bound)
        estimate = self.capability_map.pooled_compound_estimate(
            evidence, request, plan, tuple(self.endpoints[e] for e in ids), values,
            quality_forecast=float(reference.forecasts[0, z]), reasons=values["reasons"], at=at,
            quality_risk=policy.quality_risk, target=policy.minimum_quality, calibration=live, artifact_base=base)
        return plan, estimate

    def select(self, candidates, policy, *, complete, rejected=(), notes=None):
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
        if chosen is None and complete and not policy.require_certificate and policy.infeasible_behavior == InfeasibleBehavior.BEST_PREDICTED:
            # Uncertified fallback to the plan with the highest forecast quality. Spend, deadline,
            # privacy, call caps and the failure bound never relax.
            choices = [(p,e) for p,e in shortfall if policy.maximum_failure_probability >= 1 or
                       (e.failure and e.failure.upper <= policy.maximum_failure_probability)]
            scores = self._fallback_scores(choices)
            chosen = min(choices, key=lambda x: (-scores[id(x)], objective(x))) if choices else None
            if chosen: fallback = "best_predicted_uncertified"
        if chosen is None and complete and not policy.require_certificate and policy.infeasible_behavior == InfeasibleBehavior.UNCERTIFIED:
            # Explicit quality-risk relaxation only. Spend, deadline, privacy and call caps never relax.
            choices = [(p,e) for p,e in shortfall if policy.maximum_failure_probability >= 1 or
                       (e.failure and e.failure.upper <= policy.maximum_failure_probability)]
            chosen = min(choices, key=lambda x: (max(0, policy.minimum_quality-x[1].quality.lower), objective(x))) if choices else None
            if chosen: fallback = "least_shortfall_uncertified"
        return PlanningAssessment(tuple(candidates), tuple(feasible), tuple(rejected), complete, chosen, fallback, notes)

    def _fallback_scores(self, choices) -> dict:
        """Forecast quality, plus the pooled fallback margin for the best-training-mean endpoint."""
        scores = {id(x): x[1].quality.mean for x in choices}
        margin = self.capability_map.pooled_config.fallback_margin if self.pooled else 0.
        if margin and choices:
            means = [(x[1].lineage.quality_diagnostics.training_mean, x) for x in choices if _pooled_estimate(x[1])
                     and x[1].lineage.quality_diagnostics.training_mean is not None]
            if means:
                # Ties on the training mean resolve to the lexicographically first plan key.
                anchor = min(means, key=lambda m: (-m[0], m[1][0].evidence_key))[1]
                scores[id(anchor)] += margin
        return scores

    def decide(self, request: RequestContext, policy: PolicySpec, seed: int = 0,
               *, at: datetime | None = None) -> DecisionRecord:
        started, at = perf_counter(), at or datetime.now(timezone.utc)
        assessment = self.assess(request, policy, at=at)
        chosen, feasible = assessment.selection, assessment.feasible
        complete, fallback = assessment.complete, assessment.fallback
        notes = assessment.notes
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
                    (f"{POOLED}/{notes['cascade']['method'] if notes else 'hb-fixed-sequence-v1'}/binomial-p95-v3"
                     if _pooled_estimate(estimate) else
                     f"{lineage.quality_diagnostics.method}/binomial-p95-v3") if lineage.quality_diagnostics else "local-knn-v1",
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
            replay_seed=seed, decision_latency_ms=(perf_counter()-started)*1000,
            plan_metadata={"cascade": {**notes["cascade"], "served_plan_type": plan.plan_type}} if notes else None)
