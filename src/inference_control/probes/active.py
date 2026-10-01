"""Closed-loop, decision-impact probing with persistent hard daily reservations.

The acquisition score is an auditable decision-boundary heuristic, not an exact
Bayesian expected VOI claim. Failed/unknown-cost probes keep their entire reservation.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Callable
from uuid import uuid4
from inference_control.capability.conditional import EvidenceObservation
from inference_control.contracts import Call
from inference_control.util import digest


@dataclass(frozen=True)
class ProbeTask:
    probe_id: str
    endpoint_id: str
    request: object
    traffic_mass: float = 1.


@dataclass(frozen=True)
class ProbeChoice:
    task: ProbeTask
    score: float
    uncertainty: float
    boundary_sensitivity: float
    could_change_decision: bool
    reserved_cost: float
    reason: str


class DailyProbeBudget:
    def __init__(self, ledger, *, max_dollars: float, max_calls: int):
        if not math.isfinite(max_dollars) or max_dollars < 0 or max_calls < 0:
            raise ValueError("nonnegative finite probe budgets required")
        self.ledger, self.max_dollars, self.max_calls = ledger, max_dollars, max_calls

    def reserve(self, probe_key: str, bound: float, *, at=None) -> str | None:
        at = at or datetime.now(timezone.utc)
        if not math.isfinite(bound) or bound < 0: raise ValueError("invalid reservation")
        day, claim = at.astimezone(timezone.utc).date().isoformat(), str(uuid4())
        # Count-and-reserve happens under the same SQLite write transaction across processes.
        with self.ledger.lock:
            db = self.ledger.connection
            try:
                db.execute("BEGIN IMMEDIATE")
                rows = self.ledger.events("probe_reserved")
                today = [e.payload for e in rows if e.payload["day"] == day]
                if any(r["probe_key"] == probe_key for r in today):
                    db.rollback(); return None
                if len(today) >= self.max_calls or sum(r["bound"] for r in today)+bound > self.max_dollars+1e-12:
                    db.rollback(); return None
                self.ledger._append_locked("probe_reserved", {"claim":claim,"day":day,"probe_key":probe_key,"bound":bound},
                                           "measurement",f"probe:{day}:{probe_key}")
                db.commit()
                return claim
            except BaseException:
                db.rollback(); raise

    def usage(self, at=None):
        day = (at or datetime.now(timezone.utc)).astimezone(timezone.utc).date().isoformat()
        rows = [e.payload for e in self.ledger.events("probe_reserved") if e.payload["day"]==day]
        completed = {e.payload["claim"]:e.payload for e in self.ledger.events("probe_completed")}
        return {"day":day,"calls_reserved":len(rows),"dollars_reserved":sum(r["bound"] for r in rows),
                "realized_known_cost":sum(completed[r["claim"]].get("cost",0) for r in rows if r["claim"] in completed)}


class ActiveMeasurementLoop:
    def __init__(self, planner, budget: DailyProbeBudget, *, state=None, seed: int = 0):
        if planner.capability_map is None: raise ValueError("conditional capability map required")
        self.planner, self.budget, self.state, self.seed = planner, budget, state, seed

    def rank(self, tasks: list[ProbeTask], policy, *, strategy="impact", at=None) -> list[ProbeChoice]:
        if strategy not in {"impact","uncertainty","random","exhaustive"}: raise ValueError("unknown strategy")
        from inference_control.policies.validator import eligibility_reasons
        at = at or datetime.now(timezone.utc)
        choices = []
        # Comparison is between optimistic and pessimistic feasible alternatives, never oracle labels.
        cached = {}
        for task in tasks:
            if task.traffic_mass <= 0: continue
            ep = self.planner.endpoints.get(task.endpoint_id)
            if ep is None or eligibility_reasons(ep,task.request,policy): continue
            key = digest(task.request)
            if key not in cached:
                plans,_,_ = self.planner.generate(task.request,policy)
                estimates = [(p,self.planner.estimate_plan(task.request,policy,p,at,len(plans))) for p in plans]
                cached[key] = estimates
            estimates = cached[key]
            feasible = [(p,e) for p,e in estimates if e and not any(self.planner.reasons(p,e,policy))]
            incumbent = min(feasible,key=lambda x:(x[1].cost.mean,x[1].latency.upper)) if feasible else None
            match = next(((p,e) for p,e in estimates if p.plan_type=="direct" and p.steps[0].endpoint_id==task.endpoint_id),None)
            if match is None: continue
            plan, est = match
            if plan.max_spend > policy.max_absolute_spend: continue
            uncertainty = est.quality.upper-est.quality.lower if est else 1.
            boundary = max(0.,1-abs((est.quality.mean if est else .5)-policy.minimum_quality)/max(uncertainty,.001))
            optimistic_feasible = (est is None or (est.quality.upper >= policy.minimum_quality
                and est.latency.lower <= policy.deadline_ms and est.cost.lower <= policy.max_expected_spend))
            incumbent_target = bool(incumbent and any(isinstance(s,Call) and s.endpoint_id==task.endpoint_id for s in incumbent[0].steps))
            stale = est is None or (est.lineage is not None and (not est.lineage.calibrated
                or est.lineage.age_seconds >= policy.evidence_max_age_seconds/2))
            # This is a local decision-impact heuristic: hold nominal expected cost fixed
            # while checking whether quality/latency uncertainty can change the decision.
            # It is intentionally not called an exact expected value of information.
            if incumbent is None or est is None:
                competitive = True
            elif policy.objective == "quality":
                competitive = est.quality.upper > incumbent[1].quality.lower
            elif policy.objective == "latency":
                competitive = est.latency.lower < incumbent[1].latency.upper
            else:
                competitive = est.cost.mean < incumbent[1].cost.mean or incumbent_target
            boundary_crosses = est is None or est.quality.lower < policy.minimum_quality <= est.quality.upper or (
                est.latency.lower <= policy.deadline_ms < est.latency.upper)
            impact = optimistic_feasible and competitive and (stale or boundary_crosses)
            score = task.traffic_mass*(uncertainty+boundary+float(stale))*float(impact)/max(plan.max_spend,1e-9)
            if strategy == "uncertainty": score = task.traffic_mass*uncertainty/max(plan.max_spend,1e-9)
            elif strategy == "random": score = int(digest((self.seed,task.probe_id,task.endpoint_id))[:12],16)/16**12
            elif strategy == "exhaustive": score = 1.
            choices.append(ProbeChoice(task,score,uncertainty,boundary,impact,plan.max_spend,
                                       "decision_boundary" if impact else "cannot_change_current_choice"))
        return sorted(choices,key=lambda c:(-c.score,c.task.endpoint_id,c.task.probe_id))

    def run(self, tasks: list[ProbeTask], policy, measure: Callable[[ProbeTask],EvidenceObservation], *,
            strategy="impact", max_calls: int | None = None, at=None) -> dict:
        at = at or datetime.now(timezone.utc)
        completed, failed, skipped, attempted = [], [], [], 0
        for choice in self.rank(tasks,policy,strategy=strategy,at=at):
            if max_calls is not None and attempted >= max_calls: break
            if strategy=="impact" and not choice.could_change_decision:
                skipped.append(choice.task.probe_id); continue
            claim = self.budget.reserve(digest((choice.task.probe_id,choice.task.endpoint_id)),choice.reserved_cost,at=at)
            if claim is None: continue
            attempted += 1
            try:
                row = measure(choice.task)
                if row.request != choice.task.request or row.endpoint_ids != (choice.task.endpoint_id,):
                    raise ValueError("probe outcome does not match requested endpoint/request")
                expected = self.planner.endpoints[choice.task.endpoint_id].capability_revision
                if row.endpoint_revisions != (expected,): raise ValueError("probe revision mismatch")
                if row.cost > choice.reserved_cost+1e-12: raise ValueError("probe provider exceeded reserved spend")
                if row.evaluator_type in {"proxy","calibrated_judge"} and row.trusted:
                    raise ValueError("measurement requires deterministic/application/human authority; calibrate judges through outcomes")
                self.planner.capability_map.add(row)
                self.budget.ledger.append("probe_completed",{"claim":claim,"cost":row.cost,
                    "observation":row.model_dump(mode="json"),"score":choice.score,"reason":choice.reason},
                    actor="measurement",idempotency_key=f"probe-completed:{claim}")
                completed.append(row.sample_id)
            except Exception as exc:
                self.budget.ledger.append("probe_failed",{"claim":claim,"error_type":type(exc).__name__},
                    actor="measurement",idempotency_key=f"probe-failed:{claim}")
                failed.append(choice.task.probe_id)
        if self.state: self.state.store_planner(self.planner)
        return {"completed":completed,"failed":failed,"skipped_no_decision_impact":skipped,
                "attempted":attempted,"budget":self.budget.usage(at),"map_version":self.planner.map_version}
