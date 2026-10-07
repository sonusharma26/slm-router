"""Closed-loop, decision-impact probing with persistent hard daily reservations.

The acquisition score is an auditable decision-boundary heuristic, not an exact
Bayesian expected VOI claim. Failed/unknown-cost probes keep their entire reservation.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from time import perf_counter
from typing import Callable
from uuid import uuid4
from inference_control.capability.conditional import EvidenceObservation
from inference_control.adapters.providers import ProviderTimeout
from inference_control.contracts import Prediction
from inference_control.planning.planner import CapabilityEstimate
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


@dataclass(frozen=True)
class _PotentialCalibration:
    calibrated: bool = True


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
        known = {claim:row["cost"] for claim,row in completed.items() if row.get("cost") is not None}
        return {"day":day,"calls_reserved":len(rows),"dollars_reserved":sum(r["bound"] for r in rows),
                "known_cost_coverage":sum(r["claim"] in known for r in rows)/len(rows) if rows else 1.,
                "realized_known_cost":sum(known.get(r["claim"],0.) for r in rows)}


class ActiveMeasurementLoop:
    def __init__(self, planner, budget: DailyProbeBudget, *, recovery, state=None, seed: int = 0):
        if planner.capability_map is None: raise ValueError("conditional capability map required")
        if recovery.planner is not planner: raise ValueError("probe recovery must own the serving planner")
        if recovery.state is not None and recovery.state.ledger is not budget.ledger:
            raise ValueError("probe recovery and budget must share a ledger")
        self.planner, self.budget, self.state, self.seed = planner, budget, state or recovery.state, seed
        self.recovery = recovery

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
                cached[key] = self.planner.assess(task.request,policy,at=at)
            assessment = cached[key]
            estimates = assessment.candidates
            match = next(((p,e) for p,e in estimates if p.plan_type=="direct" and p.steps[0].endpoint_id==task.endpoint_id),None)
            if match is None: continue
            plan, est = match
            if plan.max_spend > policy.max_absolute_spend: continue
            uncertainty = est.quality.upper-est.quality.lower if est else 1.
            boundary = max(0.,1-abs((est.quality.mean if est else .5)-policy.minimum_quality)/max(uncertainty,.001))
            stale = est is None or (est.lineage is not None and (not est.lineage.calibrated
                or est.lineage.age_seconds >= policy.evidence_max_age_seconds/2))
            if est is None:
                optimistic = CapabilityEstimate(plan.evidence_key,Prediction(mean=1,lower=1,upper=1),
                    Prediction(mean=plan.max_spend,lower=0,upper=plan.max_spend),Prediction(mean=0,lower=0,upper=0),
                    Prediction(mean=0,lower=0,upper=0),_PotentialCalibration())
            else:
                failure = est.failure.lower if est.failure else 0.
                optimistic = replace(est,
                    quality=Prediction(mean=est.quality.upper,lower=est.quality.upper,upper=est.quality.upper),
                    latency=Prediction(mean=est.latency.lower,lower=est.latency.lower,upper=est.latency.lower),
                    failure=Prediction(mean=failure,lower=failure,upper=failure),lineage=_PotentialCalibration())
            def projected(value):
                family = tuple((p,value if p.evidence_key==plan.evidence_key else e) for p,e in estimates)
                return self.planner.select(family,policy,complete=assessment.complete)
            def identity(search):
                return (search.selection[0].evidence_key,search.fallback) if search.selection else None
            potential = projected(optimistic)
            impact = assessment.complete and (identity(potential)!=identity(assessment) or (
                stale and identity(projected(None))!=identity(assessment)))
            score = task.traffic_mass*(uncertainty+boundary+float(stale))*float(impact)/max(plan.max_spend,1e-9)
            if strategy == "uncertainty": score = task.traffic_mass*uncertainty/max(plan.max_spend,1e-9)
            elif strategy == "random": score = int(digest((self.seed,task.probe_id,task.endpoint_id))[:12],16)/16**12
            elif strategy == "exhaustive": score = 1.
            reason = "potential_independent_calibration" if impact else "cannot_change_current_choice"
            if not assessment.complete: reason = "planning_incomplete"
            choices.append(ProbeChoice(task,score,uncertainty,boundary,impact,plan.max_spend,reason))
        return sorted(choices,key=lambda c:(-c.score,c.task.endpoint_id,c.task.probe_id))

    def run(self, tasks: list[ProbeTask], policy, measure: Callable[[ProbeTask],EvidenceObservation], *,
            strategy="impact", max_calls: int | None = None, at=None) -> dict:
        realtime = at is None
        at = at or datetime.now(timezone.utc)
        completed, failed, skipped, attempted = [], [], [], 0
        drift_events = list(self.recovery.reconcile_latency())
        for choice in self.rank(tasks,policy,strategy=strategy,at=at):
            if max_calls is not None and attempted >= max_calls: break
            if strategy=="impact" and not choice.could_change_decision:
                skipped.append(choice.task.probe_id); continue
            claim = self.budget.reserve(digest((choice.task.probe_id,choice.task.endpoint_id)),choice.reserved_cost,at=at)
            if claim is None: continue
            attempted += 1
            started = perf_counter()
            sampled_at = datetime.now(timezone.utc) if realtime else at
            context = {"endpoint_id":choice.task.endpoint_id,
                       "capability_revision":self.planner.endpoints[choice.task.endpoint_id].capability_revision,
                       "slice_ids":sorted(choice.task.request.traffic_slices or {"default"}),
                       "policy_hash":digest(policy),"deadline_ms":policy.deadline_ms,
                       "observed_at":sampled_at.isoformat(),"received_at":sampled_at.isoformat(),"timed_out":False}
            try:
                row = measure(choice.task)
                if row.request != choice.task.request or row.endpoint_ids != (choice.task.endpoint_id,):
                    raise ValueError("probe outcome does not match requested endpoint/request")
                expected = self.planner.endpoints[choice.task.endpoint_id].capability_revision
                if row.endpoint_revisions != (expected,): raise ValueError("probe revision mismatch")
                if "cost" in row.metrics and row.cost > choice.reserved_cost+1e-12: raise ValueError("probe provider exceeded reserved spend")
                if row.evaluator_type in {"proxy","calibrated_judge"} and row.trusted:
                    raise ValueError("measurement requires deterministic/application/human authority; calibrate judges through outcomes")
                row = row.model_copy(update={"observed_at":min(sampled_at,row.observed_at)})
                self.planner.capability_map.add(row)
                context["observed_at"] = row.observed_at.isoformat()
                context["received_at"] = (datetime.now(timezone.utc) if realtime else max(at,row.observed_at)).isoformat()
                payload = {"claim":claim,"cost":row.cost if "cost" in row.metrics else None,
                    "cost_known":"cost" in row.metrics,"latency_context":context,
                    "observation":row.model_dump(mode="json"),"score":choice.score,"reason":choice.reason}
                self.budget.ledger.append("probe_completed",payload,
                    actor="measurement",idempotency_key=f"probe-completed:{claim}")
                completed.append(row.sample_id)
            except Exception as exc:
                context.update(duration_ms=(perf_counter()-started)*1000,timed_out=isinstance(exc,ProviderTimeout))
                context["received_at"] = (datetime.now(timezone.utc) if realtime else at).isoformat()
                payload = {"claim":claim,"error_type":type(exc).__name__,"latency_context":context}
                self.budget.ledger.append("probe_failed",payload,
                    actor="measurement",idempotency_key=f"probe-failed:{claim}")
                failed.append(choice.task.probe_id)
            drift_events.extend(self.recovery.observe_probe(payload))
        if self.state: self.state.store_planner(self.planner)
        return {"completed":completed,"failed":failed,"skipped_no_decision_impact":skipped,
                "attempted":attempted,"budget":self.budget.usage(at),"map_version":self.planner.map_version,
                "drift_events":drift_events}
