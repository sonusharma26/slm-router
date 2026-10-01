"""Durable shadow/canary orchestration; transitions run on requests/outcomes, no daemon."""
from __future__ import annotations
from dataclasses import dataclass
import math
from inference_control.util import digest


@dataclass(frozen=True)
class OfflineEvidence:
    dataset_hash: str
    split_hash: str
    policy_hash: str
    quality_lower: float
    constraint_violations: int
    evaluated_requests: int
    trusted_outcomes: int
    # Keep dataset evidence distinct from synthetic smoke diagnostics.
    evidence_kind: str = "full_information"


class LifecycleOrchestrator:
    def __init__(self, state, *, min_shadow=50, min_canary=100, max_canary_fraction=.1,
                 max_latency_violations=3):
        if min_shadow < 1 or min_canary < 1 or not 0 < max_canary_fraction <= 1:
            raise ValueError("invalid rollout bounds")
        self.state, self.min_shadow, self.min_canary = state,min_shadow,min_canary
        self.max_fraction, self.max_latency = max_canary_fraction,max_latency_violations

    def _save(self, item):
        self.state.write("lifecycle",item,key=f"lifecycle:{digest(item)}")
        return item

    def bootstrap(self, policy):
        """Explicit operator bootstrap only; never masquerades as experimentally promoted."""
        if policy.policy_id in self.state.lifecycle: return self.state.lifecycle[policy.policy_id]
        return self._save({"policy_id":policy.policy_id,"active":policy.version,"rollback":None,
            "candidate":None,"state":"active","reason":"operator_bootstrap_unvalidated",
            "fraction":0.,"shadow_count":0,"canary_samples":{},"executions":{},"history":[]})

    def propose(self, policy):
        group=dict(self.state.lifecycle.get(policy.policy_id) or {})
        if not group: raise ValueError("bootstrap an incumbent before proposing a candidate")
        if group.get("candidate"): raise ValueError("only one candidate per policy is supported")
        if policy.version == group["active"]: raise ValueError("candidate must have a new version")
        self.state.write("policy", policy.model_dump(mode="json"), key=f"policy:{digest(policy)}")
        group.update(candidate=policy.version,state="draft",reason="candidate_created",shadow_count=0,
                     canary_samples={},executions={},history=group["history"]+[{"version":policy.version,"state":"draft"}])
        return self._save(group)

    def validate_offline(self, policy, evidence: OfflineEvidence):
        row=dict(self.state.lifecycle[policy.policy_id])
        if row["state"]!="draft" or row["candidate"]!=policy.version: raise ValueError("candidate is not draft")
        if not evidence.dataset_hash or not evidence.split_hash or evidence.policy_hash!=digest(policy):
            raise ValueError("offline evidence lineage mismatch")
        if evidence.evidence_kind not in {"full_information","live_holdout"}:
            raise ValueError("synthetic/proxy evidence cannot promote production policies")
        if evidence.evaluated_requests < self.min_canary or evidence.trusted_outcomes < self.min_canary:
            raise ValueError("insufficient trusted offline evidence")
        if evidence.constraint_violations or evidence.quality_lower < policy.minimum_quality:
            raise ValueError("offline gate failed")
        row.update(state="offline_validated",reason="offline_gate_passed",offline_hash=digest(evidence))
        self._save(row)
        row={**row,"state":"shadow","reason":"automatic_shadow_start"}
        return self._save(row)

    def route(self, request, active_policy, candidate_policy, planner):
        row=dict(self.state.lifecycle[active_policy.policy_id])
        if row.get("candidate") != (candidate_policy.version if candidate_policy else None):
            raise ValueError("candidate policy version mismatch")
        if row["state"] == "stale":
            raise ValueError("policy halted by a stop rule; operator recovery required")
        incumbent=planner.decide(request,active_policy)
        if row["state"]=="shadow":
            candidate=planner.decide(request,candidate_policy)
            record={"policy_id":active_policy.policy_id,"request_id":request.request_id,
                    "incumbent":incumbent.selected_plan.evidence_key,"candidate":candidate.selected_plan.evidence_key,
                    "candidate_certificate":candidate.certificate_status,
                    "candidate_version":candidate_policy.version,"user_affected":False}
            key=f"shadow:{active_policy.policy_id}:{candidate_policy.version}:{request.request_id}"
            event=self.state.write("shadow_decision",record,key=key)
            # Counts are deduplicated from durable unique shadow events.
            shadow=[e for e in self.state.ledger.events("shadow_decision") if e.payload["policy_id"]==active_policy.policy_id
                    and e.payload["candidate_version"]==candidate_policy.version]
            row["shadow_count"]=len(shadow)
            if row["shadow_count"]>=self.min_shadow:
                row.update(state="canary",fraction=self.max_fraction,reason="shadow_gate_passed")
            self._save(row)
            return incumbent,active_policy
        if row["state"]=="canary":
            unit=request.session_id or request.request_id
            bucket=int(digest((request.application_id,unit,candidate_policy.version))[:16],16)/2**64
            if bucket < row["fraction"]:
                return planner.decide(request,candidate_policy),candidate_policy
        return incumbent,active_policy

    def observe_execution(self, decision, execution):
        row=self.state.lifecycle.get(decision.policy_id)
        if not row or decision.policy_version not in {row.get("candidate"),row.get("active")}:
            return
        row=dict(row)
        if row["state"] not in {"canary","active"}: return
        target=row.get("candidate") if row["state"]=="canary" else row.get("active")
        if decision.policy_version != target:return
        executions=dict(row.get("executions",{}))
        executions[decision.decision_id]={"state":execution.state,"violations":list(execution.constraint_violations),
            "accounting_complete":execution.accounting_complete}
        row["executions"]=executions
        hard=any(v in {"REALIZED_SPEND_EXCEEDED","INPUT_TOKEN_BOUND_EXCEEDED","OUTPUT_TOKEN_BOUND_EXCEEDED",
                      "ENDPOINT_MODEL_MISMATCH","ENDPOINT_REVISION_MISMATCH","ACCOUNTING_INCOMPLETE","MALFORMED_STRUCTURED_OUTPUT","MALFORMED_TOOL_CALL"} for v in execution.constraint_violations)
        late=sum("REALIZED_DEADLINE_EXCEEDED" in r["violations"] for r in executions.values())
        malformed=execution.state=="failed" and not execution.provider_errors
        if hard or late>=self.max_latency or malformed:
            return self.abort(row,"execution_stop_rule")
        self._save(row)

    def observe_outcome(self, decision, outcome, policy):
        row=self.state.lifecycle.get(decision.policy_id)
        if not row or row["state"]!="canary" or row["candidate"]!=decision.policy_version: return
        row=dict(row)
        samples=dict(row["canary_samples"])
        if not outcome.promotion_eligible or outcome.disputed:
            samples.pop(decision.decision_id,None)
        else:
            score=outcome.quality.get("success",outcome.quality.get("score"))
            if score is None or not 0<=score<=1:return
            samples[decision.decision_id]=score
        row["canary_samples"]=samples
        n=len(samples)
        if n:
            average=sum(samples.values())/n
            # Anytime union bound over sample-count looks; no repeated naive p-value peeking.
            alpha=policy.quality_risk/(n*(n+1))
            radius=math.sqrt(math.log(2/alpha)/(2*n))
            if n>=self.min_canary and average+radius<policy.minimum_quality:
                return self.abort(row,"quality_sequential_stop")
            if n>=self.min_canary and average-radius>=policy.minimum_quality:
                row.update(rollback=row["active"],active=row["candidate"],candidate=None,state="active",
                           fraction=0.,reason="canary_quality_gate_passed")
        return self._save(row)

    def abort(self, row, reason):
        row=dict(row)
        if row["state"]=="active":
            if not row.get("rollback"):
                row.update(state="stale",reason=reason)
            else:
                row.update(active=row["rollback"],rollback=None,state="active",reason="automatic_rollback:"+reason,
                           executions={})
        else:
            row.update(candidate=None,state="active",fraction=0.,reason="canary_rejected:"+reason,executions={})
        return self._save(row)
