"""Restart-safe control API and a deliberately thin, non-streaming Chat Completions entrypoint."""
from __future__ import annotations
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4
import hmac
import json
import math
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from inference_control.contracts import (Call, DecisionRecord, ExecutionRecord, OutcomeRecord, PolicySpec,
                                        RequestContext, Prediction)
from inference_control.execution import Executor
from inference_control.ledger import SQLiteLedger
from inference_control.ledger.state import ControlState
from inference_control.planning import Planner, CapabilityEstimate
from inference_control.outcomes import OutcomeStore
from inference_control.outcomes.trust import EvaluatorTrustStore
from inference_control.capability.conditional import ConditionalCapabilityMap, EvidenceObservation
from inference_control.lifecycle.orchestrator import LifecycleOrchestrator
from inference_control.drift.recovery import DriftRecovery
from inference_control.policies.validator import eligibility_reasons
from inference_control.util import digest
from inference_control.api.dashboard import create_dashboard_app


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

class DecideRequest(Strict):
    request: RequestContext
    policy: PolicySpec
    seed: int = 0

class ExecuteRequest(Strict):
    decision_id: str | None = None
    request: RequestContext | None = None
    policy: PolicySpec | None = None
    seed: int = 0
    messages: list[dict[str,Any]] | None = None
    response_config: dict[str,Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def source(self):
        prior=self.decision_id is not None
        inline=self.request is not None and self.policy is not None
        if prior==inline or (prior and (self.request is not None or self.policy is not None)):
            raise ValueError("provide decision_id OR request and policy")
        return self

class ChatRequest(Strict):
    model: str
    messages: list[dict[str,Any]] = Field(min_length=1,max_length=1000)
    max_tokens: int | None = Field(default=None,ge=1,le=131072)
    max_completion_tokens: int | None = Field(default=None,ge=1,le=131072)
    stream: bool = False
    response_format: dict[str,Any] | None = None
    tools: list[dict[str,Any]] | None = None
    tool_choice: Any = None
    metadata: dict[str,str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_messages(self):
        if self.max_tokens and self.max_completion_tokens and self.max_tokens!=self.max_completion_tokens:
            raise ValueError("conflicting output token limits")
        for message in self.messages:
            if message.get("role") not in {"developer","system","user","assistant","tool"}:
                raise ValueError("invalid message role")
            content=message.get("content")
            if content is not None and not isinstance(content,str):
                raise ValueError("this entrypoint accepts text messages only")
        if len(json.dumps(self.messages).encode())>1_000_000:
            raise ValueError("message payload too large")
        return self


class ControlPlane:
    def __init__(self, planner: Planner, ledger: SQLiteLedger, executor: Executor | None = None,
                 *, default_policy: PolicySpec | None = None):
        self.planner,self.ledger,self.executor = planner,ledger,executor
        self.lock=RLock()
        self.state=ControlState(ledger)
        if self.state.endpoints: self.planner.endpoints=dict(self.state.endpoints)
        if self.state.latest_map:
            self.planner.capability_map=ConditionalCapabilityMap.restore(self.state.maps[self.state.latest_map])
        elif self.state.static_estimates:
            data=self.state.static_estimates
            self.planner._map_version=data["version"]
            self.planner.estimates={r["endpoint_id"]:CapabilityEstimate(r["endpoint_id"],
                Prediction.model_validate(r["quality"]),Prediction.model_validate(r["cost"]),
                Prediction.model_validate(r["latency"]),Prediction.model_validate(r["failure"]) if r.get("failure") else None)
                for r in data["estimates"]}
        self.planner.certificates=self.state.certificates
        self.decisions,self.executions,self.policies=self.state.decisions,self.state.executions,self.state.policies
        self.outcomes=OutcomeStore()
        self.outcomes._records=self.state.outcomes
        self.outcomes._active=self.state.active_outcomes
        self.trust=EvaluatorTrustStore(self.state)
        self.lifecycle=LifecycleOrchestrator(self.state)
        self.drift=DriftRecovery(self.planner,self.state)
        # Recover observations/invalidation which were durable before a checkpoint was interrupted.
        if self.planner.capability_map is not None:
            cmap=self.planner.capability_map
            for event in ledger.events("probe_completed"):
                row=EvidenceObservation.model_validate(event.payload["observation"])
                if row.sample_id not in cmap.rows:cmap.add(row)
            for event in self.state.drifts:
                item={"endpoint_id":event["endpoint_id"],"metrics":event["metrics"],"at":event["at"],
                      "slice_id":event["slice_id"],"reason":event["reason"]}
                if item not in cmap.invalidations:cmap.invalidations.append(item);cmap._version=None
                self.planner.certificates.invalidate_dependencies(event["endpoint_id"],tuple(event["metrics"]),event["slice_id"],event["reason"],at=datetime.fromisoformat(event["at"]))
        self.drift.reconcile_latency()
        # Rebuild trusted learning rows after an outcome/checkpoint interruption.
        if self.planner.capability_map is not None:
            for saved in self.state.outcomes.values():
                if saved.disputed or self.state.active_outcomes.get(saved.decision_id) != saved.outcome_id:
                    self.planner.capability_map.supersede_outcome(saved.outcome_id)
                else:
                    self._learn_outcome(saved, monitor=False)
        for event in ledger.events():
            related = None
            if event.event_type == "outcome_disputed":
                related = self.state.outcomes.get(event.payload["outcome_id"])
            elif event.event_type == "outcome" and event.payload.get("supersedes"):
                related = self.state.outcomes.get(event.payload["outcome_id"])
            if related:
                decision = self.decisions.get(related.decision_id)
                if decision:
                    for step in decision.selected_plan.steps:
                        if isinstance(step,Call):
                            for slice_id in decision.request.traffic_slices or {None}:
                                self.planner.certificates.invalidate_dependencies(step.endpoint_id,("quality","failure"),
                                    slice_id,"outcome_authority_changed",at=datetime.fromisoformat(event.occurred_at))
        self.default_policy=default_policy
        self.state.store_planner(self.planner,default_policy)
        if default_policy:self.lifecycle.bootstrap(default_policy)
        # Bridges share live endpoint metadata, never the obsolete constructor mapping.
        if executor and hasattr(executor.adapter,"endpoints"):executor.adapter.endpoints=self.planner.endpoints

    def _record_decision(self, decision, policy):
        self.state.store_planner(self.planner,policy)
        self.state.write("decision",decision.model_dump(mode="json"),key=f"decision:{decision.decision_id}")
        return decision

    def decide(self, payload: DecideRequest) -> DecisionRecord:
        with self.lock:
            self.drift.reconcile_latency()
            if payload.request.tenant_policy_id != payload.policy.policy_id:
                raise ValueError("request tenant_policy_id does not match policy")
            prior=self.policies.get((payload.policy.policy_id,payload.policy.version))
            if prior and prior!=payload.policy:raise ValueError("policy version is immutable")
            decision=self.planner.decide(payload.request,payload.policy,payload.seed)
            return self._record_decision(decision,payload.policy)

    def execute(self, decision_id: str, *, messages=None,response_config=None,result_sink=None) -> ExecutionRecord:
        if self.executor is None:raise ValueError("executor not configured")
        with self.lock:
            self.state.refresh()
            self.drift.reconcile_latency()
            if decision_id not in self.decisions:raise KeyError(decision_id)
            if decision_id in self.state.execution_by_decision:return self.state.execution_by_decision[decision_id]
            decision=self.decisions[decision_id]
            policy=self.policies[(decision.policy_id,decision.policy_version)]
            ids=tuple(s.endpoint_id for s in decision.selected_plan.steps if isinstance(s,Call))
            snapshots=tuple(self.planner.endpoints[e].snapshot_id for e in ids if e in self.planner.endpoints)
            if snapshots!=decision.endpoint_snapshot_ids:raise ValueError("endpoint snapshot changed; request a new decision")
            for e in ids:
                if eligibility_reasons(self.planner.endpoints[e],decision.request,policy):raise ValueError("endpoint became ineligible")
            if policy.require_certificate and ids:
                reasons=self.planner.certificates.validate_decision(decision,policy,self.planner.endpoints)
                if reasons:raise ValueError(";".join(reasons))
            claim=str(uuid4())
            event=self.state.write("execution_started",{"decision_id":decision_id,"claim":claim,
                "max_spend":decision.selected_plan.max_spend},key=f"execute-once:{decision_id}")
            if event.payload["claim"]!=claim:
                raise ValueError("execution already claimed; outcome unknown or in progress; automatic re-execution is forbidden")
        # No shared mutable request context. Provider side effects happen only after a durable claim.
        try:
            record=self.executor.execute(decision,policy=policy,endpoints=self.planner.endpoints,
                certificates=self.planner.certificates,messages=messages,response_config=response_config,result_sink=result_sink)
        except Exception as exc:
            record=ExecutionRecord(decision_id=decision_id,state="failed",provider_errors=(type(exc).__name__,),
                                   reserved_spend=decision.selected_plan.max_spend,accounting_complete=False)
        with self.lock:
            self.state.write("execution",record.model_dump(mode="json"),key=f"execution-final:{decision_id}")
            self.lifecycle.observe_execution(decision,record)
            self.drift.observe_execution(decision,record,policy,observed_at=datetime.fromisoformat(event.occurred_at))
            return record

    def add_outcome(self, outcome: OutcomeRecord) -> OutcomeRecord:
        with self.lock:
            decision=self.decisions.get(outcome.decision_id)
            if decision is None:raise KeyError(outcome.decision_id)
            execution=self.state.execution_by_decision.get(outcome.decision_id)
            if outcome.execution_id and (execution is None or outcome.execution_id!=execution.execution_id):
                raise ValueError("outcome execution does not belong to this decision")
            if outcome.outcome_id in self.state.outcomes:
                previous=self.state.outcomes[outcome.outcome_id]
                if previous.model_dump(exclude={"training_eligible","promotion_eligible"})!=outcome.model_dump(exclude={"training_eligible","promotion_eligible"}):
                    raise ValueError("immutable outcome_id conflict")
                return previous
            policy=self.policies[(decision.policy_id,decision.policy_version)]
            # Qualify in a temporary store; don't mutate the read model before durable commit.
            trial=OutcomeStore();trial._records=dict(self.outcomes._records);trial._active=dict(self.outcomes._active)
            qualified=trial.add(outcome,policy)
            trusted=qualified.evaluator_type in {"deterministic","application","human"} or (
                qualified.evaluator_type=="calibrated_judge" and self.trust.trusted(qualified.evaluator_version))
            if not trusted or execution is None or execution.attempted_calls == 0:
                qualified=qualified.model_copy(update={"training_eligible":False,"promotion_eligible":False})
            self.state.write("outcome",qualified.model_dump(mode="json"),key=f"outcome:{outcome.outcome_id}")
            self._learn_outcome(qualified)
            self.state.store_planner(self.planner)
            self.lifecycle.observe_outcome(decision,qualified,policy)
            return qualified

    def _learn_outcome(self, qualified, *, monitor=True):
        cmap = self.planner.capability_map
        if cmap is None: return
        decision = self.decisions[qualified.decision_id]
        execution = self.state.execution_by_decision.get(qualified.decision_id)
        if qualified.supersedes:
            cmap.supersede_outcome(qualified.supersedes)
        ids = tuple(s.endpoint_id for s in decision.selected_plan.steps if isinstance(s,Call))
        quality = qualified.quality.get("success",qualified.quality.get("score"))
        if qualified.supersedes and monitor:
            for endpoint_id in ids:
                for slice_id in decision.request.traffic_slices or {None}:
                    self.planner.certificates.invalidate_dependencies(endpoint_id,("quality","failure"),slice_id,
                                                                      "superseded_outcome")
        sample_id = f"outcome:{qualified.outcome_id}"
        claim = self.state.claims.get(qualified.decision_id)
        observed_at = claim["started_at"] if claim else None
        prior = cmap.rows.get(sample_id)
        if prior is not None:
            if cmap.reconcile_outcome_time(sample_id, observed_at):
                for endpoint_id in prior.endpoint_ids:
                    for slice_id in prior.request.traffic_slices or {None}:
                        self.planner.certificates.invalidate_dependencies(endpoint_id,tuple(sorted(prior.metrics)),
                            slice_id,"outcome_evidence_time_reconciled")
            return
        if observed_at is None or not execution or not ids or quality is None or not qualified.training_eligible:
            return
        if qualified.disputed: return
        trust_until = None
        if qualified.evaluator_type == "calibrated_judge":
            trust = self.trust.records.get(qualified.evaluator_version)
            if not trust or not trust["trusted"]: return
            trust_until = datetime.fromisoformat(trust["valid_until"])
        snapshots = tuple(self.state.snapshots[s] for s in decision.endpoint_snapshot_ids)
        split = "calibration" if int(digest(decision.request.request_id)[:8],16)%5==0 else "train"
        row = EvidenceObservation(sample_id=sample_id,request=decision.request,
            target_id=decision.selected_plan.evidence_key,plan_type=decision.selected_plan.plan_type,
            endpoint_ids=ids,endpoint_revisions=tuple(e.capability_revision for e in snapshots),
            quality=quality if execution.state=="completed" else 0.,
            cost=execution.realized_spend,latency_ms=execution.total_latency_ms,output_tokens=execution.output_tokens,
            failed=execution.state!="completed",split=split,evaluator_type=qualified.evaluator_type,
            evaluator_version=qualified.evaluator_version,observed_at=observed_at,
            trusted=True,trusted_until=trust_until,outcome_id=qualified.outcome_id,
            metrics=frozenset({"quality","failure","latency","cost"}) if execution.accounting_complete else frozenset({"quality","failure"}))
        cmap.add(row)
        # A joint plan outcome is not a marginal endpoint-quality label.
        if monitor and len(ids)==1:
            for slice_id in decision.request.traffic_slices or {"default"}:
                if any(item["endpoint_id"] == ids[0] and "quality" in item["metrics"]
                       and item["slice_id"] in {None,slice_id}
                       and row.observed_at <= datetime.fromisoformat(item["at"])
                       for item in cmap.invalidations):
                    continue
                self.drift.observe(ids[0],"quality",slice_id,row.quality,observed_at=row.observed_at)

    def dispute_outcome(self, outcome_id: str, reason: str):
        with self.lock:
            if not reason.strip(): raise ValueError("dispute reason required")
            saved=self.state.outcomes[outcome_id]
            if saved.disputed: return saved
            self.state.write("outcome_disputed",{"outcome_id":outcome_id,"reason":reason,
                "at":datetime.now(timezone.utc).isoformat()},key=f"dispute:{outcome_id}")
            if self.planner.capability_map is not None:
                self.planner.capability_map.supersede_outcome(outcome_id)
            decision=self.decisions[saved.decision_id]
            for step in decision.selected_plan.steps:
                if isinstance(step,Call):
                    for slice_id in decision.request.traffic_slices or {None}:
                        self.planner.certificates.invalidate_dependencies(step.endpoint_id,("quality","failure"),slice_id,"disputed_outcome")
            self.state.store_planner(self.planner)
            policy=self.policies[(decision.policy_id,decision.policy_version)]
            self.lifecycle.observe_outcome(decision,self.state.outcomes[outcome_id],policy)
            return self.state.outcomes[outcome_id]

    def invalidate_evaluator(self, version: str, reason: str):
        with self.lock:
            self.trust.invalidate(version,reason)
            cmap=self.planner.capability_map
            affected=set()
            if cmap is not None:
                for sample_id,row in tuple(cmap.rows.items()):
                    if row.evaluator_version==version and row.evaluator_type=="calibrated_judge":
                        cmap.rows[sample_id]=row.model_copy(update={"trusted":False})
                        affected.update(row.endpoint_ids)
                cmap._version=cmap._index=None
                for endpoint_id in sorted(affected):
                    self.drift.invalidate(endpoint_id,("quality",),reason="evaluator_trust_invalidated:"+reason)
            self.state.store_planner(self.planner)
            return tuple(sorted(affected))

    def replay(self, decision_id: str) -> dict:
        original=self.decisions[decision_id]
        policy=self.policies[(original.policy_id,original.policy_version)]
        # Include the whole contemporaneous pool, not only the winning endpoint.
        endpoints={}
        maps={}
        static=None
        for event in self.ledger.events():
            if event.event_type=="endpoint_snapshot":
                from inference_control.contracts import EndpointSnapshot
                ep=EndpointSnapshot.model_validate(event.payload);endpoints[ep.endpoint_id]=ep
            elif event.event_type=="static_estimates":static=event.payload
            elif event.event_type=="capability_map":maps[event.payload["version"]]=event.payload["data"]
            elif event.event_type=="decision" and event.payload["decision_id"]==decision_id:break
        if original.capability_map_version in maps:
            planner=Planner(list(endpoints.values()),capability_map=ConditionalCapabilityMap.restore(maps[original.capability_map_version]))
        else:
            estimates=[]
            for r in (static or {}).get("estimates",[]):
                estimates.append(CapabilityEstimate(r["endpoint_id"],Prediction.model_validate(r["quality"]),
                    Prediction.model_validate(r["cost"]),Prediction.model_validate(r["latency"]),
                    Prediction.model_validate(r["failure"]) if r.get("failure") else None))
            planner=Planner(list(endpoints.values()),estimates,original.capability_map_version)
        result=planner.decide(original.request,policy,original.replay_seed,at=original.created_at)
        return {"decision_id":decision_id,"matched":result.decision_fingerprint==original.decision_fingerprint,
                "original":original.decision_fingerprint,"replayed":result.decision_fingerprint,
                "plan":result.selected_plan.model_dump(mode="json")}


def create_app(control: ControlPlane | None = None, *, api_key: str | None = None) -> FastAPI:
    def authorize(authorization: str | None = Header(default=None)):
        if api_key and not hmac.compare_digest(authorization or "",f"Bearer {api_key}"):
            raise HTTPException(401,"invalid API credential")
    api=FastAPI(title="SLM Router — Adaptive Inference Control",version="2.0.0a2",dependencies=[Depends(authorize)])
    # The dashboard shell contains no data or credentials. Calls from it still target
    # the bearer-authenticated API routes on this parent application.
    api.mount("/dashboard",create_dashboard_app())
    def configured():
        if control is None:raise HTTPException(503,"control plane not configured")
        return control
    @api.get("/health")
    def health():
        c=configured();return {"status":"ok","ledger_verified":c.ledger.verify(),"map_version":c.planner.map_version}
    @api.post("/v2/decide",response_model=DecisionRecord)
    def decide(payload:DecideRequest):
        try:return configured().decide(payload)
        except ValueError as exc:raise HTTPException(422,str(exc)) from None
    @api.post("/v2/execute",response_model=ExecutionRecord)
    def execute(payload:ExecuteRequest):
        c=configured()
        if not c.executor:raise HTTPException(503,"executor not configured")
        try:
            decision_id=payload.decision_id or c.decide(DecideRequest(request=payload.request,policy=payload.policy,seed=payload.seed)).decision_id
            return c.execute(decision_id,messages=payload.messages,response_config=payload.response_config)
        except KeyError:raise HTTPException(404,"decision not found") from None
        except ValueError as exc:raise HTTPException(409,str(exc)) from None
    @api.post("/v2/outcomes",response_model=OutcomeRecord,status_code=201)
    def outcome(payload:OutcomeRecord):
        try:return configured().add_outcome(payload)
        except KeyError:raise HTTPException(404,"decision not found") from None
        except ValueError as exc:raise HTTPException(422,str(exc)) from None
    @api.post("/v2/outcomes/{outcome_id}/dispute")
    def dispute(outcome_id:str, payload:dict[str,str]):
        try:return configured().dispute_outcome(outcome_id,payload.get("reason",""))
        except KeyError:raise HTTPException(404,"outcome not found") from None
        except ValueError as exc:raise HTTPException(422,str(exc)) from None
    @api.get("/v2/decisions/{decision_id}",response_model=DecisionRecord)
    def get_decision(decision_id:str):
        try:return configured().decisions[decision_id]
        except KeyError:raise HTTPException(404,"decision not found") from None
    @api.get("/v2/decisions/{decision_id}/replay")
    def replay(decision_id:str):
        try:return configured().replay(decision_id)
        except KeyError:raise HTTPException(404,"decision or historical evidence not found") from None
    @api.get("/metrics",response_class=PlainTextResponse)
    def metrics():
        from inference_control.telemetry import prometheus
        return prometheus(configured().state)
    @api.get("/v1/models")
    def models():
        c=configured()
        return {"object":"list","data":[{"id":c.default_policy.policy_id,"object":"model","owned_by":"slm-router"}] if c.default_policy else []}
    @api.post("/v1/chat/completions")
    def chat(payload:ChatRequest):
        c=configured()
        if payload.stream:raise HTTPException(400,"streaming is not supported; no inference was started")
        if not c.default_policy or not c.executor:raise HTTPException(503,"default policy/executor not configured")
        if payload.model!=c.default_policy.policy_id:raise HTTPException(404,"unknown router policy model")
        if payload.response_format or payload.tools:
            # Capability requirements affect eligibility BEFORE dispatch.
            required=set()
            if payload.response_format:required.add("structured_output")
            if payload.tools:required.add("tools")
        else:required=set()
        response_config={k:getattr(payload,k) for k in ("response_format","tools","tool_choice") if getattr(payload,k) is not None}
        # Byte bound rather than len/4: still requires operator-verified tokenizer overhead per endpoint.
        serialized=json.dumps({"messages":payload.messages,**response_config},ensure_ascii=False,separators=(",",":"))
        max_output=payload.max_completion_tokens or payload.max_tokens or 256
        text="\n".join(m.get("content") or "" for m in payload.messages)
        from inference_control.features.query import query_features
        request=RequestContext(application_id=payload.metadata.get("application_id","chat"),tenant_policy_id=c.default_policy.policy_id,
            request_id=payload.metadata.get("request_id",str(uuid4())),session_id=payload.metadata.get("session_id"),
            input_tokens=len(serialized.encode("utf-8")),max_output_tokens=max_output,feature_version="hash-text-v1",
            task_hint=payload.metadata.get("task"),traffic_slices=frozenset(filter(None,payload.metadata.get("traffic_slice","").split(","))),
            privacy_classification=payload.metadata.get("privacy","public"),required_capabilities=frozenset(required),
            query_features=query_features(text),structured_output_schema_hash=digest(payload.response_format) if payload.response_format else None,
            tool_schema_hash=digest(payload.tools) if payload.tools else None)
        group=c.state.lifecycle[c.default_policy.policy_id]
        if group["state"]=="stale":raise HTTPException(503,"active policy is stale; operator recovery required")
        active=c.policies[(c.default_policy.policy_id,group["active"])]
        candidate=c.policies.get((c.default_policy.policy_id,group.get("candidate")))
        with c.lock:
            try: decision,selected_policy=c.lifecycle.route(request,active,candidate,c.planner)
            except ValueError as exc: raise HTTPException(409,str(exc)) from None
            c._record_decision(decision,selected_policy)
        if decision.selected_plan.plan_type=="abstain":
            raise HTTPException(503,{"error":"no_feasible_plan","decision_id":decision.decision_id})
        results=[]
        try:record=c.execute(decision.decision_id,messages=payload.messages,response_config=response_config,result_sink=results.append)
        except ValueError as exc:raise HTTPException(409,str(exc)) from None
        if record.state!="completed" or not results:
            raise HTTPException(502,{"error":"execution_failed","decision_id":decision.decision_id,"execution_id":record.execution_id})
        message=results[0].message or {"role":"assistant","content":None}
        return {"id":f"chatcmpl-{record.execution_id}","object":"chat.completion","created":int(datetime.now(timezone.utc).timestamp()),
                "model":results[0].endpoint_id or payload.model,"choices":[{"index":0,"message":message,
                    "finish_reason":"tool_calls" if message.get("tool_calls") else "stop"}],
                "usage":{"prompt_tokens":record.input_tokens,"completion_tokens":record.output_tokens,
                    "total_tokens":record.input_tokens+record.output_tokens},
                "slm_router":{"decision_id":decision.decision_id,"execution_id":record.execution_id,
                    "certificate_status":decision.certificate_status,"realized_spend":record.realized_spend}}
    return api

app=create_app()
