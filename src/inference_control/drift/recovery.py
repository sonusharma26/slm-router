"""Routing-specific metric/slice invalidation and targeted recovery requests."""
from __future__ import annotations
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime, timezone
from statistics import mean, variance, NormalDist
from threading import RLock
from typing import Literal
import math
from pydantic import Field
from inference_control.contracts import Call, FrozenContract
from inference_control.util import digest
from inference_control.registry.snapshotter import Snapshotter


class EndpointLatency(FrozenContract):
    kind: Literal["endpoint"] = "endpoint"
    source_id: str
    endpoint_id: str
    slice_id: str
    capability_revision: str
    policy_hash: str
    deadline_ms: float = Field(gt=0)
    observed_at: datetime
    received_at: datetime
    duration_ms: float = Field(ge=0)
    timed_out: bool = False
    completed: bool = True


class CompoundLatency(FrozenContract):
    kind: Literal["compound"] = "compound"
    source_id: str
    plan_key: str
    endpoint_ids: tuple[str, ...]
    endpoint_revisions: tuple[str, ...]
    slice_id: str
    policy_hash: str
    deadline_ms: float = Field(gt=0)
    observed_at: datetime
    received_at: datetime
    duration_ms: float = Field(ge=0)
    timed_out: bool = False
    completed: bool = True


class DriftRecovery:
    def __init__(self, planner, state=None, *, window=30, minimum_shift=.15, alpha=.01):
        if window < 2 or not 0 < alpha < 1: raise ValueError("invalid detector configuration")
        self.planner, self.state = planner, state
        self.window, self.minimum_shift, self.alpha = window, minimum_shift, alpha
        self.reference, self.recent = {}, defaultdict(lambda:deque(maxlen=window))
        self.looks = defaultdict(int)
        self.pending = {}
        self.operational = {}
        self.receipts = set()
        self.lock = RLock()
        self._batch = None
        self._latency_sequence = 0
        self._draining_latency = False
        if state:
            self.operational = dict(state.operational_counters)
            self.receipts = set(state.latency_receipts)
            for event in state.drifts:
                if event["event_id"] not in state.recovered_events:
                    self.pending[event["event_id"]] = event
            for record in state.detectors.values():
                key = (record["endpoint_id"], record["metric"], record["slice_id"])
                if record.get("reference") is not None: self.reference[key] = tuple(record["reference"])
                self.recent[key] = deque(record["recent"], maxlen=self.window)
                self.looks[key] = record["looks"]

    def _record(self, kind, record, key):
        if self.state is None: return
        if self._batch is not None: self._batch.append((kind,record,key))
        else: self.state.write(kind,record,key=key)

    def _persist(self, key):
        if self.state is None: return
        record = {"key":digest(key), "endpoint_id":key[0], "metric":key[1], "slice_id":key[2],
                  "reference":self.reference.get(key), "recent":list(self.recent[key]), "looks":self.looks[key],
                  "window":self.window, "minimum_shift":self.minimum_shift, "alpha":self.alpha}
        self._record("drift_detector", record, f"detector:{self.state.sequence}:{digest(record)}")

    def _stale(self, endpoint_ids, metric, slice_id, observed_at):
        if self.planner.capability_map is None: return False
        return any(item["endpoint_id"] in endpoint_ids and metric in item["metrics"]
                   and (not item["slice_id"] or item["slice_id"] == slice_id)
                   and observed_at <= datetime.fromisoformat(item["at"])
                   for item in self.planner.capability_map.invalidations)

    @staticmethod
    def _latency_ids(observation):
        return (observation.endpoint_id,) if isinstance(observation,EndpointLatency) else observation.endpoint_ids

    @classmethod
    def _latency_key(cls, observation):
        direct = isinstance(observation,EndpointLatency)
        target = ("endpoint",observation.endpoint_id) if direct else ("compound",observation.plan_key)
        revisions = (observation.capability_revision,) if direct else observation.endpoint_revisions
        return digest((target,observation.slice_id,revisions,observation.policy_hash,observation.deadline_ms))

    def _save_monitor(self):
        cmap = self.planner.capability_map
        return (deepcopy(self.reference),deepcopy(self.recent),deepcopy(self.looks),
                deepcopy(self.operational),dict(self.pending),
                {key:list(versions) for key,versions in self.planner.certificates._versions.items()},
                deepcopy(cmap.invalidations) if cmap is not None else None,cmap._version if cmap is not None else None)

    def _restore_monitor(self, saved):
        self.reference,self.recent,self.looks,self.operational,self.pending,certificates,invalidations,version = saved
        self.planner.certificates._versions = certificates
        cmap = self.planner.capability_map
        if cmap is not None: cmap.invalidations,cmap._version = invalidations,version

    def _commit_monitor(self, key, events):
        cmap = self.planner.capability_map
        if cmap is not None and self.planner.map_version != self.state.latest_map:
            data = {"version":self.planner.map_version,"data":cmap.export()}
            self._batch.append(("capability_map",data,f"latency-map:{key}"))
        if events:
            data = {"records":self.planner.certificates.export()}
            if data["records"]: self._batch.append(("certificates",data,f"latency-certificates:{key}"))
        self.state.ledger.append_batch(self._batch,actor="control-plane")

    def _terminal_observations(self, event):
        if event.event_type == "execution":
            record = self.state.executions[event.payload["execution_id"]]
            decision = self.state.decisions[record.decision_id]
            claim = self.state.claims.get(record.decision_id)
            if claim:
                policy = self.state.policies[(decision.policy_id,decision.policy_version)]
                return self._execution_observations(decision,record,policy,observed_at=claim["started_at"])
        elif event.event_type in {"probe_completed","probe_failed"}:
            return self._probe_observations(event.payload)
        return ()

    def _repair_latency_order(self, ledger_events, sources):
        """Fail closed once when receipts prove a historical suffix ran first."""
        receipt_sequences = {e.payload["source_id"]:e.sequence for e in ledger_events if e.event_type == "latency_receipt"}
        covered = {}
        for event in ledger_events:
            if event.event_type == "latency_order_repair":
                for target in event.payload["targets"]:
                    pair = tuple(target)
                    covered[pair] = max(covered.get(pair,0),event.payload["through_sequence"])
        missing, latest_receipt, affected = set(), {}, set()
        for sequence,observation in sources:
            for endpoint_id in self._latency_ids(observation):
                pair = (endpoint_id,observation.slice_id)
                if sequence <= covered.get(pair,0): continue
                receipt_sequence = receipt_sequences.get(observation.source_id)
                if observation.source_id not in self.receipts:
                    missing.add(pair)
                elif pair in missing or (receipt_sequence is not None and receipt_sequence < latest_receipt.get(pair,0)):
                    affected.add(pair)
                if receipt_sequence is not None:
                    latest_receipt[pair] = max(latest_receipt.get(pair,0),receipt_sequence)
        if not affected: return ()
        expanded = True
        while expanded:
            before = len(affected)
            for _,observation in sources:
                if isinstance(observation,CompoundLatency) and any((e,observation.slice_id) in affected for e in observation.endpoint_ids):
                    affected.update((e,observation.slice_id) for e in observation.endpoint_ids)
            expanded = len(affected) != before
        skipped = [observation for _,observation in sources if observation.source_id not in self.receipts
                   and any((e,observation.slice_id) in affected for e in self._latency_ids(observation))]
        marker = {"through_sequence":max(sequence for sequence,_ in sources),"targets":[list(pair) for pair in sorted(affected)]}
        repair_id = digest(marker)
        saved, events, committed = self._save_monitor(), [], False
        self._batch = []
        try:
            at = datetime.now(timezone.utc)
            for endpoint_id,slice_id in sorted(affected):
                events.append(self.invalidate(endpoint_id,("latency",),reason="latency_history_out_of_order",
                    slice_id=slice_id,at=at,details={"detector":"durable_terminal_order","repair_id":repair_id,
                    "through_sequence":marker["through_sequence"]}))
            keys = {self._latency_key(observation) for _,observation in sources
                    if any((e,observation.slice_id) in affected for e in self._latency_ids(observation))}
            for key in sorted(keys):
                if key in self.operational:
                    counter = {**self.operational[key],"streak":0}
                    self.operational[key] = counter
                    self._record("latency_counter",counter,f"latency-counter:order-repair:{repair_id}:{key}")
            for observation in skipped:
                receipt = {"source_id":observation.source_id,"ignored_stale":True,
                    "observed_at":observation.observed_at.isoformat(),"received_at":observation.received_at.isoformat(),
                    "reason":"latency_history_out_of_order"}
                self._record("latency_receipt",receipt,f"latency-receipt:{observation.source_id}")
            self._record("latency_order_repair",marker,f"latency-order-repair:{repair_id}")
            self._commit_monitor(f"order-repair:{repair_id}",events)
            committed = True
            self.receipts.update(observation.source_id for observation in skipped)
            self.state.refresh()
            return tuple(events)
        except BaseException:
            if not committed: self._restore_monitor(saved)
            raise
        finally:
            self._batch = None

    def reconcile_latency(self):
        """Drain durable terminal order, advancing only past successfully handled facts."""
        if self.state is None: return ()
        with self.lock, self.state.ledger.lock:
            self.state.refresh()
            self.receipts.update(self.state.latency_receipts)
            self.operational.update(self.state.operational_counters)
            ledger_events = self.state.ledger.events(after=self._latency_sequence)
            terminal = [(event,self._terminal_observations(event)) for event in ledger_events
                        if event.event_type in {"execution","probe_completed","probe_failed"}]
            sources = [(event.sequence,observation) for event,observations in terminal for observation,_ in observations]
            events = list(self._repair_latency_order(ledger_events,sources))
            self._draining_latency = True
            try:
                for event,observations in terminal:
                    if any(observation.source_id not in self.receipts for observation,_ in observations):
                        if event.event_type == "execution":
                            record = self.state.executions[event.payload["execution_id"]]
                            decision = self.state.decisions[record.decision_id]
                            policy = self.state.policies[(decision.policy_id,decision.policy_version)]
                            events.extend(self.observe_execution(decision,record,policy,
                                observed_at=self.state.claims[record.decision_id]["started_at"]))
                        else:
                            events.extend(self.observe_probe(event.payload))
                    self._latency_sequence = event.sequence
                if ledger_events: self._latency_sequence = ledger_events[-1].sequence
                return tuple(events)
            finally:
                self._draining_latency = False

    def observe_latency(self, observation: EndpointLatency | CompoundLatency, *, quality=None):
        if observation.observed_at.tzinfo is None or observation.received_at.tzinfo is None:
            raise ValueError("latency timestamps require timezones")
        if observation.observed_at > observation.received_at:
            raise ValueError("latency observation cannot precede its sample")
        direct = isinstance(observation,EndpointLatency)
        ids = (observation.endpoint_id,) if direct else observation.endpoint_ids
        revisions = (observation.capability_revision,) if direct else observation.endpoint_revisions
        if not ids or len(ids) != len(revisions): raise ValueError("latency dependencies require revisions")
        target = ("endpoint",ids[0]) if direct else ("compound",observation.plan_key)
        key = self._latency_key(observation)
        with self.lock:
            if self.state:
                self.state.refresh()
                self.receipts.update(self.state.latency_receipts)
                self.operational.update(self.state.operational_counters)
            if observation.source_id in self.receipts: return ()
            saved = self._save_monitor()
            self._batch = []
            events = []
            committed = False
            try:
                stale = self._stale(ids,"latency",observation.slice_id,observation.observed_at)
                revision_stale = any(e not in self.planner.endpoints or self.planner.endpoints[e].capability_revision != revision
                                     for e,revision in zip(ids,revisions))
                stale = stale or revision_stale
                if not stale:
                    streak = self.operational.get(key,{}).get("streak",0)
                    adverse = observation.timed_out or observation.duration_ms > observation.deadline_ms
                    if adverse: streak += 1
                    elif observation.completed: streak = 0
                    if streak >= 2:
                        details = {"detector":"operational_deadline","scope":target[0],"target":target[1],
                                   "endpoint_ids":list(ids),"endpoint_revisions":list(revisions),
                                   "policy_hash":observation.policy_hash,"deadline_ms":observation.deadline_ms,
                                   "duration_ms":observation.duration_ms,"timed_out":observation.timed_out,
                                   "consecutive_adverse":streak,"observation_limit":2}
                        for endpoint_id in dict.fromkeys(ids):
                            events.append(self.invalidate(endpoint_id,("latency",),reason="operational_deadline_exceeded",
                                slice_id=observation.slice_id,at=observation.received_at,details=details))
                        streak = 0
                    elif direct and observation.completed and not observation.timed_out:
                        event = self.observe(ids[0],"latency",observation.slice_id,observation.duration_ms,
                                             at=observation.received_at,observed_at=observation.observed_at)
                        if event: events.append(event)
                    counter = {"key":key,"target":list(target),"slice_id":observation.slice_id,
                               "endpoint_revisions":list(revisions),"policy_hash":observation.policy_hash,
                               "deadline_ms":observation.deadline_ms,"streak":streak}
                    self.operational[key] = counter
                    self._record("latency_counter",counter,f"latency-counter:{observation.source_id}")
                if direct and quality is not None and not revision_stale:
                    event = self.observe(ids[0],"quality",observation.slice_id,quality,
                                         at=observation.received_at,observed_at=observation.observed_at)
                    if event: events.append(event)
                receipt = {"source_id":observation.source_id,"ignored_stale":stale,
                           "observed_at":observation.observed_at.isoformat(),"received_at":observation.received_at.isoformat()}
                self._record("latency_receipt",receipt,f"latency-receipt:{observation.source_id}")
                if self.state:
                    self._commit_monitor(observation.source_id,events)
                    committed = True
                    self.receipts.add(observation.source_id)
                    self.state.refresh()
                self.receipts.add(observation.source_id)
                return tuple(events)
            except BaseException:
                if not committed:
                    self._restore_monitor(saved)
                raise
            finally:
                self._batch = None

    def observe_execution(self, decision, record, policy, *, observed_at):
        with self.lock:
            if self.state is not None and not self._draining_latency: return self.reconcile_latency()
            return tuple(event for observation,quality in self._execution_observations(decision,record,policy,observed_at=observed_at)
                         for event in self.observe_latency(observation,quality=quality))

    def _execution_observations(self, decision, record, policy, *, observed_at):
        if record.attempted_calls == 0 or (record.total_latency_ms <= 0 and not record.timed_out_endpoint_ids) or record.state == "abstained": return ()
        ids = tuple(s.endpoint_id for s in decision.selected_plan.steps if isinstance(s,Call))
        if not ids: return ()
        snapshots = self.state.snapshots if self.state else {}
        revisions = tuple((snapshots.get(sid) or self.planner.endpoints[e]).capability_revision
                          for e,sid in zip(ids,decision.endpoint_snapshot_ids))
        observations = []
        for slice_id in sorted(decision.request.traffic_slices or {"default"}):
            common = dict(source_id=f"execution:{record.execution_id}:{slice_id}",slice_id=slice_id,
                          policy_hash=digest(policy),deadline_ms=policy.deadline_ms,observed_at=observed_at,
                          received_at=record.created_at,duration_ms=record.total_latency_ms,
                          timed_out=bool(record.timed_out_endpoint_ids),completed=not record.provider_errors)
            observation = (EndpointLatency(endpoint_id=ids[0],capability_revision=revisions[0],**common)
                if decision.selected_plan.plan_type == "direct" else
                CompoundLatency(plan_key=decision.selected_plan.evidence_key,endpoint_ids=ids,endpoint_revisions=revisions,**common))
            observations.append((observation,None))
        return tuple(observations)

    def observe_probe(self, payload):
        with self.lock:
            if self.state is not None and not self._draining_latency: return self.reconcile_latency()
            return tuple(event for observation,quality in self._probe_observations(payload)
                         for event in self.observe_latency(observation,quality=quality))

    def _probe_observations(self, payload):
        context = payload.get("latency_context")
        if context is None: return ()
        from inference_control.capability.conditional import EvidenceObservation
        row = EvidenceObservation.model_validate(payload["observation"]) if "observation" in payload else None
        if row is None and not context["timed_out"]: return ()
        observations = []
        for slice_id in context["slice_ids"]:
            observation = EndpointLatency(source_id=f"probe:{payload['claim']}:{slice_id}",
                endpoint_id=context["endpoint_id"],capability_revision=context["capability_revision"],slice_id=slice_id,
                policy_hash=context["policy_hash"],deadline_ms=context["deadline_ms"],
                observed_at=datetime.fromisoformat(context["observed_at"]),received_at=datetime.fromisoformat(context["received_at"]),
                duration_ms=row.latency_ms if row and "latency" in row.metrics else context.get("duration_ms",0),
                timed_out=context["timed_out"],completed=row is not None and not row.failed and "latency" in row.metrics)
            quality = row.quality if row and row.trusted and "quality" in row.metrics else None
            observations.append((observation,quality))
        return tuple(observations)

    def invalidate(self, endpoint_id, metrics, *, reason, slice_id=None, at=None, details=None):
        with self.lock:
            return self._invalidate(endpoint_id,metrics,reason=reason,slice_id=slice_id,at=at,details=details)

    def _invalidate(self, endpoint_id, metrics, *, reason, slice_id=None, at=None, details=None):
        at = at or datetime.now(timezone.utc)
        event = {"endpoint_id":endpoint_id,"metrics":list(metrics),"slice_id":slice_id,
                 "reason":reason,"at":at.isoformat(),"details":details or {}}
        event["event_id"] = digest(event)
        if self.planner.capability_map is not None:
            self.planner.capability_map.invalidate(endpoint_id,tuple(metrics),at=at,slice_id=slice_id,reason=reason)
        event["invalidated_certificates"] = list(self.planner.certificates.invalidate_dependencies(endpoint_id,tuple(metrics),slice_id,reason,at=at))
        self.pending[event["event_id"]] = event
        self._record("drift",event,f"drift:{event['event_id']}")
        for key in tuple(set(self.reference) | set(self.recent)):
            if key[0] == endpoint_id and key[1] in metrics and (slice_id is None or key[2] == slice_id):
                self.reference.pop(key,None); self.recent[key].clear(); self.looks[key] = 0
                self._persist(key)
        if self.state:
            if self._batch is None: self.state.store_planner(self.planner)
        return event

    def refresh_endpoint(self, snapshot, *, at=None):
        with self.lock:
            return self._refresh_endpoint(snapshot,at=at)

    def _refresh_endpoint(self, snapshot, *, at=None):
        old = self.planner.endpoints.get(snapshot.endpoint_id)
        if self.state:
            change = Snapshotter(self.state).refresh(snapshot)
            metrics = change.changed_metrics
        else:
            metrics = set()
            if old and old.capability_revision != snapshot.capability_revision: metrics.update(("quality","cost","latency","failure"))
            if old and (old.price_version,old.input_price_per_million,old.output_price_per_million) != (snapshot.price_version,snapshot.input_price_per_million,snapshot.output_price_per_million): metrics.add("cost")
            if old and (old.available,old.healthy) != (snapshot.available,snapshot.healthy): metrics.add("availability")
            metrics = tuple(sorted(metrics))
        self.planner.endpoints[snapshot.endpoint_id] = snapshot
        if metrics:
            return self.invalidate(snapshot.endpoint_id,metrics,reason="endpoint_snapshot_changed",at=at,
                                   details={"before":old.snapshot_id if old else None,"after":snapshot.snapshot_id})
        return None

    def set_reference(self, endpoint_id, metric, slice_id, values):
        with self.lock:
            return self._set_reference(endpoint_id,metric,slice_id,values)

    def _set_reference(self, endpoint_id, metric, slice_id, values):
        if len(values)<self.window: raise ValueError("insufficient reference evidence")
        key = (endpoint_id,metric,slice_id)
        self.reference[key] = tuple(values[-self.window:])
        self._persist(key)

    def observe(self, endpoint_id, metric, slice_id, value: float, *, at=None, observed_at=None):
        with self.lock:
            return self._observe(endpoint_id,metric,slice_id,value,at=at,observed_at=observed_at)

    def _observe(self, endpoint_id, metric, slice_id, value: float, *, at=None, observed_at=None):
        if metric not in {"quality","latency","failure","workload","evaluator"}: raise ValueError("unsupported statistical drift")
        if not math.isfinite(value) or value < 0: raise ValueError("finite nonnegative measurement required")
        if observed_at is not None and self._stale((endpoint_id,),metric,slice_id,observed_at): return None
        key = (endpoint_id,metric,slice_id)
        current = self.recent[key]
        current.append(value)
        if key not in self.reference:
            if len(current)==self.window:
                self.reference[key]=tuple(current); current.clear()
            self._persist(key)
            return None
        if len(current)<self.window:
            self._persist(key)
            return None
        before, after = self.reference[key], tuple(current)
        self.looks[key] += 1
        # Alpha spending over non-overlapping looks, not an uncorrected rolling p-value.
        alpha = self.alpha/(self.looks[key]*(self.looks[key]+1))
        if metric=="latency":
            # Normalize against a fixed reference cap; detector is bounded but records the raw shift.
            cap=max(before)*4 or 1
            left=[min(1,x/cap) for x in before]; right=[min(1,x/cap) for x in after]
        else:
            if max(before+after)>1: raise ValueError("non-latency detector expects [0,1]")
            left,right=before,after
        shift = mean(right)-mean(left)
        # A variance-sensitive normal approximation, not a distribution-free certificate.
        # Alpha spending controls repeated nominal looks; finite-sample calibration must be benchmarked.
        standard_error=math.sqrt(variance(left)/len(left)+variance(right)/len(right))
        z=NormalDist().inv_cdf(1-min(.49,max(1e-12,alpha/2)))
        threshold=max(self.minimum_shift,z*standard_error)
        # Workload/evaluator are two-sided; quality down and latency/failure up are adverse.
        adverse = abs(shift) if metric in {"workload","evaluator"} else (-shift if metric=="quality" else shift)
        current.clear()
        if adverse <= threshold:
            self._persist(key)
            return None
        metrics=("quality","cost","latency","failure") if metric=="workload" else (("quality",) if metric=="evaluator" else (metric,))
        self.reference[key]=after
        event = self.invalidate(endpoint_id,metrics,reason=f"{metric}_distribution_shift",slice_id=slice_id,at=at,
                               details={"shift":shift,"threshold":threshold,"samples":len(after),"look":self.looks[key],"alpha":alpha,"detector":"variance_sensitive_normal_approximation"})
        self._persist(key)
        return event

    def recovery_targets(self):
        return tuple(sorted({(e["endpoint_id"],m,e["slice_id"] or "*") for e in self.pending.values()
                             for m in e["metrics"] if m!="availability"}))

    def acknowledge_recovered(self, event_id, request, policy, *, at=None):
        with self.lock:
            return self._acknowledge_recovered(event_id,request,policy,at=at)

    def _acknowledge_recovered(self, event_id, request, policy, *, at=None):
        event=self.pending[event_id]
        if event["slice_id"] and event["slice_id"] not in (request.traffic_slices or {"default"}):
            raise ValueError("affected target has not been re-certified for this slice")
        decision=self.planner.decide(request,policy,at=at)
        from inference_control.contracts import Call
        uses_target=any(isinstance(s,Call) and s.endpoint_id==event["endpoint_id"] for s in decision.selected_plan.steps)
        if decision.certificate_status!="current" or not uses_target:
            raise ValueError("affected target has not been re-certified for this slice")
        del self.pending[event_id]
        if self.state:self.state.write("recovery",{"event_id":event_id,"certificate_id":decision.certificate_id})
        return decision
