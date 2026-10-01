"""Routing-specific metric/slice invalidation and targeted recovery requests."""
from __future__ import annotations
from collections import defaultdict, deque
from datetime import datetime, timezone
from statistics import mean, variance, NormalDist
import math
from inference_control.util import digest
from inference_control.registry.snapshotter import Snapshotter


class DriftRecovery:
    def __init__(self, planner, state=None, *, window=30, minimum_shift=.15, alpha=.01):
        if window < 2 or not 0 < alpha < 1: raise ValueError("invalid detector configuration")
        self.planner, self.state = planner, state
        self.window, self.minimum_shift, self.alpha = window, minimum_shift, alpha
        self.reference, self.recent = {}, defaultdict(lambda:deque(maxlen=window))
        self.looks = defaultdict(int)
        self.pending = {}
        if state:
            for event in state.drifts:
                if event["event_id"] not in state.recovered_events:
                    self.pending[event["event_id"]] = event
            for record in state.detectors.values():
                key = (record["endpoint_id"], record["metric"], record["slice_id"])
                if record.get("reference") is not None: self.reference[key] = tuple(record["reference"])
                self.recent[key] = deque(record["recent"], maxlen=self.window)
                self.looks[key] = record["looks"]

    def _persist(self, key):
        if self.state is None: return
        record = {"key":digest(key), "endpoint_id":key[0], "metric":key[1], "slice_id":key[2],
                  "reference":self.reference.get(key), "recent":list(self.recent[key]), "looks":self.looks[key],
                  "window":self.window, "minimum_shift":self.minimum_shift, "alpha":self.alpha}
        self.state.write("drift_detector", record, key=f"detector:{self.state.sequence}:{digest(record)}")

    def invalidate(self, endpoint_id, metrics, *, reason, slice_id=None, at=None, details=None):
        at = at or datetime.now(timezone.utc)
        event = {"endpoint_id":endpoint_id,"metrics":list(metrics),"slice_id":slice_id,
                 "reason":reason,"at":at.isoformat(),"details":details or {}}
        event["event_id"] = digest(event)
        if self.planner.capability_map is not None:
            self.planner.capability_map.invalidate(endpoint_id,tuple(metrics),at=at,slice_id=slice_id,reason=reason)
        event["invalidated_certificates"] = list(self.planner.certificates.invalidate_dependencies(endpoint_id,tuple(metrics),slice_id,reason,at=at))
        self.pending[event["event_id"]] = event
        if self.state:
            # Invalidation itself is durable first; startup replays it even if map persistence is interrupted.
            self.state.write("drift",event,key=f"drift:{event['event_id']}")
            self.state.store_planner(self.planner)
        return event

    def refresh_endpoint(self, snapshot, *, at=None):
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
        if len(values)<self.window: raise ValueError("insufficient reference evidence")
        key = (endpoint_id,metric,slice_id)
        self.reference[key] = tuple(values[-self.window:])
        self._persist(key)

    def observe(self, endpoint_id, metric, slice_id, value: float, *, at=None):
        if metric not in {"quality","latency","failure","workload","evaluator"}: raise ValueError("unsupported statistical drift")
        if not math.isfinite(value) or value < 0: raise ValueError("finite nonnegative measurement required")
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
        event=self.pending[event_id]
        decision=self.planner.decide(request,policy,at=at)
        from inference_control.contracts import Call
        uses_target=any(isinstance(s,Call) and s.endpoint_id==event["endpoint_id"] for s in decision.selected_plan.steps)
        if decision.certificate_status!="current" or not uses_target:
            raise ValueError("affected target has not been re-certified for this slice")
        del self.pending[event_id]
        if self.state:self.state.write("recovery",{"event_id":event_id,"certificate_id":decision.certificate_id})
        return decision
