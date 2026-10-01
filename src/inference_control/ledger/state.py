"""Durable read models reconstructed from immutable events, no pickle or opaque state."""
from __future__ import annotations
from inference_control.contracts import EndpointSnapshot, PolicySpec, DecisionRecord, ExecutionRecord, OutcomeRecord
from inference_control.policies import CertificateStore
from inference_control.util import digest


class ControlState:
    def __init__(self, ledger):
        self.ledger = ledger
        self.sequence = 0
        self.endpoints: dict[str, EndpointSnapshot] = {}
        self.snapshots: dict[str, EndpointSnapshot] = {}
        self.policies: dict[tuple[str,str], PolicySpec] = {}
        self.maps: dict[str, dict] = {}
        self.latest_map: str | None = None
        self.static_estimates: dict | None = None
        self.certificates = CertificateStore()
        self.decisions: dict[str, DecisionRecord] = {}
        self.executions: dict[str, ExecutionRecord] = {}
        self.execution_by_decision: dict[str, ExecutionRecord] = {}
        self.claims: dict[str, dict] = {}
        self.outcomes: dict[str, OutcomeRecord] = {}
        self.active_outcomes: dict[str, str] = {}
        self.drifts: list[dict] = []
        self.lifecycle: dict[str, dict] = {}
        self.trust: dict[str, dict] = {}
        self.detectors: dict[str, dict] = {}
        self.recovered_events: set[str] = set()
        if not ledger.verify():
            raise ValueError("ledger hash-chain verification failed")
        self.refresh()

    def refresh(self):
        for event in self.ledger.events(after=self.sequence):
            self.apply(event)

    def apply(self, event):
        p, kind = event.payload, event.event_type
        if kind == "endpoint_snapshot":
            ep = EndpointSnapshot.model_validate(p)
            self.snapshots[ep.snapshot_id] = ep
            self.endpoints[ep.endpoint_id] = ep
        elif kind == "policy":
            policy = PolicySpec.model_validate(p)
            key = (policy.policy_id, policy.version)
            if key in self.policies and self.policies[key] != policy:
                raise ValueError("immutable policy version conflict in ledger")
            self.policies[key] = policy
        elif kind == "capability_map":
            self.maps[p["version"]] = p["data"]
            self.latest_map = p["version"]
        elif kind == "static_estimates":
            self.static_estimates = p
        elif kind == "certificates":
            self.certificates.restore(p["records"])
        elif kind == "decision":
            row = DecisionRecord.model_validate(p)
            self.decisions[row.decision_id] = row
        elif kind == "execution_started":
            self.claims[p["decision_id"]] = p
        elif kind == "execution":
            row = ExecutionRecord.model_validate(p)
            self.executions[row.execution_id] = row
            self.execution_by_decision[row.decision_id] = row
        elif kind == "outcome":
            row = OutcomeRecord.model_validate(p)
            self.outcomes[row.outcome_id] = row
            self.active_outcomes[row.decision_id] = row.outcome_id
        elif kind == "outcome_disputed":
            old = self.outcomes[p["outcome_id"]]
            self.outcomes[old.outcome_id] = old.model_copy(update={"disputed":True,"training_eligible":False,"promotion_eligible":False})
        elif kind == "drift": self.drifts.append(p)
        elif kind == "lifecycle": self.lifecycle[p["policy_id"]] = p
        elif kind == "evaluator_trust": self.trust[p["evaluator_version"]] = p
        elif kind == "drift_detector": self.detectors[p["key"]] = p
        elif kind == "recovery": self.recovered_events.add(p["event_id"])
        self.sequence = event.sequence

    def write(self, kind: str, payload: dict, *, key: str | None = None, actor: str = "control-plane"):
        event = self.ledger.append(kind, payload, actor=actor, idempotency_key=key or f"{kind}:{digest(payload)}")
        self.refresh()
        return event

    def store_planner(self, planner, policy=None):
        entries = []
        for ep in planner.endpoints.values():
            if self.endpoints.get(ep.endpoint_id) != ep:
                entries.append(("endpoint_snapshot",ep.model_dump(mode="json"),f"snapshot-activation:{ep.endpoint_id}:{self.sequence}:{ep.snapshot_id}"))
        if policy:
            key = (policy.policy_id,policy.version)
            if key in self.policies and self.policies[key] != policy:
                raise ValueError("policy versions are immutable; increment version")
            entries.append(("policy", policy.model_dump(mode="json"), f"policy:{digest(policy)}"))
        if planner.capability_map is None:
            from inference_control.util import primitive
            data={"version":planner.map_version,"estimates":primitive(list(planner.estimates.values()))}
            entries.append(("static_estimates",data,f"static-estimates:{digest(data)}"))
        if planner.capability_map is not None and planner.map_version != self.latest_map:
            entries.append(("capability_map", {"version":planner.map_version,"data":planner.capability_map.export()}, f"map-activation:{self.sequence}:{planner.map_version}"))
        if planner.certificates.export():
            data = {"records":planner.certificates.export()}
            entries.append(("certificates",data,f"certificates:{digest(data)}"))
        if entries:
            self.ledger.append_batch(entries, actor="control-plane")
            self.refresh()
