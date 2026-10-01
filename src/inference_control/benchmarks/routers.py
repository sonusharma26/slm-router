"""Train/calibrate-only baselines. No choose() method has access to held-out outcomes."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from time import perf_counter
from typing import Any
import math
from inference_control.contracts import Call, PolicySpec
from inference_control.planning import Planner
from inference_control.capability.conditional import ConditionalCapabilityMap, EvidenceObservation
from inference_control.policies.validator import eligibility_reasons
from inference_control.util import digest


@dataclass(frozen=True)
class Selection:
    endpoint_id: str | None
    certificate_status: str="uncertified"
    plan_type: str="direct"
    overhead_cost: float | None=0.
    overhead_ms: float=0.
    reason: str | None=None
    metadata: dict | None=None


def eligible(query,endpoints,policy):
    return [e for e in endpoints if not eligibility_reasons(e,query.request,policy)]

def price(ep,query):
    return ((query.request.input_tokens+ep.input_token_overhead)*ep.input_price_per_million+
            query.request.max_output_tokens*ep.output_price_per_million)/1e6


class Baseline:
    def __init__(self,mode="cheapest",*,seed=0,k=32,threshold=.8):
        if mode not in {"cheapest","random","best_single","knn","threshold"}:raise ValueError("unknown baseline")
        self.name,self.seed,self.k,self.threshold=mode,seed,k,threshold
        self.training=[]
        self.average={}

    def fit(self,queries,outcomes):
        if any(q.split!="train" for q in queries):raise ValueError("baseline fit accepts training queries only")
        lookup={q.request.request_id:q for q in queries}
        if any(r.request_id not in lookup for r in outcomes):raise ValueError("training outcome leakage")
        self.training=[(lookup[r.request_id],r) for r in outcomes]
        groups={}
        for _,row in self.training:groups.setdefault(row.endpoint_id,[]).append(row.quality)
        self.average={e:mean(values) for e,values in groups.items()}

    def _estimate(self,query,endpoint_id):
        rows=[(q,r) for q,r in self.training if r.endpoint_id==endpoint_id]
        if not rows:return 0.
        def distance(item):
            q,r=item
            feature_distance=sum((x-y)**2 for x,y in zip(q.request.query_features,query.request.query_features))
            mismatch=float(q.request.task_hint!=query.request.task_hint)
            return (mismatch+feature_distance,q.request.request_id)
        neighbors=sorted(rows,key=distance)[:self.k]
        return mean(r.quality for _,r in neighbors)

    def choose(self,query,endpoints,policy):
        start=perf_counter()
        candidates=eligible(query,endpoints,policy)
        if not candidates:return Selection(None,reason="no_eligible_endpoint")
        if self.name=="cheapest":selected=min(candidates,key=lambda e:(price(e,query),e.endpoint_id))
        elif self.name=="random":selected=sorted(candidates,key=lambda e:e.endpoint_id)[int(digest((self.seed,query.request.request_id))[:16],16)%len(candidates)]
        elif self.name=="best_single":
            selected=min(candidates,key=lambda e:(-self.average.get(e.endpoint_id,0),e.endpoint_id))
        else:
            estimates={e.endpoint_id:self._estimate(query,e.endpoint_id) for e in candidates}
            if self.name=="threshold":
                cheap=min(candidates,key=lambda e:(price(e,query),e.endpoint_id))
                selected=cheap if estimates[cheap.endpoint_id]>=self.threshold else max(candidates,key=lambda e:(estimates[e.endpoint_id],-price(e,query)))
            else:
                qualified=[e for e in candidates if estimates[e.endpoint_id]>=policy.minimum_quality]
                selected=min(qualified,key=lambda e:(price(e,query),e.endpoint_id)) if qualified else max(candidates,key=lambda e:(estimates[e.endpoint_id],-price(e,query)))
        return Selection(selected.endpoint_id,overhead_ms=(perf_counter()-start)*1000)


class SLMRouter:
    name="slm-router"
    def __init__(self,endpoints,*,k=128,min_samples=20,at=None):
        self.at=at or datetime.now(timezone.utc)
        self.map=ConditionalCapabilityMap(k=k,min_samples=min_samples)
        self.planner=Planner(endpoints,capability_map=self.map)
        self.missing_latency=False

    def fit(self,queries,outcomes):
        if any(q.split not in {"train","calibration"} for q in queries):raise ValueError("SLM fit cannot see validation/test labels")
        lookup={q.request.request_id:q for q in queries}
        observations=[]
        for r in outcomes:
            if r.request_id not in lookup:raise ValueError("fit outcome leakage")
            q=lookup[r.request_id]
            ep=self.planner.endpoints[r.endpoint_id]
            self.missing_latency |= r.latency_ms is None
            observations.append(EvidenceObservation(sample_id=f"fit:{r.request_id}:{r.endpoint_id}",request=q.request,
                target_id=r.endpoint_id,endpoint_ids=(r.endpoint_id,),endpoint_revisions=(ep.capability_revision,),
                quality=r.quality,cost=r.cost,latency_ms=r.latency_ms or 0.,output_tokens=r.output_tokens,failed=r.failed,
                split=q.split,evaluator_type=r.evaluator_type if r.evaluator_type in {"deterministic","application","human"} else "benchmark",
                certification_eligible=r.evaluator_type in {"deterministic","application","human"},
                evaluator_version=r.evaluator_version,observed_at=self.at,
                metrics=frozenset({"quality","cost","failure","latency"}) if r.latency_ms is not None else frozenset({"quality","cost","failure"})))
        self.map.add_many(observations)

    def choose(self,query,endpoints,policy):
        self.planner.endpoints={e.endpoint_id:e for e in endpoints}
        start=perf_counter()
        decision=self.planner.decide(query.request,policy,at=self.at)
        calls=[s for s in decision.selected_plan.steps if isinstance(s,Call)]
        if len(calls)>1:
            raise ValueError("single-endpoint matrices cannot score unobserved compound plans")
        return Selection(calls[0].endpoint_id if calls else None,decision.certificate_status,decision.selected_plan.plan_type,
            overhead_ms=(perf_counter()-start)*1000,metadata={"decision_fingerprint":decision.decision_fingerprint,
                "quality_lower":decision.estimated_quality.lower if decision.estimated_quality else None})
