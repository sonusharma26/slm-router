"""Small deterministic fixtures. These are explicitly not public benchmark evidence."""
from __future__ import annotations
from inference_control.benchmarks.data import BenchmarkDataset, Query, Truth, grouped_split
from inference_control.contracts import EndpointSnapshot,RequestContext
from inference_control.util import digest


def uniform(*parts):return int(digest(parts)[:16],16)/2**64


def pool():
    return [EndpointSnapshot(endpoint_id=name,provider="fixture",upstream_model=name,revision="fixture-r1",
        region="local",context_window=8192,config_hash="fixture-config",price_version="fixture-price-1",
        input_price_per_million=rate,output_price_per_million=2*rate,governance=frozenset({"local"}),
        capabilities=frozenset({"structured_output","tools"})) for name,rate in (("cheap",.4),("code",1.),("strong",3.))]


def make_dataset(n=600,seed=42):
    if n<40:raise ValueError("at least 40 synthetic queries required")
    endpoints=pool();queries=[];outcomes=[]
    for i in range(n):
        task="code" if i%2 else "math"
        difficulty=uniform(seed,"difficulty",i)
        text=f"{task} exercise {i}: {'explain and prove' if difficulty>.5 else 'classify'}"
        qid=f"fixture-{seed}-{i}";group=digest(text)
        request=RequestContext(request_id=qid,application_id="benchmark",tenant_policy_id="benchmark",input_tokens=128,
            max_output_tokens=64,feature_version="fixture-query-v1",task_hint=task,traffic_slices=frozenset({task}),
            query_features=(difficulty,float(task=="code")))
        queries.append(Query(request=request,text=text,split=grouped_split(group,seed),group_id=group))
        for ep in endpoints:
            score=(.99-.65*difficulty if ep.endpoint_id=="cheap" else
                   (.98-.04*difficulty if task=="code" else .45-.15*difficulty) if ep.endpoint_id=="code" else .98-.03*difficulty)
            # Continuous bounded rubric scores make this a functional fixture, not fabricated test accuracy.
            score=max(0,min(1,score+(uniform(seed,qid,ep.endpoint_id)-.5)*.02))
            output=32
            cost=(128*ep.input_price_per_million+output*ep.output_price_per_million)/1e6
            latency={"cheap":80,"code":120,"strong":180}[ep.endpoint_id]*(.9+.2*uniform(seed,"latency",qid,ep.endpoint_id))
            outcomes.append(Truth(request_id=qid,endpoint_id=ep.endpoint_id,quality=score,cost=cost,
                latency_ms=latency,input_tokens=128,output_tokens=output,evaluator_version="synthetic-rubric-v1"))
    return BenchmarkDataset(name=f"synthetic-control-fixture-{seed}",evidence_kind="synthetic",queries=queries,
        outcomes=outcomes,endpoints=endpoints,provenance={"seed":seed,"purpose":"functional regression, not empirical superiority",
                                                       "quality":"continuous simulated rubric, not executable task accuracy"})
