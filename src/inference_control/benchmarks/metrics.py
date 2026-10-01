"""Abstention-aware metrics and paired, deterministic bootstrap intervals."""
from __future__ import annotations
from collections import defaultdict
import math
import random
from statistics import mean


def quantile(values,p):
    if not values:return None
    values=sorted(values);position=(len(values)-1)*p;lower=int(position);upper=min(len(values)-1,lower+1)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def bootstrap(values,*,seed=0,resamples=400,confidence=.95):
    if not values:return [None,None]
    rng=random.Random(seed);n=len(values)
    samples=sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples))
    return [quantile(samples,(1-confidence)/2),quantile(samples,1-(1-confidence)/2)]


def summarize(records,*,seed=0,resamples=400):
    n=len(records)
    if not n:raise ValueError("no evaluated requests")
    served=[r for r in records if not r["abstained"]]
    certified=[r for r in records if r["certified"]]
    slices=defaultdict(list)
    for r in records:slices[r["slice"]].append(r["quality"])
    costs=[r["cost"] for r in records if r["cost"] is not None]
    latency=[r["latency_ms"] for r in served if r["latency_ms"] is not None]
    overhead=[r["routing_overhead_ms"] for r in records]
    return {"requests":n,"quality_all_requests":mean(r["quality"] for r in records),
        "quality_served":mean(r["quality"] for r in served) if served else None,
        "quality_ci95":bootstrap([r["quality"] for r in records],seed=seed,resamples=resamples),
        "mean_cost":mean(costs) if len(costs)==n else None,
        "known_cost_coverage":len(costs)/n,"known_cost_sum":sum(costs),
        "cost_ci95":bootstrap(costs,seed=seed,resamples=resamples) if len(costs)==n else [None,None],
        "abstention_rate":1-len(served)/n,"certificate_coverage":len(certified)/n,
        "constraint_violation_rate":mean(float(r["constraint_violation"]) for r in served) if served else None,
        "certified_constraint_violation_rate":mean(float(r["constraint_violation"]) for r in certified) if certified else None,
        "quality_shortfall_rate":mean(float(r["quality_shortfall"]) for r in served) if served else None,
        "certified_quality_shortfall_rate":mean(float(r["quality_shortfall"]) for r in certified) if certified else None,
        "worst_slice_quality":min(mean(v) for v in slices.values()),
        "slice_quality":{s:mean(v) for s,v in sorted(slices.items())},
        "latency_coverage":len(latency)/len(served) if served else 0.,
        "latency_p50_ms":quantile(latency,.5),"latency_p95_ms":quantile(latency,.95),
        "routing_overhead_p50_ms":quantile(overhead,.5),"routing_overhead_p95_ms":quantile(overhead,.95),
        "mean_regret":mean(r["regret"] for r in records),
        "notes":["Abstentions score zero in all-request quality and regret.",
                 "A binary wrong answer is a quality shortfall, not by itself proof a population confidence bound failed.",
                 "Bootstrap intervals assume independent request groups; use grouped/blocked analysis for dependent workloads."]}


def paired_difference(a,b,*,metric="quality",seed=0,resamples=400):
    left={r["request_id"]:r for r in a};right={r["request_id"]:r for r in b}
    if set(left)!=set(right):raise ValueError("paired comparison requires identical requests")
    values=[left[k][metric]-right[k][metric] for k in sorted(left)]
    return {"mean_delta":mean(values),"ci95":bootstrap(values,seed=seed,resamples=resamples),"paired_requests":len(values)}


def matched_frontier(points):
    """Only report observed policy operating points; do not fabricate interpolated wins."""
    valid=[p for p in points if p.get("mean_cost") is not None]
    return [p for p in valid if not any(q["mean_cost"]<=p["mean_cost"] and q["quality"]>=p["quality"]
            and (q["mean_cost"]<p["mean_cost"] or q["quality"]>p["quality"]) for q in valid)]
