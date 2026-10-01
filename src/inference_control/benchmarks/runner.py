"""First-class benchmark runner with frozen split manifests and non-oracle interfaces."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean
import platform
import sys
import math
from functools import lru_cache
from importlib.metadata import version,PackageNotFoundError
from inference_control.contracts import PolicySpec
from inference_control.benchmarks.data import BenchmarkDataset
from inference_control.benchmarks.routers import Baseline, SLMRouter
from inference_control.benchmarks.metrics import summarize, paired_difference, matched_frontier
from inference_control.policies.validator import eligibility_reasons
from inference_control.util import digest, canonical


EXTERNALS=("RouteLLM","vLLM Semantic Router")


@lru_cache(maxsize=1)
def environment():
    root=Path(__file__).resolve().parents[1]
    sources={str(file.relative_to(root)):digest(file.read_text(encoding="utf-8")) for file in sorted(root.rglob("*.py"))}
    dependencies={}
    for name in ("pydantic","httpx","fastapi","typer","jsonschema","routellm"):
        try:dependencies[name]=version(name)
        except PackageNotFoundError:dependencies[name]=None
    return {"release":"2.0.0a2","python":sys.version,"platform":platform.platform(),
            "source_hash":digest(sources),"dependencies":dependencies}


def default_policy(*,minimum_quality=.65,objective="cost",missing_latency=False):
    return PolicySpec(policy_id="benchmark",version=f"bench-{minimum_quality}-{objective}",minimum_quality=minimum_quality,
        max_expected_spend=1.,max_absolute_spend=10.,deadline_ms=1e301 if missing_latency else 2500,
        objective=objective,minimum_evidence_samples=8,require_certificate=False)


def split_data(dataset, splits):
    queries=[q for q in dataset.queries if q.split in splits]
    ids={q.request.request_id for q in queries}
    return queries,[r for r in dataset.outcomes if r.request_id in ids]


def evaluate_router(router,dataset,policy,*,split="test"):
    queries=[q for q in dataset.queries if q.split==split]
    lookup={(r.request_id,r.endpoint_id):r for r in dataset.outcomes}
    records=[]
    for query in queries:
        # No row from lookup is passed to choose(); even oracle regret is computed afterwards.
        choice=router.choose(query,tuple(dataset.endpoints),policy)
        candidates=[lookup[(query.request.request_id,e.endpoint_id)] for e in dataset.endpoints
                    if not eligibility_reasons(e,query.request,policy)]
        oracle=max((r.quality for r in candidates if r.cost<=policy.max_absolute_spend and
                    (r.latency_ms is None or r.latency_ms<=policy.deadline_ms)),default=0.)
        if choice.endpoint_id is None:
            quality,cost,latency,violation,shortfall=0.,choice.overhead_cost,None,False,False
        else:
            key=(query.request.request_id,choice.endpoint_id)
            if key not in lookup:raise ValueError("router selected outside the identical model pool")
            row=lookup[key]
            ep=next(e for e in dataset.endpoints if e.endpoint_id==choice.endpoint_id)
            quality=row.quality
            cost=row.cost+choice.overhead_cost if choice.overhead_cost is not None else None
            latency=row.latency_ms+choice.overhead_ms if row.latency_ms is not None else None
            violation=bool(eligibility_reasons(ep,query.request,policy)) or row.failed or (
                cost is not None and cost>policy.max_absolute_spend) or (latency is not None and latency>policy.deadline_ms)
            shortfall=quality<policy.minimum_quality
        records.append({"request_id":query.request.request_id,"group_id":query.group_id,
            "slice":"/".join(sorted(query.request.traffic_slices)) or "default",
            "endpoint_id":choice.endpoint_id,"abstained":choice.endpoint_id is None,
            "certified":choice.certificate_status=="current","quality":quality,"cost":cost,
            "latency_ms":latency,"routing_overhead_ms":choice.overhead_ms,
            "constraint_violation":violation,"quality_shortfall":shortfall,
            "regret":max(0.,oracle-quality),"plan_type":choice.plan_type,"metadata":choice.metadata})
    return records


def run_static(dataset:BenchmarkDataset,*,policy=None,seed=42,resamples=400,competitors=None,sweep=True):
    train,train_rows=split_data(dataset,{"train"})
    fit,fit_rows=split_data(dataset,{"train","calibration"})
    if not train or not any(q.split=="calibration" for q in fit) or not any(q.split=="test" for q in dataset.queries):
        raise ValueError("benchmark needs nonempty train/calibration/test splits")
    # Group duplicates within a split would invalidate iid CIs; refuse instead of overstating confidence.
    test_groups=[q.group_id for q in dataset.queries if q.split=="test"]
    if len(test_groups)!=len(set(test_groups)):raise ValueError("test groups are dependent; deduplicate or use a grouped study")
    missing=any(r.latency_ms is None for r in fit_rows)
    policy=policy or default_policy(missing_latency=missing)
    if missing and policy.require_certificate:raise ValueError("missing latency evidence cannot support certificate benchmark claims")
    routers=[]
    for mode in ("cheapest","random","best_single","knn","threshold"):
        router=Baseline(mode,seed=seed);router.fit(train,train_rows);routers.append(router)
    slm=SLMRouter(dataset.endpoints,k=128,min_samples=8);slm.fit(fit,fit_rows);routers.append(slm)
    # Threshold selection is validation-only and frozen before test evaluation.
    validation,_=split_data(dataset,{"validation"})
    if validation:
        threshold_router=routers[-2]  # threshold baseline
        trials=[]
        for threshold in (.4,.55,.65,.75,.85,.95):
            threshold_router.threshold=threshold
            rows=evaluate_router(threshold_router,dataset,policy,split="validation")
            quality=mean(r["quality"] for r in rows);cost=mean(r["cost"] for r in rows)
            trials.append((quality>=policy.minimum_quality,cost,quality,threshold))
        feasible=[r for r in trials if r[0]]
        best=min(feasible,key=lambda r:(r[1],-r[2])) if feasible else max(trials,key=lambda r:r[2])
        threshold_router.threshold=best[3]
    routers.extend(competitors or [])
    results,records,status={}, {}, {}
    for router in routers:
        try:
            rows=evaluate_router(router,dataset,policy)
            records[router.name]=rows
            results[router.name]=summarize(rows,seed=seed,resamples=resamples)
            status[router.name]={"state":"completed","configuration":{"threshold":getattr(router,"threshold",None)}}
        except Exception as exc:
            if router in (competitors or []):
                status[router.name]={"state":"failed","reason":f"{type(exc).__name__}: {exc}"}
            else:raise
    for name in EXTERNALS:
        status.setdefault(name,{"state":"not_run","reason":"external competitor/configuration not supplied"})
    paired={name:paired_difference(records["slm-router"],rows,seed=seed,resamples=resamples)
            for name,rows in records.items() if name!="slm-router"}
    points=[]
    if sweep:
        for router in routers:
            if router in (competitors or []):continue # external threshold grids require explicit recorded configs
            for threshold in (0.,.3,.5,.65,.8,.9):
                p=policy.model_copy(update={"minimum_quality":threshold,"version":f"sweep-{threshold}"})
                rows=evaluate_router(router,dataset,p)
                points.append({"router":router.name,"minimum_quality":threshold,
                               "quality":mean(r["quality"] for r in rows),"mean_cost":mean(r["cost"] for r in rows),
                               "coverage":1-mean(float(r["abstained"]) for r in rows)})
    matched=[]
    slm_points=[p for p in points if p["router"]=="slm-router"]
    for name in results:
        if name=="slm-router":continue
        competitor_points=[p for p in points if p["router"]==name]
        for point in competitor_points:
            cost_matches=[p for p in slm_points if p["mean_cost"]<=point["mean_cost"]]
            quality_matches=[p for p in slm_points if p["quality"]>=point["quality"]]
            matched.append({"baseline":name,"baseline_cost":point["mean_cost"],"baseline_quality":point["quality"],
                "slm_quality_at_no_greater_cost":max((p["quality"] for p in cost_matches),default=None),
                "slm_cost_at_no_lower_quality":min((p["mean_cost"] for p in quality_matches),default=None)})
    blockers=[]
    if dataset.evidence_kind=="synthetic":blockers.append("SYNTHETIC_DATA_IS_NOT_EXTERNAL_BENCHMARK_EVIDENCE")
    if any(status[n]["state"]!="completed" for n in EXTERNALS):blockers.append("EXTERNAL_COMPETITORS_NOT_ALL_RUN")
    if any(v["known_cost_coverage"]<1 for v in results.values()):blockers.append("UNACCOUNTED_ROUTER_OVERHEAD_COST")
    if any(v["ci95"][0]<=0 for v in paired.values()):blockers.append("QUALITY_SUPERIORITY_NOT_ESTABLISHED_AT_THIS_OPERATING_POINT")
    blockers.append("MULTIPLE_DATASETS_SEEDS_AND_MATCHED_COST_INFERENCE_REQUIRED_FOR_BROAD_CLAIMS")
    return {"schema":"slm-benchmark-v1","kind":"static","generated_at":datetime.now(timezone.utc).isoformat(),
        "environment":environment(),"dataset":dataset.name,"evidence_kind":dataset.evidence_kind,"dataset_hash":dataset.fingerprint,
        "split_hash":dataset.split_hash,"seed":seed,"python":sys.version,"platform":platform.platform(),
        "model_pool":[e.endpoint_id for e in dataset.endpoints],"endpoint_snapshot_ids":[e.snapshot_id for e in dataset.endpoints],
        "policy":policy.model_dump(mode="json"),"policy_hash":digest(policy),"status":status,
        "results":results,"paired_quality_differences":paired,"operating_points":points,
        "pareto_frontier":matched_frontier(points),"matched_comparisons":matched,
        "claim_gate":{"passed":False,"blockers":blockers},"records":records}


def write_report(report,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/"report.json").write_text(canonical(report)+"\n",encoding="utf-8")
    lines=["# SLM Router Benchmark Report","",f"Evidence: **{report.get('evidence_kind','synthetic')}**",
           "", "Synthetic experiments validate software behavior, not real-world or competitor superiority.",""]
    if report["kind"]=="static":
        lines += ["| Router | Quality (all) | Cost/request | Abstention | Certified | p95 route ms |",
                  "|---|---:|---:|---:|---:|---:|"]
        for name,r in report["results"].items():
            cost=f"${r['mean_cost']:.6f}" if r["mean_cost"] is not None else "unknown"
            lines.append(f"| {name} | {r['quality_all_requests']:.4f} | {cost} | {r['abstention_rate']:.1%} | {r['certificate_coverage']:.1%} | {r['routing_overhead_p95_ms']:.3f} |")
        lines += ["","## External competitor status",""]
        lines += [f"- {n}: {report['status'][n]['state']}" for n in EXTERNALS]
        lines += ["","## Claim gate","",*['- '+b for b in report["claim_gate"]["blockers"]]]
    else:
        lines += ["See `report.json` for per-scenario, per-budget measurements and event traces."]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return out/"report.json"
