"""Dynamic-pool and sparse-refresh runners using the real planner, probe loop and drift path.

Hidden faults are revealed only through selected/probed outcomes. Snapshot changes are
public metadata and delivered equally to every participant, including frozen baselines.
"""
from __future__ import annotations
from datetime import datetime,timedelta,timezone
from collections import deque
from statistics import mean
import math
from inference_control.benchmarks.synthetic import make_dataset,uniform
from inference_control.benchmarks.data import Query,Truth
from inference_control.benchmarks.routers import SLMRouter,Baseline
from inference_control.benchmarks.runner import default_policy,split_data,environment
from inference_control.benchmarks.metrics import summarize
from inference_control.capability.conditional import EvidenceObservation
from inference_control.probes.active import ActiveMeasurementLoop,DailyProbeBudget,ProbeTask
from inference_control.ledger import SQLiteLedger
from inference_control.ledger.state import ControlState
from inference_control.drift.recovery import DriftRecovery,EndpointLatency
from inference_control.util import digest


SCENARIOS=("price_x2","endpoint_disappears","revision_change","latency_x3","quality_minus20pct",
           "workload_shift","delayed_rewards","biased_evaluator")


class DynamicWorld:
    def __init__(self,endpoints,*,seed,scenario,inject_at):
        if scenario not in SCENARIOS:raise ValueError("unknown dynamic scenario")
        self.original={e.endpoint_id:e for e in endpoints};self.endpoints=dict(self.original)
        self.seed,self.scenario,self.inject_at=seed,scenario,inject_at
        self.changed=False

    def advance(self,t):
        if t<self.inject_at or self.changed:return
        self.changed=True
        ep=self.endpoints["cheap"]
        updates={}
        if self.scenario=="price_x2":updates={"input_price_per_million":ep.input_price_per_million*2,
            "output_price_per_million":ep.output_price_per_million*2,"price_version":"fixture-price-2"}
        elif self.scenario=="endpoint_disappears":updates={"available":False}
        elif self.scenario=="revision_change":updates={"revision":"fixture-r2"}
        if updates:self.endpoints["cheap"]=ep.model_copy(update=updates)

    def request(self,t):
        task="code" if t%2 else "math"
        difficulty=uniform(self.seed,"dynamic-difficulty",t)
        if self.changed and self.scenario=="workload_shift":task="code";difficulty=.8+.2*difficulty
        from inference_control.contracts import RequestContext
        req=RequestContext(request_id=f"dynamic-{self.seed}-{t}",application_id="benchmark",tenant_policy_id="benchmark",
            input_tokens=128,max_output_tokens=64,feature_version="fixture-query-v1",task_hint=task,
            traffic_slices=frozenset({task}),query_features=(difficulty,float(task=="code")))
        return Query(request=req,text=f"{task} workload item {t}",split="test",group_id=req.request_id)

    def outcome(self,request,endpoint_id,t):
        ep=self.endpoints[endpoint_id];difficulty=request.query_features[0];task=request.task_hint
        quality=(.99-.65*difficulty if endpoint_id=="cheap" else
                 (.98-.04*difficulty if task=="code" else .45-.15*difficulty) if endpoint_id=="code" else .98-.03*difficulty)
        if t>=self.inject_at and endpoint_id=="cheap" and self.scenario in {"quality_minus20pct","revision_change"}:
            quality*=.8
        latency={"cheap":80,"code":120,"strong":180}[endpoint_id]
        if t>=self.inject_at and endpoint_id=="cheap" and self.scenario=="latency_x3":latency*=3
        failed=not ep.available
        return Truth(request_id=request.request_id,endpoint_id=endpoint_id,quality=0 if failed else quality,
            cost=(128*ep.input_price_per_million+32*ep.output_price_per_million)/1e6,latency_ms=latency,
            input_tokens=128,output_tokens=32,failed=failed,evaluator_version="synthetic-rubric-v1")


def run_dynamic(*,scenario="quality_minus20pct",steps=80,seed=42,strategies=("frozen","impact","random","exhaustive"),
                probe_fraction=.1,initial_requests=1200):
    if steps<8 or not 0<=probe_fraction<=1:raise ValueError("invalid workload/budget")
    data=make_dataset(initial_requests,seed)
    fit,fit_rows=split_data(data,{"train","calibration"})
    at=datetime(2026,1,1,tzinfo=timezone.utc)
    policy=default_policy(minimum_quality=.55)
    # A meaningful latency cap makes the latency shock routing-relevant.
    policy=policy.model_copy(update={"deadline_ms":220.})
    outputs={};inject_at=steps//2
    task_by_request={q.request.request_id:q.request.task_hint for q in fit}
    reference_rows={}
    for row in fit_rows:
        reference_rows.setdefault((row.endpoint_id,task_by_request[row.request_id]),[]).append(row)
    for strategy in strategies:
        if strategy not in {"frozen","cheapest","impact","uncertainty","random","exhaustive"}:raise ValueError("unknown refresh strategy")
        world=DynamicWorld(data.endpoints,seed=seed,scenario=scenario,inject_at=inject_at)
        router=SLMRouter(data.endpoints,k=128,min_samples=8,at=at);router.fit(fit,fit_rows)
        # All strategies begin with the same full initial evidence; refresh budgets are additional.
        if strategy=="cheapest":
            baseline=Baseline("cheapest");baseline.fit(*split_data(data,{"train"}))
        ledger=SQLiteLedger(":memory:")
        state=ControlState(ledger)
        state.store_planner(router.planner,policy)
        recovery=DriftRecovery(router.planner,state,window=8,minimum_shift=.1)
        for ep in data.endpoints:
            for task in ("code","math"):
                refs=reference_rows.get((ep.endpoint_id,task),[])
                if len(refs)>=8:
                    recovery.set_reference(ep.endpoint_id,"quality",task,[r.quality for r in refs])
                    recovery.set_reference(ep.endpoint_id,"latency",task,[r.latency_ms for r in refs])
        exhaustive_calls=steps*len(data.endpoints)
        calls=exhaustive_calls if strategy=="exhaustive" else math.floor(exhaustive_calls*probe_fraction)
        budget=DailyProbeBudget(ledger,max_dollars=100.,max_calls=calls)
        loop=ActiveMeasurementLoop(router.planner,budget,recovery=recovery,seed=seed)
        delayed=deque();records=[];events=[];last_snapshots=dict(world.endpoints)
        bad_after=0;first_detection=None;recovery_at=None;good_streak=0
        map_errors=[];error_observed=0;error_possible=0
        for t in range(steps):
            world.advance(t);time=at+timedelta(seconds=t+1);router.at=time
            for name,ep in world.endpoints.items():
                if ep!=last_snapshots[name]:
                    if strategy not in {"frozen","cheapest"}:
                        event=recovery.refresh_endpoint(ep,at=time)
                        if event:events.append({"step":t,**event})
                    else:router.planner.endpoints[name]=ep
                    last_snapshots[name]=ep
            query=world.request(t)
            # Delayed labels are delivered in event time and can only affect future decisions.
            while delayed and delayed[0][0]<=t:
                _,row=delayed.popleft()
                if strategy not in {"frozen","cheapest"}:
                    try:router.map.add(row)
                    except ValueError:pass
                    event=recovery.observe(row.endpoint_ids[0],"quality",row.request.task_hint,row.quality,
                                           at=time,observed_at=row.observed_at)
                    if event:events.append({"step":t,**event})
            if strategy not in {"frozen","cheapest"}:
                # Same public query catalog and cap for every acquisition strategy.
                probe_query=world.request(1_000_000+t)  # Independent measurement catalog; NEVER the current held-out request.
                tasks=[ProbeTask(f"probe-{t}-{ep}",ep,probe_query.request) for ep in sorted(world.endpoints)]
                def measure(task):
                    if task.request.request_id == query.request.request_id:
                        raise ValueError("held-out query leaked into pre-decision probes")
                    r=world.outcome(task.request,task.endpoint_id,t)
                    return EvidenceObservation(sample_id=f"probe-{strategy}-{t}-{task.endpoint_id}",request=task.request,
                        target_id=task.endpoint_id,endpoint_ids=(task.endpoint_id,),
                        endpoint_revisions=(world.endpoints[task.endpoint_id].capability_revision,),quality=r.quality,cost=r.cost,
                        latency_ms=r.latency_ms,output_tokens=r.output_tokens,failed=r.failed,
                        split="calibration" if int(digest(task.request.request_id)[:8],16)%5==0 else "train",
                        evaluator_type="deterministic",evaluator_version="synthetic-rubric-v1",observed_at=time)
                # Pace budget across time rather than spending all probes before the injected change.
                permitted=math.floor((t+1)*calls/steps)-budget.usage(time)["calls_reserved"]
                if permitted>0:
                    measured=loop.run(tasks,policy,measure,strategy=strategy,max_calls=permitted,at=time)
                    events.extend({"step":t,**event} for event in measured["drift_events"])
            choice=baseline.choose(query,tuple(world.endpoints.values()),policy) if strategy=="cheapest" else router.choose(query,tuple(world.endpoints.values()),policy)
            truths=[world.outcome(query.request,e,t) for e in world.endpoints]
            feasible=[r for r in truths if world.endpoints[r.endpoint_id].available and r.latency_ms<=policy.deadline_ms]
            oracle=max((r.quality for r in feasible),default=0.)
            selected=next((r for r in truths if r.endpoint_id==choice.endpoint_id),None)
            quality=selected.quality if selected else 0.;violation=bool(selected and (selected.failed or selected.latency_ms>policy.deadline_ms))
            shortfall=bool(selected and quality<policy.minimum_quality)
            record={"request_id":query.request.request_id,"group_id":query.group_id,"slice":query.request.task_hint,
                "endpoint_id":choice.endpoint_id,"abstained":selected is None,"certified":choice.certificate_status=="current",
                "quality":quality,"cost":selected.cost if selected else 0.,"latency_ms":selected.latency_ms if selected else None,
                "routing_overhead_ms":choice.overhead_ms,"constraint_violation":violation,"quality_shortfall":shortfall,
                "regret":max(0,oracle-quality),"step":t}
            records.append(record)
            if selected and strategy not in {"frozen","cheapest"}:
                observation=EndpointLatency(source_id=f"serving:{strategy}:{t}",endpoint_id=selected.endpoint_id,
                    capability_revision=world.endpoints[selected.endpoint_id].capability_revision,
                    slice_id=query.request.task_hint,policy_hash=digest(policy),deadline_ms=policy.deadline_ms,
                    observed_at=time,received_at=time,duration_ms=selected.latency_ms,completed=not selected.failed)
                events.extend({"step":t,**event} for event in recovery.observe_latency(observation))
                # Biased judge output is logged, but never promoted into trusted calibration.
                biased=scenario=="biased_evaluator" and t>=inject_at
                if biased:
                    ledger.append("untrusted_evaluator_observation",{"step":t,"quality":min(1.,quality+.25)},
                                  actor="simulation",idempotency_key=f"untrusted:{t}")
                else:
                    row=EvidenceObservation(sample_id=f"feedback-{t}-{selected.endpoint_id}",request=query.request,
                        target_id=selected.endpoint_id,endpoint_ids=(selected.endpoint_id,),endpoint_revisions=(world.endpoints[selected.endpoint_id].capability_revision,),
                        quality=quality,cost=selected.cost,latency_ms=selected.latency_ms,output_tokens=selected.output_tokens,failed=selected.failed,
                        split="calibration" if int(digest(query.request.request_id)[:8],16)%5==0 else "train",
                        evaluator_type="deterministic",evaluator_version="synthetic-rubric-v1",observed_at=time)
                    delay=12 if scenario=="delayed_rewards" and t>=inject_at else 1
                    delayed.append((t+delay,row))
            # Oracle-only measurement stays in the evaluator, never fed to the acquisition scorer.
            errors=[]
            for ep in world.endpoints.values():
                estimate=router.map.estimate(query.request,ep.endpoint_id,(ep,),at=time,min_samples=8)
                error_possible+=1
                if estimate:
                    error_observed+=1
                    errors.append(abs(estimate.quality.mean-world.outcome(query.request,ep.endpoint_id,t).quality))
            if errors:map_errors.append(mean(errors))
            if t>=inject_at:
                bad=violation or shortfall or selected is None
                bad_after+=int(bad);good_streak=0 if bad else good_streak+1
                if recovery_at is None and good_streak>=5:recovery_at=t-4
        post_events=[e for e in events if e["step"]>=inject_at]
        pre_events=[e for e in events if e["step"]<inject_at]
        relevant_metric={"price_x2":"cost","endpoint_disappears":"availability","revision_change":"quality",
                         "latency_x3":"latency","quality_minus20pct":"quality"}.get(scenario)
        targeted=[e for e in post_events if relevant_metric and e["endpoint_id"]=="cheap" and relevant_metric in e["metrics"]]
        first_detection=min((e["step"] for e in targeted),default=None)
        outputs[strategy]={"metrics":summarize(records,seed=seed,resamples=100),"events":events,"records":records,
            "probe_budget":budget.usage(at),"refresh_call_fraction":budget.usage(at)["calls_reserved"]/exhaustive_calls,
            "detection_delay_requests":first_detection-inject_at if first_detection is not None else None,
            "false_positive_events_before_change":len(pre_events),
            "false_positive_events_per_prechange_request":len(pre_events)/inject_at,
            "recovery_delay_requests":recovery_at-inject_at if recovery_at is not None else None,
            "bad_decisions_after_change":bad_after,"mean_capability_error":mean(map_errors) if map_errors else None,
            "capability_error_observed_fraction":error_observed/error_possible if error_possible else 0.,
            "fault_detection_instrumented":relevant_metric is not None,
            "violations_during_recovery":sum(r["constraint_violation"] for r in records if r["step"]>=inject_at and (recovery_at is None or r["step"]<=recovery_at)),
            "pending_delayed_outcomes":len(delayed)}
        ledger.close()
    return {"schema":"slm-benchmark-v1","kind":"dynamic","evidence_kind":"synthetic","scenario":scenario,"environment":environment(),
        "initial_dataset_hash":data.fingerprint,"initial_split_hash":data.split_hash,"initial_requests":initial_requests,
        "seed":seed,"steps":steps,"injected_at":inject_at,"probe_fraction":probe_fraction,"results":outputs,
        "limitations":["Synthetic fixed pool; not empirical competitor superiority.",
                       "Workload-shift, delayed-reward and biased-evaluator injections are stress tests, not automatically detected fault claims.",
                       "Online CI values are descriptive IID bootstraps; adaptive/time-correlated inference needs replicated seed-level analysis.","Refresh costs are additional to serving costs.",
                       "Null detection/recovery means not detected/recovered, not zero delay.",
                       "Recovery means five consecutive served, nonviolating requests; inspect coverage and regret too.",
                       "Workload/evaluator faults are injected; production detectors need separate calibrated studies."]}


def run_sparse(*,steps=80,seed=42,fractions=(.05,.1,.25,1.),initial_requests=1200,scenario="quality_minus20pct"):
    rows=[]
    for fraction in dict.fromkeys((0., *fractions)):
        strategies=("frozen",) if fraction==0 else (("impact","uncertainty","random") if fraction<1 else ("exhaustive",))
        report=run_dynamic(scenario=scenario,steps=steps,seed=seed,strategies=strategies,
                           probe_fraction=fraction,initial_requests=initial_requests)
        for strategy,r in report["results"].items():
            serving_cost=sum(record["cost"] for record in r["records"])
            known_cost=serving_cost+r["probe_budget"]["realized_known_cost"]
            reserved_cost=serving_cost+r["probe_budget"]["dollars_reserved"]
            rows.append({"budget_fraction":fraction,"strategy":strategy,"actual_probe_calls":r["probe_budget"]["calls_reserved"],
                "actual_probe_cost":r["probe_budget"]["realized_known_cost"],"reserved_probe_cost":r["probe_budget"]["dollars_reserved"],
                "actual_refresh_fraction":r["refresh_call_fraction"],"capability_map_error":r["mean_capability_error"],"capability_error_observed_fraction":r["capability_error_observed_fraction"],
                "mean_serving_cost":r["metrics"]["mean_cost"],
                "total_serving_cost":serving_cost,"total_known_cost":known_cost,"total_reserved_cost":reserved_cost,
                "mean_total_known_cost_per_request":known_cost/steps,
                "known_probe_cost_coverage":r["probe_budget"]["known_cost_coverage"],
                "routing_regret":r["metrics"]["mean_regret"],"quality_all_requests":r["metrics"]["quality_all_requests"],
                "abstention_rate":r["metrics"]["abstention_rate"],"constraint_violation_rate":r["metrics"]["constraint_violation_rate"]})
    return {"schema":"slm-benchmark-v1","kind":"sparse","evidence_kind":"synthetic","environment":environment(),"initial_requests":initial_requests,"seed":seed,"steps":steps,"scenario":scenario,
        "results":rows,"hypothesis":"retain routing performance at <=25% exhaustive refresh calls",
        "hypothesis_status":"not_established_automatically; inspect replicated regret, coverage and total serving+probe spend",
        "cost_accounting":"Known totals include realized serving and known probe cost; reserved totals include serving and probe reservations, including unknown billed outcomes. Shared initial evidence acquisition is excluded from both totals.",
        "exhaustive_denominator":"one refresh per endpoint per workload request; serving feedback reported separately"}
