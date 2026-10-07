"""No-network mechanics demo. Scores and provider outputs are explicitly synthetic."""
from datetime import datetime,timezone,timedelta
from pathlib import Path
from inference_control.contracts import EndpointSnapshot,RequestContext,PolicySpec,OutcomeRecord
from inference_control.capability.conditional import ConditionalCapabilityMap,EvidenceObservation
from inference_control.planning import Planner
from inference_control.execution import Executor,RuntimeResult
from inference_control.api.app import ControlPlane,DecideRequest
from inference_control.ledger import SQLiteLedger
from inference_control.probes.active import ActiveMeasurementLoop,DailyProbeBudget,ProbeTask
from inference_control.util import canonical


def run_demo(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    db=out/"demo.sqlite3"
    if db.exists():raise ValueError("use an empty demo directory; existing audit state is never overwritten")
    at=datetime.now(timezone.utc)
    endpoints=[EndpointSnapshot(endpoint_id=name,provider="local",upstream_model=name,revision="fixture-r1",
        region="local",context_window=8192,config_hash="fixture-config",price_version="fixture-price-1",
        input_price_per_million=rate,output_price_per_million=rate,governance=frozenset({"local"}))
        for name,rate in (("cheap",.1),("safe",1.))]
    policy=PolicySpec(policy_id="demo",version="1",minimum_quality=.8,max_expected_spend=.01,
        max_absolute_spend=.02,deadline_ms=1000,require_certificate=True,minimum_evidence_samples=20)
    request=RequestContext(application_id="synthetic-demo",tenant_policy_id="demo",input_tokens=128,
        max_output_tokens=64,feature_version="fixture-v1",task_hint="arithmetic",traffic_slices=frozenset({"simple"}),query_features=(.2,))
    cmap=ConditionalCapabilityMap(k=256,min_samples=20)
    def observation(ep,index,split,when,prefix):
        q=request.model_copy(update={"request_id":f"{prefix}-{split}-{index}"})
        return EvidenceObservation(sample_id=f"{prefix}:{ep.endpoint_id}:{split}:{index}",request=q,target_id=ep.endpoint_id,
            endpoint_ids=(ep.endpoint_id,),endpoint_revisions=(ep.capability_revision,),quality=.99,cost=160*ep.input_price_per_million/1e6,
            latency_ms=80 if ep.endpoint_id=="cheap" else 150,output_tokens=32,split=split,evaluator_type="deterministic",
            evaluator_version="synthetic-fixture-v1",observed_at=when)
    cmap.add_many([observation(ep,i,split,at-timedelta(seconds=2),"initial") for ep in endpoints for split in ("train","calibration") for i in range(100)])
    class MockExecution:
        def call(self,endpoint_id,generation_config):
            ep=next(e for e in endpoints if e.endpoint_id==endpoint_id)
            return RuntimeResult("fixture-output","fixture-provider-id",ep.revision,128,32,160*ep.input_price_per_million/1e6,
                None,endpoint_id=endpoint_id,message={"role":"assistant","content":"synthetic success"})
    control=ControlPlane(Planner(endpoints,capability_map=cmap),SQLiteLedger(db),Executor(MockExecution()),default_policy=policy)
    before=control.decide(DecideRequest(request=request,policy=policy))
    execution=control.execute(before.decision_id)
    replacement=endpoints[0].model_copy(update={"revision":"fixture-r2"})
    event=control.drift.refresh_endpoint(replacement)
    during=control.decide(DecideRequest(request=request.model_copy(update={"request_id":"during"}),policy=policy))
    budget=DailyProbeBudget(control.ledger,max_dollars=5,max_calls=100)
    loop=ActiveMeasurementLoop(control.planner,budget,recovery=control.drift,state=control.state)
    tasks=[]; rows={}
    for split,n in (("train",25),("calibration",75)):
        for i in range(n):
            row=observation(replacement,i,split,datetime.now(timezone.utc),"recovery")
            task=ProbeTask(row.sample_id,replacement.endpoint_id,row.request)
            tasks.append(task);rows[task.probe_id]=row
    measured=loop.run(tasks,policy,lambda task:rows[task.probe_id],max_calls=100)
    after=control.decide(DecideRequest(request=request.model_copy(update={"request_id":"after"}),policy=policy))
    recovered=control.drift.acknowledge_recovered(event["event_id"],request,policy)
    control.state.store_planner(control.planner)
    replay_before=control.replay(before.decision_id)
    replay_after=control.replay(after.decision_id)
    control.ledger.close()
    restored=ControlPlane(Planner([],capability_map=ConditionalCapabilityMap()),SQLiteLedger(db))
    report={"evidence_kind":"synthetic","before":{"plan":before.selected_plan.model_dump(mode="json"),"certificate":before.certificate_status},
        "during":{"plan":during.selected_plan.model_dump(mode="json"),"certificate":during.certificate_status},
        "after":{"plan":after.selected_plan.model_dump(mode="json"),"certificate":after.certificate_status},
        "execution":execution.model_dump(mode="json"),"drift":event,"measurement":measured,
        "ledger_verified":restored.ledger.verify(),"replay_before":replay_before["matched"],"replay_after":replay_after["matched"],
        "replay_after_restart":restored.replay(after.decision_id)["matched"],"pending_recoveries_after_restart":len(restored.drift.pending),
        "limitation":"Synthetic mechanics demo. These fixture certificates say nothing about real model quality."}
    restored.ledger.close()
    (out/"demo.json").write_text(canonical(report)+"\n",encoding="utf-8")
    return report
