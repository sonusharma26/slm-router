"""Offline release regressions: no provider accounts, model downloads, or training jobs."""
from dataclasses import replace
from datetime import datetime,timezone,timedelta
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import json
import pytest
import httpx
from fastapi.testclient import TestClient
from typer.testing import CliRunner
from inference_control.contracts import EndpointSnapshot,RequestContext,PolicySpec,Prediction,OutcomeRecord
from inference_control.planning import Planner,CapabilityEstimate
from inference_control.capability.conditional import ConditionalCapabilityMap,EvidenceObservation,stratum
from inference_control.execution import Executor,RuntimeResult
from inference_control.api.app import ControlPlane,DecideRequest,create_app
from inference_control.ledger import SQLiteLedger
from inference_control.ledger.state import ControlState
from inference_control.policies.compiler import compile_policy
from inference_control.probes.active import DailyProbeBudget,ActiveMeasurementLoop,ProbeTask
from inference_control.drift.recovery import DriftRecovery
from inference_control.lifecycle.orchestrator import LifecycleOrchestrator,OfflineEvidence
from inference_control.outcomes.trust import EvaluatorTrustStore
from inference_control.util import digest


def ep(name="a",rate=1.,**updates):
    data=dict(endpoint_id=name,provider="local",upstream_model=name,revision="r1",region="local",context_window=8192,
              config_hash="h",price_version="p1",input_price_per_million=rate,output_price_per_million=rate,
              governance=frozenset({"local"}),capabilities=frozenset({"tools","structured_output"}))
    return EndpointSnapshot(**{**data,**updates})


def request(**updates):
    return RequestContext(**{**dict(request_id="target",application_id="app",tenant_policy_id="p",input_tokens=128,
        max_output_tokens=64,query_features=(.2,),feature_version="fixed-v1",task_hint="code",traffic_slices=frozenset({"code/simple"})),**updates})


def policy(**updates):
    return PolicySpec(**{**dict(policy_id="p",version="1",minimum_quality=.8,max_expected_spend=.01,
        max_absolute_spend=.02,deadline_ms=10000,minimum_evidence_samples=10),**updates})


def estimate(target="a",q=.99,cost=.0001,lat=20):
    return CapabilityEstimate(target,Prediction(mean=q,lower=q,upper=q),Prediction(mean=cost,lower=0,upper=cost),
                              Prediction(mean=lat,lower=0,upper=lat))


def evidence(endpoints=None,n=100,at=None,quality=.99):
    endpoints=endpoints or [ep()];at=at or datetime.now(timezone.utc)-timedelta(seconds=1)
    cmap=ConditionalCapabilityMap(k=256,min_samples=10)
    rows=[]
    for endpoint in endpoints:
        for split in ("train","calibration"):
            for i in range(n):
                rows.append(EvidenceObservation(sample_id=f"{endpoint.endpoint_id}:{split}:{i}",
                    request=request(request_id=f"{split}:{i}"),target_id=endpoint.endpoint_id,endpoint_ids=(endpoint.endpoint_id,),
                    endpoint_revisions=(endpoint.capability_revision,),quality=quality,cost=.0001,latency_ms=20,
                    output_tokens=32,split=split,evaluator_type="deterministic",evaluator_version="unit-v1",observed_at=at))
    cmap.add_many(rows);return cmap


class MockExecution:
    def __init__(self):self.calls=[]
    def call(self,endpoint_id,generation_config):
        self.calls.append((endpoint_id,generation_config))
        return RuntimeResult("output-hash","provider-id","r1",128,32,.00016,None,
                             endpoint_id=endpoint_id,message={"role":"assistant","content":"ok"})


def outcome(decision,**updates):
    return OutcomeRecord(**{**dict(decision_id=decision.decision_id,quality={"success":1.},evaluator_type="human",
        evaluator_version="human-v1",source_artifact="sha256:fixture",label_confidence=1.,uncertainty=0.,
        causal_scope="plan",training_eligible=True,promotion_eligible=True),**updates})


def test_contract_nested_snapshot_immutable_and_hash_roundtrip():
    a=ep(inference_config={"stop":["END"],"temperature":0.})
    with pytest.raises(TypeError):a.inference_config["temperature"]=1
    assert digest(a)==digest(EndpointSnapshot.model_validate_json(a.model_dump_json()))
    assert digest(policy())==digest(PolicySpec.model_validate_json(policy().model_dump_json()))
    with pytest.raises(ValueError):ep(inference_config={"api_key":"do-not-store"})


@pytest.mark.parametrize("update",[{"cost":{"expected_max":3,"absolute_max":1}},
    {"unknown":True},{"plans":{"allowed":["verify_escalate"],"max_calls":2}},
    {"on_infeasible":"least_shortfall_uncertified"}])
def test_policy_compiler_rejects_unsafe_or_unknown_fields(update):
    base={"minimum_quality":.8,"cost":{"expected_max":.01,"absolute_max":.02},"latency":{"p95_max_ms":500}}
    with pytest.raises(ValueError):compile_policy({**base,**update})


def test_compiler_success_defaults_to_required_certificates():
    p=compile_policy({"minimum_quality":.8,"cost":{"expected_max":.01,"absolute_max":.02},"latency":{"p95_max_ms":500}})
    assert p.require_certificate and p.infeasible_behavior=="abstain"


def compound(kind):
    endpoints=[ep(),ep("b",2.)]
    p=policy(permitted_plan_types=frozenset({kind}),max_calls=2,verifier="check",selector="first")
    planner=Planner(endpoints)
    plan=planner.generate(request(),p)[0][0]
    planner.estimates[plan.evidence_key]=estimate(plan.evidence_key)
    return planner,p,planner.decide(request(),p)


@pytest.mark.parametrize("kind",["cascade","verify_escalate","parallel"])
def test_real_compound_search_requires_joint_evidence(kind):
    planner,p,decision=compound(kind)
    assert decision.selected_plan.plan_type==kind
    planner.estimates.clear()
    planner.estimates.update({"a":estimate("a"),"b":estimate("b")})
    absent=planner.decide(request(),p)
    assert absent.selected_plan.plan_type=="abstain"
    assert any("PLAN_EVIDENCE_MISSING" in r.reason_codes for r in absent.rejected_alternatives)


@pytest.mark.parametrize("updates",[{"max_absolute_spend":.00001,"max_expected_spend":.00001},
    {"allowed_providers":frozenset({"not-local"})},{"required_capabilities":frozenset({"vision"})},
    {"deadline_ms":1},{"data_boundary":"forbidden"}])
def test_uncertified_fallback_never_relaxes_hard_limits(updates):
    planner=Planner([ep()],[estimate(q=.3)])
    p=policy(infeasible_behavior="least_shortfall_uncertified",**updates)
    assert planner.decide(request(),p).selected_plan.plan_type=="abstain"


def test_search_truncation_abstains_instead_of_claiming_global_optimum():
    planner=Planner([ep(),ep("b")],[estimate(),estimate("b")])
    decision=planner.decide(request(),policy(max_plan_candidates=1))
    assert not decision.planning_complete and decision.selected_plan.plan_type=="abstain"


def test_conditional_strata_and_feature_vectors_change_estimates():
    cmap=evidence();assert cmap.estimate(request(),"a",(ep(),)).quality.mean>.9
    assert cmap.estimate(request(task_hint="math"),"a",(ep(),)) is None
    assert cmap.estimate(request(required_capabilities=frozenset({"tools"})),"a",(ep(),)) is None
    assert stratum(request(max_output_tokens=1024))!=stratum(request())


def test_split_isolation_duplicate_calibration_and_immutable_sample_id():
    cmap=evidence(n=4)
    row=next(iter(cmap.rows.values()))
    with pytest.raises(ValueError):cmap.add(row.model_copy(update={"sample_id":"changed","split":"calibration"}))
    with pytest.raises(ValueError):cmap.add_many([row.model_copy(update={"sample_id":"new"}),row.model_copy(update={"sample_id":"new","quality":0.})])
    cal=next(r for r in cmap.rows.values() if r.split=="calibration")
    cmap.add(cal.model_copy(update={"sample_id":"duplicate-same-request"}))
    assert cmap.estimate(request(),"a",(ep(),)).lineage.calibration_size==4


def test_certificate_expires_and_stale_metric_blocks_execution():
    endpoint=ep();cmap=evidence();planner=Planner([endpoint],capability_map=cmap)
    p=policy(require_certificate=True)
    decision=planner.decide(request(),p)
    assert decision.certificate_status=="current"
    assert not planner.certificates.validate_decision(decision,p,planner.endpoints)
    assert planner.certificates.validate_decision(decision,p,planner.endpoints,at=datetime.now(timezone.utc)+timedelta(days=2))
    planner.certificates.invalidate_dependencies("a",("latency",))
    adapter=MockExecution()
    with pytest.raises(ValueError):Executor(adapter).execute(decision,policy=p,endpoints=planner.endpoints,certificates=planner.certificates)
    assert not adapter.calls


def test_old_drift_event_does_not_revoke_newly_issued_certificate():
    planner=Planner([ep()],capability_map=evidence());p=policy(require_certificate=True)
    decision=planner.decide(request(),p)
    planner.certificates.invalidate_dependencies("a",("quality",),at=decision.created_at-timedelta(seconds=1))
    assert planner.certificates.current(decision.certificate_id)


def test_judge_trust_needs_expiry_and_expires_from_estimator():
    cmap=evidence();row=next(iter(cmap.rows.values()))
    data=row.model_dump();data.update(evaluator_type="calibrated_judge")
    with pytest.raises(ValueError):EvidenceObservation.model_validate(data)
    now=datetime.now(timezone.utc)
    trusted=evidence()
    trusted.rows={k:r.model_copy(update={"evaluator_type":"calibrated_judge","trusted_until":now+timedelta(seconds=2)}) for k,r in trusted.rows.items()}
    assert trusted.estimate(request(),"a",(ep(),),at=now)
    assert trusted.estimate(request(),"a",(ep(),),at=now+timedelta(seconds=3)) is None
    trust=EvaluatorTrustStore()
    record=trust.calibrate("judge-v1",[(1.,1.)]*100,dataset_hash="fixed",human_version="human-v1",at=now,ttl_seconds=2)
    assert record["trusted"] and trust.trusted("judge-v1",now)
    assert not trust.trusted("judge-v1",now+timedelta(seconds=3))


def test_targeted_latency_invalidation_preserves_quality_evidence():
    cmap=evidence();before=cmap.estimate(request(),"a",(ep(),))
    cmap.invalidate("a",("latency",),at=datetime.now(timezone.utc),slice_id="code/simple")
    after=cmap.estimate(request(),"a",(ep(),))
    assert before.quality==after.quality
    assert not after.lineage.calibrated and "LATENCY_MISSING" in after.lineage.reasons


def test_parallel_is_concurrent_not_sequential():
    planner,p,d=compound("parallel");barrier=Barrier(2)
    class Adapter(MockExecution):
        def call(self,e,c):
            barrier.wait(timeout=2)
            return super().call(e,c)
    adapter=Adapter()
    result=Executor(adapter,selectors={"first":lambda rows:rows[0]}).execute(d,policy=p)
    assert result.state=="completed" and result.attempted_calls==2 and len(adapter.calls)==2


def test_verifier_rejected_answer_is_not_returned_after_failed_escalation():
    planner,p,d=compound("verify_escalate")
    first=d.selected_plan.steps[0].endpoint_id
    class Adapter(MockExecution):
        def call(self,e,c):
            if e!=first:raise TimeoutError("billed timeout")
            return super().call(e,c)
    record=Executor(Adapter(),verifiers={"check":lambda result,rule:False}).execute(d,policy=p)
    assert record.state=="failed" and record.output_reference is None and record.attempted_calls==2
    assert not record.accounting_complete and record.reserved_spend==d.selected_plan.max_spend


def test_unknown_verifier_rejected_before_spend_and_output_cap_enforced():
    _,p,d=compound("verify_escalate");adapter=MockExecution()
    with pytest.raises(ValueError):Executor(adapter).execute(d,policy=p)
    assert not adapter.calls
    direct=Planner([ep()],[estimate()]).decide(request(),policy())
    Executor(adapter).execute(direct,policy=policy())
    assert adapter.calls[0][1]["max_completion_tokens"]==64


def test_provider_bridge_reports_unknown_ttft_and_checks_schema_before_spend():
    from inference_control.adapters.providers import OpenAICompatibleAdapter
    from inference_control.adapters.runtime import RuntimeProviderAdapter
    calls=[]
    def handler(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200,json={"id":"provider","model":"a","choices":[{"message":{"content":"ok"}}],
            "usage":{"prompt_tokens":20,"completion_tokens":2}})
    client=httpx.Client(transport=httpx.MockTransport(handler))
    provider=OpenAICompatibleAdapter("http://test/v1","unused",client)
    bridge=RuntimeProviderAdapter({"a":ep()},{"a":provider})
    with pytest.raises(ValueError):bridge.for_request(request(),[{"role":"user","content":"hello"}],{"tools":[{"function":{"name":"x"}}]})
    assert not calls
    bound=bridge.for_request(request(),[{"role":"user","content":"hello"}])
    result=bound.call("a",{"max_completion_tokens":64})
    assert result.time_to_first_token_ms is None and result.spend==pytest.approx(22/1e6)
    assert calls[0]["model"]=="a" and not calls[0]["stream"]


def test_provider_bridge_flags_returned_model_mismatch():
    from inference_control.adapters.providers import OpenAICompatibleAdapter
    from inference_control.adapters.runtime import RuntimeProviderAdapter
    body={"id":"provider","model":"unexpected-model","choices":[{"message":{"content":"ok"}}],
          "usage":{"prompt_tokens":20,"completion_tokens":2}}
    client=httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=body)))
    provider=OpenAICompatibleAdapter("http://test/v1","x",client)
    bridge=RuntimeProviderAdapter({"a":ep()},{"a":provider})
    result=bridge.for_request(request(),[{"role":"user","content":"hello"}]).call("a",{})
    assert "ENDPOINT_MODEL_MISMATCH" in result.constraint_violations


def test_missing_usage_and_malformed_json_fail_closed_but_preserve_known_cost():
    from inference_control.adapters.providers import OpenAICompatibleAdapter
    from inference_control.adapters.runtime import RuntimeProviderAdapter,ProviderAccountingError
    body={"id":"provider","model":"a","choices":[{"message":{"content":"not json"}}]}
    client=httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(200,json=body)))
    provider=OpenAICompatibleAdapter("http://test/v1","x",client)
    bridge=RuntimeProviderAdapter({"a":ep()},{"a":provider})
    messages=[{"role":"user","content":"hello"}]
    with pytest.raises(ProviderAccountingError):bridge.for_request(request(),messages).call("a",{})
    body["usage"]={"prompt_tokens":20,"completion_tokens":2}
    schema={"type":"json_object"}
    req=request(required_capabilities=frozenset({"structured_output"}),structured_output_schema_hash=digest(schema))
    response=bridge.for_request(req,messages,{"response_format":schema}).call("a",{})
    assert "MALFORMED_STRUCTURED_OUTPUT" in response.constraint_violations and response.spend>0


def test_restart_deduplicates_executions_and_unfinished_claims(tmp_path):
    path=tmp_path/"state.db";adapter=MockExecution();p=policy()
    control=ControlPlane(Planner([ep()],[estimate()]),SQLiteLedger(path),Executor(adapter))
    decision=control.decide(DecideRequest(request=request(),policy=p))
    record=control.execute(decision.decision_id)
    control.ledger.close()
    restored=ControlPlane(Planner([]),SQLiteLedger(path),Executor(adapter))
    assert restored.replay(decision.decision_id)["matched"]
    assert restored.execute(decision.decision_id)==record and len(adapter.calls)==1
    pending=restored.decide(DecideRequest(request=request(request_id="pending"),policy=p))
    restored.state.write("execution_started",{"decision_id":pending.decision_id,"claim":"previous-process"},key=f"execute-once:{pending.decision_id}")
    with pytest.raises(ValueError):restored.execute(pending.decision_id)
    assert len(adapter.calls)==1


def test_outcome_supersession_dispute_and_crash_recovery(tmp_path):
    path=tmp_path/"state.db";p=policy();adapter=MockExecution()
    control=ControlPlane(Planner([ep()],capability_map=evidence()),SQLiteLedger(path),Executor(adapter))
    decision=control.decide(DecideRequest(request=request(),policy=p));control.execute(decision.decision_id)
    first=control.add_outcome(outcome(decision))
    second=control.add_outcome(outcome(decision,evaluator_type="deterministic",evaluator_version="test-v1",supersedes=first.outcome_id,quality={"success":0.}))
    assert control.planner.capability_map.rows[f"outcome:{first.outcome_id}"].disputed
    assert control.planner.capability_map.rows[f"outcome:{second.outcome_id}"].quality==0
    control.dispute_outcome(second.outcome_id,"incorrect test fixture")
    assert not control.planner.capability_map.rows[f"outcome:{second.outcome_id}"].trusted
    control.ledger.close()
    recovered=ControlPlane(Planner([]),SQLiteLedger(path))
    assert recovered.state.outcomes[second.outcome_id].disputed
    assert not recovered.planner.capability_map.rows[f"outcome:{second.outcome_id}"].trusted


def test_weak_feedback_cannot_train_or_promote():
    control=ControlPlane(Planner([ep()],capability_map=evidence()),SQLiteLedger(":memory:"),Executor(MockExecution()))
    decision=control.decide(DecideRequest(request=request(),policy=policy()));control.execute(decision.decision_id)
    qualified=control.add_outcome(outcome(decision,evaluator_type="proxy",promotion_eligible=False))
    assert not qualified.training_eligible and not qualified.promotion_eligible
    assert f"outcome:{qualified.outcome_id}" not in control.planner.capability_map.rows
    with pytest.raises(ValueError):outcome(decision,quality={"success":1.2})


def test_budget_is_durable_atomic_and_conservative(tmp_path):
    path=tmp_path/"budget.db"
    a,b=SQLiteLedger(path),SQLiteLedger(path)
    budgets=[DailyProbeBudget(a,max_dollars=.3,max_calls=3),DailyProbeBudget(b,max_dollars=.3,max_calls=3)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(lambda i:budgets[i%2].reserve(str(i),.1),range(20)))
    assert sum(x is not None for x in results)==3
    assert budgets[0].usage()["calls_reserved"]==3 and a.verify()
    assert budgets[0].usage()["realized_known_cost"]==0
    assert budgets[0].usage()["dollars_reserved"]==pytest.approx(.3)


def test_probe_failure_keeps_reservation_and_never_updates_map():
    planner=Planner([ep()],capability_map=evidence());ledger=SQLiteLedger(":memory:")
    loop=ActiveMeasurementLoop(planner,DailyProbeBudget(ledger,max_dollars=1,max_calls=1),
                               recovery=DriftRecovery(planner,ControlState(ledger)))
    version=planner.map_version
    def failed(task):raise TimeoutError("may have spent money")
    result=loop.run([ProbeTask("probe","a",request())],policy(),failed,strategy="exhaustive")
    assert result["failed"]==["probe"] and result["budget"]["calls_reserved"]==1 and planner.map_version==version


def test_drift_detector_state_survives_restart(tmp_path):
    ledger=SQLiteLedger(tmp_path/"drift.db");state=ControlState(ledger)
    planner=Planner([ep()],capability_map=evidence());detector=DriftRecovery(planner,state,window=4)
    detector.set_reference("a","quality","code/simple",[1.]*4)
    detector.observe("a","quality","code/simple",0.)
    restored=DriftRecovery(planner,ControlState(ledger),window=4)
    assert tuple(restored.recent[("a","quality","code/simple")])==(0.,)
    events=[restored.observe("a","quality","code/simple",0.) for _ in range(3)]
    assert events[-1] and events[-1]["metrics"]==["quality"]


def test_lifecycle_shadow_sticky_canary_and_immediate_rollback():
    state=ControlState(SQLiteLedger(":memory:"));life=LifecycleOrchestrator(state,min_shadow=2,min_canary=5,max_canary_fraction=.25)
    old,new=policy(),policy(version="2")
    planner=Planner([ep()],[estimate()]);life.bootstrap(old);life.propose(new)
    evidence_args=dict(dataset_hash="fixture",split_hash="fixed",policy_hash=digest(new),quality_lower=.99,
                       constraint_violations=0,evaluated_requests=100,trusted_outcomes=100)
    with pytest.raises(ValueError):life.validate_offline(new,OfflineEvidence(**evidence_args,evidence_kind="synthetic"))
    # Unit fixture exercises the state protocol; no external validation claim is made.
    life.validate_offline(new,OfflineEvidence(**evidence_args))
    for i in range(2):
        d,used=life.route(request(request_id=str(i)),old,new,planner)
        assert used.version=="1"
    assert state.lifecycle["p"]["state"]=="canary"
    sticky=request(session_id="sticky")
    assert len({life.route(sticky,old,new,planner)[1].version for _ in range(10)})==1
    candidate=planner.decide(request(),new)
    from inference_control.contracts import ExecutionRecord
    life.observe_execution(candidate,ExecutionRecord(decision_id=candidate.decision_id,state="failed",constraint_violations=("REALIZED_SPEND_EXCEEDED",)))
    assert state.lifecycle["p"]["active"]=="1" and state.lifecycle["p"]["candidate"] is None


def test_chat_auth_and_streaming_rejected_without_execution():
    adapter=MockExecution();control=ControlPlane(Planner([ep()],[estimate()]),SQLiteLedger(":memory:"),Executor(adapter),default_policy=policy())
    client=TestClient(create_app(control,api_key="test-secret"))
    dashboard=client.get("/dashboard/")
    assert dashboard.status_code==200 and "SLM Router Test Bench" in dashboard.text and "test-secret" not in dashboard.text
    assert "JSON schema" in dashboard.text and "Verify replay" in dashboard.text
    assert dashboard.headers["cache-control"] == "no-store"
    assert "connect-src 'self'" in dashboard.headers["content-security-policy"]
    assert client.get("/health").status_code==401
    response=client.post("/v1/chat/completions",headers={"Authorization":"Bearer test-secret"},json={"model":"p","messages":[{"role":"user","content":"hi"}],"stream":True})
    assert response.status_code==400 and not adapter.calls
    assert client.get("/metrics",headers={"Authorization":"Bearer test-secret"}).status_code==200


def test_benchmark_fit_isolation_and_external_competitors_not_faked():
    from inference_control.benchmarks.synthetic import make_dataset
    from inference_control.benchmarks.routers import SLMRouter
    from inference_control.benchmarks.runner import run_static
    data=make_dataset(80,42)
    with pytest.raises(ValueError):SLMRouter(data.endpoints).fit([q for q in data.queries if q.split=="test"],data.outcomes)
    report=run_static(data,resamples=20,sweep=False)
    assert not report["claim_gate"]["passed"]
    assert report["status"]["RouteLLM"]["state"]=="not_run"
    assert "best_single" in report["results"] and report["results"]["slm-router"]["abstention_rate"]>0


def test_importer_preserves_missing_latency_and_never_uses_test_output_length(tmp_path):
    from inference_control.benchmarks.data import import_llmrouterbench
    path=tmp_path/"math"/"test"/"a";path.mkdir(parents=True)
    (path/"run.json").write_text(json.dumps({"records":[{"origin_query":"one plus one","score":1,"cost":.001,"prompt_tokens":5,"completion_tokens":500}]}))
    data=import_llmrouterbench(tmp_path,endpoints=[ep()],max_output_tokens=64)
    assert data.queries[0].request.max_output_tokens==64 and data.outcomes[0].latency_ms is None
    (path/"ambiguous.json").write_text((path/"run.json").read_text())
    with pytest.raises(ValueError):import_llmrouterbench(tmp_path,endpoints=[ep()])


def test_dynamic_probe_catalog_disjoint_from_heldout_requests(monkeypatch):
    from inference_control.benchmarks.dynamic import run_dynamic,DynamicWorld
    original=DynamicWorld.outcome;observed=[]
    def observe(self,req,e,t):
        if int(req.request_id.rsplit("-",1)[-1])>=1_000_000:observed.append(req.request_id)
        return original(self,req,e,t)
    monkeypatch.setattr(DynamicWorld,"outcome",observe)
    result=run_dynamic(steps=8,initial_requests=100,strategies=("exhaustive",),scenario="price_x2")
    served={r["request_id"] for r in result["results"]["exhaustive"]["records"]}
    assert observed and served.isdisjoint(observed)
    assert result["results"]["exhaustive"]["probe_budget"]["calls_reserved"]<=24


def test_competitor_pool_mapping_is_strict_and_real_route_method_used():
    from inference_control.benchmarks.competitors import RouteLLMAdapter,GatewaySelectionAdapter,replay_backend
    from inference_control.benchmarks.data import Query
    class Controller:
        def route(self,text,router,threshold):assert text=="prompt";return "strong"
    adapter=RouteLLMAdapter(Controller(),router="mf",threshold=.5,model_mapping={"strong":"a","weak":"b"})
    q=Query(request=request(),text="prompt",split="test",group_id="g")
    assert adapter.choose(q,(ep(),ep("b")),policy()).endpoint_id=="a"
    with pytest.raises(ValueError):adapter.choose(q,(ep(),),policy())
    with pytest.raises(ValueError):GatewaySelectionAdapter(name="test",base_url="http://test",router_model="auto",model_mapping={"a":"a"})
    response=TestClient(replay_backend(["a"])).post("/v1/chat/completions",json={"model":"a","messages":[]})
    assert response.json()["choices"][0]["message"]["content"]=="selection recorded"


def test_end_to_end_recovery_demo_and_cli(tmp_path):
    from inference_control.demo import run_demo
    from inference_control.cli import app
    result=run_demo(tmp_path/"demo")
    assert result["before"]["plan"]["steps"][0]["endpoint_id"]=="cheap"
    assert result["during"]["plan"]["steps"][0]["endpoint_id"]=="safe"
    assert result["after"]["plan"]["steps"][0]["endpoint_id"]=="cheap"
    assert result["replay_after_restart"] and result["ledger_verified"] and result["pending_recoveries_after_restart"]==0
    assert CliRunner().invoke(app,["--help"]).exit_code==0


def test_p95_tolerance_bound_distinguishes_prediction_from_quantile_confidence():
    from inference_control.capability.conditional import quantile_tolerance_rank
    assert quantile_tolerance_rank(58,.95,.05) is None
    assert quantile_tolerance_rank(59,.95,.05)==59
    assert quantile_tolerance_rank(71,.95,.025) is None
    assert quantile_tolerance_rank(72,.95,.025)==72


def test_certificate_is_bound_to_exact_request_not_just_coarse_slice():
    cmap=evidence()
    cmap.rows={k:r.model_copy(update={"request":r.request.model_copy(update={"query_features":(int(r.request.request_id.split(":")[1])/100,)})}) for k,r in cmap.rows.items()}
    planner=Planner([ep()],capability_map=cmap);p=policy(require_certificate=True)
    at=datetime.now(timezone.utc)
    a=planner.decide(request(query_features=(.2,)),p,at=at)
    b=planner.decide(request(request_id="other",query_features=(.8,)),p,at=at)
    assert a.certificate_id!=b.certificate_id and a.certificate_status==b.certificate_status=="current"
    transplanted=b.model_copy(update={"certificate_id":a.certificate_id})
    assert "CERTIFICATE_REQUEST_MISMATCH" in planner.certificates.validate_decision(transplanted,p,planner.endpoints)
    assert ConditionalCapabilityMap.restore(cmap.export()).version==cmap.version


def test_impact_probes_skip_stable_dominated_alternatives():
    endpoints=[ep(),ep("b",100.)];planner=Planner(endpoints,capability_map=evidence(endpoints))
    ledger=SQLiteLedger(":memory:")
    loop=ActiveMeasurementLoop(planner,DailyProbeBudget(ledger,max_dollars=5,max_calls=100),
                               recovery=DriftRecovery(planner,ControlState(ledger)))
    ranks=loop.rank([ProbeTask("one","a",request()),ProbeTask("two","b",request())],policy())
    assert all(not row.could_change_decision for row in ranks)


def test_uncertified_execution_still_rechecks_snapshot_before_spend():
    endpoint=ep();p=policy();d=Planner([endpoint],[estimate()]).decide(request(),p);adapter=MockExecution()
    with pytest.raises(ValueError):Executor(adapter).execute(d,policy=p,endpoints={"a":ep(revision="r2")})
    assert not adapter.calls


def test_external_schema_references_are_rejected_without_network():
    from inference_control.adapters.runtime import RuntimeProviderAdapter
    schema={"type":"json_schema","json_schema":{"name":"unsafe","schema":{"$ref":"http://metadata.invalid/secret"}}}
    req=request(required_capabilities=frozenset({"structured_output"}),structured_output_schema_hash=digest(schema))
    with pytest.raises(ValueError):RuntimeProviderAdapter({"a":ep()},{}).for_request(req,[{"role":"user","content":"hi"}],{"response_format":schema})
