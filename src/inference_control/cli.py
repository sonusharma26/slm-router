"""Small, no-network-by-default control-plane and Benchmark Lab CLI."""
from pathlib import Path
from typing import Optional
import json
import os
import typer
from inference_control.util import primitive

app=typer.Typer(help="SLM Router: bounded inference planning and evidence-driven evaluation.",no_args_is_help=True)
policy_app=typer.Typer(no_args_is_help=True)
benchmark_app=typer.Typer(no_args_is_help=True)
app.add_typer(policy_app,name="policy")
app.add_typer(benchmark_app,name="benchmark")


def output(value):
    typer.echo(json.dumps(primitive(value),indent=2,allow_nan=False))


@policy_app.command("validate")
def policy_validate(path:Path, endpoints:Optional[Path]=None):
    """Compile strict YAML/JSON; optionally check structural pool eligibility."""
    from inference_control.policies.compiler import compile_policy,validate_pool
    from inference_control.configuration import load_endpoints
    try:
        policy=compile_policy(path)
        result=validate_pool(policy,load_endpoints(endpoints)) if endpoints else {"valid":True}
        output({**result,"compiled":policy.model_dump(mode="json")})
    except (ValueError,OSError) as exc:
        typer.echo(str(exc),err=True);raise typer.Exit(2)


def external_routers(path,allow_external):
    """Explicit operator manifests only; no automatic downloads or best-score model selection."""
    if path is None:return []
    if not allow_external:raise ValueError("--allow-external is required; classifiers may download models or bill APIs")
    from inference_control.benchmarks.competitors import RouteLLMAdapter,VLLMSemanticRouterAdapter
    entries=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries,list):raise ValueError("competitor manifest must be an array")
    adapters=[]
    for raw in entries:
        item=dict(raw);kind=item.pop("kind")
        if kind=="routellm":
            from routellm.controller import Controller
            router=item.pop("router");strong=item.pop("strong_model");weak=item.pop("weak_model")
            controller=Controller(routers=[router],strong_model=strong,weak_model=weak)
            adapters.append(RouteLLMAdapter(controller,router=router,**item))
        else:
            cls={"vllm":VLLMSemanticRouterAdapter}.get(kind)
            if cls is None:raise ValueError("unknown competitor kind")
            env=item.pop("api_key_env",None)
            adapters.append(cls(allow_network=True,api_key=os.environ.get(env,"") if env else "",**item))
    return adapters


@benchmark_app.command("static")
def benchmark_static(dataset:Optional[Path]=None, synthetic:bool=False, requests:int=600, seed:int=42,
                     out:Path=Path("results/benchmarks/static"), minimum_quality:float=.65,
                     resamples:int=400, sweep:bool=True, competitors:Optional[Path]=None,allow_external:bool=False):
    """Run held-out direct-plan evaluation. Synthetic output is not a competitor claim."""
    from inference_control.benchmarks.data import BenchmarkDataset
    from inference_control.benchmarks.synthetic import make_dataset
    from inference_control.benchmarks.runner import run_static,write_report,default_policy
    if (dataset is None)==(not synthetic):raise typer.BadParameter("choose exactly one of --dataset and --synthetic")
    data=make_dataset(requests,seed) if synthetic else BenchmarkDataset.load(dataset)
    policy=default_policy(minimum_quality=minimum_quality,missing_latency=any(r.latency_ms is None for r in data.outcomes))
    report=run_static(data,seed=seed,policy=policy,resamples=resamples,sweep=sweep,competitors=external_routers(competitors,allow_external))
    write_report(report,out)
    output({"report":str(out/"report.json"),"claim_gate":report["claim_gate"],"results":report["results"]})


@benchmark_app.command("dynamic")
def benchmark_dynamic(scenario:str="quality_minus20pct",steps:int=80,seed:int=42,probe_fraction:float=.1,
                      initial_requests:int=1200,out:Path=Path("results/benchmarks/dynamic")):
    """Inject one named fault or all eight into the deterministic synthetic pool."""
    from inference_control.benchmarks.dynamic import run_dynamic,SCENARIOS
    from inference_control.benchmarks.runner import write_report
    names=SCENARIOS if scenario=="all" else (scenario,)
    for name in names:
        report=run_dynamic(scenario=name,steps=steps,seed=seed,probe_fraction=probe_fraction,initial_requests=initial_requests,
                           strategies=("cheapest","frozen","impact","random","exhaustive"))
        target=out/name if scenario=="all" else out
        write_report(report,target)
        output({"scenario":name,"report":str(target/"report.json"),"evidence_kind":"synthetic"})


@benchmark_app.command("sparse")
def benchmark_sparse(steps:int=80,seed:int=42,scenario:str="quality_minus20pct",
                     out:Path=Path("results/benchmarks/sparse")):
    """Measure 5%,10%,25% acquisition budgets against exhaustive refresh."""
    from inference_control.benchmarks.dynamic import run_sparse
    from inference_control.benchmarks.runner import write_report
    report=run_sparse(steps=steps,seed=seed,scenario=scenario)
    write_report(report,out)
    output({"report":str(out/"report.json"),"evidence_kind":"synthetic"})


@benchmark_app.command("import-llmrouterbench")
def import_bench(root:Path,endpoints:Path=typer.Option(...),out:Path=typer.Option(...),seed:int=42,max_output_tokens:int=1024):
    """Import a curated full matrix; never download datasets or infer absent latency."""
    from inference_control.benchmarks.data import import_llmrouterbench
    from inference_control.configuration import load_endpoints
    data=import_llmrouterbench(root,endpoints=load_endpoints(endpoints),seed=seed,max_output_tokens=max_output_tokens)
    data.save(out)
    output({"dataset":str(out),"fingerprint":data.fingerprint,"queries":len(data.queries),"models":len(data.endpoints)})


@benchmark_app.command("policy-diff")
def policy_diff(dataset:Path=typer.Option(...),current:Path=typer.Option(...),candidate:Path=typer.Option(...),out:Path=Path("results/benchmarks/policy-diff"),seed:int=42):
    from inference_control.benchmarks.data import BenchmarkDataset
    from inference_control.benchmarks.routers import SLMRouter
    from inference_control.benchmarks.runner import split_data,write_report
    from inference_control.benchmarks.counterfactual import compare_policies
    from inference_control.policies.compiler import compile_policy
    data=BenchmarkDataset.load(dataset)
    fit,truth=split_data(data,{"train","calibration"})
    def factory():
        router=SLMRouter(data.endpoints,min_samples=8);router.fit(fit,truth);return router
    report=compare_policies(data,factory,compile_policy(current),compile_policy(candidate),seed=seed)
    write_report(report,out);output({"report":str(out/"report.json"),"changed_fraction":report["changed_fraction"]})


@benchmark_app.command("replay-backend")
def replay_backend(endpoints:Path=typer.Option(...),host:str="127.0.0.1",port:int=8091):
    """Serve constant responses to a real competitor gateway; never serves truth labels."""
    import uvicorn
    from inference_control.configuration import load_endpoints
    from inference_control.benchmarks.competitors import replay_backend as factory
    uvicorn.run(factory([e.endpoint_id for e in load_endpoints(endpoints)]),host=host,port=port,workers=1)


@app.command("demo")
def demo(out:Path=Path("results/demo")):
    """Execute a synthetic certificate/drift/probe/replay demonstration without network access."""
    from inference_control.demo import run_demo
    report=run_demo(out)
    output({"report":str(out/"demo.json"),"evidence_kind":"synthetic","replay_after_restart":report["replay_after_restart"],
            "ledger_verified":report["ledger_verified"],"probe_budget":report["measurement"]["budget"]})


@app.command("replay")
def replay(ledger:Path=typer.Option(...),decision_id:str=typer.Option(...)):
    from inference_control.api.app import ControlPlane
    from inference_control.planning import Planner
    from inference_control.ledger import SQLiteLedger
    if not ledger.is_file():raise typer.BadParameter("ledger does not exist")
    control=ControlPlane(Planner([]),SQLiteLedger(ledger))
    try:output(control.replay(decision_id))
    finally:control.ledger.close()


@app.command("serve")
def serve(config:Path=typer.Option(...),host:str="127.0.0.1",port:int=8000):
    """Start one local control-plane process. Requires an explicit API credential."""
    import uvicorn
    from inference_control.configuration import load_service
    api,control=load_service(config)
    try:uvicorn.run(api,host=host,port=port,workers=1)
    finally:
        control.ledger.close()
        for adapter in getattr(control.executor.adapter,"providers",{}).values():
            if hasattr(adapter,"close"):adapter.close()


if __name__=="__main__":app()
