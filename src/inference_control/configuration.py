"""Explicit operator configuration. No arbitrary imports, policy callbacks or secret files."""
from pathlib import Path
import json
import os
from typing import Literal
import yaml
from pydantic import BaseModel, ConfigDict, Field
from inference_control.contracts import EndpointSnapshot
from inference_control.policies.compiler import compile_policy,validate_pool
from inference_control.capability.conditional import ConditionalCapabilityMap
from inference_control.planning import Planner
from inference_control.ledger import SQLiteLedger
from inference_control.adapters.providers import OpenAICompatibleAdapter
from inference_control.adapters.runtime import RuntimeProviderAdapter
from inference_control.execution import Executor
from inference_control.api.app import ControlPlane,create_app


class ProviderConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    endpoint_id:str
    kind:Literal["openai_compatible"]="openai_compatible"
    base_url:str | None=None
    api_key_env:str | None=None
    token_limit_field:Literal["max_tokens","max_completion_tokens"]="max_completion_tokens"

class ServiceConfig(BaseModel):
    model_config=ConfigDict(extra="forbid")
    endpoints:str
    policy:str
    capability_map:str | None=None
    ledger:str="slm-router.sqlite3"
    api_key_env:str="SLM_ROUTER_API_KEY"
    providers:list[ProviderConfig]=Field(default_factory=list)


def _environment_secret(name: str, *, minimum_length: int = 1) -> str:
    value = os.environ.get(name, "").strip().strip('"').strip("'")
    if len(value) < minimum_length:
        raise ValueError(f"set {name} to a credential of at least {minimum_length} characters")
    return value


def load_endpoints(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data,list):raise ValueError("endpoints file must be a JSON array")
    return [EndpointSnapshot.model_validate(e) for e in data]


def load_service(path):
    path=Path(path).resolve()
    config=ServiceConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    def file(name):return path.parent/name
    endpoints=load_endpoints(file(config.endpoints))
    policy=compile_policy(file(config.policy))
    validate_pool(policy,endpoints)
    secret=_environment_secret(config.api_key_env,minimum_length=24)
    cmap=ConditionalCapabilityMap.restore(json.loads(file(config.capability_map).read_text(encoding="utf-8"))) if config.capability_map else ConditionalCapabilityMap()
    configured={}
    for provider in config.providers:
        if provider.endpoint_id in configured:raise ValueError("duplicate provider endpoint configuration")
        if provider.endpoint_id not in {e.endpoint_id for e in endpoints}:raise ValueError("provider endpoint absent from snapshots")
        if not provider.base_url:raise ValueError("base_url required for openai_compatible")
        key=_environment_secret(provider.api_key_env) if provider.api_key_env else "local-no-key"
        configured[provider.endpoint_id]=OpenAICompatibleAdapter(provider.base_url,key,token_limit_field=provider.token_limit_field)
    if set(configured)!={e.endpoint_id for e in endpoints}:raise ValueError("every snapshot requires an explicit execution adapter")
    bridge=RuntimeProviderAdapter({e.endpoint_id:e for e in endpoints},configured)
    # Bounded built-ins are sanity checks, NOT correctness oracles. Their behavior must be included in joint evidence.
    verifiers={"nonempty":lambda result,rule:bool(result.message and (result.message.get("content") or result.message.get("tool_calls")))}
    selectors={"first_success":lambda results:results[0]}
    executor=Executor(bridge,verifiers=verifiers,selectors=selectors)
    planner=Planner(endpoints,capability_map=cmap)
    control=ControlPlane(planner,SQLiteLedger(file(config.ledger)),executor,default_policy=policy)
    # Explicit file changes become new snapshots; never mutate or silently ignore historical identities.
    desired={e.endpoint_id:e for e in endpoints}
    for name,old in tuple(control.planner.endpoints.items()):
        if name not in desired and old.available:
            control.drift.refresh_endpoint(old.model_copy(update={"available":False,"healthy":False}))
    for endpoint in endpoints:
        if control.planner.endpoints.get(endpoint.endpoint_id)!=endpoint:
            control.drift.refresh_endpoint(endpoint)
    control.state.store_planner(control.planner)
    # A config file cannot silently replace an already-active policy after restart.
    return create_app(control,api_key=secret),control
