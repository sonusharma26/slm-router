"""Real competitor seams. Unconfigured/unrun competitors are NOT synthetic baselines."""
from __future__ import annotations
from time import perf_counter
import httpx
from inference_control.benchmarks.routers import Selection


class RouteLLMAdapter:
    name="RouteLLM"
    def __init__(self, controller, *, router: str, threshold: float, model_mapping: dict[str,str],
                 overhead_cost: float | None=None):
        if not 0<=threshold<=1:raise ValueError("invalid RouteLLM threshold")
        self.controller,self.router,self.threshold=controller,router,threshold
        self.mapping,self.cost=model_mapping,overhead_cost
        self.pool=set(model_mapping.values())

    def choose(self,query,endpoints,policy):
        if {e.endpoint_id for e in endpoints}!=self.pool:
            raise ValueError("RouteLLM comparison requires the same two-model pool; restrict the whole dataset first")
        start=perf_counter()
        model=self.controller.route(query.text,router=self.router,threshold=self.threshold)
        if model not in self.mapping:raise ValueError("unmapped RouteLLM model")
        return Selection(self.mapping[model],overhead_cost=self.cost,overhead_ms=(perf_counter()-start)*1000,
                         metadata={"scope":"route_only","router":self.router,"threshold":self.threshold})


class GatewaySelectionAdapter:
    """OpenAI-compatible gateways backed by the supplied non-oracle replay stub.

    Gateway must return the selected model, not the router alias. Gateway + stub time is
    reported as such; never label it classifier-only overhead. Network use is opt-in.
    """
    def __init__(self, *, name, base_url, router_model, model_mapping, api_key="",
                 client=None, allow_network=False, replay_backend_confirmed=False,
                 classifier_cost: float | None=None):
        if not allow_network:raise ValueError("competitor network execution requires explicit opt-in")
        if not replay_backend_confirmed:raise ValueError("configure the identical model pool against the replay stub first")
        self.name,self.base,self.model=name,base_url.rstrip("/"),router_model
        self.mapping,self.key,self.cost=model_mapping,api_key,classifier_cost
        self.client=client or httpx.Client(timeout=60,follow_redirects=False)

    def choose(self,query,endpoints,policy):
        if {e.endpoint_id for e in endpoints}!=set(self.mapping.values()):raise ValueError("competitor model pool mismatch")
        start=perf_counter()
        response=self.client.post(f"{self.base}/chat/completions",headers={"Authorization":f"Bearer {self.key}"},
            json={"model":self.model,"messages":[{"role":"user","content":query.text}],
                  "max_tokens":query.request.max_output_tokens,"stream":False})
        response.raise_for_status()
        body=response.json()
        model=body.get("model")
        if model not in self.mapping:raise ValueError("gateway did not expose a mapped selected model; do not assume router alias")
        cost=body.get("metadata",{}).get("classifier_cost",self.cost)
        return Selection(self.mapping[model],overhead_cost=cost,overhead_ms=(perf_counter()-start)*1000,
                         metadata={"scope":"gateway_plus_replay_stub","competitor":self.name})


class VLLMSemanticRouterAdapter(GatewaySelectionAdapter):
    def __init__(self,**kwargs):super().__init__(name="vLLM Semantic Router",**kwargs)


def replay_backend(models: list[str]):
    """A constant-response backend: competitors cannot read benchmark labels through it."""
    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel
    app=FastAPI(title="Non-oracle benchmark selection stub")
    @app.get("/v1/models")
    def list_models():return {"object":"list","data":[{"id":m,"object":"model"} for m in models]}
    @app.post("/v1/chat/completions")
    def completion(body:dict):
        model=body.get("model")
        if model not in models:raise HTTPException(404,"model outside registered benchmark pool")
        if body.get("stream"):raise HTTPException(400,"streaming not supported by benchmark stub")
        return {"id":"selection-stub","object":"chat.completion","created":0,"model":model,
                "choices":[{"index":0,"message":{"role":"assistant","content":"selection recorded"},"finish_reason":"stop"}],
                "usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}
    return app
