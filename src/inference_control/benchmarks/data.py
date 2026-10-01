"""Full-information benchmark contract and strict LLMRouterBench import.

Truth is owned by the runner. A router receives Query and public endpoint snapshots only.
Missing latency stays missing; aggregate dataset duration is never invented as per-call latency.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal
from datetime import datetime, timezone
import json
from pydantic import BaseModel, ConfigDict, Field, model_validator
from inference_control.contracts import RequestContext, EndpointSnapshot
from inference_control.features.query import query_features
from inference_control.util import digest, canonical


class Query(BaseModel):
    model_config=ConfigDict(frozen=True,extra="forbid")
    request: RequestContext
    text: str
    split: Literal["train","calibration","validation","test"]
    group_id: str

class Truth(BaseModel):
    model_config=ConfigDict(frozen=True,extra="forbid",allow_inf_nan=False)
    request_id: str
    endpoint_id: str
    quality: float=Field(ge=0,le=1)
    cost: float=Field(ge=0)
    latency_ms: float | None=Field(default=None,ge=0)
    input_tokens: int=Field(default=0,ge=0)
    output_tokens: int=Field(default=0,ge=0)
    failed: bool=False
    evaluator_type: str="deterministic"
    evaluator_version: str="dataset-v1"

class BenchmarkDataset(BaseModel):
    model_config=ConfigDict(extra="forbid")
    name: str
    evidence_kind: Literal["synthetic","full_information"]
    queries: list[Query]
    outcomes: list[Truth]
    endpoints: list[EndpointSnapshot]
    provenance: dict=Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_full_information(self):
        ids=[q.request.request_id for q in self.queries]
        if len(ids)!=len(set(ids)):raise ValueError("duplicate query IDs")
        if not ids:raise ValueError("dataset is empty")
        pool={e.endpoint_id for e in self.endpoints}
        if len(pool)!=len(self.endpoints):raise ValueError("duplicate endpoint IDs")
        groups={}
        for q in self.queries:
            if q.group_id in groups and groups[q.group_id]!=q.split:raise ValueError("group crosses splits")
            groups[q.group_id]=q.split
        pairs=[(r.request_id,r.endpoint_id) for r in self.outcomes]
        if len(pairs)!=len(set(pairs)):raise ValueError("duplicate outcome rows")
        expected={(q,e) for q in ids for e in pool}
        if set(pairs)!=expected:raise ValueError("incomplete or mismatched full-information model matrix")
        return self

    @property
    def fingerprint(self):return digest(self)

    @property
    def split_hash(self):return digest([(q.request.request_id,q.split,q.group_id) for q in self.queries])

    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        Path(path).write_text(canonical(self)+"\n",encoding="utf-8")

    @classmethod
    def load(cls,path):return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def grouped_split(group: str, seed: int) -> str:
    # Declared before model fitting. No search for a split where SLM wins.
    bucket=int(digest((seed,group))[:8],16)/2**32
    return "train" if bucket<.5 else "calibration" if bucket<.7 else "validation" if bucket<.85 else "test"


def import_llmrouterbench(root: str | Path, *, endpoints: list[EndpointSnapshot], seed=42, max_output_tokens=1024) -> BenchmarkDataset:
    """Read results/bench/<dataset>/<split>/<model>/<timestamp>.json.

    Exactly one run per (dataset,split,model); ambiguous repeated runs fail instead of
    choosing whichever scores best. Supply a curated directory to resolve ambiguity.
    """
    root=Path(root)
    model_ids={e.endpoint_id for e in endpoints}
    runs,queries,outcomes={}, {}, []
    for path in sorted(root.rglob("*.json")):
        rel=path.relative_to(root).parts
        if len(rel)<4:continue
        dataset,source_split,model=rel[-4],rel[-3],rel[-2]
        if model not in model_ids:continue
        key=(dataset,source_split,model)
        if key in runs:raise ValueError(f"multiple runs for {key}; curate a single run per model before import")
        data=json.loads(path.read_text(encoding="utf-8"))
        if "records" not in data:continue
        runs[key]=str(path)
        for row in data["records"]:
            text=row.get("origin_query") or row.get("prompt")
            if not isinstance(text,str) or "score" not in row or "cost" not in row:
                raise ValueError(f"missing prompt/score/cost in {path}")
            # Global prompt grouping prevents duplicates leaking across dataset/split aliases.
            group=digest(" ".join(text.split()))
            request_id=digest((dataset,source_split,group))[:24]
            split=grouped_split(group,seed)
            context=RequestContext(request_id=request_id,application_id="benchmark",tenant_policy_id="benchmark",
                task_hint=dataset,traffic_slices=frozenset({dataset}),input_tokens=int(row.get("prompt_tokens",0)),
                max_output_tokens=max_output_tokens,feature_version="hash-text-v1",query_features=query_features(text))
            query=Query(request=context,text=text,split=split,group_id=group)
            if request_id in queries:
                # Tokenizers differ. Use a conservative common input bound, not endpoint-specific routing labels.
                prior=queries[request_id]
                n=max(prior.request.input_tokens,context.input_tokens)
                query=prior.model_copy(update={"request":prior.request.model_copy(update={"input_tokens":n})})
            queries[request_id]=query
            score=row["score"]
            if isinstance(score,bool):score=int(score)
            if not isinstance(score,(int,float)) or not 0<=score<=1:
                raise ValueError("score must explicitly be normalized to [0,1]; percentages are not auto-guessed")
            latency=row.get("latency_ms")
            outputs=int(row.get("completion_tokens",0))
            outcomes.append(Truth(request_id=request_id,endpoint_id=model,quality=float(score),cost=float(row["cost"]),
                latency_ms=float(latency) if latency is not None else None,input_tokens=int(row.get("prompt_tokens",0)),
                output_tokens=outputs,evaluator_type=str(row.get("evaluator_type","dataset")),
                evaluator_version=str(row.get("evaluator_version","LLMRouterBench-import"))))
    if not runs:raise ValueError("no compatible LLMRouterBench result files found")
    queries=list(queries.values())
    return BenchmarkDataset(name="LLMRouterBench-import",evidence_kind="full_information",queries=queries,outcomes=outcomes,
        endpoints=endpoints,provenance={"source":"https://github.com/ynulihao/LLMRouterBench","seed":seed,
            "run_files":sorted(runs.values()),"split_method":"global-normalized-prompt-hash-50/20/15/15",
            "output_budget":f"operator-fixed cap: {max_output_tokens}; held-out output length is not a routing feature",
            "missing_latency":"kept as null; no latency or certificate claim where absent"})
