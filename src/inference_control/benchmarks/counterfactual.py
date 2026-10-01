"""Compare two frozen policies on identical held-out full-information observations."""
from collections import Counter
from statistics import mean
from inference_control.benchmarks.runner import evaluate_router
from inference_control.benchmarks.metrics import summarize, paired_difference
from inference_control.util import digest


def compare_policies(dataset, router_factory, current, candidate, *, seed=42, resamples=400):
    """Factory must return independently fitted routers; no outcome enters choose()."""
    left=evaluate_router(router_factory(),dataset,current)
    right=evaluate_router(router_factory(),dataset,candidate)
    changed=[(a,b) for a,b in zip(left,right) if a["endpoint_id"]!=b["endpoint_id"]]
    counts=Counter(b["slice"] for _,b in changed)
    a,b=summarize(left,seed=seed,resamples=resamples),summarize(right,seed=seed,resamples=resamples)
    return {"kind":"policy_diff","evidence_kind":dataset.evidence_kind,"dataset_hash":dataset.fingerprint,
            "split_hash":dataset.split_hash,"current_policy_hash":digest(current),"candidate_policy_hash":digest(candidate),
            "changed_fraction":len(changed)/len(left),"changed_slices":dict(counts.most_common()),
            "new_abstention_fraction":sum(not x["abstained"] and y["abstained"] for x,y in zip(left,right))/len(left),
            "current":a,"candidate":b,"paired_quality_change":paired_difference(right,left,seed=seed,resamples=resamples),
            "mean_cost_change":b["mean_cost"]-a["mean_cost"] if a["mean_cost"] is not None and b["mean_cost"] is not None else None,
            "note":"Replay evidence is not automatic promotion approval; compound policies require observed joint plan results."}
