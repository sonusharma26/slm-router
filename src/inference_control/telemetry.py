"""Small Prometheus exporter over the durable ledger. No raw prompts or high-cardinality IDs."""
from __future__ import annotations
from collections import Counter


def prometheus(state) -> str:
    state.refresh()
    counts=Counter(d.selected_plan.plan_type for d in state.decisions.values())
    lines=["# TYPE slm_decisions_total counter"]
    lines.extend(f'slm_decisions_total{{plan="{kind}"}} {counts[kind]}' for kind in sorted(counts))
    metrics={"slm_certified_decisions_total":sum(d.certificate_status=="current" for d in state.decisions.values()),
             "slm_provider_errors_total":sum(len(e.provider_errors) for e in state.executions.values()),
             "slm_constraint_violations_total":sum(bool(e.constraint_violations) for e in state.executions.values()),
             "slm_realized_spend_dollars_total":sum(e.realized_spend for e in state.executions.values()),
             "slm_drift_events_total":len(state.drifts),
             "slm_pending_execution_claims":len(set(state.claims)-set(state.execution_by_decision)),
             "slm_decision_latency_ms_sum":sum(d.decision_latency_ms for d in state.decisions.values()),
             "slm_decision_latency_ms_count":len(state.decisions)}
    for name,value in metrics.items():lines.append(f"{name} {value}")
    return "\n".join(lines)+"\n"
