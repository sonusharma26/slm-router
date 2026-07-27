# Adaptive Inference Control Plane

> **Status:** breaking v2 rewrite in progress. The current verified scope covers the dataset-independent V2.2–V2.7 control-plane implementation and offline tests; empirical dataset and live-provider gates remain open. It is not production-ready or self-improving.

This project is moving from a generic SLM–LLM router to an auditable control plane for changing model pools. Requests carry explicit quality, cost, latency, privacy, provider, capability, and call constraints. The planner rejects infeasible endpoints before making a deterministic lexicographic selection; when nothing qualifies, it abstains rather than silently choosing a cheap endpoint.

See [`NEXT_PHASE_V2_ROADMAP.md`](NEXT_PHASE_V2_ROADMAP.md) for the sequenced research plan and evidence gates. Legacy v1 code remains under `slm_router` for migration only and is not the v2 API.

## Offline quickstart

```bash
uv sync --frozen
uv run inference-control-smoke
uv run pytest -q
```

The smoke command uses a local deterministic fixture and performs no network calls. Current evidence is limited to the automated tests; no live-provider, cost-saving, generalization, calibration, drift-response, or benchmark superiority claim has been validated.

## V2 packages

- `inference_control.contracts`: frozen, schema-versioned endpoint, request, policy, plan, decision, execution, and outcome records.
- `inference_control.ledger`: append-only, idempotent, hash-chained SQLite events for local use.
- `inference_control.simulation`: deterministic endpoint behavior with injectable drift.
- `inference_control.policies` and `planning`: eligibility constraints and direct/abstain planning.
- `inference_control.execution`: bounded direct, cascade, verify-escalate, parallel, and abstain runtime.
- `inference_control.capability` and `probes`: target-aware distributions, lineage, catalogs, acquisition, and hard budgets.
- `inference_control.outcomes`, `learning`, and `lifecycle`: provenance eligibility, OPE diagnostics, artifacts, shadow/canary, promotion, and rollback.
- `inference_control.drift` and `evaluation`: minimum-evidence drift controls, fault scenarios, replay, and dataset-independent gauntlet definitions.

## Legacy v1

The preserved v1 research snapshot is tagged `v1-research-snapshot`. Its oracle/FQE workflow and `/route` endpoint are legacy behavior and are not evidence for v2.
