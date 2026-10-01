# Validation record — 2.0.0a2

This record describes the final local public-release preparation cycle. It is not a production acceptance certificate.

## Local checks

| Check | Result |
|---|---|
| Full repository pytest suite | **85 passed, 0 failed**, 16.89 seconds |
| Active v2 pytest suite | **65 passed, 0 failed**, 10.50 seconds |
| Legacy tests | Included in the full suite, not replaced with new-only checks |
| Bounded policy compiler + example endpoint pool | Passed; statistical-feasibility warning correctly retained |
| CLI command discovery / service and importer help | Passed from the source checkout |
| No-network revision/recovery demo | Passed; cheap → safe → cheap routing |
| Demo SQLite hash chain | Verified |
| Replay after closing and reopening SQLite | Matched before/after decisions |
| Probe budget in demo | 100 attempted/completed calls; no failed probes; synthetic billing only |
| Static experiment | 1,200 generated requests, 207 test requests, seed 42, 400 resamples, six-point sweep |
| Dynamic experiments | All eight scenarios, five strategies each, 40 workload steps/scenario |
| Sparse-refresh experiment | 5%, 10%, 25%, 100% caps; 10 strategy/budget combinations; 40 steps |

Final commands were executed from the project root:

```bash
PYTHONPATH=src python -m pytest -q --disable-warnings --maxfail=3
PYTHONPATH=src python -m inference_control.cli policy validate examples/v2/policy.yaml --endpoints examples/v2/endpoints.json
PYTHONPATH=src python -m inference_control.cli demo --out results/final-validation-2026-10-01/demo
PYTHONPATH=src python -m inference_control.cli benchmark static --synthetic --requests 1200 --seed 42 --resamples 400 --sweep --out results/final-validation-2026-10-01/static
PYTHONPATH=src python -m inference_control.cli benchmark dynamic --scenario all --steps 40 --initial-requests 2400 --seed 42 --out results/final-validation-2026-10-01/dynamic
PYTHONPATH=src python -m inference_control.cli benchmark sparse --steps 40 --seed 42 --scenario quality_minus20pct --out results/final-validation-2026-10-01/sparse
```

Use a **new** output directory when rerunning the demo. Generated ledgers and reports remain local artifacts and are ignored by source control.

The benchmark CLI writes machine-readable reports to the requested output path. Generated reports contain synthetic records, hashes and environment metadata; review them before sharing.

## What the results do and do not show

At one static synthetic operating point, SLM Router had fixture mean quality **0.963344** and mean simulated cost **$0.00040348**. The best-single baseline had quality **0.965087** and cost **$0.00057600**. Spend fell **29.95%**, with slightly lower quality. That is not a matched-quality external benchmark win.

**Important failing scenario:** hidden latency ×3 was not detected within the 40-step run. The impact strategy incurred **8 latency violations out of 40 decisions (20%)**, including certified decisions. The underlying certificate assumptions did not remain valid through the hidden shift. This is an explicit blocker for any claim of safe adaptation through arbitrary drift; a passing unit-test suite does not erase it.

Multiple adaptive runs tied the frozen router while adding probe cost. Sparse refresh also tied across budgets on the small fixture, which does not demonstrate that refresh was useful or establish the ≤25% retention hypothesis. No such claim is made. Short, single-seed dynamic experiments are diagnostics rather than a calibrated estimate of long-horizon recovery performance or false-positive rate.

## Coverage of the automated checks

The suite checks policy rejection, bounded compound plan search, absence of invented joint-plan quality, exact query/snapshot/certificate bindings, nested immutability and stable serialization, train/calibration separation, latency tolerance sample requirements, judge trust expiry, metric-scoped invalidation, superseded/disputed outcomes, concurrent parallel dispatch, fallback rejection, provider usage and schema handling, pre-spend guards, durable execution claims, replay, atomic probe budgets, detector state recovery, rollout guards and benchmark isolation.

Old audit assertions were updated to include the newly required configuration history and pre-execution claim. Legacy research code and tests remain in the repository.

## Not run or not established

Real RouteLLM and vLLM Semantic Router execution **did not run**. Their actual adapter boundaries are implemented and protocol-tested, but mock/controller-boundary tests are not competitor measurements. No public LLMRouterBench dataset was downloaded, no model was trained/downloaded, and no live or paid provider request was made.

The final tests used `PYTHONPATH=src`. The source distribution and wheel also built successfully, the wheel installed in an isolated container, and its CLI help command passed. Python 3.11–3.13 are declared; this final cycle ran on Python 3.12, and the preceding archive validation ran on Python 3.13. The authored GitHub Actions matrix was deliberately not run during this preparation cycle. Streaming, distributed/multi-worker coherence, public multi-tenant service hardening and complete telemetry are outside the tested preview.

A repository-wide Ruff run with the newest version admitted by the current range reported existing style and modernization findings. Focused syntax/error linting passed for the compatibility fix, but repository-wide lint is not presented as a passing release gate.

## Environment and provenance

Final local Python: `3.12.14 (main, Sep 19 2026, 01:04:57) [GCC 14.2.0]` inside the isolated validation container.

Core source fingerprint:

```text
90fca2416caedc0bc7555a5f109a5a06c4356a973963d2f95d45bd23ec4d446b
```

All ten final benchmark reports shared this active `inference_control` source fingerprint. Dataset/split hashes are recorded in static and dynamic reports. Fixture-derived labels and prices are deliberately marked synthetic. Generated results and archive manifests are intentionally excluded from source control by default.

Passing tests establish the checked software behaviors, not empirical routing superiority. The release remains an adaptive evidence preview.
