# Performance results, 7 October 2026

The current router still answers none of the 29 archived held-out requests at the unchanged 0.65 quality target. Latency-shift protection improves the matching synthetic fixture, but paid probing still adds spend without quality or regret benefit. The live NVIDIA run succeeds on GPT-OSS-20B; Muse Glimmer times out on its first request.

## Current-source offline benchmarks

| Benchmark | Result |
| --- | --- |
| Recorded quality, 29 held-out requests | 0 served, 29 abstained at 0.65. Complete quality diagnostics equal the preceding replay. |
| Latency tripled, 40 requests | 2 certified constraint violations, versus 8 at baseline. Detection delay 4 requests; zero pre-change events. |
| Sparse probing versus frozen, 40 requests | Same quality 0.96354 and regret 0.002443. Impact probing makes 12 probes and increases known total cost by 6%. |

The latency fixture uses seed 42, 2,400 initial requests, impact strategy, and a 0.1 probe fraction. All 40 requests are served. Quality is 0.94553, mean regret 0.02046, and mean serving cost $0.00029472. Two adverse observations are needed to trip the reactive operational guard; these results do not establish a statistical coverage guarantee.

The sparse fixture uses `quality_minus20pct`, seed 42, 1,200 initial requests, and a 0.1 probe fraction. Frozen known total cost is $0.01536; impact cost is $0.0162816. Initial evidence acquisition is shared and excluded. Random probing costs $0.0192384 with the same quality and regret.

Observed routing overhead p95 is 2.75 ms on the recorded abstention workload and 290.30 ms on the latency simulation. These different workloads are not a speedup comparison or controlled CPU benchmark. Routing CPU work remains a separate performance concern.

Artifacts: `results/performance-2026-10-07/offline/recorded.json`, `latency.json`, and `sparse.json`.

## Live NVIDIA endpoint microbenchmark

Two endpoint workers run concurrently. Each endpoint gets one first-request warm-up followed by six serial measured requests. The workload is generated integer addition, capped at 512 output tokens per call and a 45-second transport timeout. There are no automatic retries. A failed warm-up stops that endpoint's remaining calls.

| Model | Measured calls | Exact answers | Median latency | Sampled p95 latency | Serial throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| openai/gpt-oss-20b | 6 | 6 / 6 | 2.09 s | 4.49 s | 0.40 successful requests/s |
| meta/muse-glimmer-30b | 0 | Not measured | Not measured | Not measured | Not measured |

GPT-OSS-20B's first request took 17.44 seconds. Measured mean latency is 2.50 seconds; three of six measured responses exceed the static benchmark's 2.5-second deadline. All measured calls return complete token usage and the configured model identity. Total measured usage is 486 input and 230 output tokens. No immutable model revision was observed.

Muse Glimmer's first request fails with `ProviderTimeout` after 46.17 seconds under the configured 45-second transport timeout. No HTTP status is returned. Its six measured requests are skipped, so there is no valid median or p95 for this model. This run fails its complete-measurement criterion; the failure is retained in the report.

The initial restricted-network attempt fails with `ConnectError` on both warm-ups. The network-enabled run establishes live GPT-OSS connectivity and records the Muse timeout. The earlier authorization permits loading only `NVIDIA_API_KEY` from `.env.local`; credentials are not displayed or retained in results. No packages were installed.

This small arithmetic workload measures endpoint behavior, not coding quality, certified router serving, or maximum load capacity. Six responses do not establish a population p95. The first request is not proof of a backend cold start. Actual billed dollars remain unknown; token usage is reported without assuming zero cost.

Live artifact: `results/performance-2026-10-07/nvidia-live-network/report.json`.

Reusable command, using a new output directory for each run:

```powershell
.\.venv\Scripts\python.exe scripts/run_nvidia_performance.py --out results/nvidia-performance-next --requests-per-model 6 --timeout 45
```

The full offline regression run initiated before the performance clarification also completed: 104 passed, 2 failed, 4 skipped. The two failures are the known legacy registry rejection of `nvidia-build`; the newer live adapter path is separate. No additional unit-test runs were made for this benchmark harness.
