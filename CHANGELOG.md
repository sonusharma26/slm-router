# Changelog

## 2.0.0a2 — adaptive evidence preview

Implemented bounded direct/compound plan search; conditional kNN evidence with separate calibration; exact-request certificates and confidence bounds on local p95; targeted invalidation; durable SQLite read models, detector state and execution admission; trusted delayed outcomes; shadow/canary lifecycle; actual provider/runtime bridging; strict policies; a thin non-streaming Chat Completions API; and split-safe static/dynamic/sparse benchmark tooling.

Added focused adversarial and regression tests, synthetic reproduction reports, operator examples, primary interface references and explicit statistical/deployment limits.

### Compatibility

- The complete offline suite passes on Python 3.12 with current declared dependencies: 85 tests passed. The legacy temperature-scaling objective now extracts SciPy's one-element optimization array explicitly for compatibility with recent NumPy releases.
- `slm` now invokes the adaptive `inference_control.cli`. The old command is `slm-legacy` and its optional heavy dependencies are in `.[legacy]`.
- Core dependencies no longer require the model-training stack. The declared interpreter range is Python 3.11–3.13; local release validation has run on Python 3.12 and 3.13.
- The previous `uv.lock` was removed because it described a different project graph. Generate a new deployment lock; do not follow historical `uv sync --frozen` instructions.
- Frozen default values are validated before hashing, preventing `0` versus `0.0` identity mismatches after serialization. Configuration dictionaries are recursively frozen.
- Certificates now bind the exact request. Existing loosely bound certificates from the earlier skeleton are not valid execution evidence; re-measure/reissue them. Keep a database backup and do not claim an untested production migration from an old ledger.
- The legacy full-information replay helper no longer exposes realized held-out quality/cost to the policy. Supply independent predicted arms or use the new Benchmark Lab. Its former oracle-input behavior was not a valid learned-router benchmark.
- Audit streams now include configuration/map events and durable execution claims. A legacy test expecting only `decision, execution` was updated to verify the required ordering and hash chain instead.
- No full gateway parity, streaming support, live-provider validation or competitor win is implied by the preview version.
