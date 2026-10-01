# Runtime and safety boundaries

## Deployment boundary

Run **one control-plane process and one API worker**. SQLite reservations and execution claims use serialized transactions, but live in-memory planner state is not a distributed cache-coherence protocol. Do not deploy multiple Uvicorn workers or independent service instances against the same active control plane. SQLite is the intended simple default.

The service uses one operator-owned bearer credential. `/v2/decide`, outcomes, audit reads and inference share that authority. It is not a multi-tenant public API with separate customer budgets, roles, rate limits or policy ACLs. Keep it private or behind an appropriate authenticated reverse proxy. Do not expose the constant-response benchmark backend to an untrusted network.

Provider URLs are operator configuration, not prompt inputs. An OpenAI-compatible server may be local HTTP; use HTTPS for remote traffic. There is no automatic redirect following and no application-request override of provider/model credentials. Use environment variables, not configuration snapshots, for secrets.

## Limits and accounting

A price reservation assumes the configured token prices and provider behavior are accurate. A timeout or provider error may still incur charges. A provider can also violate its declared output cap or change an alias without exposing a revision. The runtime records violations and unknown accounting; it cannot enforce the provider's billing system or prove an unobservable revision.

The chat endpoint uses serialized UTF-8 byte length as a conservative input admission heuristic plus an operator-verified per-endpoint token overhead. **This is not a universal tokenizer theorem.** Providers with special framing, multimodal tokens, hidden transformations or incompatible counting need a verified counting adapter/overhead configuration. Realized token-bound violations fail the execution and can stop rollout.

Local CPU inference, gateway/classifier cost, taxes, minimum charges, caching discounts and infrastructure spend are not automatically inferred from a per-million token price. Configure and report the relevant accounting model. Native provider retries are disabled at this layer; an externally configured gateway/provider may have retries or fallbacks of its own. Disable them or include their worst-case cost and identity in the endpoint contract.

Schema/tool requirements participate in routing eligibility. Per-request schema hashes must match the decision used for execution. JSON output is parsed; JSON-schema mode uses the declared schema; undeclared/malformed tool calls are rejected. External schema references are forbidden to avoid validation-time network access. Tool calls are returned to the application, not executed by this router. Capability declarations are operator assertions that require verification against the real deployment.

## Unsupported in this preview

Streaming is rejected **before** inference, even for direct plans. The entrypoint is a thin text-only Chat Completions subset, not every OpenAI option and not the Responses API. There is no live TTFT measurement for non-streaming responses; it stays null rather than being fabricated from total latency. Arbitrary-length cascades, provider SDK proliferation, dashboards and distributed storage are intentionally absent.

A local verifier or selector is ordinary application code; changing it without versioning its identity breaks the interpretation of old joint evidence. Built-in `nonempty` and `first_success` are mechanical examples, not semantic quality judges. Application operators must keep them bounded and validate their usefulness with joint plan observations.

## Data retention

The execution ledger does not store raw input messages or generated text. It stores request metadata/features, evidence, source references, outcome scores, hashes, IDs, costs and timing. These can still be sensitive or linkable data. `no_raw_input` is not a legal compliance assertion or a deletion implementation.

SQLite audit data is append-only and can grow. There is no production retention/deletion policy engine or encryption-at-rest/key-management service in this release. Protect the database, backups and observation exports appropriately. The demo database contains synthetic data only.

After an ambiguous external side effect, the API refuses automatic duplicate execution. Raw response bodies are ephemeral; a replayed execution record does not reconstruct the original text. Operator reconciliation is required for unresolved claims. This deliberately favors avoiding duplicate side effects over silent automatic retry.

## Statistical boundaries

Certificate guarantees are conditional on independent/exchangeable calibration, applicable request neighborhoods, correct labels, stable endpoint/evaluator behavior and the declared evidence-review protocol. They are not adversarial robustness guarantees or per-answer proofs. Statistical drift detection uses a variance-sensitive normal approximation and nominal alpha spending; false-positive and detection-delay behavior must be measured on your traffic.

Workload/evaluator scalar monitoring is available, but deriving and calibrating those signals for a real application is not automatic. The corresponding synthetic injections must not be reported as proof of production detection capability. A lack of detected drift is not evidence that no drift occurred.
