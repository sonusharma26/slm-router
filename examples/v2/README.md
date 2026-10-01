# Examples

`policy.yaml`, `endpoints.json`, and `service.yaml` are configuration templates with deliberately placeholder deployment IDs, prices and revisions. They are not model recommendations. The initial service has no calibration evidence and will abstain. No upstream inference is triggered merely by loading the configuration.

For the working no-network example, run `slm demo --out results/my-demo`; see `src/inference_control/demo.py` for the complete measurement and recovery integration.

`competitors.*.json` are opt-in experiment templates, not proof that either gateway is configured. Set `replay_backend_confirmed` only after actually routing every listed model to the constant-response benchmark backend. Null classifier cost deliberately prevents a complete cost comparison. RouteLLM installation/checkpoint setup is external and must be pinned and recorded by the experiment operator.

The service accepts `token_limit_field: max_tokens` for compatible deployments that do not implement `max_completion_tokens`. Token caps still originate in the recorded plan, never a client transport override.
