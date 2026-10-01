# Contributing

Thanks for helping improve SLM Router. The project is an alpha-stage inference-control and evaluation system, so changes should preserve its fail-closed behavior and evidence boundaries.

## Development setup

Use Python 3.11, 3.12 or 3.13:

```bash
python -m pip install -e ".[dev,legacy]"
python -m pytest -q
```

The default test suite is offline. Do not add tests that require provider credentials, paid APIs, model downloads or network access. Put live-provider experiments behind explicit operator commands and keep their generated outputs out of source control.

## Pull requests

- Keep changes focused and explain their runtime or evidence impact.
- Add regression coverage for behavior changes.
- Preserve deterministic seeds and train/calibration/test separation in benchmark code.
- Do not claim benchmark superiority from synthetic fixtures or unexecuted competitors.
- Never commit `.env` files, credentials, raw prompts, provider responses or local SQLite ledgers.
- Update the README or release documentation when public behavior or limitations change.

Before opening a pull request, run the full offline suite and the relevant CLI smoke command. Include the commands and results in the pull-request description.
