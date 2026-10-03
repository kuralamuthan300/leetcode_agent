# SESSION_STATE — resume here, do NOT re-scan repo

current_step: 1
last_verify: pending — run `uv run pytest tests/test_schemas.py -v`
files_touched: pyproject.toml, config/models.yaml, config/sandbox.yaml
next_step: 1 — Schemas + Example JSONs only (src/schemas.py, workspace/problems/examples/, workspace/jobs/examples/, tests/test_schemas.py)

## Rule
- Startup reads: AGENTS.md + this file + current Step section only.
- Allowlist = files named in current Step's Ask block. Everything else is forbidden.
- Update this file at end of each Step (current_step, last_verify, files_touched, next_step).
