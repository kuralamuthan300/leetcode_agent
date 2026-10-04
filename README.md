# LeetCode Agent (LangGraph + Ollama + Secure Sandbox)

See `docs/architecture/LEETCODE_AGENT_PLAN.md` for architecture and `docs/build/BUILD_PLAN_STEP_BY_STEP.md` for build order.

## Prerequisites

- Python 3.11+ (pinned via `.python-version`, managed by `uv`)
- `uv` (install: https://docs.astral.sh/uv/getting-started/installation/)
- Docker (for sandbox)
- Ollama running at `http://localhost:11434`

## Setup

```bash
uv sync
ollama pull gemma4:31b-cloud
ollama pull deepseek-r1:1.5b
ollama list  # verify both tags exist
docker build -f Dockerfile.sandbox -t sandbox-img .  # added in Step 4
```

## Verify Step 0

```bash
uv sync
uv run pytest -q
```

Expected: 0 tests collected, configs load.

## Verify Step 10 (Hardening + E2E + Docs)

```bash
ollama pull gemma4:31b-cloud
ollama pull deepseek-r1:1.5b
docker build -f Dockerfile.sandbox -t sandbox-img .
uv run pytest
uv run python -m src.graph
```

Expected: full suite green (`tests/test_security.py` 8/8 blocked, E2E
`tests/test_e2e_step10.py` appends `docs/reports/e2e_step10.log`),
checkpointer is `SqliteSaver` (file via `LEETCODE_CHECKPOINT_DB`, else
in-memory), human approval gates before `promote_save`
(`human_approved_promote is False` blocks) and before accepting
optimization (`human_approved_opt is False` forces rollback), every
scan/run/decision appended to `<jobdir>/audit.jsonl`. Prod pauses:
`build_creator_graph_strict()` / `build_solver_graph_strict()` /
`build_graph_strict()` compile with `interrupt_before`.
Security table: `docs/security_tests.md`.
