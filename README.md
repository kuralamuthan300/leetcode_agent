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

## Web UI (Steps 11-13)

```bash
uv run streamlit run ui/app.py
```

- Modes (sidebar radio): `creator | solver | auto | review`.
  `review` is read-only: pick a `job_id` to inspect state + `audit.jsonl` + `report.md`.
- Inputs: `category` dropdown, `difficulty` radio, `num_tests` number input
  (default 8, feeds `RequirementsSpec.num_tests` -> generated test count),
  `constraints_hint`, file upload / textarea / example picker (`two_sum_demo`).
- Live status banner (`idle|running|awaiting approval|done|needs_human` +
  `job_id, mode, attempt, opt_round`) + stage stepper streamed from
  `app.stream(stream_mode="updates")`, + tabs `Problem | Solution | Tests`
  (`test_id|input|expected|actual|pass|mean_ms`, hidden-test toggle).
- HITL-1 understanding card: summary + agent questions + editable
  `num_tests`, then `[Confirm & Continue]` / `[Answer + Continue]` —
  no heavy LLM/Docker runs before confirm.
- Mid-run clarify popup (repair stuck, attempt>=2): failing tests +
  question, `[Submit + Resume]` / `[Skip/auto-retry]`.
- HITL-2 final approval: problem + solution + pass rate + total ms, then
  `[Approve]` / `[Decline + reason]` (declines are logged to `audit.jsonl`
  and route back to repair while budget remains).
- See `docs/architecture/LEETCODE_AGENT_PLAN.md` Sec 12 (UI) + Sec 13 (HITL), build steps 11–13 in `docs/build/BUILD_PLAN_STEP_BY_STEP.md`.
