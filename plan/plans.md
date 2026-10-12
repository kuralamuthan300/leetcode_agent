# Execution Plan — LangGraph DSA Solver (steps 1–9)

Source: `architecture.md` + `guardrails.md` | Updated: 2026-10-13

## Step 1 — IntentClarifier LLM
Replace heuristic-only with `deepseek-r1:1.5b` rewrite pass; keep max-3 self-loop
+ `intent_iterations`/`intent_clear`.
Verify: `PYTHONPATH=src uv run pytest tests/test_intent.py -q`.

## Step 2 — Planner prompt (heavy)
`gemma4:31b-cloud`: algo + complexity + steps from `clarified_intent` + `feedback`;
structured `plan_steps`.
Verify: unit test with retry `feedback` input.

## Step 3 — TestCaseGen schema (small)
Emit `[{id,input,expected}]` base + edge cases via `deepseek-r1:1.5b`.
Verify: schema/edge-case test.

## Step 4 — Coder + toolbox (heavy)
`get_heavy().bind_tools([python_toolbox])`; generate code then mandatory
`python_sandbox` smoke → `coder_smoke`.
Verify: `PYTHONPATH=src uv run pytest tests/test_toolbox.py -q`.

## Step 5 — Guardrails (spec: `guardrails.md`)
1. `src/agent/tools/guardrail.py::check()` — AST allowlist, file read/write denied, fail-closed; wire into `python_toolbox` (no graph change).
2. Harden `run_python` (minimal builtins, allowlist importer, mem cap, jail cwd).
3. `guardrail_report` in state/output + blocked→retry wiring.
Verify: `PYTHONPATH=src uv run pytest tests/test_guardrails.py -q` (malicious corpus blocked, legit DSA passes).

## Step 6 — Evaluator matrix (heavy)
Run code vs all tests via `run_python`; fill `execution_results`/`eval`;
increment `iterations`; set `feedback` on fail.
Verify: pass + fail-retry cases.

## Step 7 — Resilience
Timeouts (small 30–60s, heavy 120–180s); heavy→small fallback flag;
parallel fan-out check.
Verify: timeout test.

## Step 8 — Golden E2E
Two Sum, Valid Parentheses, Binary Search via CLI;
`output/run_id_*.json` contains requirement/code/testcases.
Verify: `PYTHONPATH=src uv run pytest -q` all green + 3 CLI runs.

## Step 9 — Finalize
Update architecture status, clean `output/run_id_*.json` artifacts,
confirm `git status` only shows intended files.
