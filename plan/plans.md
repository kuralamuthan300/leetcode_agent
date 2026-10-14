# Execution Plan — LangGraph DSA Solver (steps 1–9)

Source: `architecture.md` + `guardrails.md` | Updated: 2026-10-13
Progress: Steps 1–9 complete on `task/step-1-5` (18 tests green; Two Sum, Valid Parentheses, Binary Search all 5/5 live).

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

## Step 6 — Evaluator matrix (heavy) [done]
Runs each harness via gated `run_python`; `execution_results` with actual vs
expected + error snippets in `feedback`; `iterations++`.
Verify: pass + fail-retry cases (`tests/test_nodes.py`).

## Step 7 — Resilience [done]
Timeouts (small 90s, heavy 180s via Ollama client); heavy→small fallback with
`fallback_used` flag in state/output (`try_invoke_with_fallback`, planner+coder);
planner→[tests,coder] fan-out runs parallel in LangGraph.
Verify: `test_heavy_fallback_flag`.

## Step 8 — Golden E2E [done]
Two Sum ✅ 5/5, Valid Parentheses ✅ 5/5, Binary Search ✅ 5/5 via CLI
(`--run-id` pinned); `output/run_id_*.json` contains requirement/code/testcases.

## Step 9 — Finalize [done]
Architecture status updated, `output/run_id_*.json` artifacts cleaned,
`git status` shows only intended files, 18 tests green.

## Step 10 — Coder smoke-call + error classifier [done]
`src/agent/errors.py`: `classify_error` (none/blocked/timeout/mechanical/logical),
smoke-call fails only on NameError/ImportError (dummy-input type errors noted,
not failed); `dummy_args` from signature annotations. Coder calls `solve()`
after import check (skipped without signature/stub).

## Step 11 — Local critics [done]
`critique()` in `prompts.py` + `llms.get_critic()` (gemma4, 512-token verdicts,
`PASS`/`ISSUES:`) wired as sub-steps in planner/coder/testgen: max 1
critique→regen round each, `critic_overruled` flag on exhaust. Fail-open
(critic down → proceed unflagged).

## Step 12 — Sequential chain + mechanical short-circuit [done]
`planner → coder → tests → evaluator` (testgen sees final code + signature).
Evaluator short-circuits on smoke failure (`error_class: mechanical`, no matrix
run) so planner rectifies; `error_class` none/logical/empty otherwise.

## Step 13 — State/reporter completeness [done]
`function_signature`, `critic_overruled`, `error_class` verified in state +
reporter JSON (`tests/test_reporter.py` asserts exact schema).

## Step 14 — Fullest gatherer [done]
Always-confirm on TTY (understanding + assumptions + remembered conventions,
Enter accepts, free-text corrects, max 2 rounds); disambiguation-only
questions; per-session memory (`session.py`, `/tmp` file, `--fresh` resets,
ignored off-TTY); `session_notes` in state + JSON.
