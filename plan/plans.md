# Execution Plan — LangGraph DSA Solver (steps 1–9)

Source: `architecture.md` + `guardrails.md` | Updated: 2026-10-18
Progress: Steps 1–18 complete on `plan/plan.md` (101 tests green; Two Sum, Valid Parentheses, Binary Search all 5/5 live).

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
planner→[tests,coder] fan-out ran parallel in LangGraph *(later replaced by the
sequential chain in Step 12)*.
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
*(Ordering superseded by Step 15 — see below.)*

## Step 13 — State/reporter completeness [done]
`function_signature`, `critic_overruled`, `error_class` verified in state +
reporter JSON (`tests/test_reporter.py` asserts exact schema).

## Step 14 — Fullest gatherer (memory removed) [done]
Always-confirm on TTY (understanding + assumptions, Enter accepts, free-text
corrects, max 2 rounds); disambiguation-only questions. Session memory
(`session.py`, `--fresh`, `session_notes`) removed — every run starts fresh.
Heavy model switched to `nemotron-3-super:cloud` (small/critic stay gemma4).

## Step 15 — TDD ordering + coder self-loop + blame-aware routing [done]
`planner → tests → coder → evaluator`: testgen derives from the CONTRACT (never
the coder's code) on the reasoning model (`get_testgen`, `OLLAMA_TESTGEN*`),
splits off a coder-hidden holdout test, and consumes `feedback`. Coder runs a
bounded self-loop (≤3 retries, `coder_iterations`) against the visible tests
with deterministic blame — retries on code failures, escalates when the test is
at fault. Evaluator re-runs the full matrix (visible + holdout) as the only
gate, computes dominant `eval.blame`, and routes test-blame → tests,
code/unknown-blame → planner. Full plan: `plan/plan.md`.

## Step 16 — Live terminal output (code + test I/O) [done]
`cli.py` renders rich panels during the run: the plan + contract
(signature/args/returns) after `planner`, a code panel on every coder attempt, a
test-case table (holdout labelled) on `tests`, and input/expected/actual/blame
result tables for the coder self-check (`coder_checks`) and the evaluator. On by
default; `--quiet` restores status-lines-only, `--verbose` still dumps the full
final code/tests. `ExecutionResult` entries now carry `harness`; state gains
`coder_checks`. Long code/values are clipped for readability. Full plan:
`plan/plan.md`.

## Step 17 — During-node narrator (spinner + live timer) [done]
While a node runs (LLM calls block 90-180s) `cli.py` shows a bottom-line spinner
with rotating, node-aware phrases plus a live seconds counter (`planner:
thinking…  (14s)`). The graph is drained by a worker thread (`_produce`) into a
`queue.Queue`; the main thread runs `rich.live.Live(get_renderable=...)` — a
persistent `Spinner` whose text is recomputed ~8×/s on the refresh thread — and
predicts the next node with `_next_node` (reusing the graph routers); the phrase
rotates every `_NARRATION_INTERVAL = 3s`. Every status line also ends with that
node's duration (`✓ plan (3 steps) · 12.4s`), in all modes. The spinner/timer is
shown only when panels are on and stdout is a TTY, so `--quiet`/piped runs stay
clean apart from the duration suffix. Verify: `PYTHONPATH=src uv run pytest -q
tests/test_cli.py` (`_next_node`, `_narrator` rotation/seconds, duration suffix,
forced live-path + error-path tests).

## Step 18 — Gatherer modes + batch dataset driver [done]
Two explicit gatherer modes. **interactive** (default, unchanged) shows the
understanding before the planner; **quiet** (`--gather-mode quiet`, default for
piped/headless) validates the requirement in a single pass with one
`QUIET_GATHER_SYSTEM` verdict (strict JSON) and never prompts — hard gates first
(language/scope/truncation→`incomplete`), then abort if `workable=false` or a
blocking ambiguity exists, else proceed (flagging `assumed` when the model listed
assumptions). `_route_gather` short-circuits quiet to the planner so it cannot
loop. Classified outcomes (`status` ok/assumed/aborted/`infra_error`, a pinned
`rejection_class` enum + `workability.rubric_version`) give headless runs a
stable JSON schema for an LLM-as-judge; a transport failure is `infra_error`
(bounded retry), never `rejected`. Adds the batch driver
`--problems-file problems.jsonl` (implies quiet) writing
`output/<batch_id>/<id>.json` + `summary.json` (`status_counts`,
`rejection_class_counts`, `yield`), continuing past per-row infra failures; exit
codes `0` ok/assumed/aborted, `1` infra, `2` bad input. Full plan:
`plan/plan.md` (Increment 4). Verify: `PYTHONPATH=src uv run pytest -q
tests/test_intent.py tests/test_cli.py`.

