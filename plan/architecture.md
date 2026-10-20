# Agent Architecture — LangGraph DSA Solver

Branch: `plan/plan.md` | Updated: 2026-10-18
Status: TDD chain (tests→coder) + evaluator-owned blame/diagnosis (no coder self-loop, 1 retry) + live terminal output + during-node narrator/timer + interactive/quiet gatherer modes + batch dataset driver (103 tests green).

## Goal
Terminal-driven LangGraph agent that solves data-structures & algorithms problems,
verifies with generated tests in a sandbox, retries via feedback loop, and writes
`output/run_id_xxxx.json` with requirement, code, test cases.

## Graph
`Start -> RequirementGatherer -(confirm/assume)-> Planner -> TestCaseGen -> Coder -> Evaluator -(blame-aware, ≤1 retry)-> RunReporter -> End`
- Gatherer (`gather_mode` interactive/quiet): interactive prints understanding +
  assumptions + questions and confirms inline on TTY (max 2 rounds), non-TTY
  proceeds flagged `assumed`; quiet validates once (one `QUIET_GATHER_SYSTEM`
  verdict, hard gates first) and never prompts — aborting with a
  `rejection_class` (answer-changing ambiguity ⇒ strict abort), else proceed
  (`assumed` if it had to assume). Truncated input forces questions (interactive)
  / aborts `incomplete` (quiet); a model failure is `infra_error`, not a
  rejection. Quiet never re-asks (`_route_gather` short-circuits to the planner).
- Tests derive from the Planner's CONTRACT (never the Coder's code).
- `Coder` produces **one** solution (no self-loop), gated + import/`solve()`
  smoke-checked + code-critic, then hands off.
- `Evaluator` (evaluator-only `nemotron-3-ultra`) re-runs the full matrix
  (visible + holdout) as the **only** gate, adjudicates each failure, then emits a
  holistic `failure_report` (culprit code|test + reason). `--fail--> Planner` when
  the **code** is at fault (tests retained — the planner skips `TestCaseGen` and
  goes straight to `Coder`); `--fail--> TestCaseGen` when the **tests** are at
  fault or empty (code retained — tests skip `Coder` and re-enter `Evaluator`).
  One retry total (`MAX_ITERATIONS = 2`); mechanical smoke failure short-circuits
  with `error_class` (no matrix run) straight to the planner, else `--> RunReporter`.
- `Coder -> guardrail.check() -> python_sandbox` tool (fail-closed; blocked never execs). Full spec: `guardrails.md`.

```mermaid
graph TD
    Start --> RequirementGatherer
    RequirementGatherer -.->|vague, confirms| RequirementGatherer
    RequirementGatherer -->|confirmed / assumed| Planner
    RequirementGatherer -.->|rejected: wrong language / non-DSA| RunReporter
    Planner -->|plan + CONTRACT + plan-critic / fresh| TestCaseGen
    Planner -->|replan: code blame, tests retained| Coder
    TestCaseGen -->|tests + holdout + test-critic / fresh| Coder
    TestCaseGen -->|retest: test blame, code retained| Evaluator
    Coder -->|gate + smoke-call + code-critic| Evaluator
    Evaluator -.->|blame=code| Planner
    Evaluator -.->|blame=test or empty tests| TestCaseGen
    Evaluator -->|pass / retry exhausted| RunReporter
    RunReporter --> End
```

## Models (MLflow AI gateway)
- All roles call the OpenAI-compatible MLflow gateway
  (`gateway.base_url`, default `http://127.0.0.1:5000/gateway/mlflow/v1`) with
  the endpoint name as `model`.
- Heavy + critic `nemotron-3-super`; testgen `nemotron-3-super`; small `gemma4`;
  evaluator-only `nemotron-3-ultra` (blame + diagnosis — used by the evaluator
  alone). Reasoning is server-side; the gateway returns final content only.
- `src/agent/llms.py` builds `ChatOpenAI` clients from `src/agent/config.py`:
  `get_small()` temp 0 / 4096 tokens / 90s, `get_heavy()` temp 0.2 / 4096 /
  180s, `get_critic()` temp 0 / 1024 / 60s, `get_testgen()` temp 0 / 4096 /
  180s, `get_evaluator()` temp 0 / 4096 / 240s.
- `config.yml` holds `gateway`, per-role `models` and `agent` knobs; env vars
  (`MLFLOW_HEAVY`, `MLFLOW_SMALL`, `MLFLOW_CRITIC`, `MLFLOW_TESTGEN`,
  `MLFLOW_EVALUATOR`, `MLFLOW_GATEWAY_URL`, `MLFLOW_TRACKING_URI`, …) override it.
- `src/agent/tracing.py` sets the tracking URI + experiment and enables
  `mlflow.langchain.autolog(log_traces=True)` from `cli.main()`. LLM spans are
  role-named (`model:heavy`, …), the reporter renames the trace to
  `<run_id> · <problem>` (`mlflow.traceName`) and adds `run_id`/`status`/
  `gather_mode`/`eval` tags, and the CLI prints a deep link to the trace.
  `--no-tracing` / `MLFLOW_TRACING_DISABLED=1` opt out.

## State (`src/agent/state.py`)
`run_id, requirement, gather_mode (interactive|quiet), status (ok|assumed|aborted|infra_error),`
`rejection_class, workability{verdict,reason,assumptions[],blocking_ambiguities[],confidence,rejection_class,rubric_version}, out_dir,`
`clarified_intent, intent_iterations, intent_clear,`
`confirmed, assumed, plan_steps, function_signature, arg_spec,`
`returns_spec, test_cases, holdout_tests, code,`
`coder_smoke{stdout,stderr,error}, guardrail_report{allowed,reasons[],checked_at},`
`fallback_used, critic_overruled, execution_results (replace semantics),`
`eval{passed,passed_count,total,feedback,error_class,blame}, failure_report{culprit,reason,failing[]}, next_action, feedback, iterations, output_path`.

## Nodes (`src/agent/nodes.py`)
1. RequirementGatherer: Python-only + DSA-scope gates (rejects → reporter);
   truncation forces user questions; **interactive** (default) TTY *always* shows
   understanding + assumptions and waits for Enter (max 2 correction rounds),
   non-TTY proceeds `assumed` when vague; **quiet** validates in one
   `QUIET_GATHER_SYSTEM` pass without prompting (abort on not-workable /
   blocking ambiguity / truncation, else proceed), `infra_error` on model
   failure (`_abort`, `_quiet_gather`, `_workability`).
   Emits `clarified_intent`, `confirmed`/`assumed`, `status`, `rejection_class`,
   `workability`.
2. Planner (heavy, small fallback): LLM plan + fenced JSON `CONTRACT`
   (signature + typed args/ranges + returns; legacy `SIGNATURE:` accepted);
   consumes `feedback` on retry; stub if both models fail.
3. TestCaseGen (testgen = heavy + `reasoning`, falls back to heavy): LLM JSON
   `[{id,harness,expected}]` (3-5, harness must `print(solve(...))` with
   contracted arity and in-range values, boundary coverage) derived from the
   planner CONTRACT + `feedback` — **never from the coder's code**; validated +
   guardrail-screened; splits off one **holdout** test (≥3 tests) the coder
   never sees; `[]` fallback. On a test-blame retry it consumes the evaluator's
   `failure_report` feedback, and the graph skips the coder (code retained).
4. Coder (heavy, small fallback + `python_sandbox` tool): LLM code honoring the
   planner's `function_signature` and the visible `test_cases` (AST-validated,
   repair on mismatch), mandatory guardrail gate + import smoke + `solve()`
   call-smoke (fails only on NameError/ImportError); bounded repairs (sig-fix or
   smoke-fix, then one critic regen). **No self-loop**: it emits one `code`
   (+ `coder_smoke`, `guardrail_report`) and hands off to the evaluator.
5. Evaluator (evaluator-only `nemotron-3-ultra`): re-runs every harness
   (visible + holdout) via gated `run_python` as the **only authoritative gate**;
   compares stdout to expected; adjudicates failures (`adjudicate()`: contract
   violations ⇒ test, else evaluator verdict) with per-result `blame`; repairs
   blame=test cases in place (`repair_test()`, escalates on flip-flop); computes
   the dominant `eval.blame`; then `diagnose()` reads the code + failures and emits
   `failure_report{culprit,reason,failing[]}` + `next_action` (`replan_code` /
   `fix_tests`) + a readable `feedback` string. Emits `execution_results` (replace
   semantics — latest run only; each entry carries `harness`), `eval`, `iterations`.
6. RunReporter (deterministic, no LLM): consolidate + write JSON.

Resilience: small 90s / heavy 180s / testgen 180s / evaluator 240s client
timeouts; heavy→small fallback sets `fallback_used` (planner, coder incl.
repair); `adjudicate`/`diagnose` fall back to the small model / deterministic
blame when the evaluator model is unavailable (never stall). Chain is sequential
TDD (planner→tests→coder→evaluator); the coder never runs the tests, so the
evaluator's full-matrix re-run is the only verdict. Critics: `critique()` on
nemotron (512 tokens) reviews plan/code/tests with max 1 regen each;
`critic_overruled` set when proceeding with open issues.

## CLI + Output
- `PYTHONPATH=src uv run python -m cli --problem "..." | --file p.md | stdin pipe`.
  With no args on a TTY it prompts via `input()` (blank line finishes); piped stdin still read directly.
- `run_id = run_id_YYYYMMDD_HHMMSS_<6hex>`, overridable via `--run-id`; `--out-dir` sets the output root (default `output`); `--quiet` prints status lines only, `--verbose` also dumps the full final code/tests/holdout.
- `--gather-mode {interactive,quiet}` (default: interactive on a TTY, else quiet);
  `--problems-file problems.jsonl` runs batch mode (implies quiet, mutually
  exclusive with `--problem`/`--file`).
- Batch: `output/<batch_id>/<id>.json` per row + `output/<batch_id>/summary.json`
  (`status_counts`, `rejection_class_counts`, `yield{completed,aborted,infra}`,
  `skipped`); a raising row is `infra_error` and the batch continues.
- Exit codes: `0` ok/assumed/aborted, `1` infra, `2` bad input.
- Live output: per-node status lines (`✓` done, `↻` retry, `⚠` blocked, `⛔` rejected, `(fallback)`/`(assumed)` tags), each ending with that node's duration (`· 12.4s`), plus rich panels — the planner's plan + contract (signature/args/returns), the generated code on every coder attempt, a test-case table (holdout labelled), input/expected/actual/blame result tables for the evaluator, and the evaluator's diagnosis panel (culprit + reason). Long values/code are clipped. While a node runs, a bottom-line spinner with rotating, node-aware phrases (`_NARRATION`) plus a live seconds counter (`planner: thinking…  (14s)`) is driven by a worker thread feeding a queue + main-thread `rich.live.Live(get_renderable=...)` (a persistent `Spinner` whose text is recomputed ~8×/s on the refresh thread; the phrase rotates every `_NARRATION_INTERVAL = 3s`). The spinner/timer shows only when panels are on and stdout is a TTY, so `--quiet`/piped runs stay clean (they still carry the duration suffix).
- JSON schema: `run_id, timestamp, models{heavy,small,evaluator}, requirement, gather_mode,`
  `status, rejection_class, workability{...,rubric_version}, rejected,`
  `rejection_reason, clarified_intent, intent_iterations, intent_clear,`
  `confirmed, assumed, plan_steps, function_signature, arg_spec,`
  `returns_spec, test_cases, holdout_tests, code,`
  `coder_smoke, guardrail_report, fallback_used,`
  `critic_overruled, execution_results, eval{...,blame}, failure_report{culprit,reason,failing[]}, next_action, feedback`.
- Written by RunReporter to `output/<run_id>.json` (also on exhausted retries with `passed:false`).

## Guardrails
Spec: `guardrails.md` — entry gates (Python-only language, DSA-scope, truncation
questions) + static AST gate + strict file lockdown + hardened exec, fail-closed.
Implementation: `src/agent/tools/guardrail.py`, `src/agent/errors.py`
+ `tests/test_guardrails.py`, `test_language.py`, `test_scope.py`.

## Project setup (uv)
- `pyproject.toml`: `langgraph, langchain-openai, langchain-core, pydantic, pyyaml, rich, mlflow`; dev `pytest, ruff`.
- Layout: `src/agent/{state,llms,prompts,nodes,graph,errors,tools/sandbox,tools/guardrail}.py`, `src/cli.py`,
  `tests/{test_graph,test_sandbox,test_intent,test_toolbox,test_guardrails,test_nodes,test_cli,test_scope,test_language,test_smoke,test_reporter,test_session}.py`, `output/.gitkeep`, `plan/`.
- Verify: `uv sync`, `PYTHONPATH=src uv run pytest -q` (103 passed),
  `PYTHONPATH=src uv run python -m cli --problem "two sum..."`.
