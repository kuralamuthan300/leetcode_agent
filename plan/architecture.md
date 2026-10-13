# Agent Architecture — LangGraph DSA Solver

Branch: `task/step-1-5` | Updated: 2026-10-13
Status: RequirementGatherer + signature contract + replace-results + gemma-all-models (fix-proof E2E 4/4).

## Goal
Terminal-driven LangGraph agent that solves data-structures & algorithms problems,
verifies with generated tests in a sandbox, retries via feedback loop, and writes
`output/run_id_xxxx.json` with requirement, code, test cases.

## Graph
`Start -> RequirementGatherer -(confirm/assume)-> Planner -> [TestCaseGen, Coder] -> Evaluator -> RunReporter -> End`
- Gatherer prints understanding + assumptions + questions, confirms inline on TTY
  (max 2 rounds); non-TTY proceeds flagged `assumed`. Truncated input forces questions.
- `Evaluator --fail--> Planner` (max 3 iterations), else `--> RunReporter`.
- `Coder -> guardrail.check() -> python_sandbox` tool (fail-closed; blocked never execs). Full spec: `guardrails.md`.

```mermaid
graph TD
    Start --> RequirementGatherer
    RequirementGatherer -.->|vague, confirms| RequirementGatherer
    RequirementGatherer --> Planner
    Planner --> TestCaseGen
    Planner --> Coder
    Coder -.-> PythonSandbox
    TestCaseGen --> Evaluator
    Coder --> Evaluator
    Evaluator -.->|Feedback Loop| Planner
    Evaluator --> RunReporter
    RunReporter --> End
```

## Models (Ollama)
- Heavy + small `gemma4:31b-cloud`: all nodes (gemma3 retired upstream; single
  family keeps prompts consistent).
- `src/agent/llms.py`: `get_small()` temp 0 / 4096 tokens / 90s timeout,
  `get_heavy()` temp 0.2 / 180s timeout.
- Env: `OLLAMA_HEAVY`, `OLLAMA_SMALL`, `OLLAMA_BASE_URL` (default `http://localhost:11434`).

## State (`src/agent/state.py`)
`run_id, requirement, clarified_intent, intent_iterations, intent_clear,`
`confirmed, assumed, plan_steps, function_signature, test_cases, code,`
`coder_smoke{stdout,stderr,error}, guardrail_report{allowed,reasons[],checked_at},`
`fallback_used, execution_results (replace semantics),`
`eval{passed,passed_count,total,feedback}, feedback, iterations, output_path`.

## Nodes (`src/agent/nodes.py`)
1. RequirementGatherer: Python-only + DSA-scope gates (rejects → reporter);
   truncation forces user questions; TTY confirms inline (max 2 rounds),
   non-TTY proceeds `assumed`; emits `clarified_intent`, `confirmed`/`assumed`.
2. Planner (heavy, small fallback): LLM plan + `SIGNATURE: def solve(...)`
   contract; consumes `feedback` on retry; stub if both models fail.
3. TestCaseGen (small): LLM JSON `[{id,harness,expected}]` (3-5, harness must
   `print(solve(...))` with contracted arity); validated + guardrail-screened;
   `[]` fallback.
4. Coder (heavy, small fallback + `python_sandbox` tool): LLM code honoring the
   planner's `function_signature` (AST-validated, repair on mismatch),
   mandatory guardrail gate + smoke, one repair attempt on failure;
   emits `code`, `coder_smoke`, `guardrail_report`.
5. Evaluator: runs each harness via gated `run_python`, compares stdout to
   expected; emits `execution_results` (replace semantics — latest run only),
   `eval` (failure details), `iterations`.
6. RunReporter (deterministic, no LLM): consolidate + write JSON.

Resilience: small 90s / heavy 180s client timeouts; heavy→small fallback sets
`fallback_used` (planner, coder incl. repair); planner→[tests,coder] fan-out is parallel.

## CLI + Output
- `PYTHONPATH=src uv run python -m cli --problem "..." | --file p.md | stdin pipe`.
  With no args on a TTY it prompts via `input()` (blank line finishes); piped stdin still read directly.
- `run_id = run_id_YYYYMMDD_HHMMSS_<6hex>`, overridable via `--run-id`.
- JSON schema: `run_id, timestamp, models{heavy,small}, requirement, rejected,`
  `rejection_reason, clarified_intent, intent_iterations, intent_clear,`
  `confirmed, assumed, plan_steps, function_signature, test_cases, code,`
  `coder_smoke, guardrail_report, fallback_used, execution_results, eval`.
- Written by RunReporter to `output/<run_id>.json` (also on exhausted retries with `passed:false`).

## Guardrails
Spec: `guardrails.md` (static AST gate + strict file lockdown + hardened exec,
fail-closed). Implementation: `src/agent/tools/guardrail.py` + `tests/test_guardrails.py` (see `plans.md` Step 5).

## Project setup (uv)
- `pyproject.toml`: `langgraph, langchain-ollama, langchain-core, pydantic, rich`; dev `pytest, ruff`.
- Layout: `src/agent/{state,llms,nodes,prompts,graph,tools/sandbox,tools/guardrail}.py`, `src/cli.py`,
  `tests/{test_graph,test_sandbox,test_intent,test_toolbox,test_guardrails,test_nodes,test_cli}.py`, `output/.gitkeep`, `plan/`.
- Verify: `uv sync`, `PYTHONPATH=src uv run pytest -q` (22 passed),
  `PYTHONPATH=src uv run python -m cli --problem "two sum..."`.
