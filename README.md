# leetcodeagent — LangGraph DSA solver

Terminal-driven agent that solves data-structures & algorithms problems:
clarify intent → plan → generate tests + code in parallel → verify in a
sandboxed, guardrailed runner → write `output/run_id_xxxx.json`.

Graph: `IntentClarifier (self-loop max 3) → Planner → [TestCaseGen, Coder] → Evaluator → RunReporter`
Models (Ollama): heavy `gemma4:31b-cloud` (plan/code/eval), small `gpt-oss:20b-cloud` (clarify/tests).

## Prerequisites

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- [Ollama](https://ollama.com) running locally (`ollama serve`)

## Setup

```bash
uv sync
ollama pull gemma4:31b-cloud
ollama pull gpt-oss:20b-cloud
```

Optional env overrides (`src/agent/llms.py` defaults shown):

```bash
export OLLAMA_HEAVY=gemma4:31b-cloud OLLAMA_SMALL=gpt-oss:20b-cloud
export OLLAMA_BASE_URL=http://localhost:11434
```

## Run

```bash
# inline problem
PYTHONPATH=src uv run python -m cli --problem "Return indices of the two numbers adding to target..."

# from file
PYTHONPATH=src uv run python -m cli --file problem.md

# interactive prompt (no args on a TTY; blank line finishes)
PYTHONPATH=src uv run python -m cli

# piped stdin still works
cat problem.md | PYTHONPATH=src uv run python -m cli

# pin the run id
PYTHONPATH=src uv run python -m cli --problem "..." --run-id run_id_demo
```

Each run writes `output/<run_id>.json` with requirement, clarified intent,
plan, test cases, code, sandbox results, eval report, and guardrail verdict.

## Test

```bash
PYTHONPATH=src uv run pytest -q   # 22 tests (hermetic fallbacks + one live-graph E2E)
```

## Layout

- `src/agent/{state,llms,prompts,nodes,graph}.py` — state, models, prompts, nodes, graph
- `src/agent/tools/{sandbox,guardrail}.py` — gated `python_sandbox` tool + AST security gate
- `src/cli.py` — terminal entrypoint
- `plan/{architecture,plans,guardrails}.md` — design docs
