# leetcodeagent — LangGraph DSA solver

Terminal-driven agent that solves data-structures & algorithms problems:
clarify intent → plan → generate tests → write code (bounded self-loop) →
verify in a sandboxed, guardrailed runner → write `output/run_id_xxxx.json`.

Graph: `RequirementGatherer (self-loop max 3) → Planner → TestCaseGen →
Coder (self-loop max 3) → Evaluator (blame-aware, max 3) → RunReporter`.
Models (Ollama): heavy `nemotron-3-super:cloud` (plan/code/eval), testgen
`nemotron-3-super:cloud` with reasoning on (plan tests), small
`gemma4:31b-cloud` (clarify).

## Prerequisites

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- [Ollama](https://ollama.com) running locally (`ollama serve`)

## Setup

```bash
uv sync
ollama pull nemotron-3-super:cloud
ollama pull gemma4:31b-cloud
```

Optional env overrides (`src/agent/llms.py` defaults shown):

```bash
export OLLAMA_HEAVY=nemotron-3-super:cloud OLLAMA_SMALL=gemma4:31b-cloud
export OLLAMA_TESTGEN=nemotron-3-super:cloud   # testgen model
export OLLAMA_TESTGEN_THINK=1                  # reasoning on for testgen
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

Each run streams one minimal status line per node (`✓ intent clear`, `✓ plan`,
`✓ tests`, `✓ code`, `↻ code retry`, `✓ eval 5/5`, `✓ wrote ...`, with `↻`
retries and `⚠` blocks) and writes `output/<run_id>.json` with requirement,
clarified intent, plan, test cases (plus a coder-hidden holdout), code, sandbox
results, eval report, and guardrail verdict.

Add `--verbose` to also print the final code and tests.

## Test

```bash
PYTHONPATH=src uv run pytest -q   # 72 tests (hermetic fallbacks + live-graph E2E)
```

## Layout

- `src/agent/{state,llms,prompts,nodes,graph}.py` — state, models, prompts, nodes, graph
- `src/agent/tools/{sandbox,guardrail}.py` — gated `python_sandbox` tool + AST security gate
- `src/cli.py` — terminal entrypoint
- `plan/{architecture,plans,plan,guardrails}.md` — design docs
