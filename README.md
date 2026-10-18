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

# status lines only (no code/test panels)
PYTHONPATH=src uv run python -m cli --problem "..." --quiet

# headless gather mode: validate once, never prompt (default when piped)
PYTHONPATH=src uv run python -m cli --problem "..." --gather-mode quiet

# batch dataset: one {id, requirement} per JSONL line -> output/<batch_id>/<id>.json
PYTHONPATH=src uv run python -m cli --problems-file problems.jsonl
```

Each run streams a status line per node (`✓ intent clear`, `✓ plan`, `✓ tests`,
`✓ code`, `↻ code retry`, `✓ eval 5/5`, `✓ wrote ...`, with `↻` retries and `⚠`
blocks) **plus live panels**: the plan + contract (signature/args/returns) after
the planner, the generated code on every coder attempt, the test cases (with the
coder-hidden holdout labelled), and per-test input/output result tables after the
coder self-check and the evaluator. On an interactive terminal a spinner with
rotating, node-aware phrases and a live seconds timer (`planner: thinking…  (14s)`,
`coder: writing the solution…  (58s)`) sits at the bottom while each node runs
(LLM calls block for 90-180s), and every status line ends with that node's
duration (`✓ plan (3 steps) · 12.4s`). It writes `output/<run_id>.json` with
requirement, clarified intent, plan, test cases (plus holdout), code, sandbox
results, eval report, and guardrail verdict.

Add `--quiet` for status lines only, or `--verbose` to also print the full final
code and tests.

### Gatherer modes

The requirement gatherer runs in one of two modes (`--gather-mode`):

- **interactive** (default on a TTY): rewrites the requirement, shows the
  understanding + assumptions, and waits for Enter (free text corrects it, max 2
  rounds).
- **quiet** (default when piped/headless): validates the requirement in a single
  pass and **never prompts**. It applies the hard gates first (Python-only, DSA
  scope, truncation) and then judges workability in one call — aborting when the
  requirement is not workable or has an answer-changing ambiguity, otherwise
  proceeding (flagged `assumed` if it had to assume anything).

Every run records a classified outcome in the JSON — `status`
(`ok`/`assumed`/`aborted`/`infra_error`), a `rejection_class`, and a pinned
`workability` (`rubric_version`) block — so headless runs are directly usable as
data for an LLM-as-judge. A transport/model failure is reported as `infra_error`,
**never** as a rejection.

### Batch dataset

`--problems-file problems.jsonl` (one `{"id", "requirement"}` object per line)
implies quiet gathering and runs the graph over every row, writing
`output/<batch_id>/<id>.json` per problem plus a
`output/<batch_id>/summary.json` with `status_counts`, `rejection_class_counts`
and `yield {completed, aborted, infra}`. A row that fails hard is recorded as
`infra_error` and the batch continues.

Exit codes: `0` for `ok`/`assumed`/`aborted` (the classification lives in the
JSON), `1` for infra failures, `2` for bad input.

## Test

```bash
PYTHONPATH=src uv run pytest -q   # 101 tests (hermetic fallbacks + live-graph E2E)
```

## Layout

- `src/agent/{state,llms,prompts,nodes,graph}.py` — state, models, prompts, nodes, graph
- `src/agent/tools/{sandbox,guardrail}.py` — gated `python_sandbox` tool + AST security gate
- `src/cli.py` — terminal entrypoint
- `plan/{architecture,plans,plan,guardrails}.md` — design docs
