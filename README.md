# leetcodeagent — LangGraph DSA solver

Terminal-driven agent that solves data-structures & algorithms problems:
clarify intent → plan → generate tests → write code →
verify in a sandboxed, guardrailed runner → write `output/run_id_xxxx.json`.

Graph: `RequirementGatherer (self-loop max 3) → Planner → TestCaseGen →
Coder → Evaluator (evaluator-only ultra, blame-aware, 1 retry) → RunReporter`.
The coder does **not** self-loop: it writes one solution and hands it to the
evaluator, which reads the code and the failing tests and decides whether the
**code** or the **tests** are at fault. Code blame re-runs `Planner → Coder`
(tests retained); test blame re-runs `TestCaseGen → Evaluator` (code retained).
One retry overall (1 original run + 1 retry run).
Models (Ollama): heavy `nemotron-3-super:cloud` (plan/code), testgen
`nemotron-3-super:cloud` with reasoning on (plan tests), small
`gemma4:31b-cloud` (clarify), evaluator `nemotron-3-ultra:cloud` (blame +
diagnosis; used **only** by the evaluator).

## Prerequisites

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- [Ollama](https://ollama.com) running locally (`ollama serve`)

## Setup

```bash
uv sync
ollama pull nemotron-3-super:cloud
ollama pull nemotron-3-ultra:cloud
ollama pull gemma4:31b-cloud
```

Optional env overrides (`src/agent/llms.py` defaults shown):

```bash
export OLLAMA_HEAVY=nemotron-3-super:cloud OLLAMA_SMALL=gemma4:31b-cloud
export OLLAMA_TESTGEN=nemotron-3-super:cloud   # testgen model
export OLLAMA_TESTGEN_THINK=1                  # reasoning on for testgen
export OLLAMA_EVALUATOR=nemotron-3-ultra:cloud # evaluator-only judge model
export OLLAMA_EVALUATOR_THINK=1                # reasoning on for the evaluator
export OLLAMA_BASE_URL=http://localhost:11434
```

## Run

The CLI lives in `src/` (it is not an installed package), so every invocation is
run through [`uv`](https://docs.astral.sh/uv/) with `PYTHONPATH=src`:

```bash
uv sync                                   # create/refresh the venv from uv.lock
PYTHONPATH=src uv run python -m cli ...    # run the agent inside that venv
```

`uv run` executes inside the project's managed `.venv` (no manual
`source .venv/bin/activate` needed), resolving dependencies from `uv.lock`.
Set the model env vars above first if you want to override the defaults.

### Run modes

| Mode | Command | Notes |
| --- | --- | --- |
| Inline problem | `PYTHONPATH=src uv run python -m cli --problem "Return indices of the two numbers adding to target"` | Quickest one-shot; quote the whole statement. |
| From a file | `PYTHONPATH=src uv run python -m cli --file problem.md` | Best for long statements / Markdown. |
| Interactive prompt | `PYTHONPATH=src uv run python -m cli` | No args on a TTY; type the problem, blank line to finish. |
| Piped stdin | `cat problem.md \| PYTHONPATH=src uv run python -m cli` | Reads all of stdin as the requirement. |
| Pin the run id | `... --run-id run_id_demo` | Fixes the output filename → `output/run_id_demo.json`. |
| Status lines only | `... --quiet` | No code/test/diagnosis panels; per-node status lines only. |
| Full dump | `... --verbose` | Also prints the final code, tests and holdout at the end. |
| Headless gather | `... --gather-mode quiet` | Validate once, never prompt (auto-default when piped). |
| Interactive gather | `... --gather-mode interactive` | Force the confirm-and-correct gate even when piped. |
| Custom output root | `... --out-dir runs/` | Writes `runs/<run_id>.json` instead of `output/`. |
| Batch dataset | `PYTHONPATH=src uv run python -m cli --problems-file problems.jsonl` | One `{"id","requirement"}` per line; implies quiet; see *Batch dataset*. |
| Help | `PYTHONPATH=src uv run python -m cli --help` | All flags. |

Flags combine freely — e.g. a pinned, headless, verbose run from a file:

```bash
PYTHONPATH=src uv run python -m cli --file problem.md \
  --run-id run_id_demo --gather-mode quiet --verbose
```

Each run streams a status line per node (`✓ intent clear`, `✓ plan`, `✓ tests`,
`✓ code`, `✓ eval 5/5`, `↻ eval 3/5 → retry`, `✓ wrote ...`, with `↻` retries and
`⚠` blocks) **plus live panels**: the plan + contract (signature/args/returns)
after the planner, the generated code on every coder attempt, the test cases
(with the coder-hidden holdout labelled), per-test input/output result tables
after the evaluator, and the evaluator's diagnosis panel (culprit: code vs tests
— plus the reason). On an interactive terminal a spinner with
rotating, node-aware phrases and a live seconds timer (`planner: thinking…  (14s)`,
`coder: writing the solution…  (58s)`) sits at the bottom while each node runs
(LLM calls block for 90-180s), and every status line ends with that node's
duration (`✓ plan (3 steps) · 12.4s`). It writes `output/<run_id>.json` with
requirement, clarified intent, plan, test cases (plus holdout), code, sandbox
results, eval report, failure diagnosis, and guardrail verdict.
`--quiet` (status lines only) and `--verbose` (full final code/tests) are in the
table above.

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
PYTHONPATH=src uv run pytest -q   # 103 tests (hermetic fallbacks + live-graph E2E)
```

## Layout

- `src/agent/{state,llms,prompts,nodes,graph}.py` — state, models, prompts, nodes, graph
- `src/agent/tools/{sandbox,guardrail}.py` — gated `python_sandbox` tool + AST security gate
- `src/cli.py` — terminal entrypoint
- `plan/{architecture,plans,plan,guardrails}.md` — design docs
