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
Models (MLflow AI gateway): heavy `nemotron-3-super` (plan/code), testgen
`nemotron-3-super` (plan tests), small `gemma4` (clarify), critic
`nemotron-3-super` (advisory review), evaluator `nemotron-3-ultra` (blame +
diagnosis; used **only** by the evaluator). Every call goes through the
OpenAI-compatible gateway and is traced to MLflow.

## Prerequisites

- Python ≥ 3.11, [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- An MLflow server with the AI gateway enabled, exposing the endpoints above
  (default: `http://127.0.0.1:5000/gateway/mlflow/v1`)

## Setup

```bash
uv sync
# gateway endpoints must exist on the MLflow server, named:
#   gemma4, nemotron-3-super, nemotron-3-ultra
curl -s http://127.0.0.1:5000/gateway/mlflow/v1/models
```

All model, gateway and graph settings live in [`config.yml`](config.yml); the
file is read from the repo root and can be moved with `MLFLOW_AGENT_CONFIG`.

```yaml
gateway:
  base_url: http://127.0.0.1:5000/gateway/mlflow/v1
  api_key: ""                       # gateway handles auth server-side
  tracking_uri: http://127.0.0.1:5000
  experiment: leetcodeagent

models:
  heavy:     {name: nemotron-3-super, temperature: 0.2, max_tokens: 4096, timeout_s: 180}
  small:     {name: gemma4,           temperature: 0.0, max_tokens: 4096, timeout_s: 90}
  critic:    {name: nemotron-3-super, temperature: 0.0, max_tokens: 1024, timeout_s: 60}
  testgen:   {name: nemotron-3-super, temperature: 0.0, max_tokens: 4096, timeout_s: 180}
  evaluator: {name: nemotron-3-ultra, temperature: 0.0, max_tokens: 4096, timeout_s: 240}

agent:
  max_intent_iterations: 3
  max_tests: 10
  holdout_min_tests: 3
  max_planner_retries: 1
```

Environment variables override the file (highest precedence):

| Env var | Overrides |
| --- | --- |
| `MLFLOW_GATEWAY_URL` | `gateway.base_url` |
| `MLFLOW_GATEWAY_API_KEY` | `gateway.api_key` |
| `MLFLOW_TRACKING_URI` | `gateway.tracking_uri` (standard MLflow var) |
| `MLFLOW_EXPERIMENT_NAME` | `gateway.experiment` |
| `MLFLOW_HEAVY` / `MLFLOW_SMALL` / `MLFLOW_CRITIC` / `MLFLOW_TESTGEN` / `MLFLOW_EVALUATOR` | role model (endpoint) name |
| `MLFLOW_<ROLE>_TEMPERATURE` / `_MAX_TOKENS` / `_TIMEOUT` | role knobs, e.g. `MLFLOW_HEAVY_MAX_TOKENS` |
| `MLFLOW_MAX_INTENT_ITERATIONS` / `MLFLOW_MAX_TESTS` / `MLFLOW_HOLDOUT_MIN_TESTS` / `MLFLOW_MAX_PLANNER_RETRIES` | graph knobs |
| `MLFLOW_AGENT_CONFIG` | path to this YAML file |
| `MLFLOW_TRACING_DISABLED=1` | turn tracing off |
| `MLFLOW_EXPERIMENT_ID` | use an existing experiment (skips `set_experiment`) |

Token budgets matter: reasoning tokens count against `max_tokens` on the
gateway, so a budget that is too small yields empty `content` (logged as
`empty <role> reply (finish_reason=length)`).

### Tracing

`cli.main()` calls `agent.tracing.init_tracing()`, which points MLflow at
`gateway.tracking_uri`, selects the `gateway.experiment` (default
`leetcodeagent`), and enables `mlflow.langchain.autolog(log_traces=True)`.
LangGraph nodes and every `ChatOpenAI` call are captured automatically — no
per-node decorators. Traces are visible in the MLflow UI at
`http://127.0.0.1:5000` under the configured experiment. Tracing is
best-effort: if the tracking server is down the solve continues with a warning.

## Run

The CLI lives in `src/` (it is not an installed package), so every invocation is
run through [`uv`](https://docs.astral.sh/uv/) with `PYTHONPATH=src`:

```bash
uv sync                                   # create/refresh the venv from uv.lock
PYTHONPATH=src uv run python -m cli ...    # run the agent inside that venv
```

`uv run` executes inside the project's managed `.venv` (no manual
`source .venv/bin/activate` needed), resolving dependencies from `uv.lock`.
Edit `config.yml` (or set the env vars above) to change models or endpoints.

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
PYTHONPATH=src uv run pytest -q   # 117 tests (hermetic fallbacks + live-graph E2E)
```

## Layout

- `config.yml` — gateway, per-role models/knobs, graph limits
- `src/agent/{state,llms,prompts,nodes,graph,config,tracing}.py` — state, model factory, config loader, prompts, nodes, graph, tracing
- `src/agent/tools/{sandbox,guardrail}.py` — gated `python_sandbox` tool + AST security gate
- `src/cli.py` — terminal entrypoint
- `plan/{architecture,plans,plan,guardrails}.md` — design docs
