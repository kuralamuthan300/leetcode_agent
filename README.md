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
  testgen:   {name: nemotron-3-super, temperature: 0.0, max_tokens: 8192, timeout_s: 180}
  evaluator: {name: nemotron-3-ultra, temperature: 0.0, max_tokens: 4096, timeout_s: 240}

agent:
  max_intent_iterations: 3
  max_tests: 10
  tests_per_category: 2
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
| `MLFLOW_MAX_INTENT_ITERATIONS` / `MLFLOW_MAX_TESTS` / `MLFLOW_TESTS_PER_CATEGORY` / `MLFLOW_HOLDOUT_MIN_TESTS` / `MLFLOW_MAX_PLANNER_RETRIES` | graph knobs |
| `MLFLOW_AGENT_CONFIG` | path to this YAML file |
| `MLFLOW_TRACING_DISABLED=1` | turn tracing off |
| `MLFLOW_EXPERIMENT_ID` | use an existing experiment (skips `set_experiment`) |
| `MLFLOW_EVAL_EXPERIMENT` / `MLFLOW_EVAL_DATASET` | evaluation experiment and dataset name |
| `MLFLOW_EVAL_LOG_FEEDBACK` / `MLFLOW_EVAL_LLM_JUDGE` | enable trace feedback / optional judge |
| `MLFLOW_EVAL_BASELINE_RUN_ID` | baseline run used for comparison |

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

A run trace is made readable on purpose:

- The trace is **named** `<run_id> · <problem snippet>` (via the reserved
  `mlflow.traceName` tag) instead of `LangGraph`, and carries searchable tags
  `run_id`, `status`, `gather_mode`, `eval` (e.g. `4/4`).
- LLM spans are named by **role** — `model:heavy`, `model:small`,
  `model:critic`, `model:testgen`, `model:evaluator` — so identical
  `ChatOpenAI` calls are no longer ambiguous.
- The CLI prints a deep link after every run:
  `trace: http://127.0.0.1:5000/#/experiments/<id>/traces/<trace_id>`.

Find a specific run:

```bash
# in the MLflow UI search bar, or via the SDK:
mlflow.search_traces(filter_string="tags.run_id = 'run_id_demo'")
```

**Gateway trace volume.** MLflow gateway endpoints default to
`usage_tracking: true`, which writes one trace per proxied LLM call into
`gateway/<endpoint>` experiments (`gateway/nemotron-3-super`, `gateway/gemma4`,
`gateway/nemotron-3-ultra`). Those grow fast and drown out agent traces. To
quiet them: MLflow UI → **Gateway** → open the endpoint → turn **Usage
tracking** off (repeat for all three endpoints) and delete the stale
`gateway/*` experiments. Nothing is lost for agent debugging — the agent's
`model:<role>` spans already record model, token usage, and cost. Env overrides:
`MLFLOW_TRACING_DISABLED=1` or `--no-tracing` disable tracing entirely.

## Evals

The evaluation tooling keeps the agent's internal verdict separate from the
deterministic golden verdict. Start by validating and normalizing
`evals/golden_dataset.csv`:

```bash
PYTHONPATH=src uv run python -m evals.cli prepare
PYTHONPATH=src uv run python -m evals.cli prepare --sync-mlflow
```

Preparation writes `evals/problems.jsonl` for the existing batch driver and
prints the SHA-256 digest. `--sync-mlflow` registers the snapshot as
`leetcode-golden-v1` in the `leetcodeagent-evals` experiment. Repeated syncs with
the same digest are idempotent; `--reset --sync-mlflow` intentionally recreates
the dataset.

Run a selected subset (or the full set) sequentially; repeat runs are indexed
separately for repeatability checks:

```bash
PYTHONPATH=src uv run python -m evals.cli run --subset 1,4 --repeat 1
PYTHONPATH=src uv run python -m evals.cli run --subset 1,4,8 --repeat 3 --no-mlflow
PYTHONPATH=src uv run python -m evals.cli grade --batch evals/results/<batch_id>
```

Runs and their `manifest.json` / `run_index.json` are written under
`evals/results/<batch_id>/`. `--no-mlflow` keeps local execution independent of
the tracking server. `grade` scores each stored solution against the three
golden cases and writes `grade.json`; `report --batch <dir>` writes scorecard
artifacts from it. Unavailable metrics are shown as such, never as passing
gates. See [`evals/README.md`](evals/README.md) for dataset and constraint
conventions.

The full 50-problem run is sequential and LLM-bound; expect hours rather than
minutes depending on gateway latency. Cost depends on model pricing and token
usage, so inspect trace usage/cost in MLflow rather than assuming a fixed
estimate. The scorecard distinguishes:

- **Golden strict solve rate:** problems passing all three golden cases / problems.
- **Case pass rate:** golden cases passed / golden cases evaluated, with base,
  edge, and adversarial slices.
- **False-pass rate:** internal passes that fail golden / internal passes.
- **False-fail rate:** internal failures that pass golden / internal failures.
- **Mutation, oracle, blame, constraint, infra/flaky, latency, and cost metrics:**
  see the metric contract in `src/evals/metrics_spec.py`; complexity remains an
  advisory, not a hard gate.

The evaluation MLflow run belongs to `leetcodeagent-evals`. Its artifacts include
`scorecard.md`, `metrics.json`, `per_problem.csv`, and `failures.md`; use the
per-row assessments and trace feedback to identify the individual solve. Trace
tags include `run_id`, and false-pass feedback can be queried by its assessment
name. Compare a later report with `--baseline <run_id>` to emit metric deltas and
paired strict-pass regressions.

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
| No tracing | `... --no-tracing` | Skip MLflow tracing for this run (same as `MLFLOW_TRACING_DISABLED=1`). |
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
(8 total — 2 corner, 2 easy, 2 medium, 2 hard — with one hard test held out and
labelled), per-test input/output result tables
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

The golden evaluation CSV can be validated and converted to that JSONL format
with:

```bash
PYTHONPATH=src uv run python -m evals.cli prepare
```

The command writes `evals/problems.jsonl` and prints the source CSV's SHA-256
digest. Add `--sync-mlflow` to create or update the `leetcode-golden-v1` GenAI
dataset in the `leetcodeagent-evals` experiment; the same CSV digest is a no-op
on subsequent runs. Use `--reset --sync-mlflow` only when intentionally
recreating that MLflow dataset.

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
