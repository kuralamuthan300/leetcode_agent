# Evaluation dataset and scorecard

The golden CSV is the human-edited source of truth. `evals.cli prepare`
validates 50 unique problem IDs, the three required golden case types, JSON
payloads, constraints, complexity targets, and the declared Python function
name. It also emits `problems.jsonl` in the shape consumed by the existing
headless batch CLI.

## Dataset conventions

- Every row must declare a function using `Implement function_name(...)`.
- Each case payload is JSON. Cases may be separate CSV columns or a single
  `cases` object with exactly `base_case`, `edge_case`, and `adversarial_case`.
- A case object includes its `input` (list for positional arguments or mapping
  for keyword arguments) and `expected` output.
- Linked-list and tree representations remain list encodings in v1.
- Use `prepare --sync-mlflow` to version the dataset snapshot. The CSV byte
  digest guards idempotency; `--reset` is reserved for intentional recreation.

## References and adding problems

Trusted reference solutions belong in `src/evals/reference_solutions.py`, keyed
by the canonical dataset ID. A reference accepts the same list-based input
contract as the problem statement and returns the expected public output.
Before a dataset change is evaluated, every reference must agree with all three
golden cases; a mismatch is a dataset/reference defect to investigate, not a row
to silently omit.

Add a CSV row only after adding its reference, validating its three expected
outputs, and recording any deterministic constraint rules. Keep the statement's
function name and data encoding explicit.

## Constraint rules

`constraints.yml` maps string problem IDs to rule lists. Supported static kinds
are `ast_forbid` and `ast_require`; selectors currently include `operator:Div`,
`name:<identifier>`, and `matrix_write:<identifier>`. Rules should be narrow,
explainable, and backed by a test showing both a complying and violating
solution. Complexity estimates are advisory and must not block a run.

## Running and reporting

```bash
PYTHONPATH=src uv run python -m evals.cli prepare
PYTHONPATH=src uv run python -m evals.cli run --subset 1,4,8 --repeat 1 --no-mlflow
PYTHONPATH=src uv run python -m evals.cli grade --batch evals/results/<batch_id>
PYTHONPATH=src uv run python -m evals.cli report --batch evals/results/<batch_id> --no-mlflow
```

Agent runs are sequential. `run_index.json` records per-problem/repeat paths,
durations, statuses, and trace IDs (null when tracing is disabled/unavailable).
The `grade` stage reads those outputs, grades each final solution against the
three golden cases, and writes `grade.json` (metrics, per-problem rows,
failures, false passes, constraint and complexity checks). Metrics that need a
trusted reference solution (mutation, test oracle, blame) stay `unavailable`
until `src/evals/reference_solutions.py` provides one — they are never reported
as passing. The report stage consumes `grade.json` and writes `scorecard.md`,
`metrics.json`, `per_problem.csv`, and `failures.md`. Metrics
distinguish internal agent verdicts from golden grades; the false-pass rate is
conditional on internal passes.

For MLflow, configure `evals.experiment`, `evals.dataset_name`, and feedback
settings in `config.yml` or the `MLFLOW_EVAL_*` environment variables. Dataset
sync and reporting need an MLflow 3 tracking backend with GenAI dataset support.
Local reports remain available when tracking is disabled.

The full 50-row run is sequential and may take hours depending on gateway
latency. Token cost is model-dependent; rely on trace token/cost data or an
explicit pricing table, not a fixed estimate. Metric formulas and gates are
defined centrally in `src/evals/metrics_spec.py`. Do not treat the agent's
internal verdict as the golden verdict: false-pass and false-fail are computed
by comparing the two independently.

Before merging agent changes, validate/prepare the source data and run a small
subset when a permitted dataset is available. Reserve full and repeated runs for
manual/nightly evaluation. Never turn off the solution-code guardrail to make a
golden case pass; trusted harness mode only relaxes harness-side checks
(path-looking strings and the 5k size budget that dataset literals exceed).
