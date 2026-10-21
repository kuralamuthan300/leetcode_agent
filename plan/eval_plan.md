# Eval Framework v1 — Implementation Plan

**Goal:** turn `evals/golden_dataset.csv` (50 problems × base/edge/adversarial = 150 cases)
into a repeatable scorecard that measures the agent's *outcome*, its *self-grading
calibration*, its *test-generation strength*, and its *operating cost* — with
**MLflow as the system of record** (dataset versioning, traces, per-row assessments,
metrics, artifacts, and run-to-run comparison).

**Design rule:** the agent's own `eval` verdict and the golden verdict are two
different measurements. Never collapse them.

**MLflow rule:** deterministic code execution and grading stay in our sandbox;
MLflow owns dataset versions, per-row scoring/assessments, aggregation, artifacts,
and comparison. `mlflow.genai.evaluate` orchestrates scoring over **precomputed**
outputs so it never re-runs the agent or untrusted code.

---

## 1. Recommended v1 scorecard (the contract)

These 10 metrics are the deliverable. Every later step exists to compute one of them.

| # | Metric | Spec id | Formula | Hard gate (initial) |
|---|---|---|---|---|
| 1 | **Golden Strict Solve Rate** | `golden_strict_solve_rate` | problems with all 3 golden cases passing / 50 | baseline − 5 pts, target ≥ 0.70 |
| 2 | **Golden Case Pass Rate** | `golden_case_pass_rate` | passing cases / 150, split base / edge / adversarial | report; adversarial ≥ 0.60 |
| 3 | **False-Pass Rate** | `false_pass_rate` | internal `eval.passed` but golden fails / internal passes | ≤ 0.05 |
| 4 | **False-Fail Rate + Verdict κ** | `false_fail_rate`, `cohens_kappa` | internal fails but golden passes / internal fails; Cohen's κ | ≤ 0.10; κ ≥ 0.70 |
| 5 | **Mutation Score** | `mutation_score` | mutants killed / killable mutants | ≥ 0.60 |
| 6 | **Test Oracle Correctness** | `oracle_correctness` | generated tests whose expected output matches reference / all generated tests | ≥ 0.95 |
| 7 | **Blame Accuracy** | `blame_accuracy` | correct `culprit` (code/test) / failures with derivable ground truth | ≥ 0.80 |
| 8 | **Constraint Adherence** | `constraint_adherence` | satisfied per-problem constraint rules / applicable rules (+ complexity advisory) | ≥ 0.90, advisory in v1 |
| 9 | **Infra Error Rate + Flaky Rate** | `infra_error_rate`, `flaky_rate` | `infra_error` / runs; verdict-changing problems / repeated problems | ≤ 0.02; ≤ 0.05 |
| 10 | **p95 Latency + Cost per Solved Problem** | `latency_p95`, `cost_per_solved` | p95 wall-clock/run; total LLM cost ÷ strict passes | ≤ 360 s; report-only |

The canonical contract is `src/evals/metrics_spec.py::METRICS`; the table above is
its human-readable view. Supporting numbers tracked beside these rows (all with
`primary=False` in `METRICS`) are `pass_at_1`, `pass_at_2`,
`category_balanced_solve_rate`, `golden_base_case_pass_rate`,
`golden_edge_case_pass_rate`, `golden_adversarial_case_pass_rate`,
`false_pass_share`, `verdict_agreement`, `internal_pass_rate`,
`mutation_score_macro`, `oracle_correctness_agent_tests`,
`oracle_correctness_holdout_tests`, `complexity_match_rate`, `latency_p50`, and
`cost_per_solved`. Each spec carries its numerator, denominator, direction, gate,
blocking flag, required artifacts, MLflow metric key, MLflow scorer/feedback
names, scope (per-problem vs run-level), and scorer aggregation. A metric cannot
be renamed or redefined in `report.py`/`scorers.py` without changing the spec.

**Statistics:** n=50 gives roughly ±14 pts at 95% confidence around p=0.5. Report
Wilson intervals and paired comparisons (McNemar) for config changes; do not chase
sub-5-point movements. Treat the 150 cases as clustered by problem, not independent.

**Slices for every metric:** category, complexity class, case type, assumption
status, and `iterations` (1 vs 2).

---

## 2. MLflow integration design

MLflow **3.17.1** is already installed, and the agent already emits
`mlflow.langchain.autolog` traces tagged with `run_id`. The eval framework adds a
second, versioned layer on top; it does not replace the agent's tracing.

### 2.1 Objects and run hierarchy

| Object | Name | Purpose |
|---|---|---|
| Experiment (existing) | `leetcodeagent` | Agent traces, one trace per solve |
| Experiment (new) | `leetcodeagent-evals` | Scorecard runs, datasets, assessments |
| GenAI dataset | `leetcode-golden-v1` | 50 records, versioned by digest |
| Parent eval run | `eval_<batch_id>` | Params, 10 metrics, gates, artifacts |
| Nested stage runs | `prepare` / `grade` / `mutation` / `report` | Optional isolation of stage failures |
| Trace (existing) | `<run_id> · <problem>` | Receives per-problem feedback assessments |

**Parent run tags:**

```text
eval.kind=scorecard
eval.batch_id=<batch_id>
eval.dataset_id=<id>
eval.dataset_version=<n>
eval.dataset_hash=<sha256>
eval.git_sha=<sha>
eval.config_hash=<sha256>
eval.models=heavy:nemotron-3-super,testgen:...,evaluator:...
eval.repeat=1
eval.subset=full|stratified-10
eval.baseline_run_id=<id or empty>
gate.status=pass|fail
```

**Params:** the same values plus the exact command line, Python version, and
`pricing.yml` hash. **Artifacts:** `scorecard.md`, `metrics.json`,
`per_problem.csv`, `failures.md`, `oracle.json`, `mutation.json`, `manifest.json`.

Keep the parent run as the only place gates are evaluated. Nested runs are for
debuggability, not for rolling up metrics.

### 2.2 Dataset registration and versioning

`prepare --sync-mlflow` maps each CSV row to a GenAI dataset record:

```python
{
    "inputs": {
        "problem_id": 1,
        "requirement": "<Problem Statement + Constraints + Complexity>",
        "function_name": "two_sum",
    },
    "expectations": {
        "cases": {"base_case": {...}, "edge_case": {...}, "adversarial_case": {...}},
        "complexity": "Time: O(N), Space: O(N)",
        "constraints": "...",
        "category": "Arrays / Hash Map",
    },
    "tags": {"golden_dataset_hash": "<sha256>"},
}
```

- Create once with `mlflow.genai.datasets.create_dataset(name="leetcode-golden-v1",
  experiment_id=<evals experiment>)`; append with `merge_records`.
- **Idempotency:** compare the dataset digest / recorded `golden_dataset_hash` before
  merging. `--reset` deletes and recreates for intentional dataset changes.
- Log `dataset_id`, `dataset_version`, and `dataset_hash` on every eval run.
- The CSV remains the human-editable source; the MLflow dataset is the evaluation
  snapshot. Never edit the dataset record by hand.

### 2.3 Scorer mapping (`mlflow.genai.evaluate`)

Grading runs **before** `evaluate`; the evaluation DataFrame carries precomputed
outputs and grades. This avoids re-running the agent, re-executing untrusted code
inside MLflow, and double-counting LLM cost.

| # | Scorecard metric | MLflow scorer name | Implementation | Aggregation |
|---|---|---|---|---|
| 1 | Golden Strict Solve | `golden_strict_pass` | deterministic, reads precomputed grade | mean |
| 2 | Case Pass Rate | `golden_case_pass_rate` | deterministic | mean |
| 3 | False-Pass | `false_pass` | deterministic (internal vs golden) | mean |
| 4 | False-Fail | `false_fail` | deterministic | mean |
| 5 | Mutation Score | `mutation_score` | deterministic (mutant matrix) | mean |
| 6 | Oracle Correctness | `oracle_correctness` | deterministic | mean |
| 7 | Blame Accuracy | `blame_accuracy` | deterministic | mean |
| 8 | Constraint Adherence | `constraint_adherence` | deterministic + optional judge | mean |
| 9 | Infra/Flaky | run-level metrics | `mlflow.log_metrics` | n/a |
| 10 | Latency/Cost | trace-derived metrics | `mlflow.log_metrics` | p50/p95/sum |

Example shape:

```python
import mlflow
from mlflow.genai.scorers import scorer

@scorer(name="golden_strict_pass", aggregations=["mean"])
def golden_strict_pass(inputs, outputs, expectations):
    return bool(outputs["grade"]["strict_pass"])

@scorer(name="false_pass", aggregations=["mean"])
def false_pass(inputs, outputs, expectations):
    return bool(outputs["internal_passed"] and not outputs["grade"]["strict_pass"])

result = mlflow.genai.evaluate(
    data=per_problem_df,          # inputs, outputs, expectations
    scorers=[golden_strict_pass, false_pass, ...],
)
```

- `result.metrics` is a cross-check; `metrics.json` from `report.py` remains the
  canonical scorecard with explicit denominators and Wilson CIs.
- Optional LLM judge (not a v1 gate): `mlflow.genai.make_judge(...)` or built-in
  `Correctness` for "is this solution correct?" as a second opinion. Enable with
  `--llm-judge`; log cost separately.

### 2.4 Per-problem feedback on traces

At grade time, map `output["run_id"]` → `trace_id` with
`mlflow.search_traces(experiment_ids=[<agent experiment id>],
filter_string="tags.run_id = '<run_id>'")` and attach one assessment per problem.
The agent traces live in `leetcodeagent`, while the eval run lives in
`leetcodeagent-evals`; always pass the agent experiment ID explicitly.

```python
mlflow.log_feedback(
    trace_id=trace_id,
    name="golden_strict_pass",
    value=False,
    rationale="adversarial case failed: expected 500500, got 499500",
    metadata={"problem_id": 9, "case": "adversarial", "batch_id": batch_id},
)
```

Also log `internal_pass`, `false_pass`, `oracle_correctness`, `blame_accuracy`,
`mutation_score`, and `constraint_adherence` as feedback where meaningful. This
makes failures filterable in the trace UI and turns every solve into a labeled
example for later judge/optimizer work. If tracing is off or the trace is missing,
skip feedback and record `trace_id: null` in `run_index.json`.

### 2.5 Metrics, tables, and gates

- `report.py` writes `metrics.json` / `scorecard.md` locally, then:
  - `mlflow.log_metrics({...})` for the 10 canonical metrics and each slice;
  - `mlflow.log_table(per_problem_df, artifact_file="per_problem.json")`;
  - `mlflow.log_table(slice_df, artifact_file="slices.json")`;
  - `mlflow.log_artifact(...)` for the four report files.
- Gate results are logged twice: metric `gate.<id>` = `0|1`, and run tag
  `gate.status=pass|fail` (so `search_runs` can filter).
- Log the `mlflow.genai.evaluate` result table as `eval_result.json`.

### 2.6 Latency and cost from traces

MLflow 3 exposes aggregated usage directly on `TraceInfo`:

- `trace.info.execution_time_ms` — wall-clock duration.
- `trace.info.token_usage` — `{input_tokens, output_tokens, total_tokens}`.
- `trace.info.cost` — `{input_cost, output_cost, total_cost}` (USD).

Use these first. If `token_usage`/`cost` are absent (provider/gateway does not
report usage), fall back to `evals/pricing.yml` and token counts; if neither is
available, mark the metric `unavailable` — never log `0`.

### 2.7 Configuration and offline mode

Add an `evals` block to `config.yml`:

```yaml
evals:
  experiment: leetcodeagent-evals
  dataset_name: leetcode-golden-v1
  log_feedback: true
  llm_judge: false
  baseline_run_id: ""
```

**Required code change:** `AgentConfig` in `src/agent/config.py` uses
`extra="forbid"`, so adding this YAML block without a model will make every agent
run fail validation. Add an `EvalsConfig` model (`extra="forbid"`), include it in
`AgentConfig`, `default_config()` and `_apply_env()`, then add the env overrides
(`MLFLOW_EVAL_EXPERIMENT`, `MLFLOW_EVAL_DATASET`, `MLFLOW_EVAL_LOG_FEEDBACK`,
`MLFLOW_EVAL_LLM_JUDGE`, `MLFLOW_EVAL_BASELINE_RUN_ID`) in the same style as the
existing gateway/model knobs. Add a config test that an unknown `evals` key still
fails loudly.

Environment overrides follow the existing `MLFLOW_*` convention.

Every MLflow call is wrapped in a best-effort adapter (`src/evals/mlflow_eval.py`)
with a single `mlflow_enabled` switch. `--no-mlflow` and an unreachable tracking
server must still produce the local scorecard; the report marks MLflow fields
`unavailable` and exits successfully (gate failures still exit `1`).

### 2.8 Division of responsibility

| Concern | Owner |
|---|---|
| Executing solution/test code | `src/agent/tools/sandbox.py` + `src/evals/runner.py` |
| Golden/adversarial grading | `src/evals/golden.py` |
| Per-row scoring, assessments, aggregation | `mlflow.genai.evaluate` + custom scorers |
| Dataset versioning | `mlflow.genai.datasets` |
| Trace-level feedback | `mlflow.log_feedback` |
| Metrics, tables, artifacts, comparison | MLflow run + `report.py` |
| Gate policy | `evals/gates.yml` + `report.py` |

---

## 3. Target layout

```
src/evals/
  __init__.py
  schema.py               # Problem, GoldenCase, RunRecord, GradeRecord, MetricResult
  dataset.py              # CSV load, validation, problems.jsonl, dataset hash
  adapters.py             # fn name, input/output transforms, comparison
  harness.py              # golden harness source generation + stdout parsing
  runner.py               # sandboxed runs, trusted-harness policy
  golden.py               # grade one code artifact against 3 cases
  reference_solutions.py  # 50 reference implementations (oracle)
  oracle.py               # oracle correctness + blame ground truth
  mutants.py              # AST mutants
  mutation.py             # mutation score
  constraints.py          # constraint + complexity advisory checks
  metrics.py              # scorecard math + CIs
  ops.py                  # infra/flaky/latency/cost from artifacts + traces
  mlflow_eval.py          # MLflow adapter: dataset sync, evaluate, feedback, logging
  scorers.py              # @scorer functions over precomputed outputs
  report.py               # scorecard.md / metrics.json / per_problem.csv
  cli.py                  # prepare | run | grade | report | all
evals/
  golden_dataset.csv
  constraints.yml
  pricing.yml
  gates.yml               # calibrated thresholds + hard/soft flags
  problems.jsonl          # generated, committed (small)
  results/<batch_id>/     # generated, gitignored
tests/
  test_eval_metrics_spec.py
  test_eval_dataset.py
  test_eval_harness.py
  test_eval_metrics.py
  test_eval_mutation.py
  test_eval_report.py
  test_eval_mlflow.py     # mocked tracking server / offline mode
```

`config.yml` gains the `evals:` block shown in §2.7. MLflow is optional at runtime
(`--no-mlflow`), but required for the full scorecard experience.

Run everything with the project's existing convention:

```bash
PYTHONPATH=src uv run python -m evals.cli all
```

---

## 4. Steps

### Step 1 — Freeze the metric contract

Create `src/evals/metrics_spec.py` (or a JSON/YAML file) defining, for every
scorecard metric:

- stable `id`, display name, tier (outcome / calibration / tests / ops);
- numerator and denominator in words;
- direction (`higher_is_better` / `lower_is_better`);
- gate value and whether it blocks;
- required artifacts (run JSON, grade JSON, reference);
- `mlflow_metric_key` (e.g. `golden_strict_solve_rate`), `mlflow_feedback_name`
  (e.g. `golden_strict_pass`), and whether it is a per-problem scorer or a
  run-level metric;
- scorer aggregation (`mean`, `min`, `p90`, or none).

Add a `METRICS` dict consumed by `report.py` and `scorers.py`, so a metric cannot
be renamed or redefined in one place without the other.

**Deliverable:** `src/evals/metrics_spec.py` + `plan/eval_plan.md` table above kept
in sync. The registry exports `METRICS` (all numbers), `PRIMARY_METRICS`/
`CANONICAL_IDS` (the ten scorecard rows), `SCORERS` and `FEEDBACKS` (per-problem
scorer/feedback mappings), `blocking_metrics()` (all hard gates, including the
secondary numbers in rows 4 and 9), and helpers for `report.py`/`scorers.py`.
Contract tests live in `tests/test_eval_metrics_spec.py`.

**Done when:** every scorecard number in the final report is traceable to one spec
entry with an explicit denominator, and every per-problem metric has a matching
`@scorer` name and trace feedback name.

---

### Step 2 — Normalize the dataset and build the adapter layer

`src/evals/dataset.py`:

- Read the CSV with `csv.field_size_limit(sys.maxsize)` (the adversarial cases
  exceed Python's default 128 KiB field limit — a plain `DictReader` fails today).
- Validate: 50 unique IDs, exactly `base_case` / `edge_case` / `adversarial_case`,
  JSON-parsable test payloads, non-empty constraints, parsable complexity target.
- Extract the function name with `re.search(r"Implement\s+([A-Za-z_]\w*)\s*\(", statement)`.
  All 50 currently resolve correctly (verified), e.g. `two_sum`, `my_atoi`,
  `detect_cycle`, `find_cheapest_price`.
- Emit `evals/problems.jsonl` for the existing CLI batch mode:
  `{"id": "<ID>", "requirement": "<Problem Statement>\nConstraints: <...>\nComplexity: <...>"}`.
  The requirement must contain the literal `Implement fn(...)` line and the
  encoding convention (linked lists / trees are lists) so the agent's planner
  contract matches the grader.
- Record SHA-256 of the CSV in the run manifest.
- `prepare --sync-mlflow` builds the 50 GenAI dataset records (§2.2) and
  creates/updates `leetcode-golden-v1` idempotently by digest. Log the returned
  `dataset_id`, `dataset_version`, and `dataset_hash`; `--reset` recreates it.

`src/evals/adapters.py`:

- Default adapter is identity. The statements already declare list encodings for
  linked lists and LeetCode level-order lists for trees, so no structure builder
  is required in v1.
- Keep an override registry keyed by problem ID for future cases that need
  `ListNode` / `TreeNode` construction.
- Comparison: parse actual with `ast.literal_eval`; compare with `==`; allow
  `int`/`float` equivalence and a small float tolerance for `find_median_sorted_arrays`.
- Normalize booleans (`True`/`False`) and empty-string output.

**Deliverable:** validated `evals/problems.jsonl`, dataset hash, adapter registry,
and (with `--sync-mlflow`) a versioned `leetcode-golden-v1` dataset.

**Done when:** all 50 rows load, every case produces a valid harness source, a
validation failure names the row and field, and a second `--sync-mlflow` run does
not create a duplicate dataset version.

---

### Step 3 — Reference solutions and golden-oracle validation

`src/evals/reference_solutions.py` with `REFERENCE: dict[int, callable]`, one
implementation per problem, matching the stated contract (lists in/out, in-place
semantics where stated).

This step is the largest content task (~50 small functions). Do it in category
batches (arrays → strings → bits → stacks → lists → matrices → search → trees →
graphs → DP) and review each batch.

`oracle.py` runs every reference against its three golden cases. Any mismatch is a
dataset bug, a reference bug, or an ambiguous statement — resolve it, never drop it.

**Why it is mandatory:** metrics #5 (mutation), #6 (oracle correctness), and #7
(blame ground truth) are impossible without a trusted correct implementation.

**Done when:** `150/150` golden cases pass against the references and
`evals/results/<batch>/oracle.json` records the result. This also validates the
dataset itself.

---

### Step 4 — Sandboxed golden grader

`src/evals/harness.py` builds, for each case:

```python
print("__RESULT__", two_sum(nums=[2, 7, 11, 15], target=9))
```

Constraints imposed by the existing sandbox/guardrail (`src/agent/tools/sandbox.py`,
`src/agent/tools/guardrail.py`):

- `json` is not an allowed import and `repr` is not in `_SAFE_BUILTIN_NAMES`, so
  harnesses must use `print(...)` + host-side `ast.literal_eval`.
- The guardrail rejects string constants that look like paths (`/`, `~`, `..`).
  Random string inputs (e.g. problem 4/5) can trip this. Use a **trusted harness**
  path for golden data: solution code stays fully gated (imports/names/dunders),
  while the path-like-string check is disabled for the harness only. Put this in
  `src/evals/runner.py` with a clear comment; do not weaken the production gate.
- Default sandbox timeout is 5 s, too low for adversarial cases. Grade with
  `timeout_s=30`; a timeout is a failed case with `timed_out=True`, not a crash.

`golden.py` returns per case: `{case_type, passed, actual, expected, error, timed_out,
guardrail_blocked}`.

**Done when:**
- references pass `150/150` through this grader;
- a deliberately buggy solution fails the expected cases;
- a timeout is reported as a failure;
- a guardrail-blocked solution is reported as blocked, not as a Python error.

---

### Step 5 — Batch run harness

`src/evals/runner.py` orchestrates the existing batch mode:

```bash
PYTHONPATH=src uv run python -m cli \
  --problems-file evals/problems.jsonl \
  --out-dir output/evals --quiet
```

- Do **not** create the eval run here; the batch produces agent traces only. The
  parent `eval_<batch_id>` run is created later by `report` (Step 12), after all
  grading/mutation metrics exist, so the scorecard lands in one run.
- Write `manifest.json`: dataset id/version/hash, git SHA, `config.yml` hash, model
  roles, repeat count, subset, timestamp, command.
- Write `run_index.json`: problem ID → output path, wall-clock start/end, status,
  and `trace_id` (resolved with
  `mlflow.search_traces(experiment_ids=[<agent experiment id>],
  filter_string="tags.run_id = '<run_id>'")`; `null` when tracing is off).
- Keep tracing on (needed for cost); do not depend on it for correctness.
- Support `--subset 1,4,8,14,15,21,27,33,39,45` and `--repeat 3` for the flaky
  subset. Full v1 run = **50 problems × 1 repeat + 10 problems × 3 repeats**.
- The batch is sequential and LLM-bound (90–240 s per call); plan an overnight run.
  Do not add concurrency until the gateway and MLflow tracing are proven safe
  under it.

**Done when:** one command yields 50 per-problem JSONs plus `summary.json`,
`run_index.json` carries trace IDs (or explicit nulls), and a 2-problem dry run
finishes end-to-end.

---

### Step 6 — Outcome metrics (scorecard #1, #2)

`src/evals/metrics.py`:

- `golden_strict_solve_rate` — all three cases pass.
- `case_pass_rate` — overall and per `base|edge|adversarial`.
- `pass_at_1` / `pass_at_2` — final verdict, plus whether the retry was used
  (`iterations`, `eval`).
- `category_balanced_solve_rate` — macro average across the 40 categories.
- Wilson 95% CI for each proportion; also emit numerator/denominator.

MLflow wiring:

- Build `per_problem_df` with `inputs` (problem id, requirement), `outputs`
  (verdicts, scores, code hash, `run_id`), and `expectations` (golden cases,
  complexity, constraints). Keep `trace_id` in `run_index.json` and attach feedback
  separately (§2.4) — do not mix a trace column into this DataFrame.
- Call `mlflow.genai.evaluate(data=per_problem_df, scorers=[golden_strict_pass,
  golden_case_pass_rate, ...])`; log the returned result table as
  `eval_result.json`.
- Log `golden_strict_solve_rate`, `case_pass_rate`, and the three case-type rates
  with `mlflow.log_metrics`; attach `golden_strict_pass` feedback to each trace
  (§2.4).

**Done when:** fixture-based unit tests reproduce hand-computed values, including a
problem that passes base+edge but fails adversarial, and the MLflow evaluation
result contains exactly 50 rows with the expected per-row assessments.

---

### Step 7 — Calibration metrics (scorecard #3, #4)

Define:

```text
internal_pass = status in {ok, assumed} and output["eval"]["passed"] is True
golden_pass   = all three grade cases passed

false_pass_rate = FP / internal_pass_count     # conditional risk
false_fail_rate = FN / internal_fail_count
agreement       = (TP + TN) / n
kappa           = (agreement - expected_agreement) / (1 - expected_agreement)
```

- Emit a 2×2 confusion matrix and `false_passes.json` listing escaped problem IDs
  with the failing golden case and the failing test IDs.
- Report both conditional FP rate and `FP / 50`; the conditional one answers
  "when the agent says pass, how often is it wrong?".

MLflow wiring:

- `@scorer` functions `internal_pass`, `false_pass`, `false_fail` read the
  precomputed verdicts; aggregate with `mean`.
- Log `false_pass` / `false_fail` / `internal_pass` feedback per trace with the
  failing case and test IDs in `rationale`.
- Log `false_pass_rate`, `false_fail_rate`, `verdict_agreement`, and
  `cohens_kappa` as run metrics; `false_pass_rate` is a hard gate.

**Done when:** a synthetic fixture with 2 false passes and 1 false fail yields the
exact expected rates and κ; escaped IDs appear in the report; filtering traces by
`assessments.false_pass = true` returns exactly those two problems.

---

### Step 8 — Test-oracle correctness and blame accuracy (scorecard #6, #7)

`src/evals/oracle.py`:

For every generated test (`test_cases` + `holdout_tests`):

1. Run `test["harness"]` against the reference solution.
2. `oracle_ok = (no error) and (stdout matches test["expected"])`.
3. Oracle correctness = `oracle_ok / total`; report agent tests vs holdout separately.

For every failed entry in `execution_results`:

- Locate its test by `test_id`.
- If `oracle_ok` → ground-truth blame = **code**; else = **test**.
- Compare against `entry["blame"]` and `failure_report["culprit"]`.
- Exclude/flag entries with `repaired=True` and define their treatment explicitly
  (recommended: the repaired expected output is the oracle, and the original
  failure is labelled `test`).
- Emit accuracy, a code/test confusion matrix, and a list of misblamed tests.

MLflow wiring:

- Log `oracle_correctness` and `blame_accuracy` as run metrics and as per-problem
  `@scorer` values.
- Attach `blame_accuracy` feedback per trace with `rationale` naming the test ID
  and whether the code or the expected output was wrong.

**Done when:** a fixture with one wrong expected output and one correct expected
output produces the expected accuracy, repaired entries are counted
deterministically, and the misblame is visible as trace feedback.

---

### Step 9 — Mutation score (scorecard #5)

`src/evals/mutants.py` — AST mutators, seeded and deterministic:

- comparison swap (`<` ↔ `<=`, `>` ↔ `>=`, `==` ↔ `!=`);
- arithmetic swap (`+` ↔ `-`, `*` ↔ `//` where valid);
- boolean `and` ↔ `or`;
- off-by-one on integer constants adjacent to comparisons/indexing (`n` → `n-1`, `0` → `1`);
- return-value mutation (`return x` → `return None` / `return not x`) and `break` removal.

`src/evals/mutation.py`:

1. Generate up to 5 mutants per problem from the reference.
2. Keep only **killable** mutants: those that differ from the reference on at least
   one golden case (filters syntactically broken/equivalent mutants).
3. Run every **oracle-valid** agent test against each killable mutant; a mutant is
   killed if any test errors or its stdout differs from the test's expected output.
4. Report micro score (`killed / killable`) and macro score (mean per problem),
   plus a per-problem matrix and the list of surviving mutants.

**Runtime note:** ~2.5k sandboxed runs (50 × 5 mutants × ~10 tests). No LLM calls;
cache `(mutant_id, test_id)` results in `mutation.json`.

MLflow wiring:

- Log `mutation_score` (micro and macro) as run metrics.
- Log the full mutant × test kill matrix with
  `mlflow.log_table(mutation_df, artifact_file="mutation_matrix.json")`.
- Attach per-problem `mutation_score` feedback to each trace so surviving mutants
  are visible next to the solve that produced the tests.

**Done when:** reference-vs-reference yields zero killed; a deliberately weak test
set scores low; the agent's real tests produce a stable score across two runs; the
mutation matrix and per-trace feedback are present in the eval run.

---

### Step 10 — Constraint adherence and complexity advisory (scorecard #8)

`evals/constraints.yml`, keyed by problem ID, with rule kinds:

- `ast_forbid` — e.g. problem 2: no `/` or `//` division operator.
- `ast_require` — e.g. problems 25/26: element/index assignment to `matrix`
  (in-place evidence), no wholesale copy-and-return.
- `behavior_probe` — v2, for rules that cannot be checked statically.

Keep v1 to ~6 explicit, defensible rules. `constraints.py` runs them on the final
code and reports per-rule results; the metric is per-problem binary and advisory.

`complexity.py` (advisory only): static estimate from loop nesting depth, recursion,
and builtin `sorted`/`heapq` usage; compare against the target exponent parsed from
`Complexity Targets` (`O(N)`, `O(log(min(M,N)))`, `O(N^3)`, etc.). Report
`match | better | worse | unknown`. Do **not** make this a hard gate in v1 —
empirical timing in Python is noisy and the dataset has no input generators.

MLflow wiring:

- Log `constraint_adherence` as a run metric and a per-problem scorer.
- Log the per-rule result table (`rule_id`, `problem_id`, `passed`, `reason`) as
  `constraints.json`.
- Optional `--llm-judge`: a `mlflow.genai.make_judge` judge checks constraint
  extraction from the clarified intent. Log it as a separate advisory metric and
  never let it change the deterministic score.

**Done when:** every reference solution passes its applicable constraint rules, a
known violating snippet fails, and the scorecard labels complexity as advisory;
both the deterministic table and (when enabled) the judge score are in MLflow.

---

### Step 11 — Ops metrics (scorecard #9, #10)

`src/evals/ops.py`:

- **Infra error rate:** `status == infra_error` / runs.
- **Flaky rate:** for the 10-problem × 3-repeat subset, any problem whose strict
  verdict differs across repeats is flaky.
- **Latency:** p50/p95 wall-clock per run from `run_index.json`, cross-checked
  with `trace.info.execution_time_ms` from `mlflow.get_trace(trace_id)`.
- **Cost:** read `trace.info.token_usage` and `trace.info.cost` (MLflow aggregates
  them from the `model:*` spans). If cost is absent, multiply tokens by
  `evals/pricing.yml`; if neither exists, mark `unavailable`. Emit
  `cost_per_solved = total_cost / strict_passes`.
- Log wall-clock, token, and cost totals plus p50/p95 as run metrics; include
  `repeat` in the metric key (`latency_p95_r1`, `latency_p95_r2`, ...) so flaky
  repeats do not overwrite each other.
- Secondary counters: `fallback_used`, `critic_overruled`, mean `iterations`,
  empty-reply/timeout spans.

**Done when:** a mocked trace with `execution_time_ms`, `token_usage`, and `cost`
produces the expected p95/cost, and an unavailable tracking server degrades to
`unavailable` without failing the report.

---

### Step 12 — MLflow evaluation run, scorecard, gates

`src/evals/mlflow_eval.py` is the **only** module that talks to MLflow. It exposes a
small, mockable surface:

```python
init(experiment: str) -> None
sync_dataset(records, reset=False) -> DatasetRef      # id, version, digest
start_eval_run(batch_id, manifest) -> Run
log_eval(per_problem_df, scorers) -> EvaluationResult # mlflow.genai.evaluate
log_trace_feedback(run_index, grades) -> None         # mlflow.log_feedback
log_scorecard(metrics, tables, artifacts) -> None
finish(gate_status) -> None
```

`src/evals/report.py` writes the local artifacts first, then calls the adapter:

- `metrics.json` — every metric with value, numerator, denominator, CI, gate, status.
- `scorecard.md` — the 10-row table plus slice tables and the escaped/failed lists.
- `per_problem.csv` — one row per problem: strict, base/edge/adversarial, internal
  verdict, FP/FN flag, oracle correctness, blame accuracy, mutation score, latency, cost.
- `failures.md` — grouped root causes (wrong code, wrong test, timeout, guardrail,
  infra), linking problem IDs.

MLflow sequence (single parent run, no duplicate eval runs):

1. `mlflow.set_experiment("leetcodeagent-evals")`.
2. `mlflow.start_run(run_name=f"eval_{batch_id}")`; set tags/params from §2.1.
3. `result = mlflow.genai.evaluate(data=per_problem_df, scorers=SCORERS)` using the
   precomputed outputs; log `result.tables` as `eval_result.json`.
4. `mlflow.log_metrics` for the 10 canonical metrics, each slice, and the
   `gate.<id>` 0/1 values.
5. `mlflow.log_feedback` for each available `trace_id` (§2.4).
6. `mlflow.log_table(per_problem_df, "per_problem.json")`,
   `mlflow.log_table(slice_df, "slices.json")`, `mlflow.log_table(mutation_df,
   "mutation_matrix.json")`, and `mlflow.log_artifact(...)` for the report files.
7. Compare with `eval.baseline_run_id` via `mlflow.search_runs`; log delta metrics
   and a `regressions.json` artifact.
8. Set tag `gate.status=pass|fail`, print the run URL and the trace-search filter,
   then end the run.

`src/evals/cli.py` subcommands:

```bash
python -m evals.cli prepare [--sync-mlflow] [--reset]
python -m evals.cli run --repeat 1 [--no-mlflow]
python -m evals.cli run --subset 1,4,8,... --repeat 3
python -m evals.cli grade --batch <dir>     # golden + calibration + oracle/mutation
python -m evals.cli report --batch <dir>    # local scorecard + MLflow eval run
python -m evals.cli all [--llm-judge] [--baseline <run_id>]
```

Exit code `1` if a hard gate fails (CI), `0` otherwise. `report` must be idempotent:
re-running against the same batch updates the same run (or creates a linked run with
the same `eval.batch_id`) instead of duplicating metrics.

**Done when:** `all` on a fixture batch produces both local artifacts and one MLflow
run containing the evaluation table, 10 metrics, gate tags, trace feedback, and
artifacts; `--no-mlflow` produces the same local artifacts with exit code `0`.

---

### Step 13 — Tests and CI for the eval framework

- Unit tests (no LLM): dataset validation, harness build/parse (empty string,
  bool, float, nested lists), comparison, metric math fixtures, mutant generation,
  blame ground truth, report table formatting.
- MLflow tests with a local SQLite tracking store (`sqlite:///tmp/eval-test.db`):
  dataset sync idempotency, `@scorer` contract vs `metrics_spec`, `log_feedback`
  called once per trace, gate metrics/tags, `--no-mlflow` fallback.
- Integration test: two fake agent outputs → grade → metrics → scorecard → one
  MLflow run with 2 rows.
- Slow test, `@pytest.mark.slow`: reference oracle `150/150`. Keep it out of the
  default `pytest -q` path.
- CI: default tests + oracle check + fixture scorecard. Nightly/manual: full 50-run
  eval (LLM cost).
- `ruff check src/evals tests`.

**Done when:** `PYTHONPATH=src uv run pytest -q` stays green, the eval package has
no network/LLM dependency in its default tests, and the MLflow tests pass against a
local SQLite store with no running server.

---

### Step 14 — Baseline run and gate calibration

1. Full run: 50 × 1 repeat.
2. Flaky run: 10-problem stratified subset × 3 repeats.
3. Compute the baseline scorecard and tag the run
   `eval.role=baseline`; record its run ID in `evals/gates.yml`.
4. Set gates as **baseline − margin** for hard gates and aspirational targets for
   tracked metrics. Freeze in `evals/gates.yml`.
5. Configure subsequent runs with `--baseline <run_id>`; `report.py` loads the
   baseline metrics via `mlflow.search_runs`, computes paired deltas on the same
   problem IDs, and logs `delta.<metric>` plus `regressions.json`.
6. Triage every escaped problem and every misblame into the backlog.

**Done when:** a published baseline run exists in `leetcodeagent-evals`, the 10
gates in §1 are filled with real numbers (not placeholders), and a second run shows
its deltas against the baseline in both the scorecard and the MLflow UI.

---

### Step 15 — Documentation and rollout

- Add an **Evals** section to `README.md`: commands, expected runtime/cost, artifact
  locations, metric definitions, the "internal vs golden verdict" distinction, and
  how to read the MLflow evaluation run (scorecard artifacts, per-row assessments,
  trace feedback filters, run comparison).
- Add `evals/README.md`: dataset conventions, reference-solution contract, how to add
  a problem, how to add a constraint rule, and how to sync/reset the MLflow dataset.
- `.gitignore`: `output/evals/`, `evals/results/`.
- PR checklist: `python -m evals.cli all --subset smoke --no-mlflow` before merging
  agent changes; full MLflow scorecard on the nightly/manual job.

**Done when:** a new contributor can run the scorecard locally and find the same
run, dataset version, and per-problem feedback in the MLflow UI from a clean
checkout without reading this plan.

---

## 5. Milestones

| Milestone | Steps | Outcome |
|---|---|---|
| **A. Trustworthy inputs** | 1–4 | dataset validated + versioned, references pass 150/150, grader works |
| **B. First real numbers** | 5–6 | full batch, MLflow eval run, strict/case pass rates |
| **C. Self-grading audit** | 7–8 | false pass/fail, κ, oracle correctness, blame accuracy, trace feedback |
| **D. Test strength + gates** | 9–12 | mutation matrix, constraints, ops metrics, scorecard + CI gate in MLflow |
| **E. Baseline + rollout** | 13–15 | tests, baseline run + deltas, docs |

---

## 6. Risks and decisions to lock now

1. **Sandbox payload restrictions.** No `json`, no `repr`, no path-like strings in
   harnesses. Use marker + `print` + `ast.literal_eval`; trusted-harness mode for
   golden data only.
2. **Timeouts.** Production sandbox is 5 s; golden adversarial cases need ~30 s.
   Record timeout as a failed case with a distinct flag.
3. **Contract drift.** The agent may pick a different function name or treat
   `root`/`head` as a node instead of a list. The requirement text must state the
   exact signature/encoding; a mismatch is a legitimate failure, not an eval bug.
   Consider a signature-conformance check as a v1.1 metric.
4. **Nondeterminism.** Heavy runs at temperature 0.2. v1 = 1 full repeat + 3 repeats
   on a stratified subset; do not interpret single-point differences.
5. **Reference-solution risk.** A wrong reference poisons #5–#7. The 150-case oracle
   check in Step 3 is the guard; never bypass it.
6. **Equivalent mutants.** The killable filter mitigates but does not eliminate them;
   document mutation score as a relative signal, not an absolute truth.
7. **Cost/time.** A full run is LLM-bound and can take hours. Keep the fixture and
   subset paths cheap for CI; reserve the full run for nightly/manual.
8. **MLflow API drift.** `mlflow.genai` is young (3.17.1 installed). Pin
   `mlflow>=3.17,<4`, wrap dataset/evaluate/feedback/trace calls in one adapter,
   and fail soft to `unavailable` so the local scorecard still works.
9. **Duplicate dataset versions.** `merge_records` appends. Always compare the
   dataset digest/`golden_dataset_hash` first; require `--reset` for intentional
   dataset edits, and assert record count == 50 after sync.
10. **Duplicate eval runs.** `mlflow.genai.evaluate` can create/log its own run.
    Call it **inside** the parent run and assert exactly one run per `batch_id`;
    the idempotency test in Step 13 covers this.
11. **Payload size.** Full code + tests + execution results per row can bloat the
    evaluation table. Keep `outputs` to verdicts, scores, and code hashes; store the
    full artifacts separately and link by `run_id`.
12. **Trace volume.** Per-problem feedback plus gateway usage tracking can grow the
    backend quickly. Keep gateway usage tracking off (existing README guidance),
    set a retention policy, and allow `--no-feedback` for large batches.
13. **Feature availability.** GenAI datasets/feedback need an MLflow 3 tracking
    server. Local SQLite supports the run/metrics/artifact path but may not support
    every dataset/feedback API; the adapter must probe once and degrade cleanly.

---

## 7. Definition of done (v1)

- [ ] All 50 problems normalize; references pass 150/150 golden cases.
- [ ] `leetcode-golden-v1` is registered and versioned; the eval run records
  `dataset_id`, `dataset_version`, and `dataset_hash`.
- [ ] One command runs the batch, grades, and writes `scorecard.md` + `metrics.json`.
- [ ] All 10 scorecard metrics have a numerator, denominator, CI, and gate.
- [ ] One MLflow eval run contains the `mlflow.genai.evaluate` result table, the 10
  canonical metrics, gate tags, per-problem tables, and report artifacts.
- [ ] Per-problem `golden_strict_pass` (and false-pass/misblame) feedback is attached
  to the corresponding agent traces.
- [ ] False-pass list and misblame list are produced, not just aggregate numbers.
- [ ] Mutation score computed from oracle-valid tests only; matrix logged to MLflow.
- [ ] Baseline run is tagged, gates calibrated from it, and a follow-up run shows
  deltas versus baseline.
- [ ] `--no-mlflow` produces the same local artifacts and a correct gate exit code.
- [ ] Eval package tests pass without LLM/network; full eval is marked `slow`.
- [ ] README documents how to reproduce a scorecard and read the MLflow eval run.
