# Eval Framework v1 — Implementation Plan

**Goal:** turn `evals/golden_dataset.csv` (50 problems × base/edge/adversarial = 150 cases)
into a repeatable scorecard that measures the agent's *outcome*, its *self-grading
calibration*, its *test-generation strength*, and its *operating cost*.

**Design rule:** the agent's own `eval` verdict and the golden verdict are two
different measurements. Never collapse them.

---

## 1. Recommended v1 scorecard (the contract)

These 10 metrics are the deliverable. Every later step exists to compute one of them.

| # | Metric | Formula | Hard gate (initial) |
|---|---|---|---|
| 1 | **Golden Strict Solve Rate** | problems with all 3 golden cases passing / 50 | baseline − 5 pts, target ≥ 0.70 |
| 2 | **Golden Case Pass Rate** | passing cases / 150, split base / edge / adversarial | report; adversarial ≥ 0.60 |
| 3 | **False-Pass Rate** | internal `eval.passed` but golden fails / internal passes | ≤ 0.05 |
| 4 | **False-Fail Rate + Verdict κ** | internal fails but golden passes / internal fails; Cohen's κ | ≤ 0.10; κ ≥ 0.70 |
| 5 | **Mutation Score** | mutants killed / killable mutants | ≥ 0.60 |
| 6 | **Test Oracle Correctness** | generated tests whose expected output matches reference / all generated tests | ≥ 0.95 |
| 7 | **Blame Accuracy** | correct `culprit` (code/test) / failures with derivable ground truth | ≥ 0.80 |
| 8 | **Constraint Adherence** | satisfied per-problem constraint rules / applicable rules (+ complexity advisory) | ≥ 0.90, advisory in v1 |
| 9 | **Infra Error Rate + Flaky Rate** | `infra_error` / runs; verdict-changing problems / repeated problems | ≤ 0.02; ≤ 0.05 |
| 10 | **p95 Latency + Cost per Solved Problem** | p95 wall-clock/run; total LLM cost ÷ strict passes | ≤ 360 s; report-only |

**Statistics:** n=50 gives roughly ±14 pts at 95% confidence around p=0.5. Report
Wilson intervals and paired comparisons (McNemar) for config changes; do not chase
sub-5-point movements. Treat the 150 cases as clustered by problem, not independent.

**Slices for every metric:** category, complexity class, case type, assumption
status, and `iterations` (1 vs 2).

---

## 2. Target layout

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
  ops.py                  # infra/flaky/latency/cost from artifacts + MLflow
  report.py               # scorecard.md / metrics.json / per_problem.csv
  cli.py                  # prepare | run | grade | report | all
evals/
  golden_dataset.csv
  constraints.yml
  pricing.yml
  problems.jsonl          # generated, committed (small)
  results/<batch_id>/     # generated, gitignored
tests/
  test_eval_dataset.py
  test_eval_harness.py
  test_eval_metrics.py
  test_eval_mutation.py
  test_eval_report.py
```

Run everything with the project's existing convention:

```bash
PYTHONPATH=src uv run python -m evals.cli all
```

---

## 3. Steps

### Step 1 — Freeze the metric contract

Create `src/evals/metrics_spec.py` (or a JSON/YAML file) defining, for every
scorecard metric:

- stable `id`, display name, tier (outcome / calibration / tests / ops);
- numerator and denominator in words;
- direction (`higher_is_better` / `lower_is_better`);
- gate value and whether it blocks;
- required artifacts (run JSON, grade JSON, reference, MLflow).

Add a `METRICS` dict consumed by `report.py`, so a metric cannot be renamed or
redefined in one place without the other.

**Deliverable:** `src/evals/metrics_spec.py` + `plan/eval_plan.md` table above kept
in sync.

**Done when:** every scorecard number in the final report is traceable to one spec
entry with an explicit denominator.

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

`src/evals/adapters.py`:

- Default adapter is identity. The statements already declare list encodings for
  linked lists and LeetCode level-order lists for trees, so no structure builder
  is required in v1.
- Keep an override registry keyed by problem ID for future cases that need
  `ListNode` / `TreeNode` construction.
- Comparison: parse actual with `ast.literal_eval`; compare with `==`; allow
  `int`/`float` equivalence and a small float tolerance for `find_median_sorted_arrays`.
- Normalize booleans (`True`/`False`) and empty-string output.

**Deliverable:** validated `evals/problems.jsonl`, dataset hash, adapter registry.

**Done when:** all 50 rows load, every case produces a valid harness source, and a
validation failure names the row and field.

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

- Write `manifest.json`: dataset hash, git SHA, `config.yml` hash, model roles,
  repeat count, subset, timestamp, command.
- Write `run_index.json`: problem ID → output path, wall-clock start/end, status.
- Keep tracing on (needed for cost); do not depend on it for correctness.
- Support `--subset 1,4,8,14,15,21,27,33,39,45` and `--repeat 3` for the flaky
  subset. Full v1 run = **50 problems × 1 repeat + 10 problems × 3 repeats**.
- The batch is sequential and LLM-bound (90–240 s per call); plan an overnight run.
  Do not add concurrency until the gateway and MLflow tracing are proven safe
  under it.

**Done when:** one command yields 50 per-problem JSONs plus `summary.json`, and a
2-problem dry run finishes end-to-end with the manifest populated.

---

### Step 6 — Outcome metrics (scorecard #1, #2)

`src/evals/metrics.py`:

- `golden_strict_solve_rate` — all three cases pass.
- `case_pass_rate` — overall and per `base|edge|adversarial`.
- `pass_at_1` / `pass_at_2` — final verdict, plus whether the retry was used
  (`iterations`, `eval`).
- `category_balanced_solve_rate` — macro average across the 40 categories.
- Wilson 95% CI for each proportion; also emit numerator/denominator.

**Done when:** fixture-based unit tests reproduce hand-computed values, including a
problem that passes base+edge but fails adversarial.

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

**Done when:** a synthetic fixture with 2 false passes and 1 false fail yields the
exact expected rates and κ; escaped IDs appear in the report.

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

**Done when:** a fixture with one wrong expected output and one correct expected
output produces the expected accuracy, and repaired entries are counted
deterministically.

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

**Done when:** reference-vs-reference yields zero killed; a deliberately weak test
set scores low; the agent's real tests produce a stable score across two runs.

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

**Done when:** every reference solution passes its applicable constraint rules, a
known violating snippet fails, and the scorecard labels complexity as advisory.

---

### Step 11 — Ops metrics (scorecard #9, #10)

`src/evals/ops.py`:

- **Infra error rate:** `status == infra_error` / runs.
- **Flaky rate:** for the 10-problem × 3-repeat subset, any problem whose strict
  verdict differs across repeats is flaky.
- **Latency:** p50/p95 wall-clock per run from `run_index.json`. If MLflow spans
  are available, also emit per-node p50/p95 for `model:heavy/small/critic/testgen/evaluator`.
- **Cost:** query `mlflow.search_traces(filter_string="tags.run_id = '<id>'")`,
  sum `mlflow.chat.tokenUsage` over `model:*` spans, multiply by `evals/pricing.yml`
  (USD per 1M input/output tokens). Emit tokens and cost; `cost_per_solved =
  total_cost / strict_passes`.
- If MLflow/token attributes are unavailable, mark latency/cost `unavailable` —
  never report `0`.
- Secondary counters: `fallback_used`, `critic_overruled`, mean `iterations`,
  empty-reply/timeout spans.

**Done when:** a mocked trace produces the expected p95 and cost, and an unavailable
tracking server degrades to `unavailable` without failing the report.

---

### Step 12 — Scorecard report, MLflow logging, gates

`src/evals/report.py` writes:

- `metrics.json` — every metric with value, numerator, denominator, CI, gate, status.
- `scorecard.md` — the 10-row table plus slice tables and the escaped/failed lists.
- `per_problem.csv` — one row per problem: strict, base/edge/adversarial, internal
  verdict, FP/FN flag, oracle correctness, blame accuracy, mutation score, latency, cost.
- `failures.md` — grouped root causes (wrong code, wrong test, timeout, guardrail,
  infra), linking problem IDs.

Log one MLflow run in experiment `leetcodeagent-evals` with params (dataset hash,
git SHA, config hash, models), metrics, and the artifacts above.

`src/evals/cli.py` subcommands:

```bash
python -m evals.cli prepare                 # validate CSV -> problems.jsonl
python -m evals.cli run --repeat 1          # agent batch -> output/evals/<batch>
python -m evals.cli run --subset ... --repeat 3
python -m evals.cli grade --batch <dir>     # golden + calibration + oracle/mutation
python -m evals.cli report --batch <dir>    # metrics.json + scorecard.md
python -m evals.cli all                     # full v1 pipeline
```

Exit code `1` if a hard gate fails (CI), `0` otherwise. `report` must be idempotent.

**Done when:** `all` on a fixture batch produces both artifacts and a correct gate
exit code; re-running produces byte-identical metrics.

---

### Step 13 — Tests and CI for the eval framework

- Unit tests (no LLM): dataset validation, harness build/parse (empty string,
  bool, float, nested lists), comparison, metric math fixtures, mutant generation,
  blame ground truth, report table formatting.
- Integration test: two fake agent outputs → grade → metrics → scorecard.
- Slow test, `@pytest.mark.slow`: reference oracle `150/150`. Keep it out of the
  default `pytest -q` path.
- CI: default tests + oracle check + fixture scorecard. Nightly/manual: full 50-run
  eval (LLM cost).
- `ruff check src/evals tests`.

**Done when:** `PYTHONPATH=src uv run pytest -q` stays green and the eval package
has no network/LLM dependency in its default tests.

---

### Step 14 — Baseline run and gate calibration

1. Full run: 50 × 1 repeat.
2. Flaky run: 10-problem stratified subset × 3 repeats.
3. Compute the baseline scorecard.
4. Set gates as **baseline − margin** for hard gates and aspirational targets for
   tracked metrics. Freeze in `evals/gates.yml`.
5. Triage every escaped problem and every misblame into the backlog.

**Done when:** a published baseline scorecard exists and the 10 gates in §1 are
filled with real numbers (not placeholders).

---

### Step 15 — Documentation and rollout

- Add an **Evals** section to `README.md`: commands, expected runtime/cost, artifact
  locations, metric definitions, and the "internal vs golden verdict" distinction.
- Add `evals/README.md`: dataset conventions, reference-solution contract, how to add
  a problem, how to add a constraint rule.
- `.gitignore`: `output/evals/`, `evals/results/`.
- PR checklist: run `python -m evals.cli all --subset smoke` before merging agent changes.

**Done when:** a new contributor can run the scorecard from a clean checkout without
reading this plan.

---

## 4. Milestones

| Milestone | Steps | Outcome |
|---|---|---|
| **A. Trustworthy inputs** | 1–4 | dataset validated, references pass 150/150, grader works |
| **B. First real numbers** | 5–6 | full batch + strict/case pass rates |
| **C. Self-grading audit** | 7–8 | false pass/fail, κ, oracle correctness, blame accuracy |
| **D. Test strength + gates** | 9–12 | mutation score, constraints, ops, scorecard + CI gate |
| **E. Baseline + rollout** | 13–15 | tests, calibrated baseline, docs |

---

## 5. Risks and decisions to lock now

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
8. **MLflow API drift.** Wrap trace/token queries in a compatibility layer and fail
   soft to `unavailable`.

---

## 6. Definition of done (v1)

- [ ] All 50 problems normalize; references pass 150/150 golden cases.
- [ ] One command runs the batch, grades, and writes `scorecard.md` + `metrics.json`.
- [ ] All 10 scorecard metrics have a numerator, denominator, CI, and gate.
- [ ] False-pass list and misblame list are produced, not just aggregate numbers.
- [ ] Mutation score computed from oracle-valid tests only.
- [ ] Eval package tests pass without LLM/network; full eval is marked `slow`.
- [ ] Baseline scorecard published and gates calibrated from it.
- [ ] README documents how to reproduce a scorecard.
