# Plan — TDD ordering + coder self-loop + blame-aware routing

Status: **implemented** (2026-10-18) | Branch: `plan/plan.md`
Companion docs: `architecture.md` (current design), `plans.md` (step history).

## Goal

Generate test cases **before** code, let the coder run a **bounded self-loop
(≤3 retries)** against those tests, and keep the evaluator as the
**authoritative** runner that assigns blame and routes failures to the node
responsible (tests vs code).

## Motivation

- The previous chain was `planner → coder → tests → evaluator`, so test cases
  were generated **from the coder's own output** ("Reference implementation
  (match its calls)"). A confident bug produced matching tests — a circular
  validation loop.
- Test failures always routed back to the planner, blind to whether the code
  or the test was wrong, and `test_case_gen` never saw `feedback`.
- Local failures burned a full planner round each time even when a single code
  retry would fix them.

## Target graph

```
gather → planner → tests → coder ─(self-loop ≤3)─→ evaluator ─→ reporter
                     ▲                                 │
                     └────────── blame = test ─────────┤
                     planner ←──── blame = code/unknown ┘
```

## Design invariants

1. **Evaluator is the only gate.** It unconditionally re-runs the full matrix
   (visible + holdout) and never trusts the coder's self-loop verdict. The
   self-loop is a latency optimization only.
2. **Tests derive from the contract, not the code.** `test_case_gen` consumes
   `plan_steps` + `function_signature` + `arg_spec` + `returns_spec`, never
   `state["code"]`.
3. **Two independent counters.** `iterations` (outer, ≤3, incremented by the
   evaluator) and `coder_iterations` (inner retries, ≤3, reset by
   `test_case_gen` on every entry to the coder).
4. **Feedback reaches the producer of the blame.** test-blame → `tests`;
   code/unknown-blame → `planner`; empty tests → `tests`.
5. **Inner-loop blame is deterministic** (arity/range only); full LLM
   `adjudicate` runs only in the evaluator so retries stay cheap.

## Steps (as implemented)

1. **State** (`state.py`): add `holdout_tests`, `coder_iterations`,
   `coder_status`; extend `EvalReport` with `blame`.
2. **Shared runner** (`nodes.py::run_matrix`): one function runs every harness
   and returns `(execution_results, summary)`; `adjudicate_failures=True`
   (evaluator) does full LLM blame, `False` (coder loop) uses
   `_deterministic_blame`.
3. **Testgen** (`nodes.py::test_case_gen`): derive from the contract, consume
   `feedback`, call the reasoning testgen model (`_invoke_testgen`), and split
   off a **holdout** test (≥3 tests) the coder never sees.
4. **Model** (`llms.py::get_testgen`): heavy model, `temperature=0`,
   `reasoning=True` (env `OLLAMA_TESTGEN`, `OLLAMA_TESTGEN_THINK`);
   `prompts.try_invoke` accepts `kind="testgen"`.
5. **Coder self-loop** (`nodes.py::coder`): after smoke/critic, run the visible
   tests via `run_matrix(adjudicate_failures=False)`; on code-side failure and
   budget remaining set `coder_status="retry"` + `coder_iterations`; if all
   remaining failures blame the test (or the budget is spent), `"escalate"`.
6. **Graph** (`graph.py`): wire `planner → tests → coder`, add `_route_coder`
   (retry while `coder_status == "retry"` and budget remains, else evaluator),
   and make `_route_eval` blame-aware.
7. **Evaluator** (`nodes.py::evaluator`): run visible + holdout tests, keep
   in-place `repair_test`, compute dominant `eval["blame"]`.
8. **Reporter/CLI**: emit `holdout_tests` + `coder_iterations`; status lines
   show `↻ code retry (n/3)` and `↻ code escalate to evaluator`.
9. **Tests + docs**: new routing/loop/holdout tests; README, `architecture.md`,
   `plans.md` updated.

## Anti-gaming safeguard

With tests visible to the coder and the evaluator running the same tests, a
hardcoded lookup solution could pass. `test_case_gen` therefore holds back one
test (`HOLDOUT_MIN_TESTS = 3`) that only the evaluator runs — closing the
hardcoding hole while preserving the self-loop.

## Loop budgets

- Inner: `MAX_CODER_RETRIES = 3` retries (up to 4 code generations/outer round).
- Outer: `MAX_ITERATIONS = 3` evaluator rounds. Both are module constants.

## Sandbox note

Coder and evaluator both execute through `run_python` (`tools/sandbox.py`):
static AST gate (fail-closed, blocks before spawn) + spawned child process,
allowlisted imports/builtins, 512 MB `RLIMIT_AS`, temp cwd, 5 s timeout.
Process-level isolation, **not** an OS/container sandbox. The self-loop adds
sandbox spawns but no new trust in the coder's verdict.
