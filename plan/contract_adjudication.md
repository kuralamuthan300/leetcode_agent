# Step-by-Step Plan — Contract Block + Failure Adjudication

Goal: planner decides input args + ranges; testgen builds in-range tests;
eval failures adjudicate TEST_WRONG vs CODE_WRONG with targeted repair.

## Step 1 — Planner contract block [done]
- `PLANNER_SYSTEM` ends reply with fenced JSON `CONTRACT: {...}`.
- `extract_contract(text)` in `prompts.py`: fenced JSON → legacy `SIGNATURE:`
  line → None.
- `planner()` stores `function_signature` + `arg_spec`.
- Tests: valid block, legacy fallback, absent contract.

## Step 2 — Range DSL + validation [done]
- `safe_num` / `parse_range` / `args_in_range` in `prompts.py`: `A..B` pairs
  (len- vs value-bounds by prefix), `non-empty`, `one of` enums; unknown → {}.
- `_valid_test()` drops out-of-range literal args; non-literals skip (no false drops).
- Wired through `test_case_gen` (both paths).

## Step 3 — Testgen consumes contract
- `test_case_gen()` prompt gets problem + contract + "values strictly within
  ranges" + mandatory boundary coverage (min/max/empty-if-allowed).
- Regen path reuses the same validation. No graph change.

## Step 4 — Blame adjudicator
- New `adjudicate(problem, contract, code, failing_test)` in `prompts.py`:
  deterministic contract/range check first (violation ⇒ TEST_WRONG, no LLM);
  else one small-model verdict call (`CODE_WRONG` / `TEST_WRONG` + reason).
- Record `blame` per failure into `execution_results`.

## Step 5 — Targeted repair routing
- `evaluator()`: TEST_WRONG → new `repair_test()` sub-step (regen that test
  only, code untouched); CODE_WRONG → existing coder path via planner feedback
  carrying `error_class` + blame; unsure/flip-flop → escalate to planner with
  both artifacts. Respect the existing 3-iteration budget; contract violations
  never trigger code repair.

## Step 6 — State, reporter, docs, E2E
- State: `arg_spec`; reporter JSON includes it (+ `blame` per result).
- Docs: `plan/architecture.md` (Nodes §2/§3/§5, state, schema), `plan/plans.md`
  new step, `plan/guardrails.md` contract-as-truth note.
- Full suite green + Two Sum golden E2E showing an adjudicated run.

Verify each step: hermetic unit tests first, then live E2E.
