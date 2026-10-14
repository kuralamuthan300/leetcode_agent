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

## Step 3 — Testgen consumes contract [done]
- Prompt gets problem + rendered contract + "strictly within ranges" +
  boundary coverage (min/max/empty-if-allowed); `returns_spec` stored.
- Regen path shares user + validation. No graph change.

## Step 4 — Blame adjudicator [done]
- `adjudicate()` in `prompts.py`: contract/range violations ⇒ TEST_WRONG
  without LLM; else one small-model verdict (`CODE_WRONG`/`TEST_WRONG`/`UNKNOWN`).
- `blame` recorded per failure in `execution_results` (+ surfaced in feedback).
- Verified live (subtraction bug correctly blamed on code).

## Step 5 — Targeted repair routing [done]
- `repair_test()` in `nodes.py`: blame=test cases regen'd in-evaluator
  (same id, validated, re-run); code untouched on that path.
- Still-failing → blame unknown + `[escalate]` note for planner; invalid regen
  keeps the original failure. CODE_WRONG flows via existing feedback.
- `repaired: bool` per result; 3-iteration budget unchanged.

## Step 6 — State, reporter, docs, E2E [done]
- State + reporter JSON: `arg_spec`, `returns_spec` (+ `blame`/`repaired` per result).
- Docs: `plan/architecture.md` (Nodes §2/§3/§5, state, schema, diagram),
  `plan/guardrails.md` contract-as-truth note.
- Full suite green + Two Sum golden E2E showing an adjudicated run.

Verify each step: hermetic unit tests first, then live E2E.
