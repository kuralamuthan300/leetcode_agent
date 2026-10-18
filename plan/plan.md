# Plan — TDD ordering + coder self-loop + blame-aware routing + live terminal output

Status: **implemented** (2026-10-18) | Branch: `plan/plan.md`
Companion docs: `architecture.md` (current design), `plans.md` (step history).
Increments: **1** TDD ordering + coder self-loop + blame-aware routing (below);
**2** live terminal output (plan/code/test I/O panels); **3** during-node
narrator (spinner + rotating phrases + live seconds timer); **4** gatherer modes
(interactive/quiet) + classified aborts + batch dataset driver. See `plans.md`
Steps 15–18.

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

## Increment 2 — Live terminal output (panels)

`cli.py` renders rich panels while the run streams, on by default:

1. **Plan** panel after `planner` — numbered `plan_steps` + `Contract:`
   (signature, typed args/ranges, `Returns:`) via `_render_plan`.
2. **Test cases** table after `tests` — `id | input (harness) | expected | kind`
   (holdout labelled) via `_render_tests`.
3. **code** panel on every coder attempt (`_render_code`, clipped to 80 lines) +
   a **Coder self-check** results table from `coder_checks`.
4. **Test results** table after `evaluator` — `id | input | expected | actual |
   result | blame` from `execution_results`.

`--quiet` = status lines only; `--verbose` also dumps the full final code/tests.
`ExecutionResult` gains `harness`; state gains `coder_checks`. Long values/code
are clipped (`_clip`, `_CODE_LINES`). Tests: `tests/test_cli.py`.

## Increment 3 — During-node narrator (spinner + rotating phrases + timer)

A node's LLM call blocks 90-180s and `stream_mode="updates"` emits only on
completion, so the terminal is otherwise silent. While a node runs, show a
bottom-of-screen spinner with **rotating, node-aware phrases** and a **live
seconds counter** (`planner: thinking…  (14s)`).

### Mechanism (`cli.py`)

- `_NARRATION` — per-node phrase lists; `_NARRATION_INTERVAL = 3.0`.
- `_next_node(node, state)` — predicts the next node by reusing the graph routers
  (`_route_gather`/`_route_coder`/`_route_eval`) + the static edges
  (`planner→tests`, `tests→coder`); start → `gather`, `reporter` → `None`.
- `_narrator(node, elapsed)` — pure: returns `Text("<node>: <phrase>  (<n>s)")`,
  phrase = `phrases[int(elapsed // _NARRATION_INTERVAL) % len]`.
- `_produce(graph, state, updates)` — daemon worker thread draining
  `graph.stream(...)` into a `queue.Queue`, always ending with a `done` sentinel
  (and an `error` sentinel on exception).
- `_render_patch(node, patch, show, attempt, elapsed)` — shared status line
  (suffixed with `· {elapsed:.1f}s`) + panel rendering, used by both drivers.
- `_run_graph` — two drivers: **direct** (non-TTY or `--quiet`) and **live**
  (`show and console.is_terminal`). Both measure per-node elapsed and append the
  duration to every status line. The live driver runs
  `rich.live.Live(get_renderable=tick, transient=True, refresh_per_second=8)` on
  the main thread; `tick` mutates a **persistent** `Spinner`'s `.text` (a fresh
  `Spinner` per render would freeze at frame 0), so the timer recomputes ~8×/s
  on the refresh thread and the main loop simply blocks on `updates.get()`.

### Invariants / notes

- Narrator/timer is **display-only**; it never changes routing or results.
- The spinner/timer is gated on an interactive TTY (no log spam); the `· 12.4s`
  duration suffix appears in **all** modes, so piped/`--quiet` output is no longer
  byte-for-byte identical to before.
- Hang-safe: the sentinel is enqueued in a `finally`; stream errors surface on
  the main thread and stop the loop.
- Rich `Live` stays on the main thread (only the pure producer is off-thread),
  avoiding Rich thread-safety pitfalls.
- Tests: `_next_node` routing, `_narrator` rotation + seconds formatting, the
  duration suffix, plus a forced live-path smoke test and a stream-error test
  (`tests/test_cli.py`).

## Increment 4 — Gatherer modes + batch dataset driver

Goal: make headless runs produce **machine-readable data for an LLM-as-judge
eval** (classified aborts, stable JSON schema) while keeping the interactive
human path intact.

### Two gatherer modes

`state["gather_mode"]` selects the path in `requirement_gatherer`; the default is
`interactive`, i.e. the pre-Increment-4 behaviour, so nothing else changes.

- **`interactive`** (default; TTY): today's path — rewrite, then show the
  understanding/assumptions and wait for Enter (max 2 correction rounds);
  non-TTY proceeds flagged `assumed` when vague.
- **`quiet`**: validate the requirement in **one pass, never asking the user**.
  Hard gates first (language, scope, truncation → `incomplete`), then a single
  `QUIET_GATHER_SYSTEM` call returning strict JSON
  `{clarified_intent, workable, reason, assumptions[], blocking_ambiguities[],
  confidence, rejection_class}`. Malformed/absent verdict → bounded retry, then
  `status="infra_error"` (a transport failure is **not** a rejection).
  Abort when `workable=false` **or** `blocking_ambiguities` is non-empty
  (answer-changing ambiguity ⇒ strict abort); otherwise proceed, flagging
  `assumed` when the model listed assumptions.

`--gather-mode {interactive,quiet}` (default: interactive on a TTY, else quiet).
`_route_gather` short-circuits quiet to `planner` so it can never re-ask; the
gatherer always sets `rejected`, or `assumed`/`intent_clear`, before returning.

### Status + rejection taxonomy

- `status` ∈ `ok | assumed | aborted | infra_error`.
- `rejection_class` ∈ `scope | language | incomplete | noncoding_task |
  undeliverable | under_specified | nondeterministic | contradictory |
  debug_unprovided | unsandboxable`, plus `infra` for transport failures.
  `infra_error` is excluded from the dataset; it is never `rejected`.
- `workability` (stable schema, `rubric_version` pinned) records
  `verdict, reason, assumptions[], blocking_ambiguities[], confidence,
  rejection_class` on every gather.
- Rubric aspects: task present, contract derivable, I/O determinable,
  determinism, constraints, examples, sandbox-executable — the decisive ones
  being **contract-derivability** and **answer-determinism**.

### Batch driver

`--problems-file problems.jsonl` (one `{id, requirement}` per line) implies quiet
gathering (mutually exclusive with `--problem`/`--file`/interactive) and runs the
graph sequentially per row:

- `output/<batch_id>/<id>.json` per problem via `state["out_dir"]`;
- `output/<batch_id>/summary.json` with `status_counts`,
  `rejection_class_counts`, and `yield {completed, aborted, infra}` (+ `skipped`
  for malformed lines);
- a row that raises is recorded `infra_error` and the batch continues.

### Exit codes

`0` for `ok`/`assumed`/`aborted` (the reason lives in the JSON); `1` for infra
failures (unhandled stream exception or `status="infra_error"`); `2` for bad
input (empty requirement, missing problems file, conflicting flags).

### Tests

Quiet verdict ok/assumed/blocking/not-workable/bad-class, truncation abort,
offline ⇒ `infra_error`, scope gate, quiet routing never re-asks
(`tests/test_intent.py`); gather-mode resolution, exit codes, batch summary +
infra-continue + malformed-line handling (`tests/test_cli.py`).
