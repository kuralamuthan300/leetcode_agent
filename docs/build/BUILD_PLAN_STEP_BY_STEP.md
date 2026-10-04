# LeetCode Agent — Step-by-Step Build Plan for Build Mode

> Use this file one step at a time. Each Step is sized to fit in one Build Mode response (token-safe).
> Tell Build Mode: `Implement Step N from BUILD_PLAN_STEP_BY_STEP.md, nothing else.`
> Do not skip steps. Verify each step before moving on.

## Context Budget (anti-reread)
- Per session, read: `AGENTS.md` + `docs/reports/SESSION_STATE.md` + current Step section only.
- Allowlist = files named in that Step's `Ask Build Mode` block. Forbidden = everything else (including future Steps, `src/graph.py`, `src/llm_router.py` unless listed).
- Prefer `grep` over `read` for discovery; use `read` with `offset/limit`. Delegate broad searches to `subagent explore/quick` (summary only).
- End each Step by updating `docs/reports/SESSION_STATE.md`.

Models:
- Heavy: `gemma4:31b-cloud` (via Ollama) — reasoning, problem/solution generation.
- Light: `deepseek-r1:1.5b` (via Ollama) — extraction, formatting, routing.
- If Ollama tag is invalid, Build Mode should stop and tell you. Check with `ollama list`.

Final structure being built toward (from `LEETCODE_AGENT_PLAN.md` Sec 4 + Sec 12):
```
config/models.yaml, sandbox.yaml
src/state.py, schemas.py, llm_router.py
src/tools/access_broker.py, static_scanner.py, safe_executor.py, reporter.py
src/graph.py, creator_graph.py, solver_graph.py
ui/app.py, ui/agent_client.py
workspace/jobs/<job_id>/, workspace/problems/
Dockerfile.sandbox, tests/
```

---

## LangGraph Primer (read once, high-level)

You only need 6 ideas:

1. **State:** A shared dict all steps read/write. Example: `{ job_id, problem_spec, solution_code, test_results, attempt }`. Think clipboard passed between workers.
2. **Node:** One function doing one job. Example: `draft_problem`, `run_tests`, `static_scan`. Node reads State, returns State updates. Can call LLM or be pure Python (deterministic).
3. **Edge:** Arrow from one Node to next. `A -> B` means always run B after A.
4. **Conditional Edge:** If/else router. Example: after `run_tests`, if `all_pass -> analyze`, else `-> repair`. This creates loops (retry/fix).
5. **Subgraph:** A mini-graph used as a Node in a bigger graph. We have `creator_graph` and `solver_graph` plugged into top-level `graph.py` router.
6. **Checkpointer + Human-in-loop:** Saves State after each Node so you can pause, ask user, resume. Used for `needs_human=True` approvals and crash recovery. Start with `MemorySaver`, later `SqliteSaver`.

No other LangGraph concepts needed for this project.

---

## Step 0 — Project Scaffold + Config

**Goal:** Empty runnable repo, no LLM yet.
**LangGraph concept:** None. Just files.
**Ask Build Mode:**
> Implement Step 0 from BUILD_PLAN_STEP_BY_STEP.md: create `pyproject.toml` (+ `uv.lock` via `uv lock`) with deps (langgraph, langchain-ollama, pydantic, pyyaml, docker; dev: pytest), `.python-version`, `config/models.yaml` (heavy/light tags + base_url), `config/sandbox.yaml` (timeout_ms 2000, memory 256m, cpus 1, allowed_imports list), folder structure `src/tools/ tests/ workspace/jobs/.gitkeep workspace/problems/.gitkeep`, `README.md` with uv + ollama + docker setup.

**Verify:** `uv sync && uv run pytest -q` passes (0 tests ok), `ollama list` shows both models.
**Done when:** Folders + configs exist, no code yet.

## Step 1 — Schemas + Example JSONs (Pydantic = Truth)

**Goal:** Define data shapes so LLM output is validated deterministically.
**LangGraph concept:** `State` schema preview — these Pydantic models will live inside State later.
**Ask Build Mode:**
> Implement Step 1 only: `src/schemas.py` with `RequirementsSpec, TestCase, ProblemSpec, RunResult` (Pydantic v2), plus `workspace/problems/examples/problem_id_two_sum_demo.json` and `workspace/jobs/examples/requirements_id_demo.json` matching LEETCODE_AGENT_PLAN.md Sec 5, plus `tests/test_schemas.py` (valid load, missing field rejects, duplicate test id rejects).

**Verify:** `uv run pytest tests/test_schemas.py -v` green.
**Done when:** Invalid JSON fails validation, valid example loads.

## Step 2 — Access Broker (File Jail)

**Goal:** Agent sees only `workspace/jobs/<job_id>/` with ONE json file.
**LangGraph concept:** Deterministic Node (no LLM) — safest nodes are pure Python. This will become first Node in both graphs.
**Ask Build Mode:**
> Implement Step 2 only: `src/tools/access_broker.py` with `create_job(requirements_or_problem_path) -> workdir`, `scoped_read(workdir, path)`, `scoped_write(workdir, path, content)`, `promote_to_problems(workdir, filename)`. Enforce realpath startswith workdir, block `..`, absolute, symlink escape. Add `tests/test_access_broker.py` trying `../`, `/etc/passwd`, symlink out — all must raise PermissionError.

**Verify:** `uv run pytest tests/test_access_broker.py -v` — attacks blocked, legit read/write works.
**Done when:** Cannot read/write outside job dir under any trick.

## Step 3 — Static Scanner (Pre-Run Guard)

**Goal:** Block malicious code before Docker runs it.
**LangGraph concept:** Deterministic guard Node placed before every `safe_execute`.
**Ask Build Mode:**
> Implement Step 3 only: `src/tools/static_scanner.py` using `ast` parse. Deny imports `os,sys,subprocess,socket,shutil,pathlib,ctypes,threading,multiprocessing,inspect`, calls `eval,exec,__import__,compile,open(write),input,breakpointhook`. Allowlist `math,heapq,collections,bisect,itertools,functools,typing`. Flag path strings containing `..` or starting with `/`. Return `{safe:bool, flags:list}`. Add `tests/test_static_scanner.py` with 8 malicious + 2 clean samples.

**Verify:** `uv run pytest` — 8 blocked, 2 pass.
**Done when:** `os.remove`, `shutil.rmtree`, `socket`, `subprocess`, `open('../../x')` all flagged.

## Step 4 — Secure Sandbox + Harness

**Goal:** Run untrusted `solution.py` safely with per-test timing.
**LangGraph concept:** This is the `safe_execute` Node implementation — deterministic tool called by graph, not LLM.
**Ask Build Mode:**
> Implement Step 4 only: `Dockerfile.sandbox` (python:3.11-slim, user runner, no network tools), `src/tools/_harness.py` (loads solution.function_name via importlib, runs each testcase from problem json, perf_counter_ns timing, deep-compare, JSON lines output), `src/tools/safe_executor.py` (static_scan -> docker run --network none --memory 256m --cpus 1 --pids-limit 64 --read-only -v jobdir:/work, timeout kill, sanitize traceback, return RunResults + timings). No LLM here. Add `tests/test_safe_executor.py` with mocked docker OR real docker if available (skip if no docker).

**Verify:** Run clean two_sum (passes + timings present), `while True` times out, `os.remove` never reaches docker (blocked in scan).
**Done when:** Timing table returns, infinite loop killed, container sees only `/work`.

## Step 5 — LLM Router (Ollama Heavy/Light)

**Goal:** Single entry for all LLM calls with heavy/light routing + JSON enforcement.
**LangGraph concept:** Helper every LLM Node will use. Nodes stay thin; router handles retries.
**Ask Build Mode:**
> Implement Step 5 only: `src/llm_router.py` with `ChatOllama` for both tags from `config/models.yaml`, `def generate(task: str, prompt: str, json_mode=False, retry_count=0)` auto-selecting light for `classify|extract|format|summarize` else heavy, escalate to heavy if retry_count>=1, 2x JSON parse retry with Pydantic. Add `tests/test_llm_router.py` mocked (no real Ollama needed) + manual `uv run python -m src.llm_router --healthcheck`.

**Verify:** Healthcheck connects to `http://localhost:11434`, routing picks correct model.
**Done when:** `generate(json_mode=True)` always returns valid JSON or raises after retries.

## Step 6 — Top-Level Router Graph (Minimal LangGraph)

**Goal:** First working LangGraph: State + Router + Stub subgraphs.
**LangGraph concept:** `StateGraph, START, END, add_node, add_edge, add_conditional_edges, MemorySaver`. This is the skeleton everything plugs into.
**Ask Build Mode:**
> Implement Step 6 only: `src/state.py` (AgentState TypedDict: job_id, mode, workdir, problem_spec dict|null, solution_code|null, test_results, timings_ms, attempt, errors, safety_flags, needs_human), `src/graph.py` with router node (light: if problem json path exists -> solver else creator) + two stub nodes, compiled with MemorySaver. Add `tests/test_graph.py` invoking with creator intent and solver intent.

**Verify:** `uv run pytest` shows correct branch chosen, graph image/mermaid prints.
**Done when:** You understand START->router->creator/solver->END flow. No real logic yet.

## Step 7 — Creator Subgraph

**Goal:** Requirements JSON -> validated problem JSON.
**LangGraph concept:** Subgraph + repair loop (conditional edge back to earlier node, max 3 retries then human).
**Ask Build Mode:**
> Implement Step 7 only: `src/creator_graph.py` wiring Step 2+3+4+5: nodes `intake_validate -> draft_problem(heavy) -> generate_tests(heavy) -> format_dedup(light) -> oracle_solution(heavy) -> static_scan -> safe_execute -> promote_save OR repair(heavy, max 3) -> oracle`. Expose as `creator_app`. Test with demo requirements file end-to-end (mock heavy if Ollama slow, but one real run required).

**Verify:** Produces `problem_id_*.json` that passes `test_schemas.py`, oracle passes all tests.
**Done when:** 2 categories (arrays + dp easy) generate valid problems.

## Step 8 — Solver Subgraph (Correctness Only, No Optimizer Yet)

**Goal:** Problem JSON -> passing solution + per-test timings.
**LangGraph concept:** Fix loop: `run_tests -> fail? -> repair -> static_scan -> run_tests` with attempt counter to avoid infinite loop.
**Ask Build Mode:**
> Implement Step 8 only: `src/solver_graph.py` part 1: `load_validate -> initial_solution(heavy) -> static_scan -> run_tests_timed (avg perf tests 3x) -> repair_solution(heavy, max 4)`. Return markdown table `test_id|pass|mean_ms`. No optimization yet. Test on demo two_sum JSON.

**Verify:** All tests pass, timings present, failing solution gets repaired automatically.
**Done when:** Solver solves demo problem with timing table.

## Step 9 — Optimizer + Reporter

**Goal:** Make passing solution faster, keep correctness, report before/after.
**LangGraph concept:** Second loop with rollback: only accept new State if strictly better + all pass, max 2 rounds.
**Ask Build Mode:**
> Implement Step 9 only: add to `solver_graph.py` nodes `analyze_complexity(heavy) -> propose_optimization(heavy) -> run_optimized (reuse safe_executor) -> accept_or_rollback`. Add `src/tools/reporter.py` producing `report.md` with before/after table + Big-O + what changed. Cap 2 rounds. Test on O(n^2) vs O(n) case (e.g. Two Sum brute -> hashmap) showing speedup.

**Verify:** Optimized version faster, original kept if optimized fails/slower.
**Done when:** Report shows measurable delta.

## Step 10 — Hardening + E2E + Docs

**Goal:** Prove foolproof claims.
**LangGraph concept:** Checkpointer persistence (SqliteSaver) + `needs_human` interrupts.
**Ask Build Mode:**
> Implement Step 10 only: switch to SqliteSaver, add human approval before promote_save and before accepting optimization, `audit.jsonl` logging (code_hash, flags, timings), `tests/test_security.py` 8 attacks must block, full E2E creator->solver->optimizer on fresh category, update README with exact commands: `ollama pull`, `docker build -f Dockerfile.sandbox`, `uv run pytest`, `uv run python -m src.graph`.

**Verify:** `uv run pytest` full suite green, E2E log saved, security doc lists 8/8 blocked.
**Done when:** Definition of Done from LEETCODE_AGENT_PLAN.md Sec 10 met.

---

## Step 11 — Streamlit UI Shell (Modes + Status/Stage + Problem/Solution/Tests)

**Goal:** Simple web UI (Option A: Streamlit) to interact with the agent. No graph logic changes.
**LangGraph concept:** `app.stream(..., stream_mode="updates")` — UI subscribes to node names for live stage display. `SqliteSaver` `thread_id=job_id` for resume.
**Ask Build Mode:**
> Implement Step 11 only: create `ui/app.py` (Streamlit) + `ui/agent_client.py` (thin wrapper: `run_creator(requirements_dict, thread_id)`, `run_solver(problem_dict, thread_id)`, `get_state(thread_id)`, no LLM logic in UI). Sidebar: mode radio (`creator | solver | auto | review`), `category` dropdown, `difficulty` radio, `num_tests` number input (default 8, feeds `RequirementsSpec.num_tests`), file upload / textarea / example picker (`two_sum_demo`). Main: (1) status banner pill derived from `AgentState` (`idle|running|awaiting approval|done|needs_human` + `job_id, mode, attempt, opt_round`), (2) stage stepper from streamed node names — creator: `intake_validate -> draft_problem -> generate_tests -> format_dedup -> oracle_solution -> static_scan -> run_tests -> request_promote_approval -> promote_save`; solver: `load_validate -> initial_solution -> static_scan -> run_tests_timed -> snapshot_baseline -> analyze_complexity -> propose_optimization -> scan_optimized -> run_optimized -> request_opt_approval -> accept_or_rollback -> write_report -> save_solution` (use `st.status` + `st.progress` + ✅/🔵/⚪ columns + repair badge `attempt N/4`), (3) three tabs: `Problem` (`title, statement, signature, constraints, examples`), `Solution` (`st.code(solution_code, python)` + copy), `Tests` (table `test_id|input|expected|actual|pass|mean_ms` from `test_results`/`timings_ms`, toggle hidden, `show hidden`). Add `streamlit>=1.32` dep via `pyproject.toml` + `uv lock`. Add `tests/test_ui_smoke.py` (client wrapper mocked graph: status mapping, stepper order, tabs render data without Ollama/Docker).

**Verify:** `uv run pytest tests/test_ui_smoke.py -v` green; manual `uv run streamlit run ui/app.py` shows modes, stepper advances on mocked stream, tabs show problem/solution/tests.
**Done when:** User can set `num_tests`, pick creator/solver, see live status + stage + problem statement + python solution + testcases that ran.

## Step 12 — HITL Understanding Gate + Clarify + Final Approve/Decline

**Goal:** HITL-1 after requirements (verify understanding + agent asks doubts) and HITL-2 at end (approve/decline problem/solution). Reuses Step 10 `SqliteSaver` + `interrupt_before` pattern.
**LangGraph concept:** New blocking nodes + `pending_approval` values (`understanding`, `clarify`, `promote`, `optimize`). Pause via checkpointer, resume via `update_state(thread_id, {human_answer})`.
**Ask Build Mode:**
> Implement Step 12 only: extend `src/state.py` with `understanding_summary: str|None, human_question: str|None, human_answer: str|None` (keep all Step 10 fields). In `src/creator_graph.py`: add nodes `summarize_understanding(light: renders category/difficulty/num_tests/constraints/signature expectation from requirements)` + `request_understanding_approval (sets pending_approval="understanding", needs_human=True when awaiting confirm)` wired as `intake_validate -> summarize_understanding -> request_understanding_approval -> draft_problem...`; on resume inject `human_answer` into `draft_problem` prompt context. Mirror in `src/solver_graph.py`: `summarize_problem(light)` + `request_understanding_approval` after `load_validate`. HITL-2: keep `request_promote_approval` / `request_opt_approval` but add decline path — `human_approved_*=False + human_answer=reason` routes to `repair_solution` (budget left) else `flag_human`, and logs `declined` to `audit.jsonl` (do not silently drop). Update `ui/agent_client.py` + `ui/app.py`: understanding card (summary + agent questions + editable `num_tests`/fields + `[Confirm & Continue]` / `[Answer + Continue]`), mid-run clarify popup (spec invalid or repair stuck attempt>=2: show failing tests + question + text input + `[Submit + Resume]` / `[Skip/auto-retry]`), final approval view (problem + solution + pass rate + total ms + `[Approve]` / `[Decline + reason]`). Add `tests/test_hitl_understanding.py` (mocked generate: intake pauses with `pending_approval="understanding"`, resume with answer continues to draft; decline at end routes to repair/flag + audit entry; no Docker/Ollama).

**Verify:** `uv run pytest tests/test_hitl_understanding.py -v` green; manual Streamlit: submit requirements -> understanding card blocks drafting -> confirm resumes; final approve promotes / decline logs reason.
**Done when:** No heavy LLM/Docker runs before HITL-1 confirm; every HITL-2 decision (accept/decline) is in `audit.jsonl`; safety order unchanged (scan before execute).

## Step 13 — UI E2E + Docs Polish

**Goal:** Prove UI + HITL chain works end-to-end on mocks; update user docs.
**LangGraph concept:** Checkpoint resume across two pauses in one `thread_id` (understanding -> final approval).
**Ask Build Mode:**
> Implement Step 13 only: add `tests/test_ui_e2e_step13.py` (mocked `generate` + mocked `safe_execute`: full Streamlit client flow `requirements(num_tests=5) -> understanding confirm -> streamed stages -> 5-row test table -> final approve`, plus decline-path test; assert `audit.jsonl` has `understanding_confirmed` + `promoted/accepted` entries). Update `README.md` with `uv run streamlit run ui/app.py` usage (modes, num_tests, HITL-1/HITL-2 buttons), update `docs/reports/SESSION_STATE.md` checkpoint. No architecture changes.

**Verify:** `uv run pytest -q` full suite green (Steps 0-12 still pass + new UI/HITL tests).
**Done when:** Mocked E2E passes, README documents UI + HITL buttons, SESSION_STATE points to done.

---

## How to Drive Build Mode (copy-paste)

1. `Implement Step 0 from BUILD_PLAN_STEP_BY_STEP.md, nothing else. Stop after pytest passes.`
2. Review diff, run verify command yourself.
3. Next: `Implement Step 1...` etc.
4. If response too long, say: `Continue Step N, only remaining files.`
5. Never ask for Steps N+N+1 together.

## Quick Glossary for Newcomer

- **Ollama:** Local model server. `ChatOllama(model="gemma4:31b-cloud")` calls heavy, `deepseek-r1:1.5b` calls light.
- **Pydantic:** Validator that rejects bad LLM JSON before it poisons graph.
- **AST scan:** Python parses code without running it to find banned imports — cheap safety net.
- **Docker sandbox:** Disposable container with no network, limited RAM/CPU, only `/work` writable. Deleted after each run (`--rm`).
- **Harness:** Trusted Python script inside container that times the untrusted function. LLM cannot edit it.
