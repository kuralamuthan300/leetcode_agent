# AGENTS.md — leetcodeagent

## Sources of truth
- `docs/architecture/LEETCODE_AGENT_PLAN.md` — architecture, JSON schemas, sandbox design (Sec 12 UI, Sec 13 HITL).
- `docs/build/BUILD_PLAN_STEP_BY_STEP.md` — build order. Implement exactly one Step per session (0–13), nothing else.
- `docs/reports/` — E2E logs (`e2e_step10.log`), UI/HITL verification (`steps_11_13_verification.md`). `docs/security_tests.md` (Step 10).
- `docs/reports/SESSION_STATE.md` — resume checkpoint. Read this second, after this file.

## Startup (anti-reread, read ONLY these in order)
1. `AGENTS.md` 2. `docs/reports/SESSION_STATE.md` 3. current Step section from `BUILD_PLAN_STEP_BY_STEP.md`.
4. Allowlist = files named in current Step's Ask block. Do NOT `glob src/**`, `read` full dirs, or open `docs/architecture/*` unless the Step lists it. Use `grep` for symbols, `read` with `offset/limit` for sections.
5. For lookups, delegate: `subagent explore/quick` returning function + lines only (≤10 lines), not full files to main context.

## Build rules
- One step at a time. Stop after the Step's `Verify` command passes. If output overflows, continue same Step only — never merge Steps.
- Each Step lists files to create + verify command. Run `uv run pytest -q` for Steps 0–3 before any LLM/Docker work.
- Pydantic schemas (`src/schemas.py`) reject bad LLM JSON — never bypass with manual dicts.
- Deterministic tools first: `access_broker.py` → `static_scanner.py` → `safe_executor.py`. LLM nodes depend on them.

## Ollama models (local, fixed)
- Heavy: `gemma4:31b-cloud` — problem/solution/optimization only.
- Light: `deepseek-r1:1.5b` — intake, routing, formatting, summaries.
- All calls via `src/llm_router.py`. Escalate light→heavy on retry. Verify tags with `ollama list`; on mismatch stop and report.

## Security invariants (never relax)
- Agent + sandbox see only `workspace/jobs/<job_id>/` with ONE json file. Enforce via `realpath` startswith check; block `..`, absolute paths, symlink escapes.
- Pre-run AST scan blocks `os,sys,subprocess,socket,shutil,pathlib,ctypes,threading,multiprocessing,inspect,eval,exec,__import__,compile,input`. Allowlist: `math,heapq,collections,bisect,itertools,functools,typing`.
- Docker run: `--network none --memory 256m --cpus 1 --pids-limit 64 --read-only`, mount only job dir to `/work`, `timeout` kill. Sanitize tracebacks (no host paths).

## Commands
- Setup: `uv sync` (deps from `pyproject.toml`, locked in `uv.lock`: langgraph, langchain-ollama, pydantic, pyyaml, docker, streamlit; dev: pytest)
- Health: `ollama list` + `uv run python -m src.llm_router --healthcheck`
- Sandbox: `docker build -f Dockerfile.sandbox -t sandbox-img .`
- Tests: `uv run pytest -q` (full), `uv run pytest tests/test_<name>.py -v` (single)
