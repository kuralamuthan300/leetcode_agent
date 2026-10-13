# Guardrails — Coder Sandbox Security

Updated: 2026-10-13 | Status: implemented on `task/step-1-5` (`tools/guardrail.py`, gated `run_python`, `guardrail_report` in state/output, `tests/test_guardrails.py` green).
Enforcement: fail-closed (blocked code never executes).

## Threat model
Untrusted Python from the Coder LLM (and coder-influenced harness). Must prevent:
malicious exec (reverse shell, exfil), any file read/write, resource exhaustion,
and sandbox escape via stdlib (`os`, `subprocess`, `socket`, `pathlib`, ...).

## Gate order
`Coder code + harness -> guardrail.check() -> [blocked: return error JSON, no exec] / [allowed: run_python() in hard jail]`

## 0. Python-only language gate (entry)
`check_language()` runs first in IntentClarifier: explicit non-Python
languages (java/js/ts/sql/go/rust/…), strong code markers (public class,
console.log, #include, …), or 2+ SQL keywords → `rejected: True`,
straight to RunReporter, no LLM burn. Neutral DSA defaults to Python;
explicit "python" always wins. Non-Python *code* is also backstopped by
the AST gate (foreign syntax fails parse → blocked).

## 1. Static gate (`src/agent/tools/guardrail.py::check`)
- Parse both `code` and `harness` with `ast`.
- Imports: deny all except allowlist (`math`, `typing`, `collections`,
  `itertools`, `functools`, `heapq`, `bisect`). Deny `os`, `sys`, `subprocess`,
  `socket`, `pathlib`, `shutil`, `io`, `codecs`, `pickle`, `importlib`.
- Calls/attrs: deny `open`, `file`, `eval`, `exec`, `compile`, `__import__`,
  `__subclasses__`, any dunder attribute, `os.*`, `sys.*`, `socket.*`,
  `pathlib.*`, `shutil.*`, `io.open`, `os.open`, `os.remove/unlink/mkdir/listdir/system/popen`.
- File backstop: deny path-like string args (`/`, `./`, `../`, `~`) in calls.
- Limits: code ≤15KB, harness ≤5KB.
- Returns `{allowed: bool, reasons: str[]}`; every denial logged.

## 2. File lockdown (read/write strictly prohibited)
- `open`/`file` removed from `__builtins__` allowlist; `io.open`/`os.open` nulled.
- `os`/`io`/`pathlib` unimportable via allowlist-only importer.
- Exec cwd is an empty temp jail, read-only; no mounts, no harness-supplied paths.
- Evaluator-owned harnesses only; coder cannot inject file paths.
- Container phase (later): `--read-only --network none`, no volumes.

## 3. Hardened exec (upgrade `tools/sandbox.py`)
- Keep spawn + 5s default timeout; add `setrlimit` memory cap, wall-clock kill.
- Minimal `__builtins__` allowlist (`print`, `range`, `len`, ...); no `open`/`__import__`.
- `socket` module blocked; stdout/stderr caps.
- Current gap (must fix): full `__builtins__` passed today — file/network possible until §1–§3 land.

## 4. State + output
- New `guardrail_report: {allowed, reasons[], checked_at}` in `AgentState`.
- `coder_smoke` and `execution_results` record blocked attempts as errors (no exec).
- `output/run_id_*.json` includes `guardrail_report`; blocked counts toward retry budgets.

## 5. Retry semantics
- Blocked Coder smoke → `feedback` to Coder/Planner, counts toward Evaluator max-3 loop.
- 3x blocked intent-equivalent → proceed with failure flag, never bypass gate.

## 6. Test corpus (`tests/test_guardrails.py`, all must block or fail-safe)
- `open('/etc/passwd')`, `open('x','w')`, `pathlib.Path('/tmp/x').write_text('hi')`
- `import os; os.remove('f')`, `import subprocess`, `import socket`
- `eval('1')`, `__import__('os')`, `while True: pass` (timeout), 1GB alloc (memory cap)
- Legit DSA (`def solve(...)` + pure-allowed imports) must still pass.

## Rollout
1. `guardrail.py::check()` + wire into `python_toolbox` (no graph change).
2. Harden `run_python` builtins/importer/limits.
3. `guardrail_report` in state/output + retry wiring.
4. Corpus tests green; container isolation later.
