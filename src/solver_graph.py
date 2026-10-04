"""Step 8+9: solver subgraph — problem JSON -> passing solution + timings + optimization.

Chain:
  load_validate -> initial_solution(heavy) -> static_scan
  -> run_tests_timed (safe_execute, avg over 3 runs)
  -> snapshot_baseline -> analyze_complexity(heavy)
  -> propose_optimization(heavy) -> scan_optimized -> run_optimized
  -> accept_or_rollback -> write_report -> save_solution -> END

Fix loop: scan-blocked or tests-fail -> repair_solution(heavy, max 4)
-> static_scan -> run_tests_timed. After MAX_REPAIRS -> needs_human=True.

Optimizer loop (Step 9): at most MAX_OPT_ROUNDS proposals; accept only if
all tests pass AND strictly faster, else roll back to baseline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.checkpoints import get_checkpointer
from src.llm_router import generate
from src.schemas import ProblemSpec
from src.state import AgentState
from src.tools.access_broker import create_job, scoped_read, scoped_write
from src.tools.audit import append_audit, log_decision, log_run, log_scan
from src.tools.reporter import build_report
from src.tools.safe_executor import safe_execute
from src.tools.static_scanner import scan

MAX_REPAIRS = 4
TIMED_RUNS = 3
MAX_OPT_ROUNDS = 2


class SolverState(AgentState, total=False):
    """AgentState plus solver inputs/outputs."""

    problem_path: str | None
    scan_safe: bool
    # Step 9 optimizer fields
    baseline_code: str | None
    baseline_results: list | None
    baseline_timings: list | None
    optimized_code: str | None
    optimized_results: list | None
    optimized_timings: list | None
    opt_scan_safe: bool
    complexity_before: str | None
    complexity_after: str | None
    what_changed: str | None
    opt_round: int | None
    opt_status: str | None
    opt_reason: str | None
    report_path: str | None


# ---------------------------------------------------------------- helpers

def _errors(state: SolverState) -> list:
    return list(state.get("errors") or [])


def _strip_code_fences(text: str) -> str:
    """Remove ```python fences LLMs add around raw code."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text.strip(), re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _failure_context(state: SolverState, max_chars: int = 1500) -> str:
    parts = [f"- {e}" for e in _errors(state)[-3:]]
    spec = state.get("problem_spec") or {}
    expected_by_id = {
        t.get("id"): t.get("expected")
        for t in spec.get("testcases", [])
        if isinstance(t, dict)
    }
    for r in (state.get("test_results") or [])[:4]:
        if not r.get("passed"):
            tid = r.get("test_id")
            parts.append(
                f"- test {tid} failed: expected={expected_by_id.get(tid)!r} "
                f"actual={r.get('actual')!r} error={r.get('error')!r}"
            )
    ctx = "\n".join(parts)[:max_chars]
    return f"Previous failures to fix:\n{ctx}\n" if ctx else ""


def format_results_table(test_results: list[dict]) -> str:
    """Markdown table `test_id|pass|mean_ms` for the solver output."""
    lines = ["| test_id | pass | mean_ms |", "| --- | --- | --- |"]
    for r in test_results or []:
        tid = r.get("test_id", "?")
        ok = "true" if r.get("passed") else "false"
        ms = r.get("time_ms", 0.0)
        try:
            ms_s = f"{float(ms):.3f}"
        except (TypeError, ValueError):
            ms_s = str(ms)
        lines.append(f"| {tid} | {ok} | {ms_s} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- nodes

def load_validate(state: SolverState) -> dict:
    """Create job jail from problem file (or reuse state), validate ProblemSpec."""
    errors = _errors(state)
    if state.get("workdir") and isinstance(state.get("problem_spec"), dict):
        try:
            prob = ProblemSpec.model_validate(state["problem_spec"])
        except Exception as exc:
            errors.append(f"load: invalid problem_spec in state: {exc}")
            return {"errors": errors, "needs_human": True}
        return {
            "problem_spec": prob.model_dump(),
            "job_id": prob.id,
            "mode": "solver",
            "attempt": int(state.get("attempt") or 0),
            "errors": errors,
            # Step 12: clear stale HITL gate flags on resume; downstream
            # approval nodes re-assert them when a pause is still needed.
            "needs_human": False,
            "pending_approval": None,
        }
    prob_path = state.get("problem_path")
    if not prob_path:
        errors.append("load: no problem_path given")
        return {"errors": errors, "needs_human": True}
    try:
        workdir = create_job(prob_path)
        raw = scoped_read(str(workdir), Path(prob_path).name)
        prob = ProblemSpec.model_validate(json.loads(raw))
    except Exception as exc:
        errors.append(f"load: {exc}")
        return {"errors": errors, "needs_human": True}
    return {
        "workdir": str(workdir),
        "job_id": prob.id,
        "problem_spec": prob.model_dump(),
        "mode": "solver",
        "attempt": 0,
        "errors": errors,
        "test_results": [],
        "timings_ms": [],
        "safety_flags": [],
        "needs_human": False,
    }


def summarize_problem(state: SolverState) -> dict:
    """Step 12 HITL-1: render interpreted problem + questions for review.

    Deterministic (no LLM) so the gate never depends on model availability.
    """
    errors = _errors(state)
    spec = state.get("problem_spec") or {}
    cases = spec.get("testcases") or []
    summary = (
        f"Problem: {spec.get('title', '?')} "
        f"({spec.get('difficulty', '?')}). "
        f"Function: {spec.get('function_name', '?')}. "
        f"Tests to run: {len(cases)}. "
        f"Constraints: {spec.get('constraints') or 'none'}. "
        "I will write a correct solution, then optimize it."
    )
    return {
        "understanding_summary": summary,
        "human_question": (
            "Confirm the signature/constraints above, or reply with corrections."
        ),
        "errors": errors,
    }


def request_understanding_approval(state: SolverState) -> dict:
    """Step 12 HITL-1 gate: pause before any heavy LLM/Docker work.

    Resume with human_answer (or understanding_confirmed=True) to continue.
    require_understanding=True enforces the pause; None/False auto-passes
    for headless runs (legacy compat, mirroring Step 10 gates).
    """
    errors = _errors(state)
    if state.get("human_answer") or state.get("understanding_confirmed"):
        try:
            log_decision(state.get("workdir"), stage="solver.understanding",
                         decision="understanding_confirmed",
                         reason=str(state.get("human_answer") or "confirmed"))
        except Exception:
            pass
        return {"pending_approval": None, "needs_human": False,
                "understanding_confirmed": True, "errors": errors}
    if state.get("require_understanding"):
        errors.append("understanding: awaiting human confirm before solving")
        try:
            log_decision(state.get("workdir"), stage="solver.understanding",
                         decision="pending", reason="awaiting HITL-1 confirm")
        except Exception:
            pass
        return {"pending_approval": "understanding", "needs_human": True,
                "errors": errors}
    return {"pending_approval": None, "errors": errors}


def request_clarify(state: SolverState) -> dict:
    """Step 12 mid-run clarify: repair is stuck — ask the human instead of
    silently retrying into flag_human."""
    errors = _errors(state)
    failing = _failure_context(state)
    results = state.get("test_results") or []
    bad_ids = [str(r.get("test_id", "?")) for r in results
               if isinstance(r, dict) and not r.get("passed")][:5]
    question = (
        f"Repair stuck at attempt {int(state.get('attempt') or 0)}/{MAX_REPAIRS}. "
        f"Failing tests: {bad_ids or 'see log'}. {failing}"
        "Reply with guidance (or nothing to auto-retry)."
    )
    try:
        log_decision(state.get("workdir"), stage="solver.clarify",
                     decision="pending", reason=f"stuck: {bad_ids}")
    except Exception:
        pass
    return {"pending_approval": "clarify", "needs_human": True,
            "human_question": question, "clarify_asked": True, "errors": errors}


def initial_solution(state: SolverState) -> dict:
    """Heavy: write an initial correct solution for the problem spec."""
    errors = _errors(state)
    spec = state.get("problem_spec") or {}
    human_ctx = ""
    if state.get("human_answer"):
        human_ctx = (
            "Human clarification from problem review (must respect): "
            f"{state.get('human_answer')}\n"
        )
    prompt = (
        f"Write a correct, efficient Python solution for: {spec.get('statement')}\n"
        f"Signature: {spec.get('signature')}. Examples: {spec.get('examples')}. "
        f"Constraints: {spec.get('constraints')}.\n"
        f"{human_ctx}"
        f"{_failure_context(state)}"
        "Define exactly the function above (plus any helpers). Use only allowlisted "
        "stdlib modules (math, heapq, collections, bisect, itertools, functools, typing). "
        "No I/O, no other imports. Output raw Python code only, no markdown fences."
    )
    try:
        code = generate("initial_solution", prompt)
    except Exception as exc:
        errors.append(f"initial_solution: {exc}")
        return {"errors": errors}
    code = _strip_code_fences(str(code))
    if not code:
        errors.append("initial_solution: LLM returned empty code")
        return {"errors": errors}
    return {"solution_code": code, "errors": errors}


def static_scan(state: SolverState) -> dict:
    """Deterministic guard: AST scan solution before it ever executes."""
    verdict = scan(state.get("solution_code") or "")
    try:
        log_scan(state.get("workdir"), stage="solver.static_scan",
                 code=state.get("solution_code"), flags=verdict["flags"])
    except Exception:
        pass
    return {"safety_flags": verdict["flags"], "scan_safe": verdict["safe"]}


def run_tests_timed(state: SolverState, runs: int = TIMED_RUNS) -> dict:
    """Deterministic: execute solution N times, average per-test timings."""
    errors = _errors(state)
    spec = state.get("problem_spec")
    code = state.get("solution_code")
    workdir = state.get("workdir")
    if not isinstance(spec, dict) or not code or not workdir:
        errors.append("run_tests_timed: missing problem spec, solution, or workdir")
        return {"errors": errors, "test_results": []}
    try:
        prob = ProblemSpec.model_validate(spec)
    except Exception as exc:
        errors.append(f"run_tests_timed: invalid problem spec: {exc}")
        return {"errors": errors, "test_results": []}
    all_runs: list[list[dict]] = []
    try:
        for _ in range(max(1, runs)):
            results = safe_execute(code, prob, workdir)
            all_runs.append([r.model_dump() for r in results])
    except Exception as exc:
        errors.append(f"run_tests_timed: executor error: {exc}")
        return {"errors": errors, "test_results": []}
    if not all_runs or not all_runs[0]:
        errors.append("run_tests_timed: executor returned no results")
        return {"errors": errors, "test_results": []}
    n = len(all_runs[0])
    if any(len(run) != n for run in all_runs):
        errors.append("run_tests_timed: inconsistent result counts across runs")
        base = all_runs[0]
    else:
        base = all_runs[-1]
    averaged: list[dict] = []
    timings: list[float] = []
    for i in range(n):
        mean_ms = sum(run[i].get("time_ms", 0.0) for run in all_runs) / len(all_runs)
        row = dict(base[i])
        row["time_ms"] = mean_ms
        averaged.append(row)
        timings.append(mean_ms)
    try:
        log_run(state.get("workdir"), stage="solver.run_tests_timed",
                code=code, results=averaged, timings=timings)
    except Exception:
        pass
    return {
        "test_results": averaged,
        "timings_ms": timings,
        "errors": errors,
    }


def repair_solution(state: SolverState) -> dict:
    """Heavy: fix failing/blocked solution. Counts attempt; loop goes to static_scan."""
    errors = _errors(state)
    attempt = int(state.get("attempt") or 0) + 1
    spec = state.get("problem_spec") or {}
    out: dict = {"attempt": attempt, "errors": errors}
    if state.get("human_approved_opt") is False:
        # Step 12: a declined optimization retries repair once with the
        # decline reason as context, then gets a clean approval pass.
        out["human_approved_opt"] = None
    human_ctx = ""
    if state.get("human_answer"):
        human_ctx = (f"Human feedback (must respect): {state.get('human_answer')}\n")
    prompt = (
        "The Python solution below FAILS. Fix it so all tests pass.\n"
        f"Statement: {spec.get('statement')}\n"
        f"Signature (define exactly this function): {spec.get('signature')}\n"
        f"Examples: {spec.get('examples')}. Constraints: {spec.get('constraints')}.\n"
        f"{human_ctx}"
        f"{_failure_context(state)}"
        f"Current code:\n{state.get('solution_code') or ''}\n"
        f"Safety flags (must eliminate): {state.get('safety_flags') or []}\n"
        "Use only allowlisted stdlib modules "
        "(math, heapq, collections, bisect, itertools, functools, typing). "
        "No I/O, no other imports. Output raw Python code only, no markdown fences."
    )
    try:
        code = generate(
            "repair_solution", prompt, retry_count=min(attempt, 1))
    except Exception as exc:
        errors.append(f"repair_solution: {exc}")
        out.update({"errors": errors})
        return out
    code = _strip_code_fences(str(code))
    if not code:
        errors.append("repair_solution: LLM returned empty code")
        out.update({"errors": errors})
        return out
    out.update({"solution_code": code, "errors": errors})
    return out


# ---------------------------------------------------------------- optimizer (Step 9)

def _total_ms(timings: list | None) -> float:
    total = 0.0
    for t in timings or []:
        try:
            total += float(t)
        except (TypeError, ValueError):
            continue
    return total


def snapshot_baseline(state: SolverState) -> dict:
    """Freeze the passing solution/timings before any optimization attempt.

    Sticky-accepted: once a round accepts, later snapshots keep
    opt_status=accepted so a round-2 rollback cannot downgrade the verdict.
    """
    prior_status = state.get("opt_status")
    prior_reason = state.get("opt_reason") or ""
    # First snapshot freezes the original baseline for the final report.
    # Later snapshots (round 2) keep the original baseline so report.md
    # shows original -> best delta; round-2 acceptance still requires
    # beating the current best via accept_or_rollback's total comparison
    # against timings_ms (updated on round-1 accept).
    if state.get("baseline_code") is not None and int(state.get("opt_round") or 0) > 0:
        return {
            "opt_status": prior_status if prior_status == "accepted" else "pending",
            "opt_reason": prior_reason if prior_status == "accepted" else "",
            "optimized_code": None,
            "optimized_results": None,
            "optimized_timings": None,
        }
    return {
        "baseline_code": state.get("solution_code"),
        "baseline_results": list(state.get("test_results") or []),
        "baseline_timings": list(state.get("timings_ms") or []),
        "opt_round": int(state.get("opt_round") or 0),
        "opt_status": prior_status if prior_status == "accepted" else "pending",
        "opt_reason": prior_reason if prior_status == "accepted" else "",
        "optimized_code": None,
        "optimized_results": None,
        "optimized_timings": None,
    }


def analyze_complexity(state: SolverState) -> dict:
    """Heavy: Big-O + hotspot of the baseline solution (advisory only)."""
    errors = _errors(state)
    spec = state.get("problem_spec") or {}
    prompt = (
        "Analyze the time complexity of this Python solution. "
        "Reply with one line like 'O(n^2) — nested loops over nums'.\n"
        f"Statement: {spec.get('statement')}\n"
        f"Code:\n{state.get('solution_code') or ''}\n"
    )
    try:
        raw = generate("analyze_complexity", prompt)
    except Exception as exc:
        errors.append(f"analyze_complexity: {exc}")
        return {"complexity_before": "unknown", "errors": errors}
    text = _strip_code_fences(str(raw)).splitlines()
    first = (text[0].strip() if text else "unknown")[:200]
    return {"complexity_before": first or "unknown", "errors": errors}


def propose_optimization(state: SolverState) -> dict:
    """Heavy: propose a strictly faster solution, same function signature."""
    errors = _errors(state)
    attempt = int(state.get("opt_round") or 0) + 1
    spec = state.get("problem_spec") or {}
    prompt = (
        "Optimize this Python solution to be strictly FASTER while keeping "
        "identical behavior and the exact function signature.\n"
        f"Statement: {spec.get('statement')}\n"
        f"Signature (define exactly this function): {spec.get('signature')}\n"
        f"Complexity analysis: {state.get('complexity_before') or 'unknown'}\n"
        f"Current code:\n{state.get('solution_code') or ''}\n"
        "Use only allowlisted stdlib modules "
        "(math, heapq, collections, bisect, itertools, functools, typing). "
        "No I/O, no other imports. Output raw Python code only, no markdown fences."
    )
    try:
        code = generate("propose_optimization", prompt)
    except Exception as exc:
        errors.append(f"propose_optimization: {exc}")
        return {"opt_round": attempt, "errors": errors}
    code = _strip_code_fences(str(code))
    if not code:
        errors.append("propose_optimization: LLM returned empty code")
        return {"opt_round": attempt, "errors": errors}
    return {"opt_round": attempt, "optimized_code": code, "errors": errors}


def scan_optimized(state: SolverState) -> dict:
    """Deterministic guard on the optimization candidate."""
    verdict = scan(state.get("optimized_code") or "")
    try:
        log_scan(state.get("workdir"), stage="solver.scan_optimized",
                 code=state.get("optimized_code"), flags=verdict["flags"])
    except Exception:
        pass
    return {"opt_scan_safe": verdict["safe"], "safety_flags": verdict["flags"]}


def run_optimized(state: SolverState, runs: int = TIMED_RUNS) -> dict:
    """Deterministic: execute the optimization candidate N times, averaged."""
    errors = _errors(state)
    spec = state.get("problem_spec")
    code = state.get("optimized_code")
    workdir = state.get("workdir")
    if not isinstance(spec, dict) or not code or not workdir:
        errors.append("run_optimized: missing problem spec, optimized code, or workdir")
        return {"errors": errors, "optimized_results": None, "optimized_timings": None}
    try:
        prob = ProblemSpec.model_validate(spec)
    except Exception as exc:
        errors.append(f"run_optimized: invalid problem spec: {exc}")
        return {"errors": errors, "optimized_results": None, "optimized_timings": None}
    all_runs: list[list[dict]] = []
    try:
        for _ in range(max(1, runs)):
            results = safe_execute(code, prob, workdir)
            all_runs.append([r.model_dump() for r in results])
    except Exception as exc:
        errors.append(f"run_optimized: executor error: {exc}")
        return {"errors": errors, "optimized_results": None, "optimized_timings": None}
    if not all_runs or not all_runs[0]:
        errors.append("run_optimized: executor returned no results")
        return {"errors": errors, "optimized_results": None, "optimized_timings": None}
    base = all_runs[-1]
    averaged: list[dict] = []
    timings: list[float] = []
    for i in range(len(base)):
        mean_ms = sum(run[i].get("time_ms", 0.0) for run in all_runs) / len(all_runs)
        row = dict(base[i])
        row["time_ms"] = mean_ms
        averaged.append(row)
        timings.append(mean_ms)
    try:
        log_run(state.get("workdir"), stage="solver.run_optimized",
                code=code, results=averaged, timings=timings)
    except Exception:
        pass
    return {"optimized_results": averaged, "optimized_timings": timings, "errors": errors}


def accept_or_rollback(state: SolverState) -> dict:
    """Accept candidate only if all pass AND strictly faster; else keep baseline.

    Sticky-accepted: if a previous round already accepted, a later
    rollback keeps opt_status=accepted and the current (best) solution.
    Step 10: human approval enforced upstream in request_opt_approval;
    every decision is appended to audit.jsonl.
    """
    errors = _errors(state)
    baseline_timings = state.get("baseline_timings") or []
    opt_results = state.get("optimized_results")
    opt_timings = state.get("optimized_timings") or []

    def _rollback(reason: str, note_error: bool = False) -> dict:
        if note_error:
            errors.append(f"accept_or_rollback: {reason}")
        try:
            log_decision(state.get("workdir"), stage="solver.accept_or_rollback",
                         decision="rolled-back", reason=reason)
        except Exception:
            pass
        if state.get("opt_status") == "accepted":
            return {"opt_status": "accepted", "errors": errors}
        return {
            "opt_status": "rolled-back",
            "opt_reason": reason,
            "complexity_after": state.get("complexity_before") or "unknown",
            "what_changed": "",
            "errors": errors,
        }

    if not state.get("opt_scan_safe"):
        return _rollback(f"scan-blocked: {state.get('safety_flags') or []}")
    if state.get("human_approved_opt") is False:
        errors.append("accept_or_rollback: awaiting human approval — rolling back")
        return _rollback("awaiting human approval")
    if not opt_results:
        return _rollback("executor error", note_error=True)
    if not all(r.get("passed") for r in opt_results):
        return _rollback("optimized candidate failed tests")
    before_total = _total_ms(state.get("timings_ms") or baseline_timings)
    after_total = _total_ms(opt_timings)
    if not (after_total < before_total):
        return _rollback(
            f"not faster (before={before_total:.3f}ms after={after_total:.3f}ms)"
        )
    try:
        append_audit(state.get("workdir"), {
            "stage": "solver.accept_or_rollback",
            "decision": "accepted",
            "reason": f"speedup {before_total:.3f}ms -> {after_total:.3f}ms",
            "code_hash": __import__("hashlib").sha256(
                (state.get("optimized_code") or "").encode()).hexdigest()[:16],
            "timings_ms": list(opt_timings),
            "approved": bool(state.get("human_approved_opt") is not False),
        })
    except Exception:
        pass
    return {
        "solution_code": state.get("optimized_code"),
        "test_results": list(opt_results),
        "timings_ms": list(opt_timings),
        "opt_status": "accepted",
        "opt_reason": (
            f"speedup {before_total:.3f}ms -> {after_total:.3f}ms"
        ),
        "complexity_after": "see report",
        "what_changed": "accepted faster candidate",
        "errors": errors,
    }


def write_report(state: SolverState) -> dict:
    """Deterministic: render before/after report.md into the job dir."""
    errors = _errors(state)
    workdir = state.get("workdir")
    if not workdir:
        errors.append("write_report: missing workdir")
        return {"errors": errors}
    accepted = state.get("opt_status") == "accepted"
    markdown = build_report(
        baseline_results=state.get("baseline_results"),
        optimized_results=(state.get("test_results") if accepted else None),
        baseline_timings=state.get("baseline_timings"),
        optimized_timings=(state.get("timings_ms") if accepted else state.get("baseline_timings")),
        complexity_before=state.get("complexity_before") or "unknown",
        complexity_after=state.get("complexity_after") or "unknown",
        what_changed=state.get("what_changed") or "",
        opt_status=state.get("opt_status") or "rolled-back",
        opt_reason=state.get("opt_reason") or "",
    )
    try:
        scoped_write(str(workdir), "report.md", markdown)
    except Exception as exc:
        errors.append(f"write_report: {exc}")
        return {"errors": errors}
    try:
        append_audit(str(workdir), {
            "stage": "solver.write_report",
            "decision": state.get("opt_status") or "rolled-back",
            "timings_ms": list(state.get("timings_ms") or []),
        })
    except Exception:
        pass
    return {"report_path": str(Path(workdir) / "report.md"), "errors": errors}


def save_solution(state: SolverState) -> dict:
    """Persist passing solution.py into the job dir (best-effort, non-fatal)."""
    errors = _errors(state)
    try:
        scoped_write(str(state["workdir"]), "solution.py",
                     state.get("solution_code") or "")
    except Exception as exc:
        errors.append(f"save_solution: {exc}")
        return {"errors": errors}
    try:
        append_audit(str(state["workdir"]), {
            "stage": "solver.save_solution",
            "code_hash": __import__("hashlib").sha256(
                (state.get("solution_code") or "").encode()).hexdigest()[:16],
            "timings_ms": list(state.get("timings_ms") or []),
        })
    except Exception:
        pass
    return {"errors": errors}


def request_opt_approval(state: SolverState) -> dict:
    """Step 10 human approval gate before accept_or_rollback.

    Explicit human_approved_opt=False records pending approval; the
    downstream accept_or_rollback then forces a rollback (no auto-accept)
    while still writing report.md. None/True auto-passes for automated
    runs (legacy compat); prod should use strict graph
    (interrupt_before accept) or set False then resume with True.

    Step 12: a decline (False + human_answer reason) is logged as
    "declined" and routed to repair_solution when budget remains,
    else flag_human. Never silently dropped.
    """
    errors = _errors(state)
    if state.get("human_approved_opt") is False:
        reason = state.get("human_answer") or "human declined optimization"
        try:
            log_decision(state.get("workdir"), stage="solver.opt_approval",
                         decision="declined", reason=str(reason))
        except Exception:
            pass
        return {"pending_approval": "optimize", "errors": errors}
    return {"pending_approval": None, "errors": errors}


def flag_human(state: SolverState) -> dict:
    """Max repairs exhausted — stop for human review."""
    errors = _errors(state)
    errors.append(f"repair budget exhausted after {MAX_REPAIRS} attempts")
    return {"needs_human": True, "errors": errors}


# ---------------------------------------------------------------- routing

def route_after_load(state: SolverState) -> str:
    if state.get("needs_human"):
        return "flag_human"
    return "summarize_problem"


def route_after_understanding(state: SolverState) -> str:
    return "flag_human" if state.get("needs_human") else "initial_solution"


def _repair_or_human(state: SolverState) -> str:
    if int(state.get("attempt") or 0) < MAX_REPAIRS:
        if (int(state.get("attempt") or 0) >= 2
                and state.get("allow_clarify")
                and not state.get("clarify_asked")
                and not state.get("human_answer")):
            return "request_clarify"
        return "repair_solution"
    return "flag_human"


def route_after_scan(state: SolverState) -> str:
    if state.get("scan_safe"):
        return "run_tests_timed"
    return _repair_or_human(state)


def route_after_tests(state: SolverState) -> str:
    results = state.get("test_results") or []
    if results and all(r.get("passed") for r in results):
        return "snapshot_baseline"
    return _repair_or_human(state)


def route_after_opt_scan(state: SolverState) -> str:
    if state.get("opt_scan_safe"):
        return "run_optimized"
    return "request_opt_approval"


def route_after_opt_approval(state: SolverState) -> str:
    if state.get("human_approved_opt") is False:
        # Step 12 decline: retry repair with the reason as context when
        # budget remains, else stop for human review.
        if int(state.get("attempt") or 0) < MAX_REPAIRS:
            return "repair_solution"
        return "flag_human"
    return "accept_or_rollback"


def route_after_accept(state: SolverState) -> str:
    if (
        state.get("opt_status") == "accepted"
        and int(state.get("opt_round") or 0) < MAX_OPT_ROUNDS
    ):
        # Refresh baseline so round 2 must beat round 1.
        return "snapshot_baseline"
    return "write_report"


def build_solver_graph(checkpointer=None, with_interrupt: bool = False) -> object:
    """Step 10: SqliteSaver persistence + optional human interrupt.

    with_interrupt=True pauses before accept_or_rollback for manual review.
    Default False keeps automated runs non-blocking (explicit
    human_approved_opt=False still forces rollback in accept_or_rollback).
    """
    builder = StateGraph(SolverState)
    builder.add_node("load_validate", load_validate)
    builder.add_node("summarize_problem", summarize_problem)
    builder.add_node("request_understanding_approval", request_understanding_approval)
    builder.add_node("request_clarify", request_clarify)
    builder.add_node("initial_solution", initial_solution)
    builder.add_node("static_scan", static_scan)
    builder.add_node("run_tests_timed", run_tests_timed)
    builder.add_node("repair_solution", repair_solution)
    builder.add_node("snapshot_baseline", snapshot_baseline)
    builder.add_node("analyze_complexity", analyze_complexity)
    builder.add_node("propose_optimization", propose_optimization)
    builder.add_node("scan_optimized", scan_optimized)
    builder.add_node("run_optimized", run_optimized)
    builder.add_node("request_opt_approval", request_opt_approval)
    builder.add_node("accept_or_rollback", accept_or_rollback)
    builder.add_node("write_report", write_report)
    builder.add_node("save_solution", save_solution)
    builder.add_node("flag_human", flag_human)
    builder.add_edge(START, "load_validate")
    builder.add_conditional_edges(
        "load_validate", route_after_load,
        {"summarize_problem": "summarize_problem",
         "flag_human": "flag_human"})
    builder.add_edge("summarize_problem", "request_understanding_approval")
    builder.add_conditional_edges(
        "request_understanding_approval", route_after_understanding,
        {"initial_solution": "initial_solution",
         "flag_human": "flag_human"})
    builder.add_edge("request_clarify", "flag_human")
    builder.add_edge("initial_solution", "static_scan")
    builder.add_conditional_edges(
        "static_scan", route_after_scan,
        {"run_tests_timed": "run_tests_timed",
         "repair_solution": "repair_solution",
         "request_clarify": "request_clarify",
         "flag_human": "flag_human"})
    builder.add_conditional_edges(
        "run_tests_timed", route_after_tests,
        {"snapshot_baseline": "snapshot_baseline",
         "repair_solution": "repair_solution",
         "request_clarify": "request_clarify",
         "flag_human": "flag_human"})
    builder.add_edge("repair_solution", "static_scan")
    builder.add_edge("snapshot_baseline", "analyze_complexity")
    builder.add_edge("analyze_complexity", "propose_optimization")
    builder.add_edge("propose_optimization", "scan_optimized")
    builder.add_conditional_edges(
        "scan_optimized", route_after_opt_scan,
        {"run_optimized": "run_optimized",
         "request_opt_approval": "request_opt_approval"})
    builder.add_edge("run_optimized", "request_opt_approval")
    builder.add_conditional_edges(
        "request_opt_approval", route_after_opt_approval,
        {"accept_or_rollback": "accept_or_rollback",
         "repair_solution": "repair_solution",
         "flag_human": "flag_human"})
    builder.add_conditional_edges(
        "accept_or_rollback", route_after_accept,
        {"snapshot_baseline": "snapshot_baseline",
         "write_report": "write_report"})
    builder.add_edge("write_report", "save_solution")
    builder.add_edge("save_solution", END)
    builder.add_edge("flag_human", END)
    saver = checkpointer or get_checkpointer()
    if with_interrupt:
        return builder.compile(checkpointer=saver, interrupt_before=["accept_or_rollback"])
    return builder.compile(checkpointer=saver)


solver_app = build_solver_graph()


def build_solver_graph_strict() -> object:
    """Prod graph pausing before accept_or_rollback for human approval."""
    return build_solver_graph(with_interrupt=True)
