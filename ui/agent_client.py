"""Thin UI wrapper over creator/solver LangGraph apps (Step 11).

No LLM logic here. All reasoning lives in src/creator_graph.py,
src/solver_graph.py, src/graph.py. UI only invokes apps with
thread_id=job_id on SqliteSaver and formats AgentState for display.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Iterator

CREATOR_STAGES: list[str] = [
    "intake_validate",
    "draft_problem",
    "generate_tests",
    "format_dedup",
    "oracle_solution",
    "static_scan",
    "run_tests",
    "request_promote_approval",
    "promote_save",
]

SOLVER_STAGES: list[str] = [
    "load_validate",
    "initial_solution",
    "static_scan",
    "run_tests_timed",
    "snapshot_baseline",
    "analyze_complexity",
    "propose_optimization",
    "scan_optimized",
    "run_optimized",
    "request_opt_approval",
    "accept_or_rollback",
    "write_report",
    "save_solution",
]


def stages_for_mode(mode: str) -> list[str]:
    if mode == "creator":
        return list(CREATOR_STAGES)
    if mode == "solver":
        return list(SOLVER_STAGES)
    return list(CREATOR_STAGES)


def status_for_state(state: dict[str, Any] | None) -> str:
    """Map AgentState to banner pill: idle|running|awaiting approval|done|needs_human."""
    if not state:
        return "idle"
    if state.get("needs_human"):
        pending = state.get("pending_approval")
        if pending in ("promote", "optimize", "understanding", "clarify"):
            return "awaiting approval"
        return "needs_human"
    if state.get("errors"):
        return "needs_human"
    problem_spec = state.get("problem_spec")
    solution_code = state.get("solution_code")
    test_results = state.get("test_results") or []
    if problem_spec is not None or solution_code is not None or test_results:
        return "done"
    if state.get("job_id") or state.get("mode") or state.get("workdir"):
        return "running"
    return "idle"


def stage_states(
    stages: list[str], completed: list[str]
) -> dict[str, str]:
    """Map each stage to done|active|todo for stepper rendering."""
    done = set(completed)
    out: dict[str, str] = {}
    seen_todo = False
    for i, name in enumerate(stages):
        if name in done:
            out[name] = "done"
        elif not seen_todo:
            out[name] = "active"
            seen_todo = True
        else:
            out[name] = "todo"
    # Edge: all completed -> mark last done (no active)
    if all(s in done for s in stages):
        for s in stages:
            out[s] = "done"
    _ = i
    return out


def _write_dict_to_temp(payload: dict[str, Any], suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="ui_job_")
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def run_creator(
    requirements_dict: dict[str, Any] | str | Path,
    thread_id: str,
    app: Any | None = None,
) -> dict[str, Any]:
    """Invoke creator_app. requirements_dict may be a dict or a path."""
    if app is None:
        from src.creator_graph import creator_app as app  # lazy: keeps tests docker-free
    if isinstance(requirements_dict, dict):
        req_path = _write_dict_to_temp(requirements_dict, ".json")
    else:
        req_path = str(requirements_dict)
    return app.invoke(
        {"requirements_path": req_path},
        config={"configurable": {"thread_id": thread_id}},
    )


def run_solver(
    problem_dict: dict[str, Any] | str | Path,
    thread_id: str,
    app: Any | None = None,
) -> dict[str, Any]:
    """Invoke solver_app. problem_dict may be a dict or a path."""
    if app is None:
        from src.solver_graph import solver_app as app  # lazy import
    if isinstance(problem_dict, dict):
        prob_path = _write_dict_to_temp(problem_dict, ".json")
    else:
        prob_path = str(problem_dict)
    return app.invoke(
        {"problem_path": prob_path},
        config={"configurable": {"thread_id": thread_id}},
    )


def get_state(thread_id: str, app: Any | None = None) -> dict[str, Any]:
    """Return latest snapshot values for thread_id (SqliteSaver checkpointer)."""
    if app is None:
        from src.graph import build_graph  # lazy import
        app = build_graph()
    snapshot = app.get_state(config={"configurable": {"thread_id": thread_id}})
    values = getattr(snapshot, "values", snapshot)
    if isinstance(values, dict):
        return values
    return {}


def stream_run(
    app: Any,
    payload: dict[str, Any],
    thread_id: str,
) -> Iterator[dict[str, Any]]:
    """Yield node-update chunks from app.stream(stream_mode='updates')."""
    yield from app.stream(
        payload,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    )


def completed_stages_from_stream(chunks: list[dict[str, Any]]) -> list[str]:
    """Extract node names from stream_mode='updates' chunks in order."""
    seen: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            for key in chunk:
                if key not in seen:
                    seen.append(key)
    return seen


def format_test_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Build Tests-tab rows: test_id|input|expected|actual|pass|mean_ms."""
    results = state.get("test_results") or []
    timings = state.get("timings_ms") or {}
    if isinstance(timings, list):
        timings = {}
    rows: list[dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        tid = r.get("test_id", r.get("id", "?"))
        mean_ms = timings.get(tid) if isinstance(timings, dict) else None
        if mean_ms is None:
            mean_ms = r.get("mean_ms", r.get("time_ms"))
        rows.append(
            {
                "test_id": tid,
                "input": r.get("input", ""),
                "expected": r.get("expected", ""),
                "actual": r.get("actual", r.get("output", "")),
                "pass": bool(r.get("pass", r.get("passed", False))),
                "mean_ms": mean_ms,
            }
        )
    return rows


def problem_tab_data(state: dict[str, Any]) -> dict[str, Any]:
    """Extract Problem-tab fields from AgentState.problem_spec."""
    spec = state.get("problem_spec") or {}
    if not isinstance(spec, dict):
        return {"title": "", "statement": "", "signature": "", "constraints": [], "examples": []}
    return {
        "title": spec.get("title", ""),
        "statement": spec.get("statement", ""),
        "signature": spec.get("signature", ""),
        "constraints": spec.get("constraints", []),
        "examples": spec.get("examples", []),
    }


def solution_tab_data(state: dict[str, Any]) -> str:
    """Extract python solution code for Solution tab."""
    code = state.get("solution_code")
    return code if isinstance(code, str) else ""
