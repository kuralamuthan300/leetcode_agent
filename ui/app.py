"""Streamlit UI shell (Step 11). No LLM logic; delegates to ui.agent_client."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ui.agent_client import (
    CREATOR_STAGES,
    SOLVER_STAGES,
    completed_stages_from_stream,
    format_test_rows,
    get_state,
    problem_tab_data,
    run_creator,
    run_solver,
    solution_tab_data,
    stage_states,
    stages_for_mode,
    status_for_state,
    stream_run,
)

EXAMPLE_PROBLEM = Path("workspace/problems/examples/problem_id_two_sum_demo.json")
CATEGORIES = ["arrays", "dp", "graphs", "strings", "math", "trees"]


def _load_example_problem() -> dict:
    if EXAMPLE_PROBLEM.exists():
        return json.loads(EXAMPLE_PROBLEM.read_text(encoding="utf-8"))
    return {}


def _status_banner(state: dict) -> None:
    status = status_for_state(state)
    pill = {
        "idle": "⚪ idle",
        "running": "🔵 running",
        "awaiting approval": "🟡 awaiting approval",
        "done": "✅ done",
        "needs_human": "🔴 needs_human",
    }.get(status, status)
    st.markdown(f"**Status:** {pill}")
    cols = st.columns(4)
    cols[0].write(f"job_id: `{state.get('job_id', '-')}`")
    cols[1].write(f"mode: `{state.get('mode', '-')}`")
    cols[2].write(f"attempt: `{state.get('attempt', 0)}`")
    cols[3].write(f"opt_round: `{state.get('opt_round', state.get('optRound', 0))}`")


def _stage_stepper(stages: list[str], completed: list[str], attempt: int = 0) -> None:
    mapping = stage_states(stages, completed)
    icons = {"done": "✅", "active": "🔵", "todo": "⚪"}
    cols = st.columns(len(stages) if stages else 1)
    for col, name in zip(cols, stages):
        with col:
            st.write(f"{icons[mapping[name]]}")
            st.caption(name)
    done_n = sum(1 for v in mapping.values() if v == "done")
    st.progress(done_n / max(len(stages), 1))
    if attempt and attempt >= 2:
        st.warning(f"repair badge: attempt {attempt}/4")


def main() -> None:
    st.title("LeetCode Agent")
    with st.sidebar:
        mode = st.radio("mode", ["creator", "solver", "auto", "review"], index=0)
        category = st.selectbox("category", CATEGORIES, index=0)
        difficulty = st.radio("difficulty", ["easy", "medium", "hard"], index=0)
        num_tests = st.number_input("num_tests", min_value=1, max_value=50, value=8, step=1)
        constraints_hint = st.text_input("constraints_hint", value="")
        uploaded = st.file_uploader("upload json", type=["json"])
        example_choice = st.selectbox("example picker", ["none", "two_sum_demo"], index=0)
        pasted = st.text_area("or paste json here", height=120)

    state: dict = st.session_state.get("agent_state", {})
    completed: list[str] = st.session_state.get("completed_stages", [])

    _status_banner(state)

    stages = stages_for_mode(mode if mode in ("creator", "solver") else "creator")
    if mode == "solver":
        stages = SOLVER_STAGES
    with st.status("Stages", expanded=False):
        _stage_stepper(stages, completed, attempt=int(state.get("attempt", 0) or 0))

    tab_problem, tab_solution, tab_tests = st.tabs(["Problem", "Solution", "Tests"])
    with tab_problem:
        pdata = problem_tab_data(state)
        st.subheader(pdata.get("title", "(no problem yet)"))
        st.write(pdata.get("statement", ""))
        if pdata.get("signature"):
            st.code(pdata["signature"], language="python")
        if pdata.get("constraints"):
            st.write("Constraints:")
            for c in pdata["constraints"]:
                st.write(f"- {c}")
        if pdata.get("examples"):
            st.json(pdata["examples"])
    with tab_solution:
        code = solution_tab_data(state)
        if code:
            st.code(code, language="python")
            st.button("copy", help="select code above to copy")
        else:
            st.write("(no solution yet)")
    with tab_tests:
        show_hidden = st.toggle("show hidden", value=False)
        rows = format_test_rows(state)
        spec_ids_hidden = set()
        spec = state.get("problem_spec") or {}
        if isinstance(spec, dict):
            for tc in spec.get("testcases", []) or []:
                if isinstance(tc, dict) and tc.get("is_hidden"):
                    spec_ids_hidden.add(tc.get("id"))
        if not show_hidden:
            rows = [r for r in rows if r["test_id"] not in spec_ids_hidden]
        st.write(f"{len(rows)} testcases (num_tests={int(num_tests)})")
        st.dataframe(rows, use_container_width=True)

    col_run, col_refresh = st.columns(2)
    with col_run:
        if st.button("Run"):
            thread_id = state.get("job_id") or "ui-job-1"
            try:
                payload: dict = {}
                if uploaded is not None:
                    payload = json.loads(uploaded.getvalue().decode("utf-8"))
                elif pasted.strip():
                    payload = json.loads(pasted)
                elif example_choice == "two_sum_demo":
                    payload = _load_example_problem()
                if mode in ("creator", "auto"):
                    req = payload if payload else {
                        "category": category,
                        "difficulty": difficulty,
                        "language": "python",
                        "num_tests": int(num_tests),
                        "constraints_hint": constraints_hint,
                    }
                    req.setdefault("num_tests", int(num_tests))
                    from src.creator_graph import creator_app

                    chunks = list(
                        stream_run(
                            creator_app,
                            {"requirements_path": _save_tmp(req, "req")},
                            thread_id,
                        )
                    )
                    stages = CREATOR_STAGES
                    completed = completed_stages_from_stream(chunks)
                    state = get_state(thread_id, app=creator_app)
                elif mode == "solver":
                    prob = payload if payload else _load_example_problem()
                    from src.solver_graph import solver_app

                    chunks = list(
                        stream_run(
                            solver_app,
                            {"problem_path": _save_tmp(prob, "prob")},
                            thread_id,
                        )
                    )
                    stages = SOLVER_STAGES
                    completed = completed_stages_from_stream(chunks)
                    state = get_state(thread_id, app=solver_app)
                else:  # review: read-only
                    from src.graph import build_graph

                    state = get_state(thread_id, app=build_graph())
                st.session_state["agent_state"] = state
                st.session_state["completed_stages"] = completed
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - surface to UI
                st.error(f"run failed: {exc}")
    with col_refresh:
        if st.button("Refresh state"):
            st.rerun()


def _save_tmp(payload: dict, prefix: str) -> str:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json", prefix=f"ui_{prefix}_")
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


# Thin re-exports so tests / external callers can use ui.app helpers.
__all__ = ["main", "run_creator", "run_solver", "get_state"]


if __name__ == "__main__":
    main()
