"""Streamlit UI shell (Steps 11-12). No LLM logic; delegates to ui.agent_client."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from ui.agent_client import (
    CREATOR_STAGES,
    SOLVER_STAGES,
    clarify_data,
    completed_stages_from_stream,
    final_approval_data,
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
    understanding_card_data,
    update_and_resume,
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


def _current_app(mode: str):
    if mode == "solver":
        from src.solver_graph import solver_app

        return solver_app, "solver"
    from src.creator_graph import creator_app

    return creator_app, "creator"


def _understanding_card(state: dict) -> None:
    """Step 12 HITL-1: verify understanding + agent doubts before drafting."""
    if state.get("pending_approval") != "understanding":
        return
    card = understanding_card_data(state)
    st.warning("HITL-1: confirm understanding before the agent drafts")
    st.write(card["summary"])
    if card["question"]:
        st.write(f"Agent asks: {card['question']}")
    num = st.number_input("num_tests (editable)", min_value=1, max_value=50,
                          value=int(card.get("num_tests") or 8), step=1,
                          key="hitl_num_tests")
    answer = st.text_input("corrections / answers (optional)",
                           key="hitl_understanding_answer")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Confirm & Continue", key="hitl_confirm"):
            _resume_hitl({"human_answer": answer or "confirmed",
                          "understanding_confirmed": True,
                          "hitl_num_tests": int(num)})
    with col_b:
        if st.button("Answer + Continue", key="hitl_answer"):
            _resume_hitl({"human_answer": answer,
                          "understanding_confirmed": True,
                          "hitl_num_tests": int(num)})


def _clarify_popup(state: dict) -> None:
    """Step 12 mid-run clarify: repair stuck — answer or skip/auto-retry."""
    if state.get("pending_approval") != "clarify":
        return
    info = clarify_data(state)
    st.error(f"Clarify needed (repair attempt "
             f"{info['attempt']}). Failing: "
             f"{[r.get('test_id') for r in info['failing']]}")
    if info["question"]:
        st.write(info["question"])
    answer = st.text_input("guidance for repair (optional)",
                           key="hitl_clarify_answer")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Submit + Resume", key="hitl_clarify_submit"):
            _resume_hitl({"human_answer": answer})
    with col_b:
        if st.button("Skip / auto-retry", key="hitl_clarify_skip"):
            _resume_hitl({"human_answer": ""})


def _final_approval_view(state: dict) -> None:
    """Step 12 HITL-2: approve or decline the final problem/solution."""
    if state.get("pending_approval") not in ("promote", "optimize"):
        return
    info = final_approval_data(state)
    st.warning(f"HITL-2: final approval "
               f"(pass rate {info['pass_rate']}, total {info['total_ms']} ms)")
    reason = st.text_input("decline reason (required to decline)",
                           key="hitl_decline_reason")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Approve", key="hitl_approve"):
            _resume_hitl({"human_approved_promote": True,
                          "human_approved_opt": True,
                          "human_answer": ""})
    with col_b:
        if st.button("Decline + reason", key="hitl_decline"):
            if not reason.strip():
                st.error("give a decline reason first")
            else:
                _resume_hitl({"human_approved_promote": False,
                              "human_approved_opt": False,
                              "human_answer": reason})


def _resume_hitl(values: dict) -> None:
    """Apply human values to the paused thread and resume the run."""
    thread_id = (st.session_state.get("agent_state") or {}).get("job_id") \
        or "ui-job-1"
    payload = st.session_state.get("last_payload") or {}
    mode = st.session_state.get("last_mode") or "creator"
    num = values.pop("hitl_num_tests", None)
    if num is not None:
        cur = st.session_state.get("agent_state") or {}
        req = dict(cur.get("requirements") or {})
        if req:
            values["requirements"] = {**req, "num_tests": int(num)}
    try:
        app, _ = _current_app(mode)
        new_state = update_and_resume(app, thread_id, values, payload)
    except Exception as exc:  # noqa: BLE001 - surface to UI
        st.error(f"resume failed: {exc}")
        return
    st.session_state["agent_state"] = new_state
    st.rerun()


def main() -> None:
    st.title("LeetCode Agent")
    with st.sidebar:
        with st.expander("What should I provide?", expanded=False):
            st.markdown(
                "- **creator**: pick `category` + `difficulty` + `num_tests`, "
                "optionally a `constraints_hint` (e.g. `n <= 10^5`). "
                "No file needed — the agent drafts a new problem.\n"
                "- **solver**: give a `problem_id_*.json` via upload, paste, "
                "or the `two_sum_demo` example. Sidebar category/difficulty "
                "are ignored.\n"
                "- **auto**: same as creator (router dispatch is future work).\n"
                "- **review**: read-only — enter the `job_id` of a past run "
                "to inspect its state, `audit.jsonl`, and `report.md`."
            )
        mode = st.radio(
            "mode",
            ["creator", "solver", "auto", "review"], index=0,
            help="creator: requirements -> new problem JSON. "
                 "solver: problem JSON -> solution + timings + report. "
                 "auto: currently same as creator. "
                 "review: read-only inspection of a past job_id.")
        category = st.selectbox(
            "category", CATEGORIES, index=0,
            help="Problem family to invent (e.g. arrays, dp, graphs). "
                 "Used only in creator/auto mode.")
        difficulty = st.radio(
            "difficulty", ["easy", "medium", "hard"], index=0,
            help="Target difficulty, enforced on the generated problem. "
                 "Used only in creator/auto mode.")
        num_tests = st.number_input(
            "num_tests", min_value=1, max_value=50, value=8, step=1,
            help="How many testcases to generate (default 8). "
                 "Stored as RequirementsSpec.num_tests. You can still "
                 "change it on the HITL-1 understanding card.")
        st.caption("Tip: 8 is a good default; use 4–5 for a quick trial run.")
        constraints_hint = st.text_input(
            "constraints_hint", value="",
            help="Free text like `n <= 10^5` or `must use O(1) space`. "
                 "Shapes perf-test sizes. Optional — the agent asks on the "
                 "understanding card if it is missing.")
        uploaded = st.file_uploader(
            "upload json", type=["json"],
            help="Creator: a requirements JSON "
                 "({job_id, category, difficulty, language, num_tests, "
                 "constraints_hint, extra}). Solver: a problem_id_*.json. "
                 "Leave empty to use the sidebar fields / example instead.")
        st.caption("Solver expects a problem file; creator works without one.")
        example_choice = st.selectbox(
            "example picker", ["none", "two_sum_demo"], index=0,
            help="Load the bundled Two Sum problem instead of uploading. "
                 "Handy for trying solver mode instantly.")
        pasted = st.text_area(
            "or paste json here", height=120,
            help="Paste a requirements JSON (creator) or problem JSON "
                 "(solver). Takes precedence over the example picker when "
                 "non-empty.")

    state: dict = st.session_state.get("agent_state", {})
    completed: list[str] = st.session_state.get("completed_stages", [])

    _status_banner(state)

    _understanding_card(state)
    _clarify_popup(state)
    _final_approval_view(state)

    stages = stages_for_mode(
        mode if mode in ("creator", "solver") else "creator", include_hitl=True)
    if mode == "solver":
        stages = stages_for_mode("solver", include_hitl=True)
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

                    invoke_payload = {"requirements_path": _save_tmp(req, "req"),
                                      "require_understanding": True}
                    chunks = list(
                        stream_run(creator_app, invoke_payload, thread_id))
                    stages = stages_for_mode("creator", include_hitl=True)
                    completed = completed_stages_from_stream(chunks)
                    state = get_state(thread_id, app=creator_app)
                    st.session_state["last_payload"] = invoke_payload
                    st.session_state["last_mode"] = "creator"
                elif mode == "solver":
                    prob = payload if payload else _load_example_problem()
                    from src.solver_graph import solver_app

                    invoke_payload = {"problem_path": _save_tmp(prob, "prob"),
                                      "require_understanding": True}
                    chunks = list(
                        stream_run(solver_app, invoke_payload, thread_id))
                    stages = stages_for_mode("solver", include_hitl=True)
                    completed = completed_stages_from_stream(chunks)
                    state = get_state(thread_id, app=solver_app)
                    st.session_state["last_payload"] = invoke_payload
                    st.session_state["last_mode"] = "solver"
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
