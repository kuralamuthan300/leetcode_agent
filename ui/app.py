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
    pending_action,
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
    st.warning("HITL-1: the agent has open questions — nothing runs until you answer")
    with st.expander("My understanding (FYI)", expanded=False):
        st.write(card["summary"])
    st.markdown("**Questions for you:**")
    for line in (card["question"] or "").splitlines():
        if line.strip():
            st.markdown(line)
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
                "optionally a `constraints_hint` (e.g. `n <= 10^5`) and "
                "`extra` notes. No file needed — the agent drafts a new problem.\n"
                "- **solver**: choose the `two_sum_demo` example or fill the "
                "custom-problem form (`title`, `function_name`, `signature`, "
                "`statement`, `constraints`, testcases as input/expected JSON).\n"
                "- **auto**: same as creator (router dispatch is future work).\n"
                "- **review**: read-only — enter the `job_id` of a past run "
                "to inspect its state, `audit.jsonl`, and `report.md`."
            )
        mode = st.radio(
            "mode",
            ["creator", "solver", "auto", "review"], index=0,
            help="creator: form fields -> new problem JSON. "
                 "solver: example or custom form -> solution + timings. "
                 "auto: currently same as creator. "
                 "review: read-only inspection of a past job_id.")
        # Defaults so the results section below always has values.
        category, difficulty = "arrays", "easy"
        num_tests, constraints_hint, extra, language = 8, "", "", "python"
        solver_source = "example: two_sum_demo"
        prob_title = prob_func = prob_sig = prob_stmt = ""
        prob_constraints = ""
        prob_rows: list[tuple[str, str]] = []
        review_job_id = ""
        if mode in ("creator", "auto"):
            category = st.selectbox(
                "category", CATEGORIES, index=0,
                help="Problem family to invent (e.g. arrays, dp, graphs).")
            difficulty = st.radio(
                "difficulty", ["easy", "medium", "hard"], index=0,
                help="Target difficulty, enforced on the generated problem.")
            num_tests = st.number_input(
                "num_tests", min_value=1, max_value=50, value=8, step=1,
                help="How many testcases to generate (default 8). "
                     "Stored as RequirementsSpec.num_tests. You can still "
                     "change it on the HITL-1 understanding card.")
            st.caption("Tip: 8 is a good default; use 4–5 for a quick trial run.")
            language = st.selectbox(
                "language", ["python"], index=0,
                help="Solution language. Only python is supported.")
            constraints_hint = st.text_input(
                "constraints_hint", value="",
                help="Free text like `n <= 10^5` or `must use O(1) space`. "
                     "Shapes perf-test sizes. Optional — the agent asks on the "
                     "understanding card if it is missing.")
            extra = st.text_area(
                "extra notes (optional)", height=60,
                help="Anything else the problem must satisfy, e.g. "
                     "`must run in O(n) time`.")
        elif mode == "solver":
            solver_source = st.radio(
                "problem source",
                ["example: two_sum_demo", "custom form"], index=0,
                help="Try the bundled Two Sum instantly, or describe your "
                     "own problem in the form below.")
            if solver_source == "custom form":
                prob_title = st.text_input(
                    "title", value="",
                    help="Short problem name, e.g. `Two Sum`.")
                prob_func = st.text_input(
                    "function_name", value="",
                    help="snake_case python function to implement, "
                         "e.g. `two_sum`.")
                prob_sig = st.text_input(
                    "signature", value="",
                    help="Full typed signature defining function_name, e.g. "
                         "`def two_sum(nums: list[int], target: int) -> list[int]:`.")
                prob_stmt = st.text_area(
                    "statement", height=80,
                    help="What the function must compute, in plain words.")
                prob_constraints = st.text_area(
                    "constraints (one per line)", height=60,
                    help="E.g. `2 <= n <= 10^4` newline `Only one valid answer exists.`.")
                st.markdown("**testcases** — input/expected JSON per row "
                            "(blank rows are skipped):")
                for i in range(1, 5):
                    c1, c2 = st.columns(2)
                    with c1:
                        raw_in = st.text_input(
                            f"test {i} input", value="", key=f"tc_in_{i}",
                            help='JSON object mapping parameter names to values, '
                                 'e.g. `{"nums": [2, 7], "target": 9}`.')
                    with c2:
                        raw_exp = st.text_input(
                            f"test {i} expected", value="", key=f"tc_exp_{i}",
                            help="Expected return value as JSON, e.g. `[0, 1]`.")
                    prob_rows.append((raw_in, raw_exp))
        else:  # review
            review_job_id = st.text_input(
                "job_id", value="",
                help="Thread id of a past run (shown in its status banner). "
                     "Leave empty to reuse the current session state.")

    state: dict = st.session_state.get("agent_state", {})
    completed: list[str] = st.session_state.get("completed_stages", [])

    _status_banner(state)
    elapsed = st.session_state.get("last_elapsed_s")
    if elapsed is not None:
        st.caption(f"Last run took {elapsed:.1f}s")

    action = pending_action(state)
    if action is not None:
        with st.container(border=True):
            st.subheader("Action required")
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
        if pdata.get("title"):
            st.download_button(
                "Download problem JSON",
                data=json.dumps(state.get("problem_spec") or {}, indent=2),
                file_name=f"{(state.get('problem_spec') or {}).get('id', 'problem')}.json",
                mime="application/json", key="dl_problem")
    with tab_solution:
        code = solution_tab_data(state)
        if code:
            st.code(code, language="python")
            job = state.get("job_id") or "solution"
            st.download_button(
                "Download solution.py", data=code,
                file_name=f"{job}_solution.py", mime="text/x-python",
                key="dl_solution")
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
        if mode in ("creator", "auto"):
            st.write(f"{len(rows)} testcases (num_tests={int(num_tests)})")
        else:
            st.write(f"{len(rows)} testcases")
        st.dataframe(rows, use_container_width=True)

    col_run, col_refresh = st.columns(2)
    with col_run:
        if st.button("Run"):
            thread_id = state.get("job_id") or "ui-job-1"
            try:
                if mode in ("creator", "auto"):
                    try:
                        req = build_requirements(
                            category, difficulty, int(num_tests),
                            constraints_hint, extra, language)
                    except ValueError as exc:
                        st.error(f"invalid requirements: {exc}")
                        return
                    from src.creator_graph import creator_app

                    invoke_payload = {"requirements_path": _save_tmp(req, "req"),
                                      "require_understanding": True}
                    chunks = _stream_with_progress(
                        creator_app, invoke_payload, thread_id,
                        expected=len(stages_for_mode("creator", include_hitl=True)))
                    stages = stages_for_mode("creator", include_hitl=True)
                    completed = completed_stages_from_stream(chunks)
                    state = get_state(thread_id, app=creator_app)
                    st.session_state["last_payload"] = invoke_payload
                    st.session_state["last_mode"] = "creator"
                elif mode == "solver":
                    if solver_source == "custom form":
                        try:
                            prob = build_problem(
                                prob_title, prob_func, prob_sig, prob_stmt,
                                prob_constraints.splitlines(), prob_rows)
                        except ValueError as exc:
                            st.error(f"invalid problem: {exc}")
                            return
                    else:
                        prob = _load_example_problem()
                    from src.solver_graph import solver_app

                    invoke_payload = {"problem_path": _save_tmp(prob, "prob"),
                                      "require_understanding": True}
                    chunks = _stream_with_progress(
                        solver_app, invoke_payload, thread_id,
                        expected=len(stages_for_mode("solver", include_hitl=True)))
                    stages = stages_for_mode("solver", include_hitl=True)
                    completed = completed_stages_from_stream(chunks)
                    state = get_state(thread_id, app=solver_app)
                    st.session_state["last_payload"] = invoke_payload
                    st.session_state["last_mode"] = "solver"
                else:  # review: read-only
                    from src.graph import build_graph

                    thread_id = review_job_id.strip() or thread_id
                    state = get_state(thread_id, app=build_graph())
                st.session_state["agent_state"] = state
                st.session_state["completed_stages"] = completed
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - surface to UI
                st.error(f"run failed: {exc}")
    with col_refresh:
        if st.button("Refresh state"):
            st.rerun()


def _new_job_id(prefix: str) -> str:
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def build_requirements(
    category: str,
    difficulty: str,
    num_tests: int = 8,
    constraints_hint: str = "",
    extra: str = "",
    language: str = "python",
    job_id: str | None = None,
) -> dict:
    """Build a RequirementsSpec dict purely from form fields (no upload)."""
    req = {
        "job_id": job_id or _new_job_id("req"),
        "category": (category or "").strip(),
        "difficulty": difficulty,
        "language": (language or "python").strip() or "python",
        "num_tests": int(num_tests),
        "constraints_hint": (constraints_hint or "").strip() or None,
        "extra": (extra or "").strip() or None,
    }
    if not req["category"]:
        raise ValueError("category is required")
    if req["difficulty"] not in ("easy", "medium", "hard"):
        raise ValueError("difficulty must be easy, medium, or hard")
    if not 1 <= req["num_tests"] <= 100:
        raise ValueError("num_tests must be between 1 and 100")
    return req


def build_problem(
    title: str,
    function_name: str,
    signature: str,
    statement: str,
    constraints: list[str],
    cases: list[tuple[str, str]],
) -> dict:
    """Build a ProblemSpec dict from form fields.

    cases: list of (input_json, expected_json) raw strings, one per testcase.
    """
    import re

    title = (title or "").strip()
    function_name = (function_name or "").strip()
    signature = (signature or "").strip()
    statement = (statement or "").strip()
    if not title:
        raise ValueError("title is required")
    if not function_name or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", function_name):
        raise ValueError("function_name must be a valid python identifier")
    if not signature or function_name not in signature:
        raise ValueError("signature must define function_name "
                         f"(expected something like `def {function_name}(...)`)")
    if not statement:
        raise ValueError("statement is required")
    parsed: list[tuple[object, object]] = []
    for i, (raw_in, raw_exp) in enumerate(cases, start=1):
        if not raw_in.strip() and not raw_exp.strip():
            continue  # blank row = unused
        try:
            inp = json.loads(raw_in)
        except Exception as exc:
            raise ValueError(f"testcase {i}: input is not valid JSON: {exc}")
        try:
            exp = json.loads(raw_exp)
        except Exception as exc:
            raise ValueError(f"testcase {i}: expected is not valid JSON: {exc}")
        if not isinstance(inp, dict):
            raise ValueError(
                f"testcase {i}: input must be a JSON object mapping "
                f"parameter names to values (e.g. {{\"nums\": [2, 7]}})")
        parsed.append((inp, exp))
    if not parsed:
        raise ValueError("add at least one testcase (input + expected JSON)")
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") or "problem"
    import uuid

    pid = f"{slug}_{uuid.uuid4().hex[:6]}"
    examples = [{"input": parsed[0][0], "output": parsed[0][1],
                 "explanation": ""}]
    testcases = [{"id": f"t{i}", "input": inp, "expected": exp,
                  "is_hidden": False, "timeout_ms": 2000}
                 for i, (inp, exp) in enumerate(parsed, start=1)]
    return {
        "id": pid,
        "title": title,
        "difficulty": "easy",
        "statement": statement,
        "function_name": function_name,
        "signature": signature,
        "examples": examples,
        "constraints": [c for c in (constraints or []) if c.strip()],
        "testcases": testcases,
    }


def _stream_with_progress(app, payload: dict, thread_id: str,
                          expected: int) -> list[dict]:
    """Consume app.stream with live stage updates; returns all chunks."""
    import time

    chunks: list[dict] = []
    started = time.time()
    with st.status("Starting agent…", expanded=True) as status:
        for chunk in stream_run(app, payload, thread_id):
            chunks.append(chunk)
            names = completed_stages_from_stream(chunks)
            current = names[-1] if names else "starting"
            status.update(
                label=f"Running: `{current}` "
                      f"({len(names)}/{expected} stages, "
                      f"{time.time() - started:.0f}s elapsed)")
        status.update(label=f"Run finished in {time.time() - started:.1f}s",
                      state="complete")
    st.session_state["last_elapsed_s"] = time.time() - started
    return chunks


def _save_tmp(payload: dict, prefix: str) -> str:
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json", prefix=f"ui_{prefix}_")
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


# Thin re-exports so tests / external callers can use ui.app helpers.
__all__ = ["main", "run_creator", "run_solver", "get_state",
           "build_requirements", "build_problem"]


if __name__ == "__main__":
    main()
