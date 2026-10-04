"""Step 6 tests: router picks creator vs solver, mermaid prints."""

from src.graph import app, get_mermaid


def _base_state(**overrides: object) -> dict:
    state: dict = {
        "job_id": "job_test",
        "mode": "",
        "workdir": "/tmp",
        "problem_spec": None,
        "solution_code": None,
        "test_results": [],
        "timings_ms": [],
        "attempt": 0,
        "errors": [],
        "safety_flags": [],
        "needs_human": False,
    }
    state.update(overrides)
    return state


def test_router_chooses_creator_without_problem_spec():
    out = app.invoke(_base_state(), config={"configurable": {"thread_id": "t-creator"}})
    assert out["mode"] == "creator"
    assert any("creator stub" in e for e in out["errors"])


def test_router_chooses_solver_with_problem_spec():
    out = app.invoke(
        _base_state(problem_spec={"id": "two_sum_demo"}),
        config={"configurable": {"thread_id": "t-solver"}},
    )
    assert out["mode"] == "solver"
    assert any("solver stub" in e for e in out["errors"])


def test_router_mode_solver_without_spec():
    out = app.invoke(_base_state(mode="solver"), config={"configurable": {"thread_id": "t-mode"}})
    assert out["mode"] == "solver"


def test_mermaid_shows_branches():
    mermaid = get_mermaid()
    print(mermaid)
    assert "router" in mermaid
    assert "creator" in mermaid
    assert "solver" in mermaid
