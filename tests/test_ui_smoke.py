"""Smoke tests for Step 11 UI shell (mocked graph, no Ollama/Docker)."""

import json

import pytest

from ui import agent_client as ac
import ui.app as appmod
from src.schemas import ProblemSpec, RequirementsSpec


class FakeApp:
    def __init__(self):
        self.last_payload = None
        self.last_config = None
        self._state = {}

    def invoke(self, payload, config=None):
        self.last_payload = payload
        self.last_config = config
        return {"job_id": "j1", "mode": "creator", "attempt": 1, "problem_spec": {"title": "T"}}

    def get_state(self, config=None):
        self.last_config = config

        class Snap:
            values = {"job_id": "j1", "mode": "creator", "attempt": 1}

        return Snap()

    def stream(self, payload, config=None, stream_mode=None):
        assert stream_mode == "updates"
        yield {"intake_validate": {}}
        yield {"draft_problem": {}}
        yield {"generate_tests": {}}


def test_status_mapping():
    assert ac.status_for_state(None) == "idle"
    assert ac.status_for_state({}) == "idle"
    assert ac.status_for_state({"job_id": "x"}) == "running"
    assert ac.status_for_state({"needs_human": True, "pending_approval": "promote"}) == "awaiting approval"
    assert ac.status_for_state({"needs_human": True}) == "needs_human"
    assert ac.status_for_state({"needs_human": True, "pending_approval": None}) == "needs_human"
    assert ac.status_for_state({"problem_spec": {"title": "T"}}) == "done"
    assert ac.status_for_state({"solution_code": "def f(): pass"}) == "done"


def test_stepper_order_creator_solver():
    assert ac.stages_for_mode("creator") == ac.CREATOR_STAGES
    assert ac.stages_for_mode("solver") == ac.SOLVER_STAGES
    assert len(ac.CREATOR_STAGES) == 9
    assert len(ac.SOLVER_STAGES) == 13
    assert ac.CREATOR_STAGES[0] == "intake_validate"
    assert ac.CREATOR_STAGES[-1] == "promote_save"
    assert ac.SOLVER_STAGES[0] == "load_validate"
    assert ac.SOLVER_STAGES[-1] == "save_solution"
    m = ac.stage_states(ac.CREATOR_STAGES, ["intake_validate", "draft_problem"])
    assert m["intake_validate"] == "done"
    assert m["draft_problem"] == "done"
    assert m["generate_tests"] == "active"
    assert m["promote_save"] == "todo"


def test_client_wrapper_mocked_graph():
    fake = FakeApp()
    out = ac.run_creator({"category": "arrays", "num_tests": 8}, "t1", app=fake)
    assert out["job_id"] == "j1"
    assert "requirements_path" in fake.last_payload
    assert fake.last_config == {"configurable": {"thread_id": "t1"}}

    fake2 = FakeApp()
    out2 = ac.run_solver({"id": "two_sum_demo"}, "t2", app=fake2)
    assert out2["job_id"] == "j1"
    assert "problem_path" in fake2.last_payload

    state = ac.get_state("t1", app=fake)
    assert state["job_id"] == "j1"

    chunks = list(ac.stream_run(fake, {"requirements_path": "x"}, "t1"))
    completed = ac.completed_stages_from_stream(chunks)
    assert completed == ["intake_validate", "draft_problem", "generate_tests"]


def test_tabs_render_data():
    state = {
        "problem_spec": {
            "title": "Two Sum",
            "statement": "stmt",
            "signature": "def two_sum(nums, target):",
            "constraints": ["2 <= n"],
            "examples": [{"input": {"nums": [2, 7], "target": 9}}],
            "testcases": [
                {"id": "t1", "is_hidden": False},
                {"id": "t_perf", "is_hidden": True},
            ],
        },
        "solution_code": "def two_sum(nums, target): return [0, 1]",
        "test_results": [
            {"test_id": "t1", "input": {"nums": [2, 7]}, "expected": [0, 1], "actual": [0, 1], "pass": True},
            {"test_id": "t_perf", "input": {}, "expected": [1], "actual": [1], "pass": True, "mean_ms": 3.5},
        ],
        "timings_ms": {"t1": 1.2},
    }
    pdata = ac.problem_tab_data(state)
    assert pdata["title"] == "Two Sum"
    assert "two_sum" in pdata["signature"]
    assert ac.solution_tab_data(state).startswith("def two_sum")
    rows = ac.format_test_rows(state)
    assert len(rows) == 2
    by_id = {r["test_id"]: r for r in rows}
    assert by_id["t1"]["pass"] is True
    assert by_id["t1"]["mean_ms"] == 1.2
    assert by_id["t_perf"]["mean_ms"] == 3.5
    assert set(rows[0].keys()) == {"test_id", "input", "expected", "actual", "pass", "mean_ms"}


def test_build_requirements_from_form():
    req = appmod.build_requirements("arrays", "easy", 5, "n <= 10^4", "extra notes")
    spec = RequirementsSpec.model_validate(req)  # must satisfy Pydantic
    assert spec.category == "arrays"
    assert spec.num_tests == 5
    assert spec.language == "python"
    assert req["job_id"].startswith("req_")
    with pytest.raises(ValueError):
        appmod.build_requirements("", "easy", 8)
    with pytest.raises(ValueError):
        appmod.build_requirements("arrays", "insane", 8)
    with pytest.raises(ValueError):
        appmod.build_requirements("arrays", "easy", 0)


def test_build_problem_from_form():
    prob = appmod.build_problem(
        "Two Sum", "two_sum",
        "def two_sum(nums: list[int], target: int) -> list[int]:",
        "Return indices adding to target.",
        ["2 <= n <= 10^4", ""],
        [('{"nums": [2, 7], "target": 9}', "[0, 1]"),
         ("", "")],  # blank row skipped
    )
    spec = ProblemSpec.model_validate(prob)  # must satisfy Pydantic
    assert spec.function_name == "two_sum"
    assert [t.id for t in spec.testcases] == ["t1"]
    assert spec.constraints == ["2 <= n <= 10^4"]
    with pytest.raises(ValueError):
        appmod.build_problem("T", "not a name!", "def x():", "s", [], [])
    with pytest.raises(ValueError):
        appmod.build_problem("T", "two_sum", "def other():", "s", [], [])
    with pytest.raises(ValueError):
        appmod.build_problem("T", "two_sum", "def two_sum(x):", "s", [],
                             [("not json", "1")])
    with pytest.raises(ValueError):
        appmod.build_problem("T", "two_sum", "def two_sum(x):", "s", [],
                             [("[1, 2]", "1")])  # bare-list input rejected
    with pytest.raises(ValueError):
        appmod.build_problem("T", "two_sum", "def two_sum(x):", "s", [], [])


def test_pending_action_routing():
    assert ac.pending_action({}) is None
    assert ac.pending_action(None) is None
    assert ac.pending_action({"pending_approval": "understanding"}) == "understanding"
    assert ac.pending_action({"pending_approval": "clarify"}) == "clarify"
    assert ac.pending_action({"pending_approval": "promote"}) == "final"
    assert ac.pending_action({"pending_approval": "optimize"}) == "final"
    assert ac.pending_action({"pending_approval": None}) is None
