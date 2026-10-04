"""Step 7 tests: creator subgraph end-to-end with mocked LLM (no Ollama needed)."""

import copy
import json
from pathlib import Path

import pytest

import src.creator_graph as cg
from src.schemas import ProblemSpec

ROOT = Path(__file__).resolve().parent.parent
REQ_DEMO = ROOT / "workspace" / "jobs" / "examples" / "requirements_id_demo.json"

TWO_SUM_DRAFT = {
    "id": "two_sum_mock01",
    "title": "Two Sum",
    "difficulty": "easy",
    "statement": "Return indices of the two numbers adding to target.",
    "function_name": "two_sum",
    "signature": "def two_sum(nums: list[int], target: int) -> list[int]:",
    "examples": [{"input": {"nums": [2, 7, 11, 15], "target": 9},
                  "output": [0, 1], "explanation": "nums[0]+nums[1]==9"}],
    "constraints": ["2 <= n <= 10^4", "Only one valid answer exists."],
    "testcases": [{"id": "t0", "input": {"nums": [1, 2], "target": 3},
                   "expected": [0, 1], "is_hidden": False, "timeout_ms": 2000}],
}
TWO_SUM_TESTS = [
    {"id": "t1", "input": {"nums": [2, 7, 11, 15], "target": 9},
     "expected": [0, 1], "is_hidden": False, "timeout_ms": 2000},
    {"id": "t2", "input": {"nums": [3, 2, 4], "target": 6},
     "expected": [1, 2], "is_hidden": False, "timeout_ms": 2000},
    {"id": "t3", "input": {"nums": [3, 3], "target": 6},
     "expected": [0, 1], "is_hidden": False, "timeout_ms": 2000},
]
TWO_SUM_ORACLE = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, x in enumerate(nums):\n"
    "        if target - x in seen:\n"
    "            return [seen[target - x], i]\n"
    "        seen[x] = i\n"
    "    return []\n"
)

CLIMB_DRAFT = {
    "id": "climb_mock02",
    "title": "Climbing Stairs",
    "difficulty": "easy",
    "statement": "Count distinct ways to climb n stairs taking 1 or 2 steps.",
    "function_name": "climb_stairs",
    "signature": "def climb_stairs(n: int) -> int:",
    "examples": [{"input": {"n": 3}, "output": 3,
                  "explanation": "1+1+1, 1+2, 2+1"}],
    "constraints": ["1 <= n <= 45"],
    "testcases": [{"id": "t0", "input": {"n": 2}, "expected": 2,
                   "is_hidden": False, "timeout_ms": 2000}],
}
CLIMB_TESTS = [
    {"id": "t1", "input": {"n": 2}, "expected": 2, "is_hidden": False, "timeout_ms": 2000},
    {"id": "t2", "input": {"n": 3}, "expected": 3, "is_hidden": False, "timeout_ms": 2000},
    {"id": "t3", "input": {"n": 5}, "expected": 8, "is_hidden": False, "timeout_ms": 2000},
    {"id": "t4", "input": {"n": 30}, "expected": 1346269, "is_hidden": True, "timeout_ms": 2000},
]
CLIMB_ORACLE = (
    "def climb_stairs(n):\n"
    "    a, b = 1, 1\n"
    "    for _ in range(n):\n"
    "        a, b = b, a + b\n"
    "    return a\n"
)


@pytest.fixture()
def scenario():
    return {"draft": TWO_SUM_DRAFT, "tests": TWO_SUM_TESTS,
            "oracle": TWO_SUM_ORACLE, "calls": []}


@pytest.fixture()
def hermetic(monkeypatch, tmp_path, scenario):
    """Fake LLM + isolated job/problem roots (no real Ollama, no repo writes)."""
    jobs = tmp_path / "jobs"
    problems = tmp_path / "problems"
    jobs.mkdir()
    problems.mkdir()
    monkeypatch.setattr("src.tools.access_broker._JOBS_ROOT", jobs)
    monkeypatch.setattr("src.tools.access_broker._PROBLEMS_ROOT", problems)

    def fake_generate(task, prompt, json_mode=False, **kwargs):
        scenario["calls"].append(task)
        if task == "draft_problem":
            return copy.deepcopy(scenario["draft"])
        if task == "generate_tests":
            return copy.deepcopy(scenario["tests"])
        if task == "oracle_solution":
            return scenario["oracle"]
        if task == "repair_problem":
            draft = copy.deepcopy(scenario["draft"])
            draft["testcases"] = copy.deepcopy(scenario["tests"])
            return draft
        raise AssertionError(f"unexpected LLM task: {task}")

    monkeypatch.setattr(cg, "generate", fake_generate)
    orig_exec = cg.safe_execute
    monkeypatch.setattr(cg, "safe_execute",
                        lambda code, prob, workdir, **kw: orig_exec(
                            code, prob, workdir, use_docker=False))
    return {"jobs": jobs, "problems": problems}


def _invoke(requirements_path, thread):
    return cg.creator_app.invoke(
        {"requirements_path": str(requirements_path)},
        config={"configurable": {"thread_id": thread}})


def _assert_promoted_ok(out, hermetic):
    assert not out.get("needs_human"), out.get("errors")
    spec = ProblemSpec.model_validate(out["problem_spec"])
    results = out["test_results"]
    assert results and all(r["passed"] for r in results)
    dest = Path(out["problem_path"])
    assert dest.parent == hermetic["problems"] and dest.is_file()
    assert ProblemSpec.model_validate(json.loads(dest.read_text())).id == spec.id
    return spec


def test_e2e_arrays_mocked(hermetic, scenario):
    out = _invoke(REQ_DEMO, "e2e-arrays")
    spec = _assert_promoted_ok(out, hermetic)
    assert spec.function_name == "two_sum"
    assert {"draft_problem", "generate_tests", "oracle_solution"} <= set(scenario["calls"])


def test_e2e_dp_easy_mocked(hermetic, scenario, tmp_path):
    scenario.update({"draft": CLIMB_DRAFT, "tests": CLIMB_TESTS, "oracle": CLIMB_ORACLE})
    req = {"job_id": "req_dp0001", "category": "dp", "difficulty": "easy",
           "language": "python", "num_tests": 4,
           "constraints_hint": "1 <= n <= 45", "extra": "fibonacci practice"}
    req_path = tmp_path / "requirements_dp_demo.json"
    req_path.write_text(json.dumps(req))
    out = _invoke(req_path, "e2e-dp")
    spec = _assert_promoted_ok(out, hermetic)
    assert spec.function_name == "climb_stairs"


def test_format_dedup_fixes_duplicate_ids():
    dupes = copy.deepcopy(TWO_SUM_TESTS) + [copy.deepcopy(TWO_SUM_TESTS[0])]
    assert dupes[0]["id"] == dupes[-1]["id"]
    spec = dict(TWO_SUM_DRAFT, testcases=dupes)
    out = cg.format_dedup({"problem_spec": spec, "errors": []})
    ids = [t["id"] for t in out["problem_spec"]["testcases"]]
    assert len(ids) == len(set(ids)) == len(dupes)
    ProblemSpec.model_validate(out["problem_spec"])


def test_format_dedup_wraps_bare_list_inputs():
    spec = dict(TWO_SUM_DRAFT)
    spec["examples"] = [{"input": [2, 7, 11, 15], "output": [0, 1]}]
    spec["testcases"] = [{"id": "t1", "input": [[1, 2]], "expected": [0, 1],
                           "is_hidden": False, "timeout_ms": 2000}]
    out = cg.format_dedup({"problem_spec": spec, "errors": []})
    assert out["problem_spec"]["examples"][0]["input"] == {"nums": [2, 7, 11, 15]}
    assert out["problem_spec"]["testcases"][0]["input"] == {"nums": [[1, 2]]}


def test_repair_loop_fixes_failing_oracle(hermetic, scenario, monkeypatch):
    bad = "def two_sum(nums, target):\n    return []\n"
    calls = {"n": 0}
    orig_fake = cg.generate

    def flaky(task, prompt, json_mode=False, **kwargs):
        if task == "oracle_solution":
            calls["n"] += 1
            return bad if calls["n"] == 1 else TWO_SUM_ORACLE
        return orig_fake(task, prompt, json_mode=json_mode, **kwargs)

    monkeypatch.setattr(cg, "generate", flaky)
    out = _invoke(REQ_DEMO, "repair-loop")
    assert not out.get("needs_human"), out.get("errors")
    assert out.get("attempt", 0) >= 1
    assert out["test_results"] and all(r["passed"] for r in out["test_results"])


def test_max_repairs_flags_human(hermetic, scenario):
    scenario["oracle"] = "def two_sum(nums, target):\n    return []\n"
    out = _invoke(REQ_DEMO, "max-repairs")
    assert out.get("needs_human") is True
    assert out.get("attempt", 0) == cg.MAX_REPAIRS


def test_scan_blocked_oracle_ends_human(hermetic, scenario):
    scenario["oracle"] = "import os\ndef two_sum(nums, target):\n    return os.listdir('.')\n"
    out = _invoke(REQ_DEMO, "scan-block")
    assert out.get("needs_human") is True
    assert out.get("safety_flags"), "expected scan flags recorded"
    assert out.get("problem_path") is None
