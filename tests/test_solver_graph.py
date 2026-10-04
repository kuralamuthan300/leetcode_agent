"""Step 8 tests: solver subgraph correctness loop with mocked LLM."""

import json
from pathlib import Path

import src.solver_graph as sg

ROOT = Path(__file__).resolve().parent.parent
PROBLEM_DEMO = ROOT / "workspace" / "problems" / "examples" / "problem_id_two_sum_demo.json"

GOOD_CODE = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, x in enumerate(nums):\n"
    "        if target - x in seen:\n"
    "            return [seen[target - x], i]\n"
    "        seen[x] = i\n"
    "    return []\n"
)
BAD_CODE = "def two_sum(nums, target):\n    return []\n"
EVIL_CODE = "import os\ndef two_sum(nums, target):\n    return os.listdir('.')\n"


def _hermetic(monkeypatch, tmp_path, code_by_task):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr("src.tools.access_broker._JOBS_ROOT", jobs)
    calls: list[str] = []

    def fake_generate(task, prompt, **kwargs):
        calls.append(task)
        return code_by_task(task)

    monkeypatch.setattr(sg, "generate", fake_generate)
    orig_exec = sg.safe_execute
    monkeypatch.setattr(
        sg, "safe_execute",
        lambda code, prob, workdir, **kw: orig_exec(
            code, prob, workdir, use_docker=False))
    return calls


def _invoke(problem_path, thread):
    return sg.solver_app.invoke(
        {"problem_path": str(problem_path)},
        config={"configurable": {"thread_id": thread}})


def test_solves_demo_with_timings(monkeypatch, tmp_path):
    calls = _hermetic(monkeypatch, tmp_path, lambda task: GOOD_CODE)
    out = _invoke(PROBLEM_DEMO, "solver-demo")
    assert not out.get("needs_human"), out.get("errors")
    results = out["test_results"]
    assert results and all(r["passed"] for r in results)
    assert len(out["timings_ms"]) == len(results)
    assert all(isinstance(t, float) for t in out["timings_ms"])
    assert {"initial_solution"} <= set(calls)
    table = sg.format_results_table(results)
    assert table.startswith("| test_id | pass | mean_ms |")
    assert len(table.splitlines()) == len(results) + 2
    saved = Path(out["workdir"]) / "solution.py"
    assert saved.is_file() and "def two_sum" in saved.read_text()


def test_repair_loop_fixes_failing_solution(monkeypatch, tmp_path):
    n = {"count": 0}

    def code_by_task(task):
        if task == "initial_solution":
            n["count"] += 1
            return BAD_CODE
        return GOOD_CODE

    _hermetic(monkeypatch, tmp_path, code_by_task)
    out = _invoke(PROBLEM_DEMO, "solver-repair")
    assert not out.get("needs_human"), out.get("errors")
    assert out.get("attempt", 0) >= 1
    assert out["test_results"] and all(r["passed"] for r in out["test_results"])


def test_scan_blocked_solution_goes_human(monkeypatch, tmp_path):
    _hermetic(monkeypatch, tmp_path, lambda task: EVIL_CODE)
    out = _invoke(PROBLEM_DEMO, "solver-scan-block")
    assert out.get("needs_human") is True
    assert out.get("safety_flags"), "expected scan flags recorded"
    assert out.get("attempt", 0) == sg.MAX_REPAIRS


def test_max_repairs_flags_human(monkeypatch, tmp_path):
    _hermetic(monkeypatch, tmp_path, lambda task: BAD_CODE)
    out = _invoke(PROBLEM_DEMO, "solver-max-repairs")
    assert out.get("needs_human") is True
    assert out.get("attempt", 0) == sg.MAX_REPAIRS


def test_format_results_table_shape():
    rows = [
        {"test_id": "t1", "passed": True, "time_ms": 1.23456},
        {"test_id": "t2", "passed": False, "time_ms": 2.0},
    ]
    table = sg.format_results_table(rows)
    assert table.splitlines() == [
        "| test_id | pass | mean_ms |",
        "| --- | --- | --- |",
        "| t1 | true | 1.235 |",
        "| t2 | false | 2.000 |",
    ]


def test_timings_averaged_over_runs(monkeypatch, tmp_path):
    _hermetic(monkeypatch, tmp_path, lambda task: GOOD_CODE)
    spec = json.loads(PROBLEM_DEMO.read_text())
    from src.schemas import ProblemSpec as PS
    from src.tools import safe_executor as se

    seen = {"n": 0}
    orig = se.safe_execute

    def counting(code, prob, workdir, **kw):
        seen["n"] += 1
        return orig(code, prob, workdir, use_docker=False)

    monkeypatch.setattr(sg, "safe_execute", counting)
    from src.tools.access_broker import create_job

    workdir = create_job(PROBLEM_DEMO)
    out = sg.run_tests_timed({
        "problem_spec": PS.model_validate(spec).model_dump(),
        "solution_code": GOOD_CODE,
        "workdir": str(workdir),
        "errors": [],
    })
    assert seen["n"] == sg.TIMED_RUNS
    assert len(out["timings_ms"]) == len(out["test_results"]) == len(spec["testcases"])
