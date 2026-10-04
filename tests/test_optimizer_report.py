"""Step 9 tests: optimizer accept/rollback + reporter, mocked LLM + executor timings."""

from pathlib import Path

import src.solver_graph as sg

ROOT = Path(__file__).resolve().parent.parent
PROBLEM_DEMO = ROOT / "workspace" / "problems" / "examples" / "problem_id_two_sum_demo.json"

BRUTE = (
    "def two_sum(nums, target):\n"
    "    for i in range(len(nums)):\n"
    "        for j in range(i + 1, len(nums)):\n"
    "            if nums[i] + nums[j] == target:\n"
    "                return [i, j]\n"
    "    return []\n"
)
OPT = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, x in enumerate(nums):\n"
    "        if target - x in seen:\n"
    "            return [seen[target - x], i]\n"
    "        seen[x] = i\n"
    "    return []\n"
)
BROKEN = "def two_sum(nums, target):\n    return []\n"


def _stub_executor(monkeypatch, baseline_ms: float, optimized_ms: float):
    """Deterministic safe_execute: timing depends on which code runs."""
    from src.schemas import RunResult

    def fake_execute(code, prob, workdir, **kw):
        per_test = baseline_ms if "range(len(nums))" in code else optimized_ms
        out = []
        for t in prob.testcases:
            passed = True
            if "return []" in code and "range(len(nums))" not in code and "seen" not in code:
                passed = False  # BROKEN candidate
            out.append(
                RunResult(
                    test_id=t.id, passed=passed, actual=t.expected,
                    error=None if passed else "wrong", time_ms=per_test,
                )
            )
        return out

    monkeypatch.setattr(sg, "safe_execute", fake_execute)


def _stub_llm(monkeypatch, tmp_path, initial_code: str, optimized_code: str):
    jobs = tmp_path / "jobs"
    jobs.mkdir(exist_ok=True)
    monkeypatch.setattr("src.tools.access_broker._JOBS_ROOT", jobs)
    calls: list[str] = []

    def fake_generate(task, prompt, **kwargs):
        calls.append(task)
        if task == "initial_solution":
            return initial_code
        if task == "analyze_complexity":
            return "O(n^2) — nested loops over nums"
        if task == "propose_optimization":
            return optimized_code
        return optimized_code

    monkeypatch.setattr(sg, "generate", fake_generate)
    return calls


def _invoke(thread):
    return sg.solver_app.invoke(
        {"problem_path": str(PROBLEM_DEMO)},
        config={"configurable": {"thread_id": thread}},
    )


def test_optimizer_accepts_faster_candidate(monkeypatch, tmp_path):
    calls = _stub_llm(monkeypatch, tmp_path, BRUTE, OPT)
    _stub_executor(monkeypatch, baseline_ms=10.0, optimized_ms=2.0)
    out = _invoke("opt-accept")
    assert not out.get("needs_human"), out.get("errors")
    assert out.get("opt_status") == "accepted", out
    assert out.get("opt_round", 0) <= sg.MAX_OPT_ROUNDS
    assert "seen" in (out.get("solution_code") or "")
    report = Path(out["workdir"]) / "report.md"
    assert report.is_file()
    text = report.read_text()
    assert "O(n^2)" in text
    assert "before" in text.lower() and "after" in text.lower()
    assert "Speedup" in text
    assert calls.count("propose_optimization") <= sg.MAX_OPT_ROUNDS


def test_optimizer_rolls_back_broken_candidate(monkeypatch, tmp_path):
    _stub_llm(monkeypatch, tmp_path, BRUTE, BROKEN)
    _stub_executor(monkeypatch, baseline_ms=10.0, optimized_ms=1.0)
    out = _invoke("opt-rollback-broken")
    assert not out.get("needs_human"), out.get("errors")
    assert out.get("opt_status") == "rolled-back"
    assert "range(len(nums))" in (out.get("solution_code") or "")
    report = Path(out["workdir"]) / "report.md"
    assert report.is_file()


def test_optimizer_rolls_back_slower_candidate(monkeypatch, tmp_path):
    _stub_llm(monkeypatch, tmp_path, BRUTE, OPT)
    _stub_executor(monkeypatch, baseline_ms=2.0, optimized_ms=50.0)
    out = _invoke("opt-rollback-slower")
    assert out.get("opt_status") == "rolled-back"
    assert "not faster" in (out.get("opt_reason") or "")
    assert "range(len(nums))" in (out.get("solution_code") or "")


def test_reporter_build_report_shape():
    from src.tools.reporter import build_report

    md = build_report(
        baseline_results=[{"test_id": "t1", "passed": True, "time_ms": 10.0}],
        optimized_results=[{"test_id": "t1", "passed": True, "time_ms": 2.0}],
        baseline_timings=[10.0],
        optimized_timings=[2.0],
        complexity_before="O(n^2)",
        complexity_after="O(n)",
        what_changed="hashmap lookup",
        opt_status="accepted",
        opt_reason="speedup",
    )
    assert "O(n^2)" in md and "O(n)" in md
    assert "hashmap lookup" in md
    assert "| t1 | true |" in md
