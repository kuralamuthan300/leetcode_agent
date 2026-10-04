"""Step 10 E2E: creator -> solver -> optimizer on fresh category (strings).

Mocked LLM only; real local executor (use_docker=False). Asserts promotion,
solving, report.md, and audit.jsonl in both job dirs. Appends a short log
line to docs/reports/e2e_step10.log for the Verify step.
"""

import copy
import json
from pathlib import Path

import src.creator_graph as cg
import src.solver_graph as sg
from src.schemas import ProblemSpec

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "docs" / "reports" / "e2e_step10.log"

STR_DRAFT = {
    "id": "reverse_mock10",
    "title": "Reverse String",
    "difficulty": "easy",
    "statement": "Return the reverse of the input string.",
    "function_name": "reverse_string",
    "signature": "def reverse_string(s: str) -> str:",
    "examples": [{"input": {"s": "hello"}, "output": "olleh",
                  "explanation": "reverse hello"}],
    "constraints": ["0 <= len(s) <= 10^5"],
    "testcases": [{"id": "t0", "input": {"s": "ab"}, "expected": "ba",
                   "is_hidden": False, "timeout_ms": 2000}],
}
STR_TESTS = [
    {"id": "t1", "input": {"s": "hello"}, "expected": "olleh",
     "is_hidden": False, "timeout_ms": 2000},
    {"id": "t2", "input": {"s": ""}, "expected": "",
     "is_hidden": False, "timeout_ms": 2000},
    {"id": "t3", "input": {"s": "a"}, "expected": "a",
     "is_hidden": False, "timeout_ms": 2000},
    {"id": "t4", "input": {"s": "abc" * 100}, "expected": ("abc" * 100)[::-1],
     "is_hidden": True, "timeout_ms": 2000},
]
STR_ORACLE = "def reverse_string(s):\n    return s[::-1]\n"
STR_OPT = "def reverse_string(s):\n    return ''.join(reversed(s))\n"


def test_e2e_strings_creator_to_optimizer(monkeypatch, tmp_path):
    jobs = tmp_path / "jobs"
    problems = tmp_path / "problems"
    jobs.mkdir()
    problems.mkdir()
    monkeypatch.setattr("src.tools.access_broker._JOBS_ROOT", jobs)
    monkeypatch.setattr("src.tools.access_broker._PROBLEMS_ROOT", problems)

    req = {"job_id": "req_strings001", "category": "strings", "difficulty": "easy",
           "language": "python", "num_tests": 4,
           "constraints_hint": "0 <= len(s) <= 10^5", "extra": "fresh category e2e"}
    req_path = tmp_path / "requirements_strings_demo.json"
    req_path.write_text(json.dumps(req))

    def fake_creator(task, prompt, json_mode=False, **kw):
        if task == "draft_problem":
            return copy.deepcopy(STR_DRAFT)
        if task == "generate_tests":
            return copy.deepcopy(STR_TESTS)
        if task in ("oracle_solution", "repair_problem"):
            return STR_ORACLE
        raise AssertionError(task)

    monkeypatch.setattr(cg, "generate", fake_creator)
    orig_c_exec = cg.safe_execute
    monkeypatch.setattr(cg, "safe_execute",
                        lambda code, prob, workdir, **kw: orig_c_exec(
                            code, prob, workdir, use_docker=False))

    cout = cg.creator_app.invoke(
        {"requirements_path": str(req_path)},
        config={"configurable": {"thread_id": "e2e10-creator"}})
    assert not cout.get("needs_human"), cout.get("errors")
    promoted = Path(cout["problem_path"])
    assert promoted.is_file()
    ProblemSpec.model_validate(json.loads(promoted.read_text()))
    creator_audit = Path(cout["workdir"]) / "audit.jsonl"
    assert creator_audit.is_file()

    def fake_solver(task, prompt, **kw):
        if task == "initial_solution":
            return STR_ORACLE
        if task == "repair_solution":
            return STR_ORACLE
        if task == "analyze_complexity":
            return "O(n) — single pass over s"
        if task == "propose_optimization":
            return STR_OPT
        return STR_OPT

    monkeypatch.setattr(sg, "generate", fake_solver)
    orig_s_exec = sg.safe_execute
    monkeypatch.setattr(sg, "safe_execute",
                        lambda code, prob, workdir, **kw: orig_s_exec(
                            code, prob, workdir, use_docker=False))

    sout = sg.solver_app.invoke(
        {"problem_path": str(promoted)},
        config={"configurable": {"thread_id": "e2e10-solver"}})
    assert not sout.get("needs_human"), sout.get("errors")
    assert sout["test_results"] and all(r["passed"] for r in sout["test_results"])
    assert sout.get("opt_status") in ("accepted", "rolled-back")
    report = Path(sout["workdir"]) / "report.md"
    assert report.is_file()
    solver_audit = Path(sout["workdir"]) / "audit.jsonl"
    assert solver_audit.is_file()
    assert solver_audit.read_text().strip()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            f"e2e10 category=strings problem={cout['problem_spec']['id']} "
            f"tests={len(sout['test_results'])} pass=all "
            f"opt={sout.get('opt_status')} reason={sout.get('opt_reason')}\n"
        )
