"""Step 13: UI E2E over ui.agent_client with mocked LLM/Docker.

Covers the full Streamlit client flow:
requirements(num_tests=5) -> understanding confirm -> streamed stages ->
5-row test table -> final approve, plus the decline path.
"""

import copy
import json
from pathlib import Path

from ui import agent_client as ac
import src.creator_graph as cg
import src.solver_graph as sg

ROOT = Path(__file__).resolve().parent.parent
PROBLEM_DEMO = ROOT / "workspace" / "problems" / "examples" / "problem_id_two_sum_demo.json"

DRAFT = {
    "id": "two_sum_e2e13",
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
TESTS_5 = [
    {"id": f"t{i}", "input": {"nums": [2, 7, 11, 15], "target": 9},
     "expected": [0, 1], "is_hidden": False, "timeout_ms": 2000}
    for i in range(1, 6)
]
GOOD_CODE = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, x in enumerate(nums):\n"
    "        if target - x in seen:\n"
    "            return [seen[target - x], i]\n"
    "        seen[x] = i\n"
    "    return []\n"
)

REQ_5 = {"job_id": "req_e2e1301", "category": "arrays", "difficulty": "easy",
        "language": "python", "num_tests": 5,
        "constraints_hint": "n <= 10^4", "extra": ""}


def _hermetic(monkeypatch, tmp_path):
    jobs = tmp_path / "jobs"
    problems = tmp_path / "problems"
    jobs.mkdir()
    problems.mkdir()
    monkeypatch.setattr("src.tools.access_broker._JOBS_ROOT", jobs)
    monkeypatch.setattr("src.tools.access_broker._PROBLEMS_ROOT", problems)

    def fake_creator(task, prompt, json_mode=False, **kwargs):
        if task == "draft_problem":
            return copy.deepcopy(DRAFT)
        if task == "generate_tests":
            return copy.deepcopy(TESTS_5)
        if task in ("oracle_solution", "repair_problem"):
            return GOOD_CODE
        raise AssertionError(f"unexpected creator task: {task}")

    def fake_solver(task, prompt, **kwargs):
        return GOOD_CODE

    monkeypatch.setattr(cg, "generate", fake_creator)
    monkeypatch.setattr(sg, "generate", fake_solver)
    orig_c, orig_s = cg.safe_execute, sg.safe_execute
    monkeypatch.setattr(
        cg, "safe_execute",
        lambda c, p, w, **kw: orig_c(c, p, w, use_docker=False))
    monkeypatch.setattr(
        sg, "safe_execute",
        lambda c, p, w, **kw: orig_s(c, p, w, use_docker=False))
    req_file = tmp_path / "requirements_e2e13.json"
    req_file.write_text(json.dumps(REQ_5))
    return {"req_file": req_file}


def _audit(workdir):
    path = Path(workdir) / "audit.jsonl"
    assert path.is_file(), "audit.jsonl must exist"
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def test_creator_e2e_understanding_to_approved(monkeypatch, tmp_path):
    h = _hermetic(monkeypatch, tmp_path)

    # 1. Submit requirements -> HITL-1 understanding card blocks drafting.
    paused = ac.run_creator(REQ_5, "e2e13-creator", app=cg.creator_app)
    assert paused.get("pending_approval") == "understanding"
    card = ac.understanding_card_data(paused)
    assert card["waiting"] is True
    assert card["num_tests"] == 5
    assert "arrays" in card["summary"]

    # 2. Streamed stages advance in order on an auto-pass thread.
    chunks = list(ac.stream_run(
        cg.creator_app,
        {"requirements_path": str(h["req_file"]), "require_understanding": False},
        "e2e13-stream"))
    stages = ac.completed_stages_from_stream(chunks)
    assert stages.index("intake_validate") < \
        stages.index("summarize_understanding") < \
        stages.index("draft_problem") < stages.index("promote_save")

    # 3. Confirm & continue -> full run completes.
    payload = {"requirements_path": str(h["req_file"]),
               "require_understanding": True}
    final = ac.update_and_resume(
        cg.creator_app, "e2e13-creator",
        {"human_answer": "confirmed, proceed", "understanding_confirmed": True},
        payload)
    assert not final.get("needs_human"), final.get("errors")
    assert Path(final["problem_path"]).is_file()

    # 4. Five-row test table + final approval view.
    rows = ac.format_test_rows(final)
    assert len(rows) == 5, [r["test_id"] for r in rows]
    assert all(r["pass"] for r in rows)
    approval = ac.final_approval_data(final)
    assert approval["pass_rate"] == "5/5"

    # 5. Audit chain covers the two pauses.
    decisions = _audit(final["workdir"])
    assert any(e.get("decision") == "understanding_confirmed"
               for e in decisions)
    assert any(e.get("decision") == "promoted" for e in decisions)


def test_creator_e2e_decline_path(monkeypatch, tmp_path):
    h = _hermetic(monkeypatch, tmp_path)
    paused = ac.run_creator(REQ_5, "e2e13-decline", app=cg.creator_app)
    assert paused.get("pending_approval") == "understanding"

    payload = {"requirements_path": str(h["req_file"]),
               "require_understanding": True}
    final = ac.update_and_resume(
        cg.creator_app, "e2e13-decline",
        {"human_approved_promote": False,
         "human_answer": "too easy, needs harder edge cases",
         "understanding_confirmed": True},
        payload)
    assert final.get("attempt", 0) >= 1, "decline must route to repair"
    assert Path(final["problem_path"]).is_file()
    decisions = _audit(final["workdir"])
    assert any(e.get("decision") == "understanding_confirmed"
               for e in decisions)
    declined = [e for e in decisions if e.get("decision") == "declined"]
    assert declined and any("too easy" in str(e.get("reason"))
                            for e in declined)
    assert any(e.get("decision") == "promoted" for e in decisions)


def test_solver_e2e_understanding_to_done(monkeypatch, tmp_path):
    _hermetic(monkeypatch, tmp_path)
    paused = ac.run_solver(str(PROBLEM_DEMO), "e2e13-solver", app=sg.solver_app)
    assert paused.get("pending_approval") == "understanding"
    card = ac.understanding_card_data(paused)
    assert card["waiting"] is True
    assert "Two Sum" in card["summary"] or "two_sum" in card["summary"]

    payload = {"problem_path": str(PROBLEM_DEMO),
               "require_understanding": True}
    final = ac.update_and_resume(
        sg.solver_app, "e2e13-solver",
        {"human_answer": "confirmed", "understanding_confirmed": True},
        payload)
    assert not final.get("needs_human"), final.get("errors")
    rows = ac.format_test_rows(final)
    assert rows and all(r["pass"] for r in rows)
    approval = ac.final_approval_data(final)
    assert approval["pass_rate"].endswith(f"/{len(rows)}")
    decisions = _audit(final["workdir"])
    assert any(e.get("decision") == "understanding_confirmed"
               for e in decisions)
    assert any(e.get("decision") in ("accepted", "rolled-back")
               for e in decisions)
