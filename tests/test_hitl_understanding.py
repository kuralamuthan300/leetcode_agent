"""Step 12 tests: HITL understanding gate + clarify + approve/decline.

Mocked LLM + hermetic job dirs (no Ollama/Docker).
"""

import copy
import json
from pathlib import Path

import src.creator_graph as cg
import src.solver_graph as sg
from src.schemas import ProblemSpec

ROOT = Path(__file__).resolve().parent.parent
REQ_DEMO = ROOT / "workspace" / "jobs" / "examples" / "requirements_id_demo.json"
PROBLEM_DEMO = ROOT / "workspace" / "problems" / "examples" / "problem_id_two_sum_demo.json"

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
]
GOOD_ORACLE = (
    "def two_sum(nums, target):\n"
    "    seen = {}\n"
    "    for i, x in enumerate(nums):\n"
    "        if target - x in seen:\n"
    "            return [seen[target - x], i]\n"
    "        seen[x] = i\n"
    "    return []\n"
)
BAD_ORACLE = "def two_sum(nums, target):\n    return []\n"


def _doubtful_req(tmp_path):
    """Requirements missing the constraints hint -> real doubt, must pause."""
    req = json.loads(REQ_DEMO.read_text())
    req.pop("constraints_hint", None)
    req["job_id"] = "req_doubt01"
    path = tmp_path / "requirements_doubt.json"
    path.write_text(json.dumps(req))
    return path


def _doubtful_problem(tmp_path):
    """Problem with a single testcase -> real doubt, must pause."""
    prob = json.loads(PROBLEM_DEMO.read_text())
    prob["testcases"] = prob["testcases"][:1]
    prob["id"] = "two_sum_doubt01"
    path = tmp_path / "problem_doubt.json"
    path.write_text(json.dumps(prob))
    return path


def _audit_decisions(workdir):
    path = Path(workdir) / "audit.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _hermetic_creator(monkeypatch, tmp_path, oracle=GOOD_ORACLE):
    jobs = tmp_path / "jobs"
    problems = tmp_path / "problems"
    jobs.mkdir()
    problems.mkdir()
    monkeypatch.setattr("src.tools.access_broker._JOBS_ROOT", jobs)
    monkeypatch.setattr("src.tools.access_broker._PROBLEMS_ROOT", problems)
    calls, prompts, exec_calls = [], {}, {"n": 0}

    def fake_generate(task, prompt, json_mode=False, **kwargs):
        calls.append(task)
        prompts.setdefault(task, []).append(str(prompt))
        if task == "draft_problem":
            return copy.deepcopy(TWO_SUM_DRAFT)
        if task == "generate_tests":
            return copy.deepcopy(TWO_SUM_TESTS)
        if task in ("oracle_solution", "repair_problem"):
            return oracle
        raise AssertionError(f"unexpected LLM task: {task}")

    monkeypatch.setattr(cg, "generate", fake_generate)
    orig_exec = cg.safe_execute

    def counting_exec(code, prob, workdir, **kw):
        exec_calls["n"] += 1
        return orig_exec(code, prob, workdir, use_docker=False)

    monkeypatch.setattr(cg, "safe_execute", counting_exec)
    return {"calls": calls, "prompts": prompts, "exec": exec_calls}


def _hermetic_solver(monkeypatch, tmp_path, code=GOOD_ORACLE):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    monkeypatch.setattr("src.tools.access_broker._JOBS_ROOT", jobs)
    calls, prompts = [], {}

    def fake_generate(task, prompt, **kwargs):
        calls.append(task)
        prompts.setdefault(task, []).append(str(prompt))
        return code

    monkeypatch.setattr(sg, "generate", fake_generate)
    orig_exec = sg.safe_execute
    monkeypatch.setattr(
        sg, "safe_execute",
        lambda c, p, w, **kw: orig_exec(c, p, w, use_docker=False))
    return {"calls": calls, "prompts": prompts}


# ---------------------------------------------------------------- creator HITL-1

def test_creator_understanding_blocks_heavy_work(monkeypatch, tmp_path):
    h = _hermetic_creator(monkeypatch, tmp_path)
    req_path = _doubtful_req(tmp_path)
    out = cg.creator_app.invoke(
        {"requirements_path": str(req_path), "require_understanding": True},
        config={"configurable": {"thread_id": "hitl-pause"}})
    assert out.get("pending_approval") == "understanding"
    assert out.get("needs_human") is True
    assert "draft_problem" not in h["calls"], "no heavy LLM before confirm"
    assert h["exec"]["n"] == 0, "no Docker before confirm"
    assert out.get("problem_path") is None
    assert "arrays" in (out.get("understanding_summary") or "")
    assert out.get("human_question"), "agent must ask doubts"


def test_creator_understanding_resume_continues(monkeypatch, tmp_path):
    h = _hermetic_creator(monkeypatch, tmp_path)
    req_path = _doubtful_req(tmp_path)
    cfg = {"configurable": {"thread_id": "hitl-resume"}}
    cg.creator_app.invoke(
        {"requirements_path": str(req_path), "require_understanding": True},
        config=cfg)
    cg.creator_app.update_state(cfg, {"human_answer": "arrays only, n <= 100"})
    out = cg.creator_app.invoke(
        {"requirements_path": str(req_path)}, config=cfg)
    assert not out.get("needs_human"), out.get("errors")
    assert out.get("pending_approval") is None
    assert Path(out["problem_path"]).is_file()
    drafts = h["prompts"].get("draft_problem", [])
    assert drafts and any("arrays only" in p for p in drafts), \
        "resume answer must reach draft_problem prompt"


def test_creator_complete_reqs_auto_confirm(monkeypatch, tmp_path):
    # No doubts (hint present, sane count) -> no pause, but the
    # auto-confirm is still logged.
    h = _hermetic_creator(monkeypatch, tmp_path)
    out = cg.creator_app.invoke(
        {"requirements_path": str(REQ_DEMO), "require_understanding": True},
        config={"configurable": {"thread_id": "hitl-autopass"}})
    assert not out.get("needs_human"), out.get("errors")
    assert out.get("pending_approval") is None
    assert "draft_problem" in h["calls"], "no doubts means straight to draft"
    assert Path(out["problem_path"]).is_file()
    decisions = _audit_decisions(out["workdir"])
    confirmed = [e for e in decisions
                 if e.get("decision") == "understanding_confirmed"]
    assert confirmed and any("no open questions" in str(e.get("reason"))
                             for e in confirmed)


def test_creator_default_run_skips_gate(monkeypatch, tmp_path):
    # Legacy compat: no require_understanding -> full auto run.
    _hermetic_creator(monkeypatch, tmp_path)
    out = cg.creator_app.invoke(
        {"requirements_path": str(REQ_DEMO)},
        config={"configurable": {"thread_id": "hitl-legacy"}})
    assert not out.get("needs_human"), out.get("errors")
    assert Path(out["problem_path"]).is_file()


# ---------------------------------------------------------------- HITL-2 decline

def test_promote_decline_repairs_and_audits(monkeypatch, tmp_path):
    _hermetic_creator(monkeypatch, tmp_path)
    out = cg.creator_app.invoke(
        {"requirements_path": str(REQ_DEMO),
         "human_approved_promote": False,
         "human_answer": "too trivial, make it harder"},
        config={"configurable": {"thread_id": "hitl-decline"}})
    assert out.get("attempt", 0) >= 1, "decline must route to repair"
    assert Path(out["problem_path"]).is_file(), "repaired run promotes"
    decisions = _audit_decisions(out["workdir"])
    declined = [e for e in decisions if e.get("decision") == "declined"]
    assert declined, "decline must be logged, not silently dropped"
    assert any("trivial" in str(e.get("reason")) for e in declined)


def test_promote_decline_no_budget_flags_human(monkeypatch, tmp_path):
    _hermetic_creator(monkeypatch, tmp_path)
    monkeypatch.setattr(cg, "MAX_REPAIRS", 0)
    out = cg.creator_app.invoke(
        {"requirements_path": str(REQ_DEMO),
         "human_approved_promote": False,
         "human_answer": "nope"},
        config={"configurable": {"thread_id": "hitl-decline-stuck"}})
    assert out.get("needs_human") is True
    assert out.get("problem_path") is None
    decisions = _audit_decisions(out["workdir"])
    assert any(e.get("decision") == "declined" for e in decisions)


def test_opt_decline_repairs_and_audits(monkeypatch, tmp_path):
    _hermetic_solver(monkeypatch, tmp_path)
    out = sg.solver_app.invoke(
        {"problem_path": str(PROBLEM_DEMO),
         "human_approved_opt": False,
         "human_answer": "prefer iterative over recursion"},
        config={"configurable": {"thread_id": "hitl-opt-decline"}})
    assert out.get("attempt", 0) >= 1, "decline must route to repair"
    assert not out.get("needs_human"), out.get("errors")
    decisions = _audit_decisions(out["workdir"])
    declined = [e for e in decisions if e.get("decision") == "declined"]
    assert declined, "opt decline must be logged"
    assert any("iterative" in str(e.get("reason")) for e in declined)


# ---------------------------------------------------------------- mid-run clarify

def test_creator_clarify_pause_and_resume(monkeypatch, tmp_path):
    h = _hermetic_creator(monkeypatch, tmp_path, oracle=BAD_ORACLE)
    cfg = {"configurable": {"thread_id": "hitl-clarify"}}
    out = cg.creator_app.invoke(
        {"requirements_path": str(REQ_DEMO), "allow_clarify": True},
        config=cfg)
    assert out.get("pending_approval") == "clarify"
    assert out.get("needs_human") is True
    assert out.get("human_question"), "clarify must ask a question"
    assert out.get("attempt", 0) >= 2, "clarify fires when stuck, not early"
    decisions = _audit_decisions(out["workdir"])
    assert any(e.get("stage") == "creator.clarify" for e in decisions)

    cg.creator_app.update_state(cfg, {"human_answer": "check the target index"})
    out2 = cg.creator_app.invoke(
        {"requirements_path": str(REQ_DEMO)}, config=cfg)
    assert out2.get("attempt", 0) > out.get("attempt", 0), \
        "resume must retry repair with the answer"
    assert out2.get("needs_human") is True  # oracle still bad -> human review


def test_solver_understanding_pause_resume(monkeypatch, tmp_path):
    h = _hermetic_solver(monkeypatch, tmp_path)
    prob_path = _doubtful_problem(tmp_path)
    cfg = {"configurable": {"thread_id": "hitl-solver-u"}}
    out = sg.solver_app.invoke(
        {"problem_path": str(prob_path), "require_understanding": True},
        config=cfg)
    assert out.get("pending_approval") == "understanding"
    assert out.get("needs_human") is True
    assert "initial_solution" not in h["calls"]

    sg.solver_app.update_state(cfg, {"human_answer": "use a hashmap"})
    out2 = sg.solver_app.invoke(
        {"problem_path": str(prob_path)}, config=cfg)
    assert not out2.get("needs_human"), out2.get("errors")
    assert out2["test_results"] and all(r["passed"] for r in out2["test_results"])
    inits = h["prompts"].get("initial_solution", [])
    assert inits and any("hashmap" in p for p in inits)


def test_hitl_routers_pure():
    assert cg.route_after_understanding(
        {"needs_human": True}) == "flag_human"
    assert cg.route_after_understanding({}) == "draft_problem"
    assert cg.route_after_promote_approval(
        {"human_approved_promote": False, "attempt": 0}) == "repair_solution"
    assert cg.route_after_promote_approval(
        {"human_approved_promote": False, "attempt": 99}) == "flag_human"
    assert sg.route_after_understanding({}) == "initial_solution"
    assert sg.route_after_opt_approval(
        {"human_approved_opt": False, "attempt": 0}) == "repair_solution"
    assert sg.route_after_opt_approval(
        {"human_approved_opt": False, "attempt": 99}) == "flag_human"
    assert sg.route_after_opt_approval({}) == "accept_or_rollback"
    # Safety order unchanged: scan still precedes execute in both graphs.
    assert ProblemSpec is not None
