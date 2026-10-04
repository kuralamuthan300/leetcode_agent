"""Step 7: creator subgraph — requirements JSON -> validated problem JSON.

Chain:
  intake_validate -> draft_problem(heavy) -> generate_tests(heavy)
  -> format_dedup(deterministic) -> oracle_solution(heavy) -> static_scan
  -> run_tests(safe_execute) -> promote_save -> END

Repair loop: scan-blocked or oracle-tests-fail -> repair_solution(heavy,
max 3) -> oracle_solution. After MAX_REPAIRS failures -> needs_human=True.

Note: format_dedup is deterministic Python (dedup + TestCase validation),
not an LLM call — safer than asking light to reformat valid JSON.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.checkpoints import get_checkpointer
from src.llm_router import generate
from src.schemas import ProblemSpec, RequirementsSpec, TestCase
from src.state import AgentState
from src.tools.access_broker import (
    create_job,
    promote_to_problems,
    scoped_read,
    scoped_write,
)
from src.tools.audit import append_audit, log_decision, log_run, log_scan
from src.tools.safe_executor import safe_execute
from src.tools.static_scanner import scan

MAX_REPAIRS = 3


class CreatorState(AgentState, total=False):
    """AgentState plus creator inputs/outputs."""

    requirements: dict | None
    requirements_path: str | None
    problem_path: str | None
    scan_safe: bool


# ---------------------------------------------------------------- helpers

def _errors(state: CreatorState) -> list:
    return list(state.get("errors") or [])


def _strip_code_fences(text: str) -> str:
    """Remove ```python fences LLMs add around raw code."""
    m = re.search(r"```(?:python)?\s*(.*?)```", text.strip(), re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _as_testcase_list(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [t for t in payload if isinstance(t, dict)]
    if isinstance(payload, dict):
        for key in ("testcases", "tests", "cases"):
            if isinstance(payload.get(key), list):
                return [t for t in payload[key] if isinstance(t, dict)]
    return []


def _first_param(signature: str | None) -> str | None:
    """Best-effort first parameter name from a `def f(a: ..., b: ...)` signature."""
    m = re.search(r"def\s+\w+\s*\((.*?)\)", signature or "", re.DOTALL)
    if not m:
        return None
    for raw in m.group(1).split(","):
        name = raw.split(":")[0].split("=")[0].strip()
        if name and name not in ("self", "cls", "*", "/"):
            return name.lstrip("*")
    return None


def _normalize_mapping_input(item: dict, first_param: str | None) -> dict:
    """LLMs sometimes emit a bare list for input; wrap it as {first_param: value}."""
    if not isinstance(item.get("input"), dict) and first_param:
        item = dict(item)
        item["input"] = {first_param: item["input"]}
    return item


def _failure_context(state: CreatorState, max_chars: int = 1500) -> str:
    parts = [f"- {e}" for e in _errors(state)[-3:]]
    spec = state.get("problem_spec") or {}
    expected_by_id = {t.get("id"): t.get("expected") for t in spec.get("testcases", []) if isinstance(t, dict)}
    for r in (state.get("test_results") or [])[:4]:
        if not r.get("passed"):
            tid = r.get("test_id")
            parts.append(
                f"- test {tid} failed: expected={expected_by_id.get(tid)!r} "
                f"actual={r.get('actual')!r} error={r.get('error')!r}"
            )
    ctx = "\n".join(parts)[:max_chars]
    return f"Previous failures to fix:\n{ctx}\n" if ctx else ""


# ---------------------------------------------------------------- nodes

def intake_validate(state: CreatorState) -> dict:
    """Create job jail from requirements file, validate RequirementsSpec."""
    errors = _errors(state)
    if state.get("workdir") and state.get("requirements"):
        try:
            req = RequirementsSpec.model_validate(state["requirements"])
        except Exception as exc:
            errors.append(f"intake: invalid requirements in state: {exc}")
            return {"errors": errors, "needs_human": True}
        return {
            "requirements": req.model_dump(),
            "job_id": req.job_id,
            "mode": "creator",
            "attempt": int(state.get("attempt") or 0),
            "errors": errors,
            # Step 12: clear stale HITL gate flags on resume; downstream
            # approval nodes re-assert them when a pause is still needed.
            "needs_human": False,
            "pending_approval": None,
        }
    req_path = state.get("requirements_path")
    if not req_path:
        errors.append("intake: no requirements_path given")
        return {"errors": errors, "needs_human": True}
    try:
        workdir = create_job(req_path)
        raw = scoped_read(str(workdir), Path(req_path).name)
        req = RequirementsSpec.model_validate(json.loads(raw))
    except Exception as exc:
        errors.append(f"intake: {exc}")
        return {"errors": errors, "needs_human": True}
    return {
        "workdir": str(workdir),
        "job_id": req.job_id,
        "requirements": req.model_dump(),
        "mode": "creator",
        "attempt": 0,
        "errors": errors,
        "test_results": [],
        "timings_ms": [],
        "safety_flags": [],
        "needs_human": False,
    }


def summarize_understanding(state: CreatorState) -> dict:
    """Step 12 HITL-1: render interpreted requirements + questions for review.

    Deterministic (no LLM) so the gate never depends on model availability.
    """
    errors = _errors(state)
    req = state.get("requirements") or {}
    category = req.get("category", "?")
    difficulty = req.get("difficulty", "?")
    num_tests = req.get("num_tests", 8)
    hint = req.get("constraints_hint") or "none given"
    extra = req.get("extra") or "none"
    summary = (
        f"Category: {category}. Difficulty: {difficulty}. "
        f"Tests to generate: {num_tests}. Constraints hint: {hint}. "
        f"Extra: {extra}. "
        "I will draft a NEW original LeetCode-style problem with a snake_case "
        "function name, typed signature, 1-3 worked examples, and 2+ constraints."
    )
    questions = []
    if not req.get("constraints_hint"):
        questions.append(
            "No constraints hint given — what max input size (n) should tests target?"
        )
    questions.append(
        "Confirm category/difficulty/num_tests, or reply with corrections."
    )
    return {
        "understanding_summary": summary,
        "human_question": " ".join(questions),
        "errors": errors,
    }


def request_understanding_approval(state: CreatorState) -> dict:
    """Step 12 HITL-1 gate: pause before any heavy LLM/Docker work.

    Resume with human_answer (or understanding_confirmed=True) to continue.
    require_understanding=True enforces the pause; None/False auto-passes
    for headless runs (legacy compat, mirroring Step 10 gates).
    """
    errors = _errors(state)
    if state.get("human_answer") or state.get("understanding_confirmed"):
        try:
            log_decision(state.get("workdir"), stage="creator.understanding",
                         decision="understanding_confirmed",
                         reason=str(state.get("human_answer") or "confirmed"))
        except Exception:
            pass
        return {"pending_approval": None, "needs_human": False,
                "understanding_confirmed": True, "errors": errors}
    if state.get("require_understanding"):
        errors.append("understanding: awaiting human confirm before drafting")
        try:
            log_decision(state.get("workdir"), stage="creator.understanding",
                         decision="pending", reason="awaiting HITL-1 confirm")
        except Exception:
            pass
        return {"pending_approval": "understanding", "needs_human": True,
                "errors": errors}
    return {"pending_approval": None, "errors": errors}


def request_clarify(state: CreatorState) -> dict:
    """Step 12 mid-run clarify: repair is stuck — ask the human instead of
    silently retrying into flag_human. Resume with human_answer to retry
    repair with the answer as context, or skip to auto-retry."""
    errors = _errors(state)
    failing = _failure_context(state)
    results = state.get("test_results") or []
    bad_ids = [str(r.get("test_id")) for r in results
               if isinstance(r, dict) and not r.get("passed")][:5]
    question = (
        f"Repair stuck at attempt {int(state.get('attempt') or 0)}/{MAX_REPAIRS}. "
        f"Failing tests: {bad_ids or 'see log'}. {failing}"
        "Reply with guidance (or nothing to auto-retry)."
    )
    try:
        log_decision(state.get("workdir"), stage="creator.clarify",
                     decision="pending", reason=f"stuck: {bad_ids}")
    except Exception:
        pass
    return {"pending_approval": "clarify", "needs_human": True,
            "human_question": question, "clarify_asked": True, "errors": errors}


def draft_problem(state: CreatorState) -> dict:
    """Heavy: invent problem shell (testcases are placeholders, replaced next)."""
    errors = _errors(state)
    req = state.get("requirements") or {}
    human_ctx = ""
    if state.get("human_answer"):
        human_ctx = (
            "\nHuman clarification from requirements review (must respect): "
            f"{state.get('human_answer')}\n"
        )
    prompt = (
        "You create LeetCode-style coding problems as JSON.\n"
        f"Category: {req.get('category')}. Difficulty: {req.get('difficulty')}. Language: python.\n"
        f"Constraints hint: {req.get('constraints_hint')}. Extra notes: {req.get('extra')}.\n"
        f"{human_ctx}"
        "Invent a NEW original problem in this category (do not copy known problems).\n"
        "Pick a short snake_case function name and a typed signature like "
        '"def solve(nums: list[int]) -> int:".\n'
        "Include 1-3 worked examples {input, output, explanation} and 2+ constraint strings.\n"
        "Each example input MUST be an object mapping parameter names to values "
        '(e.g. {"nums": [...], "target": 9}), never a bare list.\n'
        'Include a "testcases" key with exactly 2 simple placeholder cases '
        "(they will be replaced later).\n"
        "Return JSON only with keys: id, title, difficulty, statement, "
        "function_name, signature, examples, constraints, testcases. "
        '"id" must be a short unique slug like "<name>_<6 hex chars>".'
    )
    try:
        draft = generate("draft_problem", prompt, json_mode=True)
    except Exception as exc:
        errors.append(f"draft_problem: {exc}")
        return {"errors": errors}
    if not isinstance(draft, dict):
        errors.append("draft_problem: LLM did not return a JSON object")
        return {"errors": errors}
    draft = dict(draft)
    draft["difficulty"] = req.get("difficulty", "easy")  # enforce, LLM may drift
    return {"problem_spec": draft, "errors": errors}


def generate_tests(state: CreatorState) -> dict:
    """Heavy: generate num_tests diverse testcases for the drafted problem."""
    errors = _errors(state)
    spec = state.get("problem_spec")
    if not isinstance(spec, dict):
        errors.append("generate_tests: no problem draft to attach tests to")
        return {"errors": errors}
    req = state.get("requirements") or {}
    n = int(req.get("num_tests") or 8)
    prompt = (
        f"Given this problem (signature: {spec.get('signature')}; "
        f"statement: {spec.get('statement')}; function: {spec.get('function_name')}):\n"
        f"Generate exactly {n} diverse test cases as a JSON list. Each item: "
        '{"id": "t1".., "input": {param: value...}, "expected": value, '
        '"is_hidden": bool, "timeout_ms": 2000}.\n'
        "Each input MUST be an object keyed by parameter name, never a bare list.\n"
        "Input keys MUST match the signature parameter names. Cover: worked examples, "
        "edge cases (empty, single element, min/max), and at least one large perf case "
        "marked is_hidden=true. Expected values must be exactly correct. "
        "Return a JSON list only."
    )
    try:
        payload = generate("generate_tests", prompt, json_mode=True)
    except Exception as exc:
        errors.append(f"generate_tests: {exc}")
        return {"errors": errors}
    cases = _as_testcase_list(payload)
    if not cases:
        errors.append("generate_tests: LLM returned no usable testcases")
    spec = dict(spec)
    spec["testcases"] = cases
    return {"problem_spec": spec, "errors": errors}


def format_dedup(state: CreatorState) -> dict:
    """Deterministic: unique testcase ids, defaults, full ProblemSpec check."""
    errors = _errors(state)
    spec = state.get("problem_spec")
    if not isinstance(spec, dict):
        errors.append("format_dedup: no problem draft present")
        return {"errors": errors}
    spec = dict(spec)
    first_param = _first_param(spec.get("signature"))
    spec["examples"] = [
        _normalize_mapping_input(dict(e), first_param)
        for e in (spec.get("examples", []) or []) if isinstance(e, dict)
    ]
    seen: set[str] = set()
    fixed: list[dict] = []
    for i, tc in enumerate(spec.get("testcases", []) or []):
        if not isinstance(tc, dict):
            continue
        tc = _normalize_mapping_input(dict(tc), first_param)
        tid = str(tc.get("id") or f"t{i + 1}")
        suffix = 2
        while tid in seen:
            tid = f"{tid.rsplit('_dup', 1)[0]}_dup{suffix}"
            suffix += 1
        seen.add(tid)
        tc["id"] = tid
        tc.setdefault("timeout_ms", 2000)
        tc.setdefault("is_hidden", False)
        fixed.append(tc)
    spec["testcases"] = fixed
    try:
        ProblemSpec.model_validate(spec)
    except Exception as exc:
        errors.append(f"format_dedup: invalid problem spec: {exc}")
    return {"problem_spec": spec, "errors": errors}


def oracle_solution(state: CreatorState) -> dict:
    """Heavy: write reference solution for the current spec (with failure ctx)."""
    errors = _errors(state)
    spec = state.get("problem_spec") or {}
    prompt = (
        f"Write a correct, efficient Python solution for: {spec.get('statement')}\n"
        f"Signature: {spec.get('signature')}. Examples: {spec.get('examples')}. "
        f"Constraints: {spec.get('constraints')}.\n"
        f"{_failure_context(state)}"
        "Define exactly the function above (plus any helpers). Use only allowlisted "
        "stdlib modules (math, heapq, collections, bisect, itertools, functools, typing). "
        "No I/O, no other imports. Output raw Python code only, no markdown fences."
    )
    try:
        code = generate("oracle_solution", prompt)
    except Exception as exc:
        errors.append(f"oracle_solution: {exc}")
        return {"errors": errors}
    code = _strip_code_fences(str(code))
    if not code:
        errors.append("oracle_solution: LLM returned empty code")
        return {"errors": errors}
    return {"solution_code": code, "errors": errors}


def static_scan(state: CreatorState) -> dict:
    """Deterministic guard: AST scan oracle before it ever executes."""
    verdict = scan(state.get("solution_code") or "")
    try:
        log_scan(state.get("workdir"), stage="creator.static_scan",
                 code=state.get("solution_code"), flags=verdict["flags"])
    except Exception:
        pass
    return {"safety_flags": verdict["flags"], "scan_safe": verdict["safe"]}


def run_tests(state: CreatorState) -> dict:
    """Deterministic: execute oracle against spec testcases via sandbox."""
    errors = _errors(state)
    spec = state.get("problem_spec")
    code = state.get("solution_code")
    workdir = state.get("workdir")
    if not isinstance(spec, dict) or not code or not workdir:
        errors.append("run_tests: missing problem spec, solution, or workdir")
        return {"errors": errors, "test_results": []}
    try:
        prob = ProblemSpec.model_validate(spec)
    except Exception as exc:
        errors.append(f"run_tests: invalid problem spec: {exc}")
        return {"errors": errors, "test_results": []}
    try:
        results = safe_execute(code, prob, workdir)
    except Exception as exc:
        errors.append(f"run_tests: executor error: {exc}")
        return {"errors": errors, "test_results": []}
    dumped = [r.model_dump() for r in results]
    timings = [r["time_ms"] for r in dumped]
    try:
        log_run(workdir, stage="creator.run_tests", code=code,
                results=dumped, timings=timings)
    except Exception:
        pass
    return {
        "test_results": dumped,
        "timings_ms": timings,
        "errors": errors,
    }


def repair_solution(state: CreatorState) -> dict:
    """Heavy: fix invalid spec (oracle is regenerated next node). Counts attempt."""
    errors = _errors(state)
    attempt = int(state.get("attempt") or 0) + 1
    spec = state.get("problem_spec")
    req = state.get("requirements") or {}
    out: dict = {"attempt": attempt, "errors": errors}
    if state.get("human_approved_promote") is False:
        # Step 12: a declined promotion retries repair once with the
        # decline reason as context, then gets a clean approval pass.
        out["human_approved_promote"] = None
    human_ctx = ""
    if state.get("human_answer"):
        human_ctx = (f"Human feedback (must respect): {state.get('human_answer')}\n")
    try:
        ProblemSpec.model_validate(spec)
        out.update({"errors": errors})
        return out  # spec ok; oracle retries
    except Exception as exc:
        errors.append(f"repair: spec invalid, regenerating: {exc}")
    prompt = (
        "The problem JSON below is INVALID or inconsistent. Errors:\n"
        f"{errors[-2:]}\nRequirements: category={req.get('category')}, "
        f"difficulty={req.get('difficulty')}, num_tests={req.get('num_tests')}.\n"
        f"{human_ctx}"
        f"Current spec: {json.dumps(spec)[:3000]}\n"
        "Return a corrected FULL problem JSON (keys: id, title, difficulty, statement, "
        "function_name, signature, examples, constraints, testcases) with unique "
        "testcase ids. Every examples[].input and testcases[].input MUST be an object "
        'mapping parameter names to values (e.g. {"nums": [...], "target": 9}), '
        "never a bare list. Return JSON only."
    )
    try:
        fixed = generate("repair_problem", prompt, json_mode=True,
                          retry_count=min(attempt, 1))
    except Exception as exc:
        errors.append(f"repair: {exc}")
        out.update({"errors": errors})
        return out
    if isinstance(fixed, dict):
        fixed = dict(fixed)
        fixed["difficulty"] = req.get("difficulty", "easy")
        try:
            TestCase.model_validate(fixed.get("testcases", [{}])[0])
        except Exception:
            pass
        out.update({"problem_spec": fixed, "errors": errors})
        return out
    errors.append("repair: LLM did not return a JSON object")
    out.update({"errors": errors})
    return out


def promote_save(state: CreatorState) -> dict:
    """Validate final spec, save problem JSON to job dir, promote to problems/.

    Step 10: writes audit.jsonl entry (code_hash, flags, timings) and
    records approval state. Human approval is enforced in
    request_promote_approval before this node runs.
    """
    errors = _errors(state)
    try:
        prob = ProblemSpec.model_validate(state.get("problem_spec"))
    except Exception as exc:
        errors.append(f"promote: invalid spec: {exc}")
        return {"errors": errors, "needs_human": True}
    digest = hashlib.sha256((state.get("solution_code") or "").encode()).hexdigest()[:16]
    final = prob.model_dump()
    final["oracle_solution_hash"] = digest
    prob = ProblemSpec.model_validate(final)  # hash field is part of schema
    filename = f"problem_id_{prob.id}.json"
    try:
        scoped_write(str(state["workdir"]), filename,
                     json.dumps(final, indent=2))
        dest = promote_to_problems(str(state["workdir"]), filename)
    except Exception as exc:
        errors.append(f"promote: {exc}")
        return {"errors": errors, "needs_human": True}
    try:
        append_audit(str(state["workdir"]), {
            "stage": "creator.promote_save",
            "code_hash": digest,
            "problem_id": prob.id,
            "flags": list(state.get("safety_flags") or []),
            "timings_ms": list(state.get("timings_ms") or []),
            "decision": "promoted",
            "approved": bool(state.get("human_approved_promote") is not False),
        })
    except Exception:
        pass
    return {
        "problem_spec": prob.model_dump(),
        "problem_path": str(dest),
        "errors": errors,
    }


def request_promote_approval(state: CreatorState) -> dict:
    """Step 10 human approval gate before promote_save.

    Explicit human_approved_promote=False blocks for review (needs_human).
    None/True auto-passes for automated runs (legacy compat); prod CLI
    should set False initially and resume with True after review, or use
    build_creator_graph(with_interrupt=True) which pauses before promote.

    Step 12: a decline (False + human_answer reason) is logged as
    "declined" and routed to repair_solution when budget remains,
    else flag_human. Never silently dropped.
    """
    errors = _errors(state)
    if state.get("human_approved_promote") is False:
        reason = state.get("human_answer") or "human declined promotion"
        try:
            log_decision(state.get("workdir"), stage="creator.promote_approval",
                         decision="declined", reason=str(reason))
        except Exception:
            pass
        return {"pending_approval": "promote", "errors": errors}
    return {"pending_approval": None, "errors": errors}


def flag_human(state: CreatorState) -> dict:
    """Max repairs exhausted — stop for human review."""
    errors = _errors(state)
    errors.append(f"repair budget exhausted after {MAX_REPAIRS} attempts")
    return {"needs_human": True, "errors": errors}


# ---------------------------------------------------------------- routing

def route_after_intake(state: CreatorState) -> str:
    if state.get("needs_human"):
        return "flag_human"
    return "summarize_understanding"


def route_after_understanding(state: CreatorState) -> str:
    return "flag_human" if state.get("needs_human") else "draft_problem"


def _repair_or_human(state: CreatorState) -> str:
    if int(state.get("attempt") or 0) < MAX_REPAIRS:
        if (int(state.get("attempt") or 0) >= 2
                and state.get("allow_clarify")
                and not state.get("clarify_asked")
                and not state.get("human_answer")):
            return "request_clarify"
        return "repair_solution"
    return "flag_human"


def route_after_scan(state: CreatorState) -> str:
    if state.get("scan_safe"):
        return "run_tests"
    return _repair_or_human(state)


def route_after_tests(state: CreatorState) -> str:
    results = state.get("test_results") or []
    if results and all(r.get("passed") for r in results):
        return "request_promote_approval"
    return _repair_or_human(state)


def route_after_promote_approval(state: CreatorState) -> str:
    if state.get("human_approved_promote") is False:
        # Step 12 decline: retry repair with the reason as context when
        # budget remains, else stop for human review. Decline is logged
        # in request_promote_approval, never silently dropped.
        if int(state.get("attempt") or 0) < MAX_REPAIRS:
            return "repair_solution"
        return "flag_human"
    return "flag_human" if state.get("needs_human") else "promote_save"


def build_creator_graph(checkpointer=None, with_interrupt: bool = False) -> object:
    """Step 10: SqliteSaver persistence + optional human interrupt.

    with_interrupt=True pauses before promote_save for manual review.
    Default False keeps automated runs non-blocking (approval gate still
    blocks when human_approved_promote is explicitly False).
    """
    builder = StateGraph(CreatorState)
    builder.add_node("intake_validate", intake_validate)
    builder.add_node("summarize_understanding", summarize_understanding)
    builder.add_node("request_understanding_approval", request_understanding_approval)
    builder.add_node("request_clarify", request_clarify)
    builder.add_node("draft_problem", draft_problem)
    builder.add_node("generate_tests", generate_tests)
    builder.add_node("format_dedup", format_dedup)
    builder.add_node("oracle_solution", oracle_solution)
    builder.add_node("static_scan", static_scan)
    builder.add_node("run_tests", run_tests)
    builder.add_node("request_promote_approval", request_promote_approval)
    builder.add_node("promote_save", promote_save)
    builder.add_node("repair_solution", repair_solution)
    builder.add_node("flag_human", flag_human)
    builder.add_edge(START, "intake_validate")
    builder.add_conditional_edges(
        "intake_validate", route_after_intake,
        {"summarize_understanding": "summarize_understanding",
         "flag_human": "flag_human"})
    builder.add_edge("summarize_understanding", "request_understanding_approval")
    builder.add_conditional_edges(
        "request_understanding_approval", route_after_understanding,
        {"draft_problem": "draft_problem", "flag_human": "flag_human"})
    builder.add_edge("request_clarify", "flag_human")
    builder.add_edge("draft_problem", "generate_tests")
    builder.add_edge("generate_tests", "format_dedup")
    builder.add_edge("format_dedup", "oracle_solution")
    builder.add_edge("oracle_solution", "static_scan")
    builder.add_conditional_edges(
        "static_scan", route_after_scan,
        {"run_tests": "run_tests", "repair_solution": "repair_solution",
         "request_clarify": "request_clarify", "flag_human": "flag_human"})
    builder.add_conditional_edges(
        "run_tests", route_after_tests,
        {"request_promote_approval": "request_promote_approval",
         "repair_solution": "repair_solution",
         "request_clarify": "request_clarify",
         "flag_human": "flag_human"})
    builder.add_conditional_edges(
        "request_promote_approval", route_after_promote_approval,
        {"promote_save": "promote_save", "flag_human": "flag_human",
         "repair_solution": "repair_solution"})
    builder.add_edge("repair_solution", "oracle_solution")
    builder.add_edge("promote_save", END)
    builder.add_edge("flag_human", END)
    saver = checkpointer or get_checkpointer()
    if with_interrupt:
        return builder.compile(checkpointer=saver, interrupt_before=["promote_save"])
    return builder.compile(checkpointer=saver)


creator_app = build_creator_graph()


def build_creator_graph_strict() -> object:
    """Prod graph pausing before promote_save for human approval."""
    return build_creator_graph(with_interrupt=True)
