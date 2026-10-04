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

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.llm_router import generate
from src.schemas import ProblemSpec, RequirementsSpec, TestCase
from src.state import AgentState
from src.tools.access_broker import (
    create_job,
    promote_to_problems,
    scoped_read,
    scoped_write,
)
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


def draft_problem(state: CreatorState) -> dict:
    """Heavy: invent problem shell (testcases are placeholders, replaced next)."""
    errors = _errors(state)
    req = state.get("requirements") or {}
    prompt = (
        "You create LeetCode-style coding problems as JSON.\n"
        f"Category: {req.get('category')}. Difficulty: {req.get('difficulty')}. Language: python.\n"
        f"Constraints hint: {req.get('constraints_hint')}. Extra notes: {req.get('extra')}.\n"
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
    return {
        "test_results": dumped,
        "timings_ms": [r["time_ms"] for r in dumped],
        "errors": errors,
    }


def repair_solution(state: CreatorState) -> dict:
    """Heavy: fix invalid spec (oracle is regenerated next node). Counts attempt."""
    errors = _errors(state)
    attempt = int(state.get("attempt") or 0) + 1
    spec = state.get("problem_spec")
    req = state.get("requirements") or {}
    try:
        ProblemSpec.model_validate(spec)
        return {"attempt": attempt, "errors": errors}  # spec ok; oracle retries
    except Exception as exc:
        errors.append(f"repair: spec invalid, regenerating: {exc}")
    prompt = (
        "The problem JSON below is INVALID or inconsistent. Errors:\n"
        f"{errors[-2:]}\nRequirements: category={req.get('category')}, "
        f"difficulty={req.get('difficulty')}, num_tests={req.get('num_tests')}.\n"
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
        return {"attempt": attempt, "errors": errors}
    if isinstance(fixed, dict):
        fixed = dict(fixed)
        fixed["difficulty"] = req.get("difficulty", "easy")
        try:
            TestCase.model_validate(fixed.get("testcases", [{}])[0])
        except Exception:
            pass
        return {"attempt": attempt, "problem_spec": fixed, "errors": errors}
    errors.append("repair: LLM did not return a JSON object")
    return {"attempt": attempt, "errors": errors}


def promote_save(state: CreatorState) -> dict:
    """Validate final spec, save problem JSON to job dir, promote to problems/."""
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
    return {
        "problem_spec": prob.model_dump(),
        "problem_path": str(dest),
        "errors": errors,
    }


def flag_human(state: CreatorState) -> dict:
    """Max repairs exhausted — stop for human review."""
    errors = _errors(state)
    errors.append(f"repair budget exhausted after {MAX_REPAIRS} attempts")
    return {"needs_human": True, "errors": errors}


# ---------------------------------------------------------------- routing

def route_after_intake(state: CreatorState) -> str:
    return "flag_human" if state.get("needs_human") else "draft_problem"


def _repair_or_human(state: CreatorState) -> str:
    if int(state.get("attempt") or 0) < MAX_REPAIRS:
        return "repair_solution"
    return "flag_human"


def route_after_scan(state: CreatorState) -> str:
    if state.get("scan_safe"):
        return "run_tests"
    return _repair_or_human(state)


def route_after_tests(state: CreatorState) -> str:
    results = state.get("test_results") or []
    if results and all(r.get("passed") for r in results):
        return "promote_save"
    return _repair_or_human(state)


def build_creator_graph() -> object:
    builder = StateGraph(CreatorState)
    builder.add_node("intake_validate", intake_validate)
    builder.add_node("draft_problem", draft_problem)
    builder.add_node("generate_tests", generate_tests)
    builder.add_node("format_dedup", format_dedup)
    builder.add_node("oracle_solution", oracle_solution)
    builder.add_node("static_scan", static_scan)
    builder.add_node("run_tests", run_tests)
    builder.add_node("promote_save", promote_save)
    builder.add_node("repair_solution", repair_solution)
    builder.add_node("flag_human", flag_human)
    builder.add_edge(START, "intake_validate")
    builder.add_conditional_edges(
        "intake_validate", route_after_intake,
        {"draft_problem": "draft_problem", "flag_human": "flag_human"})
    builder.add_edge("draft_problem", "generate_tests")
    builder.add_edge("generate_tests", "format_dedup")
    builder.add_edge("format_dedup", "oracle_solution")
    builder.add_edge("oracle_solution", "static_scan")
    builder.add_conditional_edges(
        "static_scan", route_after_scan,
        {"run_tests": "run_tests", "repair_solution": "repair_solution",
         "flag_human": "flag_human"})
    builder.add_conditional_edges(
        "run_tests", route_after_tests,
        {"promote_save": "promote_save", "repair_solution": "repair_solution",
         "flag_human": "flag_human"})
    builder.add_edge("repair_solution", "oracle_solution")
    builder.add_edge("promote_save", END)
    builder.add_edge("flag_human", END)
    return builder.compile(checkpointer=MemorySaver())


creator_app = build_creator_graph()
