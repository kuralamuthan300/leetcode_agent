"""LangGraph nodes: Steps 1-4 LLM-backed, Step 5 guardrailed, Step 6 evaluator.

Every LLM call falls back to deterministic behavior so tests stay hermetic.
"""

from __future__ import annotations

from . import prompts
from .prompts import (
    CODER_REPAIR_SYSTEM,
    CODER_SYSTEM,
    INTENT_SYSTEM,
    PLANNER_SYSTEM,
    TESTGEN_SYSTEM,
    extract_code,
    extract_json_array,
    try_invoke,
    try_invoke_with_fallback,
)

MAX_INTENT_ITERATIONS = 3
MAX_TESTS = 10

_CLEAR_KEYWORDS = (
    "input",
    "output",
    "return",
    "given",
    "example",
    "constraint",
    "array",
    "string",
    "number",
    "list",
    "tree",
    "graph",
    "sort",
    "search",
)


def is_intent_clear(text: str) -> bool:
    t = text.strip().lower()
    if len(t) < 30:
        return False
    return any(k in t for k in _CLEAR_KEYWORDS)


def intent_clarifier(state: dict) -> dict:
    from .prompts import SCOPE_SYSTEM, try_invoke
    from .tools.guardrail import check_language, check_scope

    n = state.get("intent_iterations", 0) + 1
    req = state.get("requirement", "").strip()
    lang = check_language(req)
    if not lang["allowed"]:
        return {
            "clarified_intent": req,
            "intent_iterations": n,
            "intent_clear": False,
            "rejected": True,
            "rejection_reason": lang["reason"] + " — this agent solves in Python only",
        }
    scope = check_scope(req)
    if scope["allowed"] is False:
        return {
            "clarified_intent": req,
            "intent_iterations": n,
            "intent_clear": False,
            "rejected": True,
            "rejection_reason": scope["reason"] + " — this agent solves DSA coding tasks only",
        }
    if scope["allowed"] is None:
        verdict = try_invoke("small", f"Request:\n{req}", SCOPE_SYSTEM)
        if not verdict or not verdict.strip().upper().startswith("ALLOW"):
            return {
                "clarified_intent": req,
                "intent_iterations": n,
                "intent_clear": False,
                "rejected": True,
                "rejection_reason": (verdict or "scope unclear").strip()[:200]
                + " — this agent solves DSA coding tasks only",
            }
    prior = state.get("clarified_intent", "") or req
    if n == 1:
        user = f"Requirement:\n{req}"
    else:
        user = (
            "The previous clarification is still too vague. Make it MORE specific: "
            f"add input/output formats and constraints.\nPrevious:\n{prior}"
        )
    rewritten = try_invoke("small", user, INTENT_SYSTEM)
    text = rewritten if rewritten else prior
    return {
        "clarified_intent": text,
        "intent_iterations": n,
        "intent_clear": is_intent_clear(text),
    }


def _parse_plan(text: str) -> list[str]:
    import re

    steps = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
        if line:
            steps.append(line)
    return steps[:10] or [text.strip()[:500]]


def planner(state: dict) -> dict:
    intent = state.get("clarified_intent", "")
    feedback = state.get("feedback", "")
    user = f"Problem:\n{intent}"
    if feedback:
        user += f"\n\nPrevious attempt failed with this feedback — address it:\n{feedback[:1000]}"
    text, fell_back = try_invoke_with_fallback(user, PLANNER_SYSTEM)
    if text:
        out: dict = {"plan_steps": _parse_plan(text)}
        if fell_back:
            out["fallback_used"] = True
        return out
    base = [f"Solve: {intent[:120]}"]
    if feedback:
        base.append(f"Address feedback: {feedback[:500]}")
    return {"plan_steps": base}


def _valid_test(item: object) -> dict | None:
    if not isinstance(item, dict):
        return None
    tid = str(item.get("id", "")).strip()
    harness = str(item.get("harness", "")).strip()
    expected = item.get("expected", "")
    if not (tid and harness and isinstance(expected, str) and expected.strip()):
        return None
    if "print" not in harness or "solve" not in harness:
        return None
    from .tools.guardrail import check

    if not check("", harness)["allowed"]:
        return None
    return {"id": tid, "harness": harness, "expected": expected.strip()}


def test_case_gen(state: dict) -> dict:
    intent = state.get("clarified_intent", "")
    plan = "\n".join(state.get("plan_steps", []))
    user = f"Problem:\n{intent}\n\nPlan:\n{plan}"
    text = try_invoke("small", user, TESTGEN_SYSTEM)
    if not text:
        return {"test_cases": []}
    parsed = extract_json_array(text)
    if not parsed:
        return {"test_cases": []}
    tests = [t for item in parsed[:MAX_TESTS] if (t := _valid_test(item))]
    return {"test_cases": tests}


def _smoke(code: str) -> tuple[dict, dict]:
    from .tools import guardrail_report as gate
    from .tools.sandbox import run_python

    report = gate(code, "print('smoke: imports ok')")
    smoke = run_python(code, harness="print('smoke: imports ok')")
    return smoke, report


def coder(state: dict) -> dict:
    """Generate code with heavy LLM, gate + smoke it, one repair attempt."""
    intent = state.get("clarified_intent", "")
    plan = "\n".join(state.get("plan_steps", []))
    feedback = state.get("feedback", "") or state.get("eval", {}).get("feedback", "")
    previous = state.get("code", "")
    user = f"Problem:\n{intent}\n\nPlan:\n{plan}"
    if feedback:
        user += f"\n\nMust address this feedback:\n{feedback[:1000]}"
    if previous and not previous.strip().startswith("# TODO"):
        user += f"\n\nPrevious code to improve:\n{previous[:4000]}"

    text, fell_back = try_invoke_with_fallback(user, CODER_SYSTEM)
    code = extract_code(text) if text else "# TODO: implement solution\n"
    smoke, report = _smoke(code)

    if smoke["error"] and not smoke["error"].startswith("blocked:") and text:
        repair_user = (
            f"Problem:\n{intent}\n\nFailing code:\n{code[:4000]}"
            f"\n\nSmoke error:\n{smoke['error'][:1000]}"
        )
        fixed, repair_fell_back = try_invoke_with_fallback(repair_user, CODER_REPAIR_SYSTEM)
        fell_back = fell_back or repair_fell_back
        if fixed:
            code = extract_code(fixed)
            smoke, report = _smoke(code)

    out: dict = {"code": code, "coder_smoke": smoke, "guardrail_report": report}
    if fell_back:
        out["fallback_used"] = True
    return out


def evaluator(state: dict) -> dict:
    """Run each test harness against the code; fail-closed on empty tests."""
    from .tools.sandbox import run_python

    code = state.get("code", "")
    tests = state.get("test_cases", [])
    iterations = state.get("iterations", 0) + 1
    results = []
    for t in tests:
        r = run_python(code, harness=t["harness"])
        ok = not r["error"] and r["stdout"].strip() == t["expected"].strip()
        results.append(
            {
                "test_id": t["id"],
                "passed": ok,
                "actual": r["stdout"].strip()[:2000],
                "expected": t["expected"].strip()[:2000],
                "error": r["error"][:2000],
            }
        )
    passed_count = sum(1 for r in results if r["passed"])
    if not tests:
        ev = {"passed": False, "passed_count": 0, "total": 0, "feedback": "no test cases generated"}
    else:
        failed = [r for r in results if not r["passed"]]
        details = "; ".join(
            f"{r['test_id']}: expected {r['expected'][:200]!r} got {r['actual'][:200]!r}"
            + (f" error {r['error'][:200]!r}" if r["error"] else "")
            for r in failed[:5]
        )
        ev = {
            "passed": passed_count == len(tests),
            "passed_count": passed_count,
            "total": len(tests),
            "feedback": "" if not failed else f"failed {len(failed)}: {details}"[:2000],
        }
    return {"execution_results": results, "eval": ev, "iterations": iterations}


def run_reporter(state: dict) -> dict:
    """Consolidate everything and write output/run_id_xxxx.json."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    from .llms import HEAVY_MODEL, SMALL_MODEL

    run_id = state.get("run_id", "run_id_unknown")
    out = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {"heavy": HEAVY_MODEL, "small": SMALL_MODEL},
        "requirement": state.get("requirement", ""),
        "rejected": state.get("rejected", False),
        "rejection_reason": state.get("rejection_reason", ""),
        "clarified_intent": state.get("clarified_intent", ""),
        "intent_iterations": state.get("intent_iterations", 0),
        "intent_clear": state.get("intent_clear", False),
        "plan_steps": state.get("plan_steps", []),
        "test_cases": state.get("test_cases", []),
        "code": state.get("code", ""),
        "coder_smoke": state.get("coder_smoke", {}),
        "guardrail_report": state.get("guardrail_report", {}),
        "fallback_used": state.get("fallback_used", False),
        "execution_results": state.get("execution_results", []),
        "eval": state.get("eval", {}),
    }
    path = Path("output") / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return {"output_path": str(path)}
