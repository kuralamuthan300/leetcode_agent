"""LangGraph nodes: Steps 1-4 LLM-backed, Step 5 guardrailed, Step 6 evaluator.

Every LLM call falls back to deterministic behavior so tests stay hermetic.
"""

from __future__ import annotations

from . import prompts
from .tools.guardrail import is_truncated
from .prompts import (
    CODER_REPAIR_SYSTEM,
    CODER_SYSTEM,
    INTENT_SYSTEM,
    PLANNER_SYSTEM,
    TESTGEN_SYSTEM,
    args_in_range,
    code_defines_signature,
    critique,
    extract_code,
    extract_contract,
    extract_json_array,
    harness_matches,
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


def _missing_questions(req: str, text: str) -> list[str]:
    low = f"{req}\n{text}".lower()
    qs = []
    if is_truncated(req):
        tail = " ".join(req.strip().split()[-4:])
        qs.append(f"Your request seems cut off after '{tail}' — what comes next?")
    if "array" in low and not any(w in low for w in ("matrix", "matrices", "1-d", "2-d", "1d", "2d", "one-dimension", "two-dimension")):
        qs.append("1-D array, matrix, or another structure?")
    if "rotat" in low and not any(w in low for w in ("clockwise", "counter", "left", "right", " by ", " k")):
        qs.append("Rotate by how much (k), and in which direction?")
    if not any(w in low for w in ("input", "output", "return", "example")):
        qs.append("What are the exact input and output formats (with an example)?")
    if not any(w in low for w in ("constraint", "complexity", "o(n", "o(1", "limit")):
        qs.append("Any constraints or required complexity?")
    return qs[:3] or ["What are the constraints and an example?"]


def _assumptions(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip().lower().startswith("assumption")][:5]


def _confirm_with_user(understanding: str, assumptions: list[str], questions: list[str]) -> tuple[str, bool]:
    """Print understanding, ask inline. Returns (extra_info, confirmed)."""
    print("\nMy understanding:")
    print(understanding.strip()[:1500])
    if assumptions:
        print("\nAssumptions:")
        for a in assumptions:
            print(f"- {a[:200]}")
    if questions:
        print("\nOpen questions:")
        for i, q in enumerate(questions, 1):
            print(f"{i}. {q}")
    else:
        print("\nNo open questions.")
    try:
        answer = input("Confirm (Enter=yes) or type corrections: ")
    except EOFError:
        return "", False
    if answer.strip().lower() in ("", "ok", "yes", "y", "looks good", "confirmed", "correct", "lgtm"):
        return "", True
    return answer, False


def requirement_gatherer(state: dict) -> dict:
    import sys

    from .prompts import SCOPE_SYSTEM, try_invoke
    from .tools.guardrail import check_language, check_scope, is_truncated

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

    needs_confirmation = is_truncated(req) or not is_intent_clear(text)
    confirmed, assumed = True, False
    session_notes: list[str] = []
    remembered: list[str] = []
    if sys.stdin.isatty():
        from . import session as session_mem

        remembered = session_mem.load()
        if remembered:
            session_notes = [f"(remembered) {c}" for c in remembered]
            ctx = "Session conventions (already confirmed, apply them):\n" + "\n".join(f"- {c}" for c in remembered)
            follow = try_invoke("small", f"{ctx}\n\nRequirement:\n{req}", INTENT_SYSTEM)
            if follow:
                text = follow
        questions = _missing_questions(req, text) if needs_confirmation else []
        for _ in range(2):
            extra, ok = _confirm_with_user(text, _assumptions(text) + session_notes, questions)
            if ok:
                break
            req = f"{req}\nUser clarification: {extra.strip()}"
            session_mem.remember([extra.strip()])
            follow = try_invoke("small", f"Requirement:\n{req}", INTENT_SYSTEM)
            text = follow if follow else text
            questions = _missing_questions(req, text)
        confirmed = ok
        assumed = not ok
        if ok:
            session_mem.remember(_assumptions(text))
    elif needs_confirmation:
        assumed = True
    return {
        "clarified_intent": text,
        "requirement": req,
        "intent_iterations": n,
        "intent_clear": is_intent_clear(text) and not assumed,
        "confirmed": confirmed,
        "assumed": assumed,
        "session_notes": session_notes,
    }


def _parse_plan(text: str) -> list[str]:
    import re

    steps = []
    in_fence = False
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or s.upper().startswith("CONTRACT:") or s.upper().startswith("SIGNATURE:"):
            continue
        if s.startswith(("{", "}", '"')):
            continue
        line = re.sub(r"^[\s\-\*\d\.\)]+", "", raw).strip()
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
    if not text:
        base = [f"Solve: {intent[:120]}"]
        if feedback:
            base.append(f"Address feedback: {feedback[:500]}")
        return {"plan_steps": base}
    overruled = False
    verdict = critique("plan", text, intent)
    if verdict and not verdict["passed"]:
        text2, fb2 = try_invoke_with_fallback(
            user + f"\n\nReviewer issues (fix all):\n{verdict['issues']}", PLANNER_SYSTEM)
        if text2:
            text, fell_back = text2, fell_back or fb2
            verdict2 = critique("plan", text, intent)
            overruled = bool(verdict2 and not verdict2["passed"])
        else:
            overruled = True
    out: dict = {"plan_steps": _parse_plan(text)}
    contract = extract_contract(text)
    if contract:
        out["function_signature"] = contract["signature"]
        out["arg_spec"] = contract["args"]
        out["returns_spec"] = contract["returns"]
    if fell_back:
        out["fallback_used"] = True
    if overruled:
        out["critic_overruled"] = True
    return out


def _valid_test(item: object, sig: str | None = None, arg_spec: list | None = None) -> dict | None:
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
    if sig and not harness_matches(harness, sig):
        return None
    if arg_spec and not args_in_range(arg_spec, harness):
        return None
    return {"id": tid, "harness": harness, "expected": expected.strip()}


def _contract_brief(sig: str, arg_spec: list, returns_spec: str = "") -> str:
    """Render the planner contract for the testgen prompt."""
    lines = [f"Contract: {sig or 'def solve(...)'}"]
    for a in arg_spec or []:
        if isinstance(a, dict) and a.get("name"):
            rng = f", range: {a['range']}" if a.get("range") else ""
            typ = f": {a['type']}" if a.get("type") else ""
            lines.append(f"- {a['name']}{typ}{rng}")
    if returns_spec:
        lines.append(f"Returns: {returns_spec}")
    return "\n".join(lines)


def test_case_gen(state: dict) -> dict:
    intent = state.get("clarified_intent", "")
    plan = "\n".join(state.get("plan_steps", []))
    sig = state.get("function_signature", "")
    arg_spec = state.get("arg_spec", [])
    code = state.get("code", "")
    user = f"Problem:\n{intent}\n\nPlan:\n{plan}\n\n{_contract_brief(sig, arg_spec, state.get('returns_spec', ''))}"
    user += (
        "\nGenerate values strictly within these ranges. "
        "Cover boundaries: a minimum-boundary case, a maximum-boundary case, "
        "and an empty-input case only if the contract allows empty."
    )
    if sig:
        user += f"\n\nCall the function exactly as contracted: {sig}"
    if code and not code.strip().startswith("# TODO"):
        user += f"\n\nReference implementation (match its calls):\n{code[:2000]}"
    text = try_invoke("small", user, TESTGEN_SYSTEM)
    if not text:
        return {"test_cases": []}
    parsed = extract_json_array(text)
    if not parsed:
        return {"test_cases": []}
    tests = [t for item in parsed[:MAX_TESTS] if (t := _valid_test(item, sig or None, arg_spec or None))]
    overruled = False
    if tests:
        import json as _json

        verdict = critique("tests", _json.dumps(tests)[:4000], intent)
        if verdict and not verdict["passed"]:
            text2 = try_invoke("small", user + f"\n\nReviewer issues (fix all):\n{verdict['issues']}", TESTGEN_SYSTEM)
            if text2:
                parsed2 = extract_json_array(text2)
                if parsed2:
                    tests = [t for item in parsed2[:MAX_TESTS] if (t := _valid_test(item, sig or None, arg_spec or None))]
                    verdict2 = critique("tests", _json.dumps(tests)[:4000], intent)
                    overruled = bool(verdict2 and not verdict2["passed"])
                else:
                    overruled = True
            else:
                overruled = True
    out: dict = {"test_cases": tests}
    if overruled:
        out["critic_overruled"] = True
    return out


def _smoke(code: str, sig: str = "") -> tuple[dict, dict]:
    from .errors import dummy_args, is_call_failure
    from .tools import guardrail_report as gate
    from .tools.sandbox import run_python

    report = gate(code, "print('smoke: imports ok')")
    smoke = run_python(code, harness="print('smoke: imports ok')")
    if smoke["error"] or not sig:
        return smoke, report
    # Call-smoke: actually invoke solve() to catch body NameErrors.
    # Other dummy-input exceptions are noted, not failed (see errors.py).
    call = run_python(code, harness=f"print(solve({', '.join(dummy_args(sig))}))")
    if is_call_failure(call["error"]):
        return {"stdout": call["stdout"], "stderr": call["stderr"], "error": call["error"]}, report
    if call["error"]:
        smoke = dict(smoke)
        smoke["note"] = f"call-smoke (ignored): {call['error'][:200]}"
    return smoke, report


def coder(state: dict) -> dict:
    """Generate code honoring the planner's signature, gate + smoke, repair once."""
    intent = state.get("clarified_intent", "")
    plan = "\n".join(state.get("plan_steps", []))
    sig = state.get("function_signature", "")
    feedback = state.get("feedback", "") or state.get("eval", {}).get("feedback", "")
    previous = state.get("code", "")
    user = f"Problem:\n{intent}\n\nPlan:\n{plan}"
    if sig:
        user += f"\n\nYou MUST define exactly this function (same name and parameters): {sig}"
    if feedback:
        user += f"\n\nMust address this feedback:\n{feedback[:1000]}"
    if previous and not previous.strip().startswith("# TODO"):
        user += f"\n\nPrevious code to improve:\n{previous[:4000]}"

    text, fell_back = try_invoke_with_fallback(user, CODER_SYSTEM)
    code = extract_code(text) if text else "# TODO: implement solution\n"
    repaired = False

    if sig and text and not code_defines_signature(code, sig):
        correction = (
            f"Problem:\n{intent}\n\nYour code must define exactly: {sig}\n"
            f"Failing code:\n{code[:4000]}"
        )
        fixed, repair_fell_back = try_invoke_with_fallback(correction, CODER_REPAIR_SYSTEM)
        fell_back = fell_back or repair_fell_back
        if fixed:
            code = extract_code(fixed)
        repaired = True
    smoke, report = _smoke(code, sig)

    if smoke["error"] and not smoke["error"].startswith("blocked:") and text and not repaired:
        repair_user = (
            f"Problem:\n{intent}\n\nFailing code:\n{code[:4000]}"
            f"\n\nSmoke error:\n{smoke['error'][:1000]}"
        )
        fixed, repair_fell_back = try_invoke_with_fallback(repair_user, CODER_REPAIR_SYSTEM)
        fell_back = fell_back or repair_fell_back
        if fixed:
            code = extract_code(fixed)
            smoke, report = _smoke(code, sig)
        repaired = True

    overruled = False
    if text and "def solve" in code:
        verdict = critique("code", code, f"{intent}\n\nPlan:\n{plan}")
        if verdict and not verdict["passed"]:
            if not repaired:
                regen_user = (
                    f"Problem:\n{intent}\n\nYour code:\n{code[:4000]}"
                    f"\n\nReviewer issues (fix all):\n{verdict['issues']}"
                )
                fixed, repair_fell_back = try_invoke_with_fallback(regen_user, CODER_REPAIR_SYSTEM)
                fell_back = fell_back or repair_fell_back
                if fixed:
                    code = extract_code(fixed)
                    smoke, report = _smoke(code, sig)
                    verdict2 = critique("code", code, intent)
                    overruled = bool(verdict2 and not verdict2["passed"])
                else:
                    overruled = True
            else:
                overruled = True

    out: dict = {"code": code, "coder_smoke": smoke, "guardrail_report": report}
    if fell_back:
        out["fallback_used"] = True
    if overruled:
        out["critic_overruled"] = True
    return out


def evaluator(state: dict) -> dict:
    """Run each test harness against the code; fail-closed on empty tests.

    Mechanical smoke failure short-circuits: no matrix run, straight back
    to planner with error_class set (Step 3).
    """
    from .errors import classify_error
    from .prompts import adjudicate
    from .tools.sandbox import run_python

    code = state.get("code", "")
    tests = state.get("test_cases", [])
    sig = state.get("function_signature", "")
    arg_spec = state.get("arg_spec", [])
    problem = state.get("clarified_intent", "")
    iterations = state.get("iterations", 0) + 1
    smoke_err = (state.get("coder_smoke") or {}).get("error", "")
    if smoke_err and classify_error(smoke_err) in ("mechanical", "blocked", "timeout"):
        ev = {
            "passed": False,
            "passed_count": 0,
            "total": len(tests),
            "feedback": f"smoke failed ({classify_error(smoke_err)}), fix code first: {smoke_err[:500]}",
            "error_class": "mechanical",
        }
        return {"execution_results": [], "eval": ev, "iterations": iterations}
    results = []
    for t in tests:
        r = run_python(code, harness=t["harness"])
        ok = not r["error"] and r["stdout"].strip() == t["expected"].strip()
        entry = {
            "test_id": t["id"],
            "passed": ok,
            "actual": r["stdout"].strip()[:2000],
            "expected": t["expected"].strip()[:2000],
            "error": r["error"][:2000],
            "blame": "none",
        }
        if not ok:
            verdict = adjudicate(problem, sig, arg_spec, code, entry)
            entry["blame"] = verdict["blame"]
            entry["error"] = (f"[blame:{verdict['blame']}] {verdict['reason']} " + entry["error"])[:2000]
        results.append(entry)
    passed_count = sum(1 for r in results if r["passed"])
    if not tests:
        ev = {"passed": False, "passed_count": 0, "total": 0, "feedback": "no test cases generated", "error_class": "empty"}
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
            "error_class": "none" if not failed else "logical",
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
        "confirmed": state.get("confirmed", False),
        "assumed": state.get("assumed", False),
        "plan_steps": state.get("plan_steps", []),
        "function_signature": state.get("function_signature", ""),
        "test_cases": state.get("test_cases", []),
        "code": state.get("code", ""),
        "coder_smoke": state.get("coder_smoke", {}),
        "guardrail_report": state.get("guardrail_report", {}),
        "fallback_used": state.get("fallback_used", False),
        "critic_overruled": state.get("critic_overruled", False),
        "session_notes": state.get("session_notes", []),
        "execution_results": state.get("execution_results", []),
        "eval": state.get("eval", {}),
    }
    path = Path("output") / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return {"output_path": str(path)}
