"""Node stubs — full prompts land in the build-out step."""

from __future__ import annotations

MAX_INTENT_ITERATIONS = 3

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
    n = state.get("intent_iterations", 0) + 1
    req = state.get("requirement", "").strip()
    prior = state.get("clarified_intent", "") or req
    # TODO: small-LLM rewrite pass (deepseek-r1:1.5b) appended here;
    # stub keeps prior text and only scores clarity.
    clear = is_intent_clear(prior)
    return {
        "clarified_intent": prior,
        "intent_iterations": n,
        "intent_clear": clear,
    }


def planner(state: dict) -> dict:
    feedback = state.get("feedback", "")
    base = [f"Solve: {state.get('clarified_intent', '')[:120]}"]
    if feedback:
        base.append(f"Address feedback: {feedback[:500]}")
    return {"plan_steps": base}


def test_case_gen(state: dict) -> dict:
    return {"test_cases": []}


def coder(state: dict) -> dict:
    """Generate code (stub: heavy-LLM later), then verify via python toolbox."""
    from .tools.sandbox import run_python

    # TODO: replace stub with get_heavy().bind_tools([python_toolbox]) call.
    code = state.get("code") or "# TODO: implement solution\n"
    if code.strip().startswith("# TODO"):
        smoke = run_python(code, harness="print('smoke: imports ok')")
    else:
        smoke = run_python(code, harness="print('smoke: imports ok')")
    return {"code": code, "coder_smoke": smoke}


def evaluator(state: dict) -> dict:
    tests = state.get("test_cases", [])
    total = len(tests)
    return {
        "execution_results": [],
        "eval": {
            "passed": total == 0,
            "passed_count": 0,
            "total": total,
            "feedback": "" if total == 0 else "no test cases generated yet",
        },
    }


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
        "clarified_intent": state.get("clarified_intent", ""),
        "intent_iterations": state.get("intent_iterations", 0),
        "intent_clear": state.get("intent_clear", False),
        "plan_steps": state.get("plan_steps", []),
        "test_cases": state.get("test_cases", []),
        "code": state.get("code", ""),
        "coder_smoke": state.get("coder_smoke", {}),
        "execution_results": state.get("execution_results", []),
        "eval": state.get("eval", {}),
    }
    path = Path("output") / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return {"output_path": str(path)}
