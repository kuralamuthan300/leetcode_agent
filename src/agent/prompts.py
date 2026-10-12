"""Prompts + lenient output parsing for Steps 1-4.

All LLM calls go through `try_invoke`, which returns None on any failure so
nodes can fall back to deterministic behavior (tests stay hermetic / offline).
"""

from __future__ import annotations

import json
import re

SOLVE_CONTRACT = (
    "Define plain top-level Python code. The entry point MUST be a top-level "
    "function named `solve`. Use only the standard library modules: math, "
    "typing, collections, itertools, functools, heapq, bisect. No file, "
    "network, OS, subprocess, or shell access. No input() — the harness calls "
    "`solve` directly."
)

INTENT_SYSTEM = (
    "You clarify data-structures & algorithms problem statements. "
    "Rewrite the user's requirement as 2-4 crisp sentences covering: goal, "
    "input format, output format, and any constraints/examples. "
    "If details are missing, state sensible defaults explicitly as "
    "'Assumption: ...'. Reply with ONLY the clarified statement."
)

PLANNER_SYSTEM = (
    "You plan solutions to DSA problems. Given a clarified problem (and optional "
    "feedback from a failed attempt), output a short plan: one approach, its "
    "time/space complexity, then numbered implementation steps. "
    "Reply with ONLY the plan, one item per line."
)

TESTGEN_SYSTEM = (
    "You write test cases for DSA problems. Given the problem and plan, output "
    "a JSON array of 3 to 5 test objects. Each object has exactly these keys: "
    "id (short string), harness (a Python snippet that calls `solve` and MUST "
    "use print to output the result), expected (the exact printed output as a "
    "string, never null or an array). Example:\n"
    '[{"id": "base", "harness": "print(solve([2, 7, 11, 15], 9))", '
    '"expected": "[0, 1]"}]\n'
    "Reply with ONLY the JSON array, no prose."
)

CODER_SYSTEM = (
    "You write Python solutions to DSA problems. " + SOLVE_CONTRACT + " "
    "Reply with ONLY the code (a python fenced block is fine)."
)

CODER_REPAIR_SYSTEM = (
    "You fix Python solutions. " + SOLVE_CONTRACT + " "
    "The previous code failed its smoke run; the error is given. "
    "Reply with ONLY the corrected code."
)


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning traces (some models inline them)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_code(text: str) -> str:
    """Pull code out of a fenced block if present, else return text as-is."""
    text = strip_think(text)
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    return (m.group(1) if m else text).strip() + "\n"


def extract_json_array(text: str) -> list | None:
    """Parse a JSON array from raw or fenced LLM output; None if unparseable."""
    text = strip_think(text)
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    candidate = m.group(1) if m else text.strip()
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def try_invoke(kind: str, user: str, system: str, retries: int = 1) -> str | None:
    """Invoke small/heavy model; retry on empty; None on persistent failure."""
    from . import llms

    for _ in range(retries + 1):
        try:
            model = llms.get_small() if kind == "small" else llms.get_heavy()
            msg = model.invoke([("system", system), ("human", user)])
            content = msg.content if isinstance(msg.content, str) else ""
            if strip_think(content):
                return strip_think(content)
        except Exception:
            return None
    return None


def try_invoke_with_fallback(user: str, system: str) -> tuple[str | None, bool]:
    """Heavy first, small on failure. Returns (text, fell_back)."""
    text = try_invoke("heavy", user, system)
    if text:
        return text, False
    text = try_invoke("small", user, system)
    return text, text is not None
