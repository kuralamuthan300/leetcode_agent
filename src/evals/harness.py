"""Trusted harness generation for golden cases."""

from __future__ import annotations

import ast
from typing import Any

RESULT_MARKER = "__RESULT__"


def _literal(value: Any) -> str:
    return repr(value)


def build_harness(function_name: str, case: Any) -> str:
    """Generate a call/print harness from a case's input and expected fields."""
    if not function_name.isidentifier():
        raise ValueError(f"invalid function name: {function_name!r}")
    if not isinstance(case, dict):
        raise ValueError("golden case must be an object")  # noqa: TRY004 - invalid dataset case

    if "kwargs" in case:
        kwargs = case["kwargs"]
        if not isinstance(kwargs, dict):
            raise ValueError("case field 'kwargs' must be an object")
        call = f"{function_name}(**{_literal(kwargs)})"
    elif "args" in case:
        args = case["args"]
        if not isinstance(args, (list, tuple)):
            raise ValueError("case field 'args' must be a list")
        call = f"{function_name}({', '.join(_literal(item) for item in args)})"
    elif isinstance(case.get("input"), dict):
        kwargs = case["input"]
        call = f"{function_name}(**{_literal(kwargs)})"
    elif isinstance(case.get("input"), (list, tuple)):
        args = case["input"]
        call = f"{function_name}({', '.join(_literal(item) for item in args)})"
    elif "input" in case:
        call = f"{function_name}({_literal(case['input'])})"
    else:
        args = [value for key, value in case.items() if key != "expected"]
        call = f"{function_name}({', '.join(_literal(item) for item in args)})"

    return f"print({RESULT_MARKER!r}, {call})"


def parse_harness_output(stdout: str) -> Any:
    """Parse the value following the last result marker in sandbox stdout."""
    matches = [
        line.split(RESULT_MARKER, 1)[1].strip()
        for line in stdout.splitlines()
        if RESULT_MARKER in line
    ]
    if not matches:
        raise ValueError(f"missing {RESULT_MARKER} output marker")
    try:
        return ast.literal_eval(matches[-1])
    except (ValueError, SyntaxError) as exc:
        if not matches[-1]:
            return ""
        raise ValueError(f"result is not a Python literal: {matches[-1]!r}") from exc
