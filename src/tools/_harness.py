"""Trusted harness (mounted at /work/_harness.py). LLM cannot edit it.

Usage: python3 /work/_harness.py /work/problem.json /work/solution.py
Reads problem JSON, imports solution.function_name via importlib,
runs each testcase, prints one JSON object per line to stdout.
"""

import contextlib
import importlib.util
import io
import json
import sys
import time
import traceback


def _to_jsonable(value):
    try:
        json.dumps(value)
        return value
    except Exception:
        try:
            return repr(value)
        except Exception:
            return "<unrepresentable>"


def _call(func, test_input):
    if isinstance(test_input, dict):
        try:
            return func(**test_input)
        except TypeError as exc:
            # Fallback: single-arg function receiving the whole dict,
            # or single-value dict passed positionally.
            if len(test_input) == 1:
                return func(next(iter(test_input.values())))
            raise exc
    return func(test_input)


def main(problem_path: str, solution_path: str) -> int:
    with open(problem_path, encoding="utf-8") as f:
        problem = json.load(f)

    function_name = problem["function_name"]
    testcases = problem.get("testcases", [])

    spec = importlib.util.spec_from_file_location("solution", solution_path)
    if spec is None or spec.loader is None:
        for tc in testcases:
            print(json.dumps({
                "test_id": tc.get("id", "unknown"),
                "passed": False,
                "time_ms": 0.0,
                "actual": None,
                "error": "harness: cannot load solution.py",
                "stdout": "",
            }))
        return 0

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        err = traceback.format_exc(limit=3)
        for tc in testcases:
            print(json.dumps({
                "test_id": tc.get("id", "unknown"),
                "passed": False,
                "time_ms": 0.0,
                "actual": None,
                "error": f"import-error: {err.splitlines()[-1] if err.splitlines() else err}",
                "stdout": "",
            }))
        return 0

    if not hasattr(module, function_name):
        for tc in testcases:
            print(json.dumps({
                "test_id": tc.get("id", "unknown"),
                "passed": False,
                "time_ms": 0.0,
                "actual": None,
                "error": f"harness: function '{function_name}' not found",
                "stdout": "",
            }))
        return 0

    func = getattr(module, function_name)

    for tc in testcases:
        test_id = tc.get("id", "unknown")
        test_input = tc.get("input")
        expected = tc.get("expected")
        buf = io.StringIO()
        start = time.perf_counter_ns()
        actual = None
        error = None
        try:
            with contextlib.redirect_stdout(buf):
                actual = _call(func, test_input)
        except Exception:
            tb = traceback.format_exc(limit=5)
            # Keep only exception line + frame context; sanitized further by safe_executor.
            lines = tb.splitlines()
            error = lines[-1] if lines else "exception"
        end = time.perf_counter_ns()
        time_ms = (end - start) / 1e6
        try:
            passed = (error is None) and (actual == expected)
        except Exception:
            passed = False
        out = buf.getvalue()
        if len(out) > 2000:
            out = out[:2000] + "...<truncated>"
        print(json.dumps({
            "test_id": test_id,
            "passed": bool(passed),
            "time_ms": float(time_ms),
            "actual": _to_jsonable(actual),
            "error": error,
            "stdout": out,
        }))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    prob = sys.argv[1] if len(sys.argv) > 1 else "/work/problem.json"
    sol = sys.argv[2] if len(sys.argv) > 2 else "/work/solution.py"
    raise SystemExit(main(prob, sol))
