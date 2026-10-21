"""Deterministic grading of solution source against the golden cases."""

from __future__ import annotations

from typing import Any

from agent.tools.sandbox import run_python
from evals.adapters import adapt_input, adapt_output, outputs_equal
from evals.dataset import CASE_NAMES, GoldenProblem
from evals.harness import build_harness, parse_harness_output

GOLDEN_TIMEOUT_S = 30


def grade_code(
    problem: GoldenProblem,
    code: str,
    *,
    timeout_s: int = GOLDEN_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a solution against all golden cases and return a JSON-ready grade."""
    results: list[dict[str, Any]] = []
    for case_type in CASE_NAMES:
        case = problem.cases[case_type]
        harness_case = dict(case) if isinstance(case, dict) else case
        if isinstance(harness_case, dict):
            for input_field in ("input", "args", "kwargs"):
                if input_field in harness_case:
                    harness_case[input_field] = adapt_input(
                        problem.id,
                        harness_case[input_field],
                    )
        harness = build_harness(problem.function_name, harness_case)
        execution = run_python(code, harness, timeout_s, trusted_harness=True)
        error = str(execution.get("error") or "")
        blocked = error.startswith("blocked:")
        timed_out = "timeout after " in error
        actual: Any = None
        if not error:
            try:
                actual = parse_harness_output(str(execution.get("stdout") or ""))
            except ValueError as exc:
                error = str(exc)
        expected = case.get("expected") if isinstance(case, dict) else None
        actual = adapt_output(problem.id, actual)
        passed = not error and outputs_equal(
            actual,
            expected,
            function_name=problem.function_name,
        )
        results.append(
            {
                "case_type": case_type,
                "passed": passed,
                "actual": actual,
                "expected": expected,
                "error": error,
                "timed_out": timed_out,
                "guardrail_blocked": blocked,
            }
        )
    return {
        "problem_id": problem.id,
        "strict_pass": all(result["passed"] for result in results),
        "cases": results,
    }
