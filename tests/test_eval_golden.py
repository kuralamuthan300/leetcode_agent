from __future__ import annotations

from evals.dataset import GoldenProblem
from evals.golden import grade_code


def _problem() -> GoldenProblem:
    return GoldenProblem(
        id="1",
        statement="Implement add(a, b)",
        constraints="integers",
        complexity="O(1)",
        function_name="add",
        cases={
            "base_case": {"input": [2, 3], "expected": 5},
            "edge_case": {"input": [1, 1], "expected": 2},
            "adversarial_case": {"input": [-100, 2], "expected": -98},
        },
    )


def test_golden_grader_passes_all_cases_for_correct_solution():
    grade = grade_code(_problem(), "def add(a, b):\n    return a + b")
    assert grade["strict_pass"] is True
    assert [case["passed"] for case in grade["cases"]] == [True, True, True]


def test_golden_grader_fails_buggy_solution():
    grade = grade_code(_problem(), "def add(a, b):\n    return a - b")
    assert grade["strict_pass"] is False
    assert all(not case["passed"] for case in grade["cases"])


def test_golden_grader_reports_guardrail_block_as_blocked():
    grade = grade_code(_problem(), "import os\ndef add(a, b):\n    return a + b")
    assert all(case["guardrail_blocked"] for case in grade["cases"])
    assert all(case["passed"] is False for case in grade["cases"])


def test_golden_grader_reports_timeout_as_failed_case(monkeypatch):
    from evals import golden

    monkeypatch.setattr(
        golden,
        "run_python",
        lambda *args, **kwargs: {
            "stdout": "",
            "stderr": "",
            "error": "timeout after 30s",
        },
    )
    grade = grade_code(_problem(), "def add(a, b): return a + b")
    assert not grade["strict_pass"]
    assert all(case["timed_out"] and not case["passed"] for case in grade["cases"])
