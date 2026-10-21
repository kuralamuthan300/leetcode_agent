from __future__ import annotations

import pytest

from evals.calibration import confusion_metrics
from evals.dataset import GoldenProblem
from evals.metrics import compute_outcome_metrics, wilson_interval


def _problem(problem_id: str, category: str = "array") -> GoldenProblem:
    return GoldenProblem(
        id=problem_id,
        statement="Implement solve(x)",
        constraints="x is an integer",
        complexity="O(1)",
        function_name="solve",
        cases={},
        category=category,
    )


def test_outcome_rates_and_case_breakdown_include_adversarial_failures():
    problems = [_problem("1"), _problem("2")]
    grades = {
        "1": {
            "strict_pass": False,
            "cases": [
                {"case_type": "base_case", "passed": True},
                {"case_type": "edge_case", "passed": True},
                {"case_type": "adversarial_case", "passed": False},
            ],
        },
        "2": {
            "strict_pass": True,
            "cases": [
                {"case_type": "base_case", "passed": True},
                {"case_type": "edge_case", "passed": True},
                {"case_type": "adversarial_case", "passed": True},
            ],
        },
    }
    metrics = compute_outcome_metrics(problems, grades)
    assert metrics["golden_strict_solve_rate"]["value"] == 0.5
    assert metrics["golden_case_pass_rate"]["value"] == 5 / 6
    assert metrics["golden_adversarial_case_pass_rate"]["numerator"] == 1
    assert metrics["golden_adversarial_case_pass_rate"]["denominator"] == 2
    assert metrics["golden_strict_solve_rate"]["ci95"] == wilson_interval(1, 2)


def test_repeat_aware_pass_at_metrics_and_category_balanced_macro():
    problems = [_problem("1", "A"), _problem("2", "A"), _problem("3", "B")]
    grades = {problem.id: {"strict_pass": False, "cases": []} for problem in problems}
    runs = {
        "1": [{"repeat": 1, "internal_passed": True, "iterations": 1}],
        "2": [{"repeat": 1, "internal_passed": True, "iterations": 2}],
        "3": [{"repeat": 1, "internal_passed": False, "iterations": 2}],
    }
    metrics = compute_outcome_metrics(problems, grades, runs)
    assert metrics["pass_at_1"]["value"] == 1 / 3
    assert metrics["pass_at_2"]["value"] == 2 / 3
    assert metrics["category_balanced_solve_rate"]["value"] == 0


def test_confusion_rates_kappa_and_escaped_problem_ids():
    grades = {
        "false-pass-1": {
            "strict_pass": False,
            "cases": [{"case_type": "edge_case", "passed": False}],
        },
        "false-pass-2": {"strict_pass": False, "cases": []},
        "false-fail": {"strict_pass": True, "cases": []},
        "true-pass": {"strict_pass": True, "cases": []},
        "true-negative": {"strict_pass": False, "cases": []},
    }
    runs = {
        problem_id: [
            {
                "status": "ok" if passed else "aborted",
                "eval": {"passed": passed},
                "execution_results": [{"test_id": "t1", "passed": False}],
            }
        ]
        for problem_id, passed in {
            "false-pass-1": True,
            "false-pass-2": True,
            "false-fail": False,
            "true-pass": True,
            "true-negative": False,
        }.items()
    }
    result = confusion_metrics(grades, runs)
    assert result["confusion_matrix"] == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 2,
        "false_negative": 1,
    }
    assert result["false_pass_rate"] == 2 / 3
    assert result["false_pass_share"] == 2 / 5
    assert result["false_fail_rate"] == 1 / 2
    assert result["cohens_kappa"] == pytest.approx(-2 / 13)
    assert {item["problem_id"] for item in result["false_passes"]} == {
        "false-pass-1",
        "false-pass-2",
    }
    assert result["false_passes"][0]["failing_test_ids"] == ["t1"]
