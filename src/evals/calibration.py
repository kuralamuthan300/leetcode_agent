"""Calibration metrics comparing internal and golden verdicts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def internal_pass(run: Mapping[str, Any]) -> bool:
    status = run.get("status")
    evaluation = run.get("eval") or {}
    passed = run.get("internal_passed", evaluation.get("passed", False))
    return status in {"ok", "assumed"} and passed is True


def confusion_metrics(
    golden_grades: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Compare one run per problem and return rates plus escaped failures."""
    tp = tn = fp = fn = 0
    false_passes: list[dict[str, Any]] = []
    per_problem: dict[str, dict[str, bool]] = {}
    problem_ids = sorted(set(golden_grades) | set(runs))
    for problem_id in problem_ids:
        grade = golden_grades.get(problem_id, {})
        golden_passed = bool(grade.get("strict_pass", False))
        records = sorted(runs.get(problem_id, []), key=lambda item: item.get("repeat", 1))
        run = records[0] if records else {}
        internally_passed = internal_pass(run)
        per_problem[problem_id] = {
            "internal_passed": internally_passed,
            "golden_passed": golden_passed,
        }
        if internally_passed and golden_passed:
            tp += 1
        elif not internally_passed and not golden_passed:
            tn += 1
        elif internally_passed:
            fp += 1
            failing_cases = [
                item.get("case_type")
                for item in grade.get("cases", [])
                if not item.get("passed")
            ]
            false_passes.append(
                {
                    "problem_id": problem_id,
                    "failing_golden_cases": failing_cases,
                    "failing_test_ids": [
                        item.get("test_id")
                        for item in (run.get("execution_results") or [])
                        if not item.get("passed")
                    ],
                }
            )
        else:
            fn += 1

    total = tp + tn + fp + fn
    internal_passes = tp + fp
    internal_fails = tn + fn
    agreement = (tp + tn) / total if total else None
    expected_agreement = (
        ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total * total)
        if total
        else None
    )
    kappa = (
        (agreement - expected_agreement) / (1 - expected_agreement)
        if agreement is not None and expected_agreement is not None and expected_agreement != 1
        else (1.0 if agreement == 1 else None)
    )
    return {
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
        },
        "false_pass_rate": fp / internal_passes if internal_passes else None,
        "false_pass_share": fp / total if total else None,
        "false_fail_rate": fn / internal_fails if internal_fails else None,
        "verdict_agreement": agreement,
        "cohens_kappa": kappa,
        "false_passes": false_passes,
        "per_problem": per_problem,
        "denominator": total,
    }
