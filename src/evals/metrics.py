"""Outcome scorecard metrics and binomial confidence intervals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from evals.dataset import CASE_NAMES, GoldenProblem


def wilson_interval(successes: int, total: int, *, z: float = 1.96) -> tuple[float, float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * (
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) ** 0.5 / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _metric(successes: int, total: int) -> dict[str, Any]:
    return {
        "value": successes / total if total else None,
        "numerator": successes,
        "denominator": total,
        "ci95": wilson_interval(successes, total),
    }


def compute_outcome_metrics(
    problems: Sequence[GoldenProblem],
    grades: Mapping[str, Mapping[str, Any]],
    runs: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Compute strict/case rates and repeat-aware support metrics."""
    runs = runs or {}
    strict = {
        problem.id: bool(grades.get(problem.id, {}).get("strict_pass", False))
        for problem in problems
    }
    cases_by_type: dict[str, list[bool]] = {name: [] for name in CASE_NAMES}
    for problem in problems:
        case_results = {
            result.get("case_type"): bool(result.get("passed"))
            for result in grades.get(problem.id, {}).get("cases", [])
        }
        for case_type in CASE_NAMES:
            cases_by_type[case_type].append(case_results.get(case_type, False))

    category_results: dict[str, list[bool]] = defaultdict(list)
    for problem in problems:
        category_results[problem.category or "uncategorized"].append(strict[problem.id])
    category_rates = [
        sum(values) / len(values) for values in category_results.values() if values
    ]

    first_passes = 0
    within_two_passes = 0
    for problem in problems:
        problem_runs = sorted(runs.get(problem.id, []), key=lambda item: item.get("repeat", 1))
        run = next((item for item in problem_runs if item.get("repeat", 1) == 1), None)
        if run is None:
            continue
        evaluation = run.get("eval") or {}
        passed = bool(run.get("internal_passed", evaluation.get("passed", False)))
        iterations = int(run.get("iterations", evaluation.get("iterations", 1)) or 1)
        if passed and iterations == 1:
            first_passes += 1
        if passed:
            within_two_passes += 1

    all_case_passes = sum(sum(values) for values in cases_by_type.values())
    all_cases = sum(len(values) for values in cases_by_type.values())
    total = len(problems)
    return {
        "golden_strict_solve_rate": _metric(sum(strict.values()), total),
        "golden_case_pass_rate": _metric(all_case_passes, all_cases),
        "golden_base_case_pass_rate": _metric(
            sum(cases_by_type["base_case"]), len(cases_by_type["base_case"])
        ),
        "golden_edge_case_pass_rate": _metric(
            sum(cases_by_type["edge_case"]), len(cases_by_type["edge_case"])
        ),
        "golden_adversarial_case_pass_rate": _metric(
            sum(cases_by_type["adversarial_case"]), len(cases_by_type["adversarial_case"])
        ),
        "pass_at_1": _metric(first_passes, total),
        "pass_at_2": _metric(within_two_passes, total),
        "category_balanced_solve_rate": {
            "value": sum(category_rates) / len(category_rates) if category_rates else None,
            "numerator": sum(category_rates),
            "denominator": len(category_rates),
            "ci95": None,
        },
        "counts": {
            "problems": total,
            "cases": all_cases,
            "categories": len(category_rates),
        },
    }
