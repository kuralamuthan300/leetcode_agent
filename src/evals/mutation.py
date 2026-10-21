"""Deterministic mutation score computation and kill matrices."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from evals.adapters import outputs_equal, parse_output
from evals.harness import RESULT_MARKER
from evals.mutants import Mutant

RunHarness = Callable[[str, str], Mapping[str, Any]]


def _run_case(source: str, harness: str, run_harness: RunHarness) -> tuple[Any, str]:
    execution = run_harness(source, harness)
    error = str(execution.get("error") or "")
    if error:
        return None, error
    stdout = str(execution.get("stdout") or "")
    try:
        value = stdout.rsplit(RESULT_MARKER, 1)[1].strip() if RESULT_MARKER in stdout else stdout.strip()
        return parse_output(value), ""
    except (IndexError, ValueError, SyntaxError) as exc:
        return None, str(exc)


def score_mutants(
    reference_source: str,
    mutants: Sequence[Mutant],
    golden_cases: Sequence[Mapping[str, Any]],
    oracle_valid_tests: Sequence[Mapping[str, Any]],
    *,
    run_harness: RunHarness,
    limit: int = 5,
) -> dict[str, Any]:
    """Filter non-killable mutants, then score oracle-valid tests against them."""
    selected = list(mutants[:limit])
    killable: list[Mutant] = []
    for mutant in selected:
        for case in golden_cases:
            reference_actual, reference_error = _run_case(
                reference_source, str(case["harness"]), run_harness
            )
            mutant_actual, mutant_error = _run_case(mutant.source, str(case["harness"]), run_harness)
            if not reference_error and (mutant_error or not outputs_equal(
                reference_actual, mutant_actual
            )):
                killable.append(mutant)
                break

    matrix: list[dict[str, Any]] = []
    killed = 0
    for mutant in killable:
        killed_by: list[str] = []
        for test in oracle_valid_tests:
            actual, error = _run_case(mutant.source, str(test["harness"]), run_harness)
            if error or not outputs_equal(actual, test.get("expected")):
                killed_by.append(str(test.get("id", "")))
            matrix.append(
                {
                    "mutant_id": mutant.mutant_id,
                    "test_id": str(test.get("id", "")),
                    "killed": bool(error or not outputs_equal(actual, test.get("expected"))),
                    "error": error,
                }
            )
        if killed_by:
            killed += 1

    survivors = [
        mutant.mutant_id
        for mutant in killable
        if not any(
            row["mutant_id"] == mutant.mutant_id and row["killed"]
            for row in matrix
        )
    ]
    return {
        "killed": killed,
        "killable": len(killable),
        "mutation_score": killed / len(killable) if killable else None,
        "mutation_score_macro": killed / len(killable) if killable else None,
        "matrix": matrix,
        "surviving_mutants": survivors,
        "mutants": [
            {"mutant_id": mutant.mutant_id, "mutation": mutant.mutation}
            for mutant in selected
        ],
    }
