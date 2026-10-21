"""Generated-test oracle checks and deterministic blame ground truth."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from evals.adapters import outputs_equal, parse_output
from evals.harness import RESULT_MARKER

RunHarness = Callable[[str, str], Mapping[str, Any]]


def check_generated_tests(
    reference_code: str,
    tests: Sequence[Mapping[str, Any]],
    *,
    run_harness: RunHarness,
) -> dict[str, Any]:
    """Run generated tests against trusted reference code and score expected values."""
    results: list[dict[str, Any]] = []
    for test in tests:
        test_id = str(test.get("id", ""))
        expected = test.get("expected")
        harness = str(test.get("harness", ""))
        execution = run_harness(reference_code, harness)
        error = str(execution.get("error") or "")
        actual: Any = None
        stdout = str(execution.get("stdout") or "")
        if not error:
            try:
                if RESULT_MARKER in stdout:
                    actual = parse_output(stdout.rsplit(RESULT_MARKER, 1)[1].strip())
                else:
                    actual = parse_output(stdout.strip())
            except (ValueError, SyntaxError) as exc:
                error = f"could not parse reference output: {exc}"
        oracle_ok = not error and outputs_equal(actual, expected)
        results.append(
            {
                "test_id": test_id,
                "oracle_ok": oracle_ok,
                "actual": actual,
                "expected": expected,
                "error": error,
                "repaired": bool(test.get("repaired", False)),
            }
        )

    valid = sum(result["oracle_ok"] for result in results)
    return {
        "oracle_correctness": valid / len(results) if results else None,
        "oracle_valid": valid,
        "total": len(results),
        "tests": results,
    }


def audit_blame(
    execution_results: Sequence[Mapping[str, Any]],
    oracle_tests: Sequence[Mapping[str, Any]],
    failure_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare evaluator blame with per-test oracle-derived blame."""
    oracle_by_id = {str(test.get("test_id")): test for test in oracle_tests}
    code_correct = code_incorrect = test_correct = test_incorrect = 0
    misblamed: list[dict[str, Any]] = []
    truths: list[str] = []
    for entry in execution_results:
        if entry.get("passed"):
            continue
        test_id = str(entry.get("test_id", ""))
        oracle = oracle_by_id.get(test_id)
        if oracle is None:
            continue
        ground_truth = "test" if entry.get("repaired") or oracle.get("repaired") else (
            "code" if oracle.get("oracle_ok") else "test"
        )
        predicted = entry.get("blame")
        truths.append(ground_truth)
        if ground_truth == "code":
            if predicted == ground_truth:
                code_correct += 1
            else:
                code_incorrect += 1
        elif predicted == ground_truth:
            test_correct += 1
        else:
            test_incorrect += 1
        if predicted != ground_truth:
            misblamed.append(
                {
                    "test_id": test_id,
                    "predicted": predicted,
                    "ground_truth": ground_truth,
                    "oracle_ok": bool(oracle.get("oracle_ok")),
                    "repaired": bool(entry.get("repaired") or oracle.get("repaired")),
                }
            )

    total = code_correct + code_incorrect + test_correct + test_incorrect
    dominant_truth = (
        "code" if truths.count("code") > truths.count("test") else "test"
    ) if truths else None
    predicted_culprit = (failure_report or {}).get("culprit")
    holistic = (
        predicted_culprit == dominant_truth
        if dominant_truth and predicted_culprit in {"code", "test"}
        else None
    )
    return {
        "accuracy": (code_correct + test_correct) / total if total else None,
        "correct": code_correct + test_correct,
        "total": total,
        "confusion_matrix": {
            "code_correct": code_correct,
            "code_mislabeled_test": code_incorrect,
            "test_correct": test_correct,
            "test_mislabeled_code": test_incorrect,
        },
        "misblamed": misblamed,
        "dominant_ground_truth": dominant_truth,
        "failure_report_correct": holistic,
    }


def score_test_oracle(
    reference_code: str,
    test_cases: Sequence[Mapping[str, Any]],
    holdout_tests: Sequence[Mapping[str, Any]],
    *,
    run_harness: RunHarness,
) -> dict[str, Any]:
    agent = check_generated_tests(reference_code, test_cases, run_harness=run_harness)
    holdout = check_generated_tests(reference_code, holdout_tests, run_harness=run_harness)
    total = agent["total"] + holdout["total"]
    valid = agent["oracle_valid"] + holdout["oracle_valid"]
    return {
        "oracle_correctness": valid / total if total else None,
        "oracle_correctness_agent_tests": agent["oracle_correctness"],
        "oracle_correctness_holdout_tests": holdout["oracle_correctness"],
        "agent_tests": agent,
        "holdout_tests": holdout,
    }
