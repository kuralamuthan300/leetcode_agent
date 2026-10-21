"""MLflow GenAI scorers over precomputed evaluation outputs."""

from __future__ import annotations

from typing import Any


def build_scorers() -> list[Any]:
    """Create scorer objects lazily so local evaluation never requires MLflow."""
    from mlflow.genai.scorers import scorer

    @scorer(name="golden_strict_pass", aggregations=["mean"])
    def golden_strict_pass(inputs, outputs, expectations):
        return bool((outputs.get("grade") or {}).get("strict_pass", False))

    @scorer(name="golden_case_pass_rate", aggregations=["mean"])
    def golden_case_pass_rate(inputs, outputs, expectations):
        cases = (outputs.get("grade") or {}).get("cases", [])
        return (
            sum(bool(case.get("passed")) for case in cases) / len(cases)
            if cases
            else 0.0
        )

    @scorer(name="false_pass", aggregations=["mean"])
    def false_pass(inputs, outputs, expectations):
        return bool(
            outputs.get("internal_passed")
            and not (outputs.get("grade") or {}).get("strict_pass", False)
        )

    @scorer(name="internal_pass", aggregations=["mean"])
    def internal_pass(inputs, outputs, expectations):
        return bool(outputs.get("internal_passed"))

    @scorer(name="false_fail", aggregations=["mean"])
    def false_fail(inputs, outputs, expectations):
        return bool(
            not outputs.get("internal_passed")
            and (outputs.get("grade") or {}).get("strict_pass", False)
        )

    @scorer(name="mutation_score", aggregations=["mean"])
    def mutation_score(inputs, outputs, expectations):
        return float(outputs.get("mutation_score") or 0.0)

    @scorer(name="oracle_correctness", aggregations=["mean"])
    def oracle_correctness(inputs, outputs, expectations):
        value = outputs.get("oracle_correctness")
        return float(value) if value is not None else 0.0

    @scorer(name="blame_accuracy", aggregations=["mean"])
    def blame_accuracy(inputs, outputs, expectations):
        value = outputs.get("blame_accuracy")
        return float(value) if value is not None else 0.0

    @scorer(name="constraint_adherence", aggregations=["mean"])
    def constraint_adherence(inputs, outputs, expectations):
        value = outputs.get("constraint_adherence")
        return float(value) if value is not None else 0.0

    return [
        golden_strict_pass,
        golden_case_pass_rate,
        internal_pass,
        false_pass,
        false_fail,
        mutation_score,
        oracle_correctness,
        blame_accuracy,
        constraint_adherence,
    ]
