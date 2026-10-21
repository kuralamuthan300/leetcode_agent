"""Contract tests for ``evals.metrics_spec`` (Eval Framework v1, Step 1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.metrics_spec import (
    CANONICAL_IDS,
    FEEDBACKS,
    METRICS,
    METRICS_BY_MLFLOW_KEY,
    PRIMARY_METRICS,
    SCORERS,
    Aggregation,
    Artifact,
    Direction,
    Scope,
    Tier,
    blocking_metrics,
    feedback_specs,
    get_metric,
    metric_ids_for_row,
    per_problem_metrics,
    primary_metrics,
    run_level_metrics,
    scorer_specs,
)

PLAN = Path(__file__).resolve().parents[1] / "plan" / "eval_plan.md"


def test_ten_primary_metrics_one_per_scorecard_row():
    assert len(PRIMARY_METRICS) == 10
    assert len(CANONICAL_IDS) == 10
    assert CANONICAL_IDS == tuple(spec.id for spec in PRIMARY_METRICS)
    assert [spec.scorecard_row for spec in PRIMARY_METRICS] == list(range(1, 11))
    assert primary_metrics() is PRIMARY_METRICS


def test_registry_keys_are_stable_unique_ids():
    assert METRICS
    for metric_id, spec in METRICS.items():
        assert metric_id == spec.id
        assert spec.id.startswith(
            (
                "golden_",
                "false_",
                "pass_",
                "category_",
                "mutation_",
                "oracle_",
                "blame_",
                "constraint_",
                "complexity_",
                "infra_",
                "flaky_",
                "latency_",
                "cost_",
                "verdict_",
                "cohens_",
                "internal_",
            )
        )


def test_mlflow_keys_are_unique():
    keys = [spec.mlflow_metric_key for spec in METRICS.values()]
    assert len(keys) == len(set(keys))
    assert set(keys) == set(METRICS_BY_MLFLOW_KEY)
    for spec in METRICS.values():
        assert METRICS_BY_MLFLOW_KEY[spec.mlflow_metric_key] is spec


def test_per_problem_metrics_have_matching_scorer_and_feedback():
    for spec in per_problem_metrics():
        assert spec.scope is Scope.PER_PROBLEM
        assert spec.mlflow_scorer_name
        assert spec.mlflow_feedback_name
        assert spec.aggregation in {Aggregation.MEAN, Aggregation.MIN, Aggregation.P90}
        assert SCORERS[spec.mlflow_scorer_name] is spec
        assert FEEDBACKS[spec.mlflow_feedback_name] is spec
        assert scorer_specs()[spec.mlflow_scorer_name] is spec
        assert feedback_specs()[spec.mlflow_feedback_name] is spec


def test_run_level_metrics_have_no_scorer_or_feedback():
    for spec in run_level_metrics():
        assert spec.scope is Scope.RUN
        assert spec.mlflow_scorer_name is None
        assert spec.mlflow_feedback_name is None
        assert spec.aggregation is Aggregation.NONE
        assert spec.scorer_aggregation == "none"


def test_strict_solve_contract_matches_mlflow_mapping():
    spec = get_metric("golden_strict_solve_rate")
    assert spec.tier is Tier.OUTCOME
    assert spec.direction is Direction.HIGHER
    assert spec.gate_value == pytest.approx(0.70)
    assert spec.blocking is True
    assert spec.baseline_margin == pytest.approx(0.05)
    assert spec.scope is Scope.PER_PROBLEM
    assert spec.aggregation is Aggregation.MEAN
    assert spec.mlflow_metric_key == "golden_strict_solve_rate"
    assert spec.mlflow_scorer_name == "golden_strict_pass"
    assert spec.mlflow_feedback_name == "golden_strict_pass"
    assert spec.gate_operator == ">="


def test_initial_gate_values_match_the_plan_table():
    expected = {
        "golden_strict_solve_rate": (0.70, True, Direction.HIGHER),
        "golden_adversarial_case_pass_rate": (0.60, True, Direction.HIGHER),
        "false_pass_rate": (0.05, True, Direction.LOWER),
        "false_fail_rate": (0.10, True, Direction.LOWER),
        "cohens_kappa": (0.70, True, Direction.HIGHER),
        "mutation_score": (0.60, True, Direction.HIGHER),
        "oracle_correctness": (0.95, True, Direction.HIGHER),
        "blame_accuracy": (0.80, True, Direction.HIGHER),
        "constraint_adherence": (0.90, False, Direction.HIGHER),
        "infra_error_rate": (0.02, True, Direction.LOWER),
        "flaky_rate": (0.05, True, Direction.LOWER),
        "latency_p95": (360.0, False, Direction.LOWER),
    }
    for metric_id, (value, blocking, direction) in expected.items():
        spec = get_metric(metric_id)
        assert spec.gate_value == pytest.approx(value), metric_id
        assert spec.blocking is blocking, metric_id
        assert spec.direction is direction, metric_id
        assert spec.gate_operator == (">=" if direction is Direction.HIGHER else "<=")


def test_report_only_metrics_have_no_gate():
    for metric_id in (
        "golden_case_pass_rate",
        "golden_base_case_pass_rate",
        "golden_edge_case_pass_rate",
        "false_pass_share",
        "verdict_agreement",
        "internal_pass_rate",
        "mutation_score_macro",
        "oracle_correctness_agent_tests",
        "oracle_correctness_holdout_tests",
        "pass_at_1",
        "pass_at_2",
        "category_balanced_solve_rate",
        "complexity_match_rate",
        "latency_p50",
        "cost_per_solved",
    ):
        assert get_metric(metric_id).gate_value is None, metric_id


def test_blocking_metrics_include_secondary_hard_gates():
    ids = {spec.id for spec in blocking_metrics()}
    assert {
        "golden_strict_solve_rate",
        "golden_adversarial_case_pass_rate",
        "false_pass_rate",
        "false_fail_rate",
        "cohens_kappa",
        "mutation_score",
        "oracle_correctness",
        "blame_accuracy",
        "infra_error_rate",
        "flaky_rate",
    } <= ids
    assert "constraint_adherence" not in ids
    assert "latency_p95" not in ids
    assert all(spec.blocking and spec.gate_value is not None for spec in blocking_metrics())


def test_every_metric_has_explicit_denominator_and_artifacts():
    for spec in METRICS.values():
        assert spec.numerator.strip()
        assert spec.denominator.strip()
        assert spec.required_artifacts
        assert all(isinstance(artifact, Artifact) for artifact in spec.required_artifacts)
        assert spec.unit in {"ratio", "score", "seconds", "usd"}


def test_rows_group_primary_and_supporting_metrics():
    row_2 = metric_ids_for_row(2)
    assert [spec.id for spec in row_2] == [
        "golden_case_pass_rate",
        "golden_base_case_pass_rate",
        "golden_edge_case_pass_rate",
        "golden_adversarial_case_pass_rate",
    ]
    assert sum(spec.primary for spec in row_2) == 1
    assert sum(spec.primary for spec in metric_ids_for_row(10)) == 1
    with pytest.raises(ValueError):
        metric_ids_for_row(0)


def test_primary_metrics_are_documented_in_the_plan():
    text = PLAN.read_text(encoding="utf-8")
    for spec in PRIMARY_METRICS:
        assert f"`{spec.id}`" in text, f"{spec.id} is missing from plan/eval_plan.md"
    assert "src/evals/metrics_spec.py" in text


def test_spec_serializes_to_json_primitives():
    import json

    for spec in METRICS.values():
        payload = spec.to_dict()
        assert json.loads(json.dumps(payload)) == payload
        assert payload["scope"] in {"per_problem", "run"}
        assert payload["tier"] in {"outcome", "calibration", "tests", "ops"}


def test_unknown_metric_raises_helpful_error():
    with pytest.raises(KeyError, match="unknown eval metric"):
        get_metric("golden_vibes")
