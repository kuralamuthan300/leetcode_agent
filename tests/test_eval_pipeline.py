from __future__ import annotations

import json

import pytest

from evals.metrics_spec import per_problem_metrics
from evals.pipeline import report_batch
from evals.scorers import build_scorers


def test_local_report_pipeline_writes_scorecard_without_mlflow(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"batch_id": "fixture"}))
    (tmp_path / "run_index.json").write_text("{}")
    (tmp_path / "grade.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "golden_strict_solve_rate": {
                        "value": 0.8,
                        "numerator": 4,
                        "denominator": 5,
                    }
                },
                "grades": {},
                "per_problem": [{"problem_id": "1", "strict_pass": True}],
            }
        )
    )

    result = report_batch(tmp_path, no_mlflow=True)

    assert result["gate_status"] == "unavailable"
    assert (tmp_path / "scorecard.md").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "per_problem.csv").exists()
    assert (tmp_path / "failures.md").exists()


def test_report_pipeline_requires_completed_grade_stage(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the grade stage"):
        report_batch(tmp_path, no_mlflow=True)


def test_report_pipeline_degrades_to_local_artifacts_if_tracking_is_unavailable(
    tmp_path,
    monkeypatch,
):
    from evals import pipeline

    (tmp_path / "grade.json").write_text(
        json.dumps({"metrics": {}, "grades": {}, "per_problem": []})
    )
    monkeypatch.setattr(
        pipeline,
        "start_eval_run",
        lambda *args: (_ for _ in ()).throw(ConnectionError("tracking server offline")),
    )

    result = report_batch(tmp_path)

    assert result["mlflow_status"] == "unavailable"
    assert "tracking server offline" in result["mlflow_error"]
    assert (tmp_path / "scorecard.md").exists()


def test_mlflow_scorer_names_match_per_problem_metric_contract():
    names = {scorer.name for scorer in build_scorers()}
    expected = {metric.mlflow_scorer_name for metric in per_problem_metrics()}
    assert names == expected
