"""The isolated MLflow adapter for eval datasets, runs, feedback, and artifacts."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _mlflow():
    import mlflow

    return mlflow


def init(experiment: str = "leetcodeagent-evals") -> None:
    mlflow = _mlflow()
    from agent.config import get_config

    mlflow.set_tracking_uri(get_config().gateway.tracking_uri)
    mlflow.set_experiment(experiment)


def sync_dataset(
    records: list[dict[str, Any]],
    digest: str,
    *,
    reset: bool = False,
    dataset_name: str = "leetcode-golden-v1",
    experiment_name: str = "leetcodeagent-evals",
) -> dict[str, Any]:
    mlflow = _mlflow()
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(experiment_name)
    )
    api = mlflow.genai.datasets
    existing = [
        dataset
        for dataset in api.search_datasets(
            experiment_ids=experiment_id,
            max_results=100,
        )
        if dataset.name == dataset_name
    ]
    dataset = existing[0] if existing else None
    if reset and dataset is not None:
        api.delete_dataset(dataset_id=dataset.dataset_id)
        dataset = None
    if dataset is None:
        dataset = api.create_dataset(name=dataset_name, experiment_id=experiment_id)

    tags = dataset.tags or {}
    stored_hash = tags.get("golden_dataset_hash")
    try:
        stored_version = int(tags.get("golden_dataset_version", "0"))
    except (TypeError, ValueError):
        stored_version = 0
    if stored_hash != digest:
        dataset = dataset.merge_records(records)
        dataset_version = getattr(dataset.version, "version", dataset.version)
        version = int(dataset_version) if dataset_version is not None else stored_version + 1
        api.set_dataset_tags(
            dataset.dataset_id,
            {"golden_dataset_hash": digest, "golden_dataset_version": str(version)},
        )
    else:
        dataset_version = getattr(dataset.version, "version", dataset.version)
        version = int(dataset_version) if dataset_version is not None else stored_version
    try:
        record_count = len(dataset.to_df())
    except (AttributeError, TypeError):
        record_count = None
    if record_count is not None and record_count != len(records):
        raise RuntimeError(
            f"MLflow dataset {dataset_name!r} has {record_count} records; "
            f"expected {len(records)}"
        )
    return {
        "dataset_id": dataset.dataset_id,
        "dataset_version": version,
        "dataset_hash": digest,
    }


def start_eval_run(batch_id: str, manifest: dict[str, Any]):
    mlflow = _mlflow()
    from agent.config import get_config

    experiment_name = get_config().evals.experiment
    init(experiment_name)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError("MLflow evaluation experiment could not be initialized")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.eval.batch_id = '{batch_id}'",
        max_results=10,
        order_by=["attributes.start_time DESC"],
    )
    if runs:
        active = mlflow.start_run(run_id=runs[0].info.run_id)
    else:
        active = mlflow.start_run(run_name=f"eval_{batch_id}")
    tags = {
        "eval.kind": "scorecard",
        "eval.batch_id": batch_id,
        "eval.dataset_id": str(manifest.get("dataset_id") or ""),
        "eval.dataset_version": str(manifest.get("dataset_version") or ""),
        "eval.dataset_hash": str(manifest.get("dataset_hash") or ""),
        "eval.git_sha": str(manifest.get("git_sha") or ""),
        "eval.config_hash": str(manifest.get("config_hash") or ""),
        "eval.models": json.dumps(manifest.get("models") or {}, sort_keys=True),
        "eval.repeat": str(manifest.get("repeat") or 1),
        "eval.subset": str(manifest.get("subset") or "full"),
        "eval.baseline_run_id": str(manifest.get("baseline_run_id") or ""),
    }
    mlflow.set_tags({key: value for key, value in tags.items() if value})
    params = {
        "python_version": manifest.get("python_version", ""),
        "command": json.dumps(manifest.get("command") or []),
        "pricing_hash": manifest.get("pricing_hash", ""),
    }
    mlflow.log_params({key: str(value) for key, value in params.items() if value})
    return active


def log_eval(data, scorers):
    return _mlflow().genai.evaluate(data=data, scorers=scorers)


def result_tables(result: Any) -> dict[str, Any]:
    tables = getattr(result, "tables", None) or {}
    return dict(tables) if isinstance(tables, dict) else {"eval_result.json": tables}


def compare_baseline(
    baseline_run_id: str,
    metrics: dict[str, Any],
    per_problem: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare aggregate metrics and paired strict verdicts to a prior run."""
    import csv

    mlflow = _mlflow()
    client = mlflow.MlflowClient()
    baseline_run = client.get_run(baseline_run_id)
    baseline_metrics = baseline_run.data.metrics
    deltas: dict[str, float] = {}
    for metric_id, value in metrics.items():
        current = value.get("value") if isinstance(value, dict) else value
        baseline = baseline_metrics.get(metric_id)
        if isinstance(current, (int, float)) and isinstance(baseline, (int, float)):
            deltas[f"delta.{metric_id}"] = float(current) - float(baseline)

    baseline_problems: dict[str, bool] = {}
    try:
        artifact_path = client.download_artifacts(baseline_run_id, "per_problem.csv")
        with Path(artifact_path).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                baseline_problems[str(row["problem_id"])] = row.get("strict_pass", "").lower() == "true"
    except (OSError, KeyError, FileNotFoundError):
        logger.info("baseline run %s has no readable per_problem.csv artifact", baseline_run_id)

    current_problems = {
        str(row.get("problem_id")): bool(row.get("strict_pass"))
        for row in per_problem
        if row.get("problem_id") is not None
    }
    paired = baseline_problems.keys() & current_problems.keys()
    regressions = [
        problem_id
        for problem_id in sorted(paired)
        if baseline_problems[problem_id] and not current_problems[problem_id]
    ]
    improvements = [
        problem_id
        for problem_id in sorted(paired)
        if not baseline_problems[problem_id] and current_problems[problem_id]
    ]
    if deltas:
        mlflow.log_metrics(deltas)
    return {
        "baseline_run_id": baseline_run_id,
        "deltas": deltas,
        "paired_problems": len(paired),
        "regressions": regressions,
        "improvements": improvements,
    }


def log_trace_feedback(run_index: dict[str, Any], grades: dict[str, Any]) -> int:
    mlflow = _mlflow()
    count = 0
    for problem_id, entries in run_index.items():
        grade = grades.get(problem_id, {})
        for entry in entries:
            trace_id = entry.get("trace_id")
            if not trace_id:
                continue
            internal = entry.get("internal_passed", grade.get("internal_passed"))
            assessments = [
                (
                    "golden_strict_pass",
                    bool(grade.get("strict_pass")),
                    "All three golden cases passed."
                    if grade.get("strict_pass")
                    else "At least one golden case failed.",
                ),
            ]
            if internal is not None:
                assessments.extend(
                    [
                        (
                            "internal_pass",
                            bool(internal),
                            "Internal agent evaluation verdict.",
                        ),
                        (
                            "false_pass",
                            bool(internal and not grade.get("strict_pass")),
                            "Internal and golden verdict comparison.",
                        ),
                        (
                            "false_fail",
                            bool(not internal and grade.get("strict_pass")),
                            "Internal and golden verdict comparison.",
                        ),
                    ]
                )
            for key in (
                "oracle_correctness",
                "blame_accuracy",
                "mutation_score",
                "constraint_adherence",
            ):
                if grade.get(key) is not None:
                    assessments.append((key, grade[key], f"Per-problem {key.replace('_', ' ')}."))
            for name, value, rationale in assessments:
                mlflow.log_feedback(
                    trace_id=str(trace_id),
                    name=name,
                    value=value,
                    rationale=rationale,
                )
                count += 1
    return count


def log_scorecard(
    metrics: dict[str, Any],
    tables: dict[str, Any],
    artifact_paths: list[str | Path],
) -> None:
    mlflow = _mlflow()
    scalar_metrics = {}
    for name, payload in metrics.items():
        value = payload.get("value") if isinstance(payload, dict) else payload
        if isinstance(value, (int, float)) and math.isfinite(value):
            scalar_metrics[name] = float(value)
    if scalar_metrics:
        mlflow.log_metrics(scalar_metrics)
    for name, table in tables.items():
        mlflow.log_table(table, artifact_file=name)
    for path in artifact_paths:
        if Path(path).is_file():
            mlflow.log_artifact(str(path))


def finish(gate_status: str) -> None:
    mlflow = _mlflow()
    mlflow.set_tag("gate.status", gate_status)
    mlflow.end_run()
