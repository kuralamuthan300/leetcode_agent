"""Local grading artifact collation and scorecard reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from evals.mlflow_eval import (
    compare_baseline,
    finish,
    log_eval,
    log_scorecard,
    log_trace_feedback,
    result_tables,
    start_eval_run,
)
from evals.report import write_report
from evals.scorers import build_scorers

logger = logging.getLogger(__name__)


def report_batch(
    batch_dir: str | Path,
    *,
    no_mlflow: bool = False,
    baseline_run_id: str | None = None,
) -> dict[str, Any]:
    directory = Path(batch_dir)
    required = directory / "grade.json"
    if not required.is_file():
        raise FileNotFoundError(
            f"missing {required}; run the grade stage before generating a scorecard"
        )
    grade_doc = json.loads(required.read_text(encoding="utf-8"))
    manifest_path = directory / "manifest.json"
    run_index_path = directory / "run_index.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    if baseline_run_id:
        manifest["baseline_run_id"] = baseline_run_id
    run_index = json.loads(run_index_path.read_text(encoding="utf-8")) if run_index_path.exists() else {}
    metrics = grade_doc.get("metrics") or {}
    per_problem = grade_doc.get("per_problem") or []
    failures = grade_doc.get("failures") or []
    artifacts = {
        name: grade_doc[name]
        for name in ("oracle.json", "mutation.json", "constraints.json", "false_passes.json")
        if name in grade_doc
    }
    report = write_report(
        directory,
        metrics=metrics,
        per_problem=per_problem,
        failures=failures,
        artifacts=artifacts,
    )
    if no_mlflow:
        report["mlflow_status"] = "disabled"
        return report

    active_run = None
    try:
        active_run = start_eval_run(str(manifest.get("batch_id") or directory.name), manifest)
        eval_rows = []
        for row in per_problem:
            grade = grade_doc.get("grades", {}).get(str(row.get("problem_id")), {})
            eval_rows.append(
                {
                    "inputs": {
                        "problem_id": str(row.get("problem_id", "")),
                        "requirement": row.get("requirement", ""),
                    },
                    "outputs": {
                        "grade": grade,
                        "internal_passed": row.get("internal_passed", False),
                        "mutation_score": row.get("mutation_score"),
                        "oracle_correctness": row.get("oracle_correctness"),
                        "blame_accuracy": row.get("blame_accuracy"),
                        "constraint_adherence": row.get("constraint_adherence"),
                    },
                    "expectations": row.get("expectations", {}),
                }
            )
        tables: dict[str, Any] = {}
        tables["per_problem.json"] = per_problem
        mutation = grade_doc.get("mutation.json")
        if isinstance(mutation, dict) and mutation.get("matrix") is not None:
            tables["mutation_matrix.json"] = mutation["matrix"]
        if eval_rows:
            import pandas as pd

            evaluation = log_eval(pd.DataFrame(eval_rows), build_scorers())
            tables.update(result_tables(evaluation))
        baseline_id = manifest.get("baseline_run_id")
        if baseline_id:
            comparison = compare_baseline(str(baseline_id), metrics, per_problem)
            artifacts["regressions.json"] = comparison
            report["baseline"] = comparison
            (directory / "regressions.json").write_text(
                json.dumps(comparison, indent=2),
                encoding="utf-8",
            )
            tables["regressions.json"] = comparison
        from agent.config import get_config

        report["trace_feedback_count"] = (
            log_trace_feedback(run_index, grade_doc.get("grades", {}))
            if get_config().evals.log_feedback
            else 0
        )
        for metric_id, gate in report["gates"].items():
            metrics[f"gate.{metric_id}"] = {
                "value": 1.0 if gate["status"] == "pass" else 0.0
                if gate["status"] == "fail"
                else None
            }
        log_scorecard(
            metrics,
            tables,
            [
                directory / "scorecard.md",
                directory / "metrics.json",
                directory / "per_problem.csv",
                directory / "failures.md",
                *(directory / name for name in artifacts),
                *([directory / "regressions.json"] if baseline_id else []),
            ],
        )
        finish(report["gate_status"])
        report["mlflow_run_id"] = active_run.info.run_id
        report["mlflow_status"] = "available"
    except Exception as exc:  # noqa: BLE001 - MLflow is optional; local artifacts are authoritative
        if active_run is not None:
            try:
                finish("error")
            except Exception as finish_exc:  # noqa: BLE001 - preserve the original adapter failure
                logger.warning("could not close failed MLflow eval run: %s", finish_exc)
        logger.warning("MLflow reporting unavailable; local scorecard was written: %s", exc)
        report["mlflow_status"] = "unavailable"
        report["mlflow_error"] = str(exc)
        return report
    return report
