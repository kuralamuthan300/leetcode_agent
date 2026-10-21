"""Command-line entry point for evaluation dataset preparation."""

from __future__ import annotations

import argparse
import json
import sys

from evals.dataset import dataset_hash, load_dataset, sync_mlflow_dataset, write_problems_jsonl
from evals.grade import grade_batch
from evals.pipeline import report_batch
from evals.runner import run_batch


def _prepare(args: argparse.Namespace) -> int:
    try:
        problems = load_dataset(args.dataset)
        digest = dataset_hash(args.dataset)
        output = write_problems_jsonl(problems, args.output)
    except (OSError, ValueError) as exc:
        print(f"prepare failed: {exc}", file=sys.stderr)
        return 2

    result = {"problems": len(problems), "dataset_hash": digest, "problems_file": str(output)}
    if args.sync_mlflow:
        result.update(sync_mlflow_dataset(problems, digest, reset=args.reset))
    print(json.dumps(result, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare LeetCode evaluation datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="validate and normalize the golden CSV")
    prepare.add_argument("--dataset", default="evals/golden_dataset.csv")
    prepare.add_argument("--output", default="evals/problems.jsonl")
    prepare.add_argument("--sync-mlflow", action="store_true")
    prepare.add_argument("--reset", action="store_true", help="recreate the MLflow dataset")
    run = subparsers.add_parser("run", help="run the agent over the golden dataset")
    run.add_argument("--dataset", default="evals/golden_dataset.csv")
    run.add_argument("--results-root", default="evals/results")
    run.add_argument("--subset", default=None, help="comma-separated problem IDs or 'full'")
    run.add_argument("--repeat", type=int, default=1)
    run.add_argument("--no-mlflow", action="store_true")
    grade = subparsers.add_parser("grade", help="grade a completed batch against the golden cases")
    grade.add_argument("--batch", required=True)
    grade.add_argument("--dataset", default="evals/golden_dataset.csv")
    grade.add_argument("--constraints", default="evals/constraints.yml")
    report = subparsers.add_parser("report", help="write local scorecard and optional MLflow run")
    report.add_argument("--batch", required=True)
    report.add_argument("--no-mlflow", action="store_true")
    report.add_argument("--baseline", default=None, help="baseline MLflow run ID")
    args = parser.parse_args(argv)

    if getattr(args, "reset", False) and not args.sync_mlflow:
        parser.error("--reset requires --sync-mlflow")
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "grade":
        try:
            grade_path = grade_batch(
                args.batch,
                dataset_path=args.dataset,
                constraints_path=args.constraints,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"grade failed: {exc}", file=sys.stderr)
            return 2
        print(grade_path)
        return 0
    if args.command == "run":
        try:
            batch_dir = run_batch(
                dataset_path=args.dataset,
                subset=args.subset,
                repeat=args.repeat,
                results_root=args.results_root,
                no_mlflow=args.no_mlflow,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"run failed: {exc}", file=sys.stderr)
            return 2
        print(batch_dir)
        return 0
    if args.command == "report":
        try:
            result = report_batch(
                args.batch,
                no_mlflow=args.no_mlflow,
                baseline_run_id=args.baseline,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"report failed: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, default=str))
        return 1 if result["gate_status"] == "fail" else 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
