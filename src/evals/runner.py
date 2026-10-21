"""Sequential batch execution for the evaluation dataset."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agent.config import get_config
from evals.dataset import GoldenProblem, dataset_hash, load_dataset, sync_mlflow_dataset

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _trace_id(run_id: str, experiment_name: str) -> str | None:
    try:
        import mlflow

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None
        traces = mlflow.search_traces(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.run_id = '{run_id}'",
            max_results=10,
        )
        if traces is None or traces.empty:
            return None
        return str(traces.iloc[0]["trace_id"])
    except Exception as exc:  # noqa: BLE001 - tracing lookup must not fail a batch
        logger.warning("could not resolve trace for %s: %s", run_id, exc)
        return None


def _select_problems(
    problems: list[GoldenProblem],
    subset: str | None,
) -> list[GoldenProblem]:
    if not subset or subset == "full":
        return problems
    requested = {part.strip() for part in subset.split(",") if part.strip()}
    if not requested:
        raise ValueError("subset must contain problem IDs")
    by_id = {problem.id: problem for problem in problems}
    missing = requested - by_id.keys()
    if missing:
        raise ValueError(f"unknown problem ID(s) in subset: {', '.join(sorted(missing))}")
    return [problem for problem in problems if problem.id in requested]


def run_batch(
    *,
    dataset_path: str | Path = "evals/golden_dataset.csv",
    subset: str | None = None,
    repeat: int = 1,
    results_root: str | Path = "evals/results",
    no_mlflow: bool = False,
) -> Path:
    """Run selected dataset records sequentially and persist manifest/index files."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    problems = _select_problems(load_dataset(dataset_path), subset)
    digest = dataset_hash(dataset_path)
    batch_id = f"eval_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    batch_dir = Path(results_root) / batch_id
    batch_dir.mkdir(parents=True, exist_ok=False)

    config = get_config()
    dataset_ref = (
        sync_mlflow_dataset(
            problems,
            digest,
            dataset_name=config.evals.dataset_name,
            experiment_name=config.evals.experiment,
        )
        if not no_mlflow
        else {"dataset_id": None, "dataset_version": None, "dataset_hash": digest}
    )
    config_path = Path(os.getenv("MLFLOW_AGENT_CONFIG", _REPO_ROOT / "config.yml"))
    manifest: dict[str, Any] = {
        "batch_id": batch_id,
        **dataset_ref,
        "git_sha": _git_sha(),
        "config_hash": _sha256_file(config_path),
        "models": {role: setting.name for role, setting in config.models.items()},
        "repeat": repeat,
        "subset": subset or "full",
        "created_at": datetime.now(UTC).isoformat(),
        "baseline_run_id": config.evals.baseline_run_id or None,
        "no_mlflow": no_mlflow,
        "command": [
            sys.executable,
            "-m",
            "evals.cli",
            "run",
            "--repeat",
            str(repeat),
            *(["--subset", subset] if subset else []),
            *(["--no-mlflow"] if no_mlflow else []),
        ],
    }
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    run_index: dict[str, list[dict[str, Any]]] = {}
    with tempfile.TemporaryDirectory(prefix="eval-problems-") as temp_dir:
        for repeat_index in range(1, repeat + 1):
            run_rows = [
                {"id": f"{problem.id}__r{repeat_index}", "requirement": problem.requirement}
                for problem in problems
            ]
            input_path = Path(temp_dir) / f"problems-r{repeat_index}.jsonl"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in run_rows),
                encoding="utf-8",
            )
            output_root = batch_dir / "agent_runs" / f"repeat_{repeat_index}"
            command = [
                sys.executable,
                "-m",
                "cli",
                "--problems-file",
                str(input_path),
                "--out-dir",
                str(output_root),
                "--quiet",
            ]
            env = os.environ.copy()
            src_path = str(_REPO_ROOT / "src")
            env["PYTHONPATH"] = os.pathsep.join(
                [src_path, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [src_path]
            )
            if no_mlflow:
                env["MLFLOW_TRACING_DISABLED"] = "1"
            started = datetime.now(UTC)
            completed = subprocess.run(
                command,
                cwd=_REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            ended = datetime.now(UTC)
            if completed.returncode:
                raise RuntimeError(
                    f"agent batch failed ({completed.returncode}): "
                    f"{completed.stderr[-2000:] or completed.stdout[-2000:]}"
                )

            summaries = sorted(output_root.glob("batch_*/summary.json"))
            if not summaries:
                raise RuntimeError(f"agent batch did not create summary.json under {output_root}")
            agent_batch_dir = summaries[-1].parent
            batch_summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
            run_details = {
                str(item.get("run_id")): item for item in batch_summary.get("runs", [])
            }
            for problem in problems:
                run_id = f"{problem.id}__r{repeat_index}"
                details = run_details.get(run_id, {})
                output_path = Path(
                    details.get("output_path") or agent_batch_dir / f"{run_id}.json"
                )
                record: dict[str, Any] = {
                    "problem_id": problem.id,
                    "repeat": repeat_index,
                    "output_path": str(output_path),
                    "started_at": details.get("started_at") or started.isoformat(),
                    "ended_at": details.get("ended_at") or ended.isoformat(),
                    "duration_s": details.get("duration_s"),
                    "status": details.get("status", "missing"),
                    "trace_id": None,
                }
                if output_path.is_file():
                    try:
                        output = json.loads(output_path.read_text(encoding="utf-8"))
                        record["status"] = output.get("status", record["status"])
                    except (OSError, json.JSONDecodeError) as exc:
                        record["status"] = "infra_error"
                        record["read_error"] = str(exc)
                if not no_mlflow:
                    record["trace_id"] = _trace_id(run_id, config.gateway.experiment)
                run_index.setdefault(problem.id, []).append(record)

    (batch_dir / "run_index.json").write_text(json.dumps(run_index, indent=2), encoding="utf-8")
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest["run_index"] = "run_index.json"
    manifest["problem_count"] = len(problems)
    (batch_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return batch_dir
