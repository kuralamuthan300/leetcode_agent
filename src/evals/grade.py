"""Golden grading stage: score agent outputs and write ``grade.json``.

The batch runner stores one agent output per problem/repeat and an index of
their paths. This stage reads those outputs, grades the final solution against
the three golden cases, and collates the deterministic metrics that
``evals.cli report`` consumes. Reference-based metrics (mutation, test oracle,
blame) stay unavailable when no trusted reference solution exists, rather than
being faked as passing.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.calibration import confusion_metrics
from evals.complexity import estimate_complexity
from evals.constraints import check_constraints, load_constraint_rules
from evals.dataset import CASE_NAMES, GoldenProblem, dataset_hash, load_dataset
from evals.golden import grade_code
from evals.metrics import compute_outcome_metrics, wilson_interval
from evals.ops import compute_ops_metrics

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Run fields copied out of the agent output JSON into the grading record.
_RUN_FIELDS = (
    "status",
    "rejection_class",
    "rejection_reason",
    "eval",
    "execution_results",
    "iterations",
    "test_cases",
    "holdout_tests",
    "failure_report",
    "fallback_used",
    "critic_overruled",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_output_path(record: Mapping[str, Any], batch_dir: Path) -> Path | None:
    raw = record.get("output_path")
    if not raw:
        return None
    candidate = Path(str(raw))
    if candidate.is_absolute():
        return candidate
    for base in (Path.cwd(), _REPO_ROOT, batch_dir):
        resolved = base / candidate
        if resolved.is_file():
            return resolved
    return Path.cwd() / candidate


def _rate(successes: int, total: int) -> dict[str, Any]:
    return {
        "value": successes / total if total else None,
        "numerator": successes,
        "denominator": total,
        "ci95": wilson_interval(successes, total),
    }


def _ungraded_grade(problem: GoldenProblem, reason: str) -> dict[str, Any]:
    return {
        "problem_id": problem.id,
        "strict_pass": False,
        "ungraded": True,
        "reason": reason,
        "cases": [
            {
                "case_type": case_type,
                "passed": False,
                "actual": None,
                "expected": problem.cases[case_type].get("expected"),
                "error": reason,
                "timed_out": False,
                "guardrail_blocked": False,
            }
            for case_type in CASE_NAMES
        ],
    }


def _run_record(
    record: Mapping[str, Any],
    output: Mapping[str, Any],
    batch_dir: Path,
) -> dict[str, Any]:
    enriched = dict(record)
    for field in _RUN_FIELDS:
        if field in output:
            enriched[field] = output[field]
    if not enriched.get("status"):
        enriched["status"] = output.get("status", record.get("status", "missing"))
    return enriched


def _failure_reason(grade: Mapping[str, Any]) -> str:
    reasons = [
        f"{case.get('case_type')}: {case.get('error') or 'output mismatch'}"
        for case in grade.get("cases", [])
        if not case.get("passed")
    ]
    return "; ".join(reasons) or str(grade.get("reason") or "strict golden verdict failed")


def _constraint_summary(
    problems: Mapping[str, GoldenProblem],
    codes: Mapping[str, str],
    rules: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    checks: dict[str, Any] = {}
    for problem_id, code in codes.items():
        if problem_id not in rules or not rules[problem_id]:
            continue
        checks[problem_id] = check_constraints(problem_id, code, rules)
    applicable = [check for check in checks.values() if check.get("applicable_rules")]
    if not applicable:
        return checks, None
    adherent = sum(bool(check.get("adherent")) for check in applicable)
    return checks, _rate(adherent, len(applicable))


def _complexity_summary(
    problems: Mapping[str, GoldenProblem],
    codes: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    estimates: dict[str, Any] = {}
    for problem_id, code in codes.items():
        problem = problems.get(problem_id)
        if problem is None:
            continue
        estimates[problem_id] = estimate_complexity(code, problem.complexity)
    comparable = [
        item
        for item in estimates.values()
        if item.get("status") in {"match", "better", "worse"}
    ]
    if not comparable:
        return estimates, None
    matches = sum(item.get("status") == "match" for item in comparable)
    return estimates, _rate(matches, len(comparable))


def grade_batch(
    batch_dir: str | Path,
    *,
    dataset_path: str | Path = "evals/golden_dataset.csv",
    constraints_path: str | Path = "evals/constraints.yml",
) -> Path:
    """Grade one completed batch and write ``grade.json`` next to its index."""
    directory = Path(batch_dir)
    index_path = directory / "run_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"missing {index_path}; run `evals.cli run` before grading"
        )

    run_index = _load_json(index_path)
    manifest = _load_json(directory / "manifest.json")
    problems = {problem.id: problem for problem in load_dataset(dataset_path)}
    rules = load_constraint_rules(constraints_path)

    grades: dict[str, Any] = {}
    runs: dict[str, list[dict[str, Any]]] = {}
    codes: dict[str, str] = {}
    verdicts: dict[str, list[bool]] = {}
    durations: dict[str, float | None] = {}
    per_problem: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for problem_id in sorted(run_index, key=lambda value: (len(value), value)):
        problem = problems.get(problem_id)
        if problem is None:
            logger.warning("run_index references unknown problem %s", problem_id)
            continue
        records = sorted(run_index[problem_id], key=lambda item: item.get("repeat", 1))
        graded_repeats: list[dict[str, Any]] = []
        problem_runs: list[dict[str, Any]] = []
        for record in records:
            path = _resolve_output_path(record, directory)
            output = _load_json(path) if path is not None else {}
            run = _run_record(record, output, directory)
            problem_runs.append(run)
            code = str(output.get("code") or "")
            if code.strip():
                graded = grade_code(problem, code)
                codes.setdefault(problem_id, code)
            else:
                graded = _ungraded_grade(
                    problem,
                    f"no solution code (status={run.get('status', 'missing')})",
                )
            graded["repeat"] = record.get("repeat", 1)
            graded_repeats.append(graded)
            durations.setdefault(problem_id, record.get("duration_s"))
        grades[problem_id] = graded_repeats[0]
        runs[problem_id] = problem_runs
        verdicts[problem_id] = [bool(item.get("strict_pass")) for item in graded_repeats]

    evaluated = [problems[problem_id] for problem_id in grades]
    metrics: dict[str, Any] = compute_outcome_metrics(evaluated, grades, runs)
    calibration = confusion_metrics(grades, runs)
    matrix = calibration["confusion_matrix"]
    total = calibration["denominator"]
    internal_passes = matrix["true_positive"] + matrix["false_positive"]
    internal_fails = matrix["true_negative"] + matrix["false_negative"]
    shares = {
        "false_pass_rate": (matrix["false_positive"], internal_passes),
        "false_pass_share": (matrix["false_positive"], total),
        "false_fail_rate": (matrix["false_negative"], internal_fails),
        "verdict_agreement": (matrix["true_positive"] + matrix["true_negative"], total),
        "cohens_kappa": (None, None),
    }
    for name, (numerator, denominator) in shares.items():
        metrics[name] = {
            "value": calibration[name],
            "numerator": numerator,
            "denominator": denominator,
        }
    metrics["internal_pass_rate"] = _rate(internal_passes, total)
    metrics.update(
        compute_ops_metrics(run_index, strict_verdicts=verdicts)
    )

    checks, constraint_metric = _constraint_summary(problems, codes, rules)
    if constraint_metric is not None:
        metrics["constraint_adherence"] = constraint_metric
    complexity, complexity_metric = _complexity_summary(problems, codes)
    if complexity_metric is not None:
        metrics["complexity_match_rate"] = complexity_metric

    for problem_id, grade in grades.items():
        run = next(
            (item for item in runs[problem_id] if item.get("repeat", 1) == grade.get("repeat", 1)),
            {},
        )
        case_results = {
            case.get("case_type"): bool(case.get("passed"))
            for case in grade.get("cases", [])
        }
        passed_cases = sum(case_results.values())
        check = checks.get(problem_id) or {}
        per_problem.append(
            {
                "problem_id": problem_id,
                "requirement": problems[problem_id].requirement,
                "category": problems[problem_id].category,
                "repeat": grade.get("repeat", 1),
                "status": run.get("status"),
                "rejection_class": run.get("rejection_class"),
                "internal_passed": bool(
                    calibration["per_problem"].get(problem_id, {}).get("internal_passed")
                ),
                "golden_strict_pass": bool(grade.get("strict_pass")),
                "cases_passed": passed_cases,
                "cases_total": len(case_results),
                **{case_type: case_results.get(case_type, False) for case_type in CASE_NAMES},
                "iterations": run.get("iterations"),
                "duration_s": durations.get(problem_id),
                "constraint_adherence": check.get("adherent"),
                "complexity_status": (complexity.get(problem_id) or {}).get("status"),
                "complexity_estimate": (complexity.get(problem_id) or {}).get("estimate"),
                "failing_cases": ",".join(
                    case_type for case_type, passed in case_results.items() if not passed
                ),
            }
        )
        if grade.get("strict_pass") and calibration["per_problem"].get(problem_id, {}).get("internal_passed"):
            continue
        failures.append(
            {
                "problem_id": problem_id,
                "category": (
                    "aborted"
                    if run.get("status") == "aborted"
                    else "infra_error"
                    if run.get("status") == "infra_error"
                    else "false_pass"
                    if calibration["per_problem"].get(problem_id, {}).get("internal_passed")
                    else "false_fail"
                    if grade.get("strict_pass")
                    else "golden_fail"
                ),
                "reason": (
                    str(run.get("rejection_reason") or "agent aborted the requirement")
                    if run.get("status") == "aborted"
                    else _failure_reason(grade)
                ),
            }
        )

    document = {
        "batch_id": manifest.get("batch_id") or directory.name,
        "dataset_hash": dataset_hash(dataset_path),
        "graded_at": datetime.now(UTC).isoformat(),
        "repeat": manifest.get("repeat", 1),
        "metrics": metrics,
        "grades": grades,
        "per_problem": per_problem,
        "failures": failures,
        "false_passes.json": {
            "false_passes": calibration["false_passes"],
            "false_pass_rate": calibration["false_pass_rate"],
            "false_pass_share": calibration["false_pass_share"],
        },
        "constraints.json": {"checks": checks},
        "complexity.json": {"estimates": complexity},
    }
    destination = directory / "grade.json"
    destination.write_text(json.dumps(document, indent=2, default=str), encoding="utf-8")
    return destination
