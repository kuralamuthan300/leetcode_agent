"""Local scorecard artifacts and hard-gate evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from evals.metrics_spec import METRICS, PRIMARY_METRICS, blocking_metrics


def evaluate_gates(metrics: dict[str, Any]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for spec in blocking_metrics():
        raw = metrics.get(spec.id)
        value = raw.get("value") if isinstance(raw, dict) else raw
        if value is None:
            gates[spec.id] = {
                "status": "unavailable",
                "value": None,
                "threshold": spec.gate_value,
                "blocking": spec.blocking,
            }
            continue
        passed = value >= spec.gate_value if spec.direction.value == "higher_is_better" else (
            value <= spec.gate_value
        )
        gates[spec.id] = {
            "status": "pass" if passed else "fail",
            "value": value,
            "threshold": spec.gate_value,
            "blocking": spec.blocking,
        }
    return gates


def _status(value: Any, gate: dict[str, Any] | None) -> str:
    if value is None:
        return "unavailable"
    if gate is None:
        return "report"
    return gate["status"]


def _scorecard_markdown(metrics: dict[str, Any], gates: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Scorecard",
        "",
        "| # | Metric | Value | Numerator / denominator | Gate | Status |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for spec in PRIMARY_METRICS:
        raw = metrics.get(spec.id)
        item = raw if isinstance(raw, dict) else {"value": raw}
        value = item.get("value")
        rendered_value = "unavailable" if value is None else f"{value:.4f}"
        numerator = item.get("numerator", "—")
        denominator = item.get("denominator", "—")
        gate = gates.get(spec.id)
        threshold = f"{spec.gate_operator} {spec.gate_value}" if spec.gate_value is not None else "—"
        lines.append(
            f"| {spec.scorecard_row} | {spec.display_name} | {rendered_value} | "
            f"{numerator} / {denominator} | {threshold} | {_status(value, gate)} |"
        )
    lines.extend(["", "## Gate details", ""])
    for metric_id, gate in gates.items():
        lines.append(
            f"- `{metric_id}`: {gate['status']} "
            f"(value={gate['value']}, threshold={gate['threshold']})"
        )
    return "\n".join(lines) + "\n"


def _write_per_problem(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    batch_dir: str | Path,
    *,
    metrics: dict[str, Any],
    per_problem: list[dict[str, Any]] | None = None,
    failures: list[dict[str, Any]] | None = None,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the stable local report artifacts before optional tracking."""
    directory = Path(batch_dir)
    directory.mkdir(parents=True, exist_ok=True)
    gates = evaluate_gates(metrics)
    gate_status = "fail" if any(gate["status"] == "fail" for gate in gates.values()) else (
        "unavailable" if any(gate["status"] == "unavailable" for gate in gates.values()) else "pass"
    )
    enriched: dict[str, Any] = {}
    for metric_id, spec in METRICS.items():
        value = metrics.get(metric_id)
        if isinstance(value, dict):
            enriched[metric_id] = {
                **value,
                "spec": spec.to_dict(),
                "gate": gates.get(metric_id),
                "status": _status(value.get("value"), gates.get(metric_id)),
            }
        else:
            enriched[metric_id] = {
                "value": value,
                "spec": spec.to_dict(),
                "gate": gates.get(metric_id),
                "status": _status(value, gates.get(metric_id)),
            }
    payload = {
        "metrics": enriched,
        "gates": gates,
        "gate_status": gate_status,
        **(artifacts or {}),
    }
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (directory / "scorecard.md").write_text(_scorecard_markdown(metrics, gates), encoding="utf-8")
    _write_per_problem(directory / "per_problem.csv", per_problem or [])

    failure_lines = ["# Failures", ""]
    for failure in failures or []:
        failure_lines.append(
            f"- Problem `{failure.get('problem_id', '?')}`: "
            f"{failure.get('category', 'unclassified')} — {failure.get('reason', '')}"
        )
    if not failures:
        failure_lines.append("No failures recorded.")
    (directory / "failures.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    for name, content in (artifacts or {}).items():
        if name.endswith(".json"):
            (directory / name).write_text(
                json.dumps(content, indent=2, default=str),
                encoding="utf-8",
            )
    return {"gate_status": gate_status, "gates": gates, "metrics": enriched}
