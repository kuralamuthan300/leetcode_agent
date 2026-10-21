from __future__ import annotations

import csv
import json

from evals.report import evaluate_gates, write_report


def test_report_writes_scorecard_metrics_and_failures(tmp_path):
    metrics = {
        "golden_strict_solve_rate": {"value": 0.8, "numerator": 40, "denominator": 50},
        "golden_adversarial_case_pass_rate": {"value": 0.5, "numerator": 25, "denominator": 50},
        "false_pass_rate": {"value": 0.02, "numerator": 1, "denominator": 50},
    }
    result = write_report(
        tmp_path,
        metrics=metrics,
        per_problem=[{"problem_id": "x/../unsafe", "strict_pass": False}],
        failures=[{"problem_id": "x/../unsafe", "category": "wrong code", "reason": "failed"}],
        artifacts={"oracle.json": {"valid": 1}},
    )
    assert result["gate_status"] == "fail"
    payload = json.loads((tmp_path / "metrics.json").read_text())
    assert payload["gates"]["golden_adversarial_case_pass_rate"]["status"] == "fail"
    assert (tmp_path / "oracle.json").is_file()
    assert "Golden Strict Solve Rate" in (tmp_path / "scorecard.md").read_text()
    with (tmp_path / "per_problem.csv").open() as stream:
        assert next(csv.DictReader(stream))["problem_id"] == "x/../unsafe"
    assert "wrong code" in (tmp_path / "failures.md").read_text()


def test_unavailable_gates_do_not_look_like_success():
    gates = evaluate_gates({})
    assert gates["golden_strict_solve_rate"]["status"] == "unavailable"
