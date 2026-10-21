from __future__ import annotations

import csv
import json

import pytest

from evals.dataset import load_dataset
from evals.grade import grade_batch

_TWO_SUM = """
def two_sum(nums, target):
    seen = {}
    for index, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index
    return []
"""


def _write_bundle_dataset(path, *, count: int = 50) -> None:
    fields = [
        "ID",
        "Problem Statement",
        "Constraints",
        "Complexity Targets",
        "Category",
        "Test Cases & Expected Outputs",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(1, count + 1):
            cases = {
                "base_case": {"input": [index], "output": index},
                "edge_case": {"input": [], "output": 0},
                "adversarial_case": {"input": [index], "output": index},
            }
            writer.writerow(
                {
                    "ID": str(index),
                    "Problem Statement": f"Implement solve_{index}(nums)\nReturn the result.",
                    "Constraints": "1 <= len(nums) <= 100",
                    "Complexity Targets": "Time: O(N), Space: O(N)",
                    "Category": "Arrays",
                    "Test Cases & Expected Outputs": json.dumps(cases),
                }
            )


def _write_batch(
    directory,
    problem_id: str,
    code: str,
    *,
    status: str = "ok",
    extra: dict | None = None,
) -> None:
    output_path = directory / "agent_runs" / f"{problem_id}__r1.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": f"{problem_id}__r1",
        "status": status,
        "code": code,
        "eval": {"passed": True, "iterations": 1},
        "execution_results": [],
        "iterations": 1,
        **(extra or {}),
    }
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    (directory / "manifest.json").write_text(
        json.dumps({"batch_id": "fixture", "repeat": 1}),
        encoding="utf-8",
    )
    (directory / "run_index.json").write_text(
        json.dumps(
            {
                problem_id: [
                    {
                        "problem_id": problem_id,
                        "repeat": 1,
                        "output_path": str(output_path),
                        "status": status,
                        "duration_s": 12.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_combined_case_column_and_output_alias_load(tmp_path):
    source = tmp_path / "golden.csv"
    _write_bundle_dataset(source)

    problems = load_dataset(source)

    assert len(problems) == 50
    case = problems[0].cases["base_case"]
    assert case["expected"] == 1
    assert "output" not in case


def test_grade_stage_writes_grade_json_and_outcome_metrics(tmp_path):
    _write_batch(tmp_path, "1", _TWO_SUM)

    destination = grade_batch(tmp_path, dataset_path="evals/golden_dataset.csv")

    grade = json.loads(destination.read_text(encoding="utf-8"))
    assert grade["grades"]["1"]["strict_pass"] is True
    assert grade["metrics"]["golden_strict_solve_rate"]["value"] == 1.0
    assert grade["metrics"]["false_pass_rate"]["value"] == 0.0
    assert grade["metrics"]["internal_pass_rate"]["value"] == 1.0
    assert grade["per_problem"][0]["problem_id"] == "1"
    # Reference-free metrics stay absent so the report shows them as unavailable.
    assert "mutation_score" not in grade["metrics"]
    assert "oracle_correctness" not in grade["metrics"]
    assert "false_passes.json" in grade


def test_grade_stage_flags_false_pass_when_internal_verdict_is_wrong(tmp_path):
    _write_batch(tmp_path, "1", "def two_sum(nums, target):\n    return [0, 0]\n")

    destination = grade_batch(tmp_path, dataset_path="evals/golden_dataset.csv")

    grade = json.loads(destination.read_text(encoding="utf-8"))
    assert grade["grades"]["1"]["strict_pass"] is False
    assert grade["metrics"]["false_pass_rate"]["value"] == 1.0
    assert grade["metrics"]["false_pass_share"]["value"] == 1.0
    assert grade["failures"][0]["category"] == "false_pass"
    assert grade["false_passes.json"]["false_passes"][0]["problem_id"] == "1"


def test_grade_stage_handles_missing_code_without_crashing(tmp_path):
    _write_batch(tmp_path, "1", "", status="infra_error")

    destination = grade_batch(tmp_path, dataset_path="evals/golden_dataset.csv")

    grade = json.loads(destination.read_text(encoding="utf-8"))
    assert grade["grades"]["1"]["ungraded"] is True
    assert grade["metrics"]["infra_error_rate"] == 1.0
    assert grade["failures"][0]["category"] == "infra_error"


def test_grade_stage_records_aborted_requirements_separately(tmp_path):
    _write_batch(
        tmp_path,
        "1",
        "",
        status="aborted",
        extra={"rejection_class": "scope", "rejection_reason": "not a DSA coding task (letter)"},
    )

    destination = grade_batch(tmp_path, dataset_path="evals/golden_dataset.csv")

    grade = json.loads(destination.read_text(encoding="utf-8"))
    assert grade["per_problem"][0]["rejection_class"] == "scope"
    assert grade["failures"][0]["category"] == "aborted"
    assert "letter" in grade["failures"][0]["reason"]


def test_grade_stage_requires_a_run_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="evals.cli run"):
        grade_batch(tmp_path, dataset_path="evals/golden_dataset.csv")
