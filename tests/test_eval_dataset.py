from __future__ import annotations

import csv
import json
import sys
from types import SimpleNamespace

import pytest

from evals.adapters import (
    adapt_input,
    outputs_equal,
    parse_output,
    register_adapter,
)
from evals.dataset import dataset_hash, load_dataset, write_problems_jsonl


def _write_dataset(path, *, count=50, overrides=None):
    fields = [
        "Problem ID",
        "Problem Statement",
        "Constraints",
        "Complexity Targets",
        "Category",
        "Base Case",
        "Edge Case",
        "Adversarial Case",
    ]
    overrides = overrides or {}
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(1, count + 1):
            row = {
                "Problem ID": str(index),
                "Problem Statement": f"Implement solve_{index}(nums)\nReturn the result.",
                "Constraints": "1 <= len(nums) <= 100",
                "Complexity Targets": "Time: O(N), Space: O(N)",
                "Category": "Arrays",
                "Base Case": json.dumps({"input": [index], "expected": index}),
                "Edge Case": json.dumps({"input": [], "expected": 0}),
                "Adversarial Case": json.dumps({"input": [index, index], "expected": index}),
            }
            row.update(overrides.get(index, {}))
            writer.writerow(row)


def test_loads_valid_dataset_and_builds_batch_requirements(tmp_path):
    source = tmp_path / "golden.csv"
    _write_dataset(source)

    problems = load_dataset(source)

    assert len(problems) == 50
    problem = problems[0]
    assert problem.id == "1"
    assert problem.function_name == "solve_1"
    assert problem.cases["base_case"] == {"input": [1], "expected": 1}
    assert "Implement solve_1(nums)" in problem.requirement
    assert "Constraints: 1 <= len(nums) <= 100" in problem.requirement
    assert "Complexity: Time: O(N), Space: O(N)" in problem.requirement
    assert "linked lists and trees are represented as lists" in problem.requirement

    output = write_problems_jsonl(problems, tmp_path / "out" / "problems.jsonl")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 50
    assert rows[0] == problem.to_problem_record()
    assert len(dataset_hash(source)) == 64


def test_csv_field_limit_supports_large_adversarial_payload(tmp_path):
    source = tmp_path / "large.csv"
    huge_payload = json.dumps({"input": "x" * (128 * 1024 + 100), "expected": ""})
    _write_dataset(source, overrides={7: {"Adversarial Case": huge_payload}})

    problems = load_dataset(source)

    assert len(problems[6].cases["adversarial_case"]["input"]) > 128 * 1024


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({1: {"Constraints": ""}}, r"row 2: field 'Constraints'"),
        ({1: {"Base Case": "{invalid"}}, r"row 2: field 'Base Case' is not valid JSON"),
        ({1: {"Base Case": json.dumps({"input": [1]})}}, r"row 2: field 'Base Case' is missing 'expected'"),
        ({1: {"Complexity Targets": "linear"}}, r"row 2: field 'Complexity Targets'"),
        ({1: {"Problem Statement": "No signature"}}, r"row 2: field 'Problem Statement'"),
    ],
)
def test_validation_errors_identify_row_and_field(tmp_path, overrides, message):
    source = tmp_path / "invalid.csv"
    _write_dataset(source, overrides=overrides)

    with pytest.raises(ValueError, match=message):
        load_dataset(source)


def test_duplicate_ids_and_wrong_dataset_size_are_rejected(tmp_path):
    source = tmp_path / "invalid.csv"
    _write_dataset(source, overrides={2: {"Problem ID": "1"}})
    with pytest.raises(ValueError, match=r"row 3: field 'Problem ID' duplicates ID"):
        load_dataset(source)

    _write_dataset(source, count=49)
    with pytest.raises(ValueError, match="expected exactly 50"):
        load_dataset(source)


def test_combined_cases_json_column_is_supported(tmp_path):
    source = tmp_path / "cases.csv"
    fields = ["id", "statement", "constraints", "complexity", "cases"]
    cases = {
        "base_case": {"input": [1], "expected": 1},
        "edge_case": {"input": [], "expected": 0},
        "adversarial_case": {"input": [1, 1], "expected": 1},
    }
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(1, 51):
            writer.writerow(
                {
                    "id": index,
                    "statement": f"Implement solve_{index}(values)",
                    "constraints": "values is a list",
                    "complexity": "O(N)",
                    "cases": json.dumps(cases),
                }
            )

    assert load_dataset(source)[-1].cases == cases


def test_combined_cases_must_contain_exactly_the_three_case_types(tmp_path):
    source = tmp_path / "extra-case.csv"
    cases = {
        "base_case": {},
        "edge_case": {},
        "adversarial_case": {},
        "unexpected_case": {},
    }
    with source.open("w", encoding="utf-8", newline="") as stream:
        fields = ["id", "statement", "constraints", "complexity", "cases"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(1, 51):
            writer.writerow(
                {
                    "id": index,
                    "statement": f"Implement solve_{index}(values)",
                    "constraints": "values is a list",
                    "complexity": "O(N)",
                    "cases": json.dumps(cases),
                }
            )

    with pytest.raises(ValueError, match="must contain exactly"):
        load_dataset(source)


def test_output_parsing_and_problem_specific_numeric_tolerance():
    assert parse_output("True") is True
    assert parse_output("[]") == []
    assert parse_output("") == ""
    assert outputs_equal("1", 1.0)
    assert not outputs_equal("True", 1)
    assert outputs_equal("2.0000000001", 2.0, function_name="find_median_sorted_arrays")
    assert not outputs_equal("2.0001", 2.0, function_name="find_median_sorted_arrays")
    assert not outputs_equal("2.0000000001", 2.0, function_name="other_function")


def test_problem_adapter_override_is_keyed_by_problem_id():
    register_adapter("999", input_transform=lambda value: tuple(value))
    assert adapt_input("999", [1, 2]) == (1, 2)
    assert adapt_input("998", [1, 2]) == [1, 2]


def test_prepare_cli_writes_jsonl_and_prints_manifest(tmp_path, capsys):
    from evals.cli import main

    source = tmp_path / "golden.csv"
    output = tmp_path / "normalized.jsonl"
    _write_dataset(source)

    assert main(["prepare", "--dataset", str(source), "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["problems"] == 50
    assert result["dataset_hash"] == dataset_hash(source)
    assert result["problems_file"] == str(output)
    assert len(output.read_text().splitlines()) == 50


def test_mlflow_sync_is_digest_idempotent_and_reset_recreates(monkeypatch):
    class FakeClient:
        def get_experiment_by_name(self, name):
            assert name == "leetcodeagent-evals"
            return SimpleNamespace(experiment_id="eval-exp")

        def create_experiment(self, name):
            raise AssertionError("existing experiment should be reused")

    datasets = {}
    state = {"created": 0, "deleted": 0, "merged": 0}

    class FakeDataset:
        def __init__(self, dataset_id, name):
            self.dataset_id = dataset_id
            self.name = name
            self.tags = {}
            self.version = 0

        def merge_records(self, records):
            assert len(records) == 50
            assert records[0]["inputs"]["function_name"] == "solve_1"
            assert records[0]["expectations"]["cases"]["base_case"]["expected"] == 1
            assert records[0]["tags"]["golden_dataset_hash"] in {"digest", "new-digest"}
            self.version += 1
            state["merged"] += 1
            return self

    class FakeDatasets:
        @staticmethod
        def search_datasets(experiment_ids, max_results):
            assert experiment_ids == "eval-exp"
            assert max_results == 100
            return list(datasets.values())

        @staticmethod
        def create_dataset(name, experiment_id):
            state["created"] += 1
            dataset = FakeDataset(f"dataset-{state['created']}", name)
            datasets[dataset.dataset_id] = dataset
            return dataset

        @staticmethod
        def delete_dataset(dataset_id):
            state["deleted"] += 1
            del datasets[dataset_id]

        @staticmethod
        def set_dataset_tags(dataset_id, tags):
            datasets[dataset_id].tags.update(tags)

    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(MlflowClient=FakeClient, genai=SimpleNamespace(datasets=FakeDatasets)),
    )
    problems = [
        SimpleNamespace(
            to_genai_record=lambda digest, index=index: {
                "inputs": {"function_name": f"solve_{index}"},
                "expectations": {"cases": {"base_case": {"expected": 1}}},
                "tags": {"golden_dataset_hash": digest},
            }
        )
        for index in range(1, 51)
    ]
    from evals.dataset import sync_mlflow_dataset

    first = sync_mlflow_dataset(problems, "digest")
    second = sync_mlflow_dataset(problems, "digest")
    assert first["dataset_id"] == second["dataset_id"]
    assert first["dataset_version"] == second["dataset_version"] == 1
    assert state == {"created": 1, "deleted": 0, "merged": 1}

    changed = sync_mlflow_dataset(problems, "new-digest")
    assert changed["dataset_id"] == first["dataset_id"]
    assert changed["dataset_version"] == 2
    assert state == {"created": 1, "deleted": 0, "merged": 2}

    reset = sync_mlflow_dataset(problems, "digest", reset=True)
    assert reset["dataset_id"] != first["dataset_id"]
    assert reset["dataset_version"] == 1
    assert state == {"created": 2, "deleted": 1, "merged": 3}
