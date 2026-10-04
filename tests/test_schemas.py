"""Step 1 tests: valid loads pass, bad JSON is rejected."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schemas import ProblemSpec, RequirementsSpec, RunResult

ROOT = Path(__file__).resolve().parent.parent
REQ_DEMO = ROOT / "workspace" / "jobs" / "examples" / "requirements_id_demo.json"
PROB_DEMO = ROOT / "workspace" / "problems" / "examples" / "problem_id_two_sum_demo.json"


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def test_valid_requirements_loads():
    spec = RequirementsSpec.model_validate(_load(REQ_DEMO))
    assert spec.job_id == "req_a1b2c3d4"
    assert spec.difficulty == "easy"
    assert spec.num_tests == 8


def test_valid_problem_loads():
    spec = ProblemSpec.model_validate(_load(PROB_DEMO))
    assert spec.id == "two_sum_a1b2c3d4"
    assert spec.function_name == "two_sum"
    assert len(spec.testcases) >= 3


def test_missing_field_rejects():
    doc = _load(PROB_DEMO)
    doc.pop("statement")
    with pytest.raises(ValidationError):
        ProblemSpec.model_validate(doc)


def test_duplicate_test_id_rejects():
    doc = _load(PROB_DEMO)
    doc = copy.deepcopy(doc)
    doc["testcases"].append(copy.deepcopy(doc["testcases"][0]))
    with pytest.raises(ValidationError, match="duplicate test case ids"):
        ProblemSpec.model_validate(doc)


def test_valid_run_result_loads():
    r = RunResult.model_validate({"test_id": "t1", "passed": True, "time_ms": 1.5, "actual": [0, 1]})
    assert r.passed is True
    with pytest.raises(ValidationError):
        RunResult.model_validate({"test_id": "t1", "passed": True})  # missing time_ms
