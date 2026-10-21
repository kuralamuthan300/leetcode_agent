"""Loading, validating, and exporting the versioned golden evaluation dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CASE_NAMES = ("base_case", "edge_case", "adversarial_case")
_FUNCTION_RE = re.compile(r"Implement\s+([A-Za-z_]\w*)\s*\(")
_COMPLEXITY_RE = re.compile(r"\bO\s*\(\s*[^)\s][^)]*\)")
_FIELD_ALIASES = {
    "id": {"id", "problemid", "problemnumber", "number"},
    "statement": {"problemstatement", "statement", "problem"},
    "constraints": {"constraints", "constraint"},
    "complexity": {
        "complexity",
        "complexitytarget",
        "complexitytargets",
        "targetcomplexity",
    },
    "category": {"category", "topic", "topics"},
    "cases": {
        "cases",
        "testcases",
        "goldencases",
        # The human-edited CSV keeps all three cases in one JSON column.
        "testcasesexpectedoutputs",
        "testcasesandexpectedoutputs",
        "goldencasesexpectedoutputs",
    },
}

# The CSV's case objects spell the expected result "output"; the loader's
# canonical key is "expected". Accept both and normalize on load.
_EXPECTED_ALIASES = ("expected", "output", "expectedoutput", "expectedresult")


@dataclass(frozen=True, slots=True)
class GoldenProblem:
    id: str
    statement: str
    constraints: str
    complexity: str
    function_name: str
    cases: dict[str, Any]
    category: str = ""

    @property
    def requirement(self) -> str:
        return (
            f"{self.statement.rstrip()}\n"
            f"Constraints: {self.constraints.strip()}\n"
            f"Complexity: {self.complexity.strip()}\n"
            "Encoding convention: linked lists and trees are represented as lists."
        )

    def to_problem_record(self) -> dict[str, str]:
        return {"id": self.id, "requirement": self.requirement}

    def to_genai_record(self, dataset_hash: str) -> dict[str, Any]:
        return {
            "inputs": {
                "problem_id": self.id,
                "requirement": self.requirement,
                "function_name": self.function_name,
            },
            "expectations": {
                "cases": self.cases,
                "complexity": self.complexity,
                "constraints": self.constraints,
                "category": self.category,
            },
            "tags": {"golden_dataset_hash": dataset_hash},
        }


def dataset_hash(path: str | Path) -> str:
    """Return the SHA-256 digest of the CSV's exact bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _column(
    fieldnames: list[str],
    logical_name: str,
    aliases: set[str] | None = None,
) -> str | None:
    normalized = {_normalize_header(name): name for name in fieldnames}
    candidates = aliases or _FIELD_ALIASES[logical_name]
    return next(
        (
            normalized[normalized_alias]
            for alias in candidates
            if (normalized_alias := _normalize_header(alias)) in normalized
        ),
        None,
    )


def _required_value(row: dict[str, str | None], field: str, row_number: int) -> str:
    value = row.get(field)
    if value is None or not value.strip():
        raise ValueError(f"row {row_number}: field '{field}' must not be empty")
    return value.strip()


def _parse_json_payload(value: str | None, row_number: int, field: str) -> Any:
    if value is None or not value.strip():
        raise ValueError(f"row {row_number}: field '{field}' must not be empty")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"row {row_number}: field '{field}' is not valid JSON: {exc.msg}"
        ) from exc
    return payload


def _normalize_case(payload: Any, row_number: int, field: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(  # noqa: TRY004 - malformed dataset content is a validation error
            f"row {row_number}: field '{field}' must be a JSON object"
        )
    normalized = dict(payload)
    aliases = [key for key in normalized if _normalize_header(key) in _EXPECTED_ALIASES]
    if not aliases:
        raise ValueError(f"row {row_number}: field '{field}' is missing 'expected'")
    chosen = next(
        (key for key in aliases if _normalize_header(key) == "expected"),
        aliases[0],
    )
    expected = normalized[chosen]
    for key in aliases:
        normalized.pop(key)
    normalized["expected"] = expected
    return normalized


def _parse_case(value: str | None, row_number: int, field: str) -> dict[str, Any]:
    return _normalize_case(_parse_json_payload(value, row_number, field), row_number, field)


def _parse_case_bundle(value: str | None, row_number: int, field: str) -> dict[str, Any]:
    payload = _parse_json_payload(value, row_number, field)
    if not isinstance(payload, dict):
        raise ValueError(  # noqa: TRY004 - malformed dataset content is a validation error
            f"row {row_number}: field '{field}' must be a JSON object"
        )
    normalized_cases: dict[str, Any] = {}
    for key, case in payload.items():
        normalized = _normalize_header(str(key))
        if normalized in normalized_cases:
            raise ValueError(f"row {row_number}: field '{field}' repeats case '{key}'")
        normalized_cases[normalized] = case
    expected = {_normalize_header(name) for name in CASE_NAMES}
    if normalized_cases.keys() != expected:
        missing = next((name for name in CASE_NAMES if _normalize_header(name) not in normalized_cases), None)
        if missing:
            raise ValueError(f"row {row_number}: field '{field}' is missing '{missing}'")
        raise ValueError(f"row {row_number}: field '{field}' must contain exactly {', '.join(CASE_NAMES)}")
    return {
        _name: _normalize_case(
            normalized_cases[_normalize_header(_name)],
            row_number,
            f"{field}[{_name}]",
        )
        for _name in CASE_NAMES
    }


def load_dataset(
    path: str | Path = "evals/golden_dataset.csv",
) -> list[GoldenProblem]:
    """Read and validate the golden CSV, reporting errors with row and field context."""
    source = Path(path)
    csv.field_size_limit(sys.maxsize)
    problems: list[GoldenProblem] = []
    seen_ids: set[str] = set()

    with source.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("row 1: CSV is missing a header")

        columns = {
            name: _column(reader.fieldnames, name)
            for name in ("id", "statement", "constraints", "complexity", "category", "cases")
        }
        for required in ("id", "statement", "constraints", "complexity"):
            if columns[required] is None:
                raise ValueError(
                    f"row 1: required field '{required}' is missing from the CSV header"
                )

        case_columns = {
            name: _column(
                reader.fieldnames,
                name,
                {name, name.replace("_", ""), name.removesuffix("_case")},
            )
            for name in CASE_NAMES
        }
        has_bundle = columns["cases"] is not None
        if not has_bundle and any(case_columns[name] is None for name in CASE_NAMES):
            missing = next(name for name, column in case_columns.items() if column is None)
            raise ValueError(
                f"row 1: required field '{missing}' (or a JSON cases column) is missing"
            )

        for row_number, row in enumerate(reader, 2):
            row_id = _required_value(row, columns["id"], row_number)
            if row_id in seen_ids:
                raise ValueError(f"row {row_number}: field '{columns['id']}' duplicates ID {row_id!r}")
            seen_ids.add(row_id)
            statement = _required_value(row, columns["statement"], row_number)
            constraints = _required_value(row, columns["constraints"], row_number)
            complexity = _required_value(row, columns["complexity"], row_number)
            if not _COMPLEXITY_RE.search(complexity):
                raise ValueError(
                    f"row {row_number}: field '{columns['complexity']}' has no parsable "
                    "Big-O complexity target"
                )
            function_match = _FUNCTION_RE.search(statement)
            if function_match is None:
                raise ValueError(
                    f"row {row_number}: field '{columns['statement']}' has no "
                    "'Implement function_name(' declaration"
                )

            if has_bundle:
                cases = _parse_case_bundle(row.get(columns["cases"]), row_number, columns["cases"])
            else:
                cases = {
                    name: _parse_case(row.get(case_columns[name]), row_number, case_columns[name])
                    for name in CASE_NAMES
                }

            category_column = columns["category"]
            category = (row.get(category_column) or "").strip() if category_column else ""
            problems.append(
                GoldenProblem(
                    id=row_id,
                    statement=statement,
                    constraints=constraints,
                    complexity=complexity,
                    function_name=function_match.group(1),
                    cases=cases,
                    category=category,
                )
            )

    if len(problems) != 50:
        raise ValueError(
            f"dataset contains {len(problems)} rows; expected exactly 50"
        )
    return problems


def write_problems_jsonl(
    problems: list[GoldenProblem],
    path: str | Path = "evals/problems.jsonl",
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for problem in problems:
            stream.write(json.dumps(problem.to_problem_record(), ensure_ascii=False) + "\n")
    return destination


def sync_mlflow_dataset(
    problems: list[GoldenProblem],
    digest: str,
    *,
    reset: bool = False,
    dataset_name: str = "leetcode-golden-v1",
    experiment_name: str = "leetcodeagent-evals",
) -> dict[str, Any]:
    """Create or idempotently update the GenAI evaluation dataset in MLflow."""
    from evals.mlflow_eval import sync_dataset

    records = [problem.to_genai_record(digest) for problem in problems]
    return sync_dataset(
        records,
        digest,
        reset=reset,
        dataset_name=dataset_name,
        experiment_name=experiment_name,
    )
