"""Pydantic v2 schemas — source of truth for LLM JSON validation (Step 1)."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_FUNCTION_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ensure_json_serializable(value: Any, field_name: str) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be JSON-serializable: {exc}") from exc
    return value


class RequirementsSpec(BaseModel):
    """Input requirements JSON: requirements_id_xxxx.json (Sec 5.1)."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    language: str = "python"
    num_tests: int = Field(default=8, ge=1, le=100)
    constraints_hint: str | None = None
    extra: str | None = None

    @field_validator("job_id", "category", "language", mode="before")
    @classmethod
    def _strip_nonempty(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("must be a non-empty string")
        return v


class TestCase(BaseModel):
    """Single testcase inside ProblemSpec."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    input: dict[str, Any]
    expected: Any
    is_hidden: bool = False
    timeout_ms: int = Field(default=2000, ge=1)

    @field_validator("input", "expected", mode="after")
    @classmethod
    def _must_be_serializable(cls, v: Any) -> Any:
        return _ensure_json_serializable(v, "testcase field")


class ProblemExample(BaseModel):
    """Worked example inside ProblemSpec (Sec 5.2 examples list)."""

    model_config = ConfigDict(extra="forbid")

    input: dict[str, Any]
    output: Any
    explanation: str | None = None

    @field_validator("input", "output", mode="after")
    @classmethod
    def _must_be_serializable(cls, v: Any) -> Any:
        return _ensure_json_serializable(v, "example field")


class ProblemSpec(BaseModel):
    """Output problem JSON: problem_id_xxxx.json (Sec 5.2)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    statement: str = Field(min_length=1)
    function_name: str = Field(min_length=1)
    signature: str = Field(min_length=1)
    examples: list[ProblemExample] = Field(min_length=1)
    constraints: list[str] = Field(min_length=1)
    testcases: list[TestCase] = Field(min_length=1)
    oracle_solution_hash: str | None = None

    @field_validator("function_name", mode="after")
    @classmethod
    def _valid_identifier(cls, v: str) -> str:
        if not _FUNCTION_NAME_RE.match(v):
            raise ValueError("function_name must be a valid Python identifier")
        return v

    @model_validator(mode="after")
    def _check_consistency(self) -> ProblemSpec:
        ids = [t.id for t in self.testcases]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate test case ids: {dupes}")
        if self.function_name not in self.signature:
            raise ValueError("function_name must appear in signature")
        return self


class RunResult(BaseModel):
    """Per-test execution result (minimal timed result)."""

    model_config = ConfigDict(extra="forbid")

    test_id: str = Field(min_length=1)
    passed: bool
    time_ms: float = Field(ge=0)
    actual: Any | None = None
    error: str | None = None
    stdout: str | None = None
