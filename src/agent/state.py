from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class ExecutionResult(TypedDict):
    test_id: str
    passed: bool
    actual: str
    expected: str
    error: str


class EvalReport(TypedDict):
    passed: bool
    passed_count: int
    total: int
    feedback: str


class AgentState(TypedDict, total=False):
    run_id: str
    requirement: str
    clarified_intent: str
    intent_iterations: int
    intent_clear: bool
    plan_steps: list[str]
    test_cases: list[dict[str, Any]]
    code: str
    coder_smoke: dict[str, str]
    guardrail_report: dict
    execution_results: Annotated[list[ExecutionResult], add]
    eval: EvalReport
    feedback: str
    iterations: int
    fallback_used: bool
    output_path: str
