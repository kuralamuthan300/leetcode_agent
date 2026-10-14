from __future__ import annotations

from typing import Any, TypedDict

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
    error_class: str  # none | logical | mechanical | empty


class AgentState(TypedDict, total=False):
    run_id: str
    requirement: str
    clarified_intent: str
    intent_iterations: int
    intent_clear: bool
    confirmed: bool
    assumed: bool
    plan_steps: list[str]
    function_signature: str
    test_cases: list[dict[str, Any]]
    code: str
    coder_smoke: dict[str, str]
    guardrail_report: dict
    execution_results: list[ExecutionResult]  # replace semantics: latest run only
    eval: EvalReport
    feedback: str
    iterations: int
    fallback_used: bool
    rejected: bool
    rejection_reason: str
    critic_overruled: bool
    output_path: str
