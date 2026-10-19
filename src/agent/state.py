from __future__ import annotations

from typing import Any, TypedDict

class ExecutionResult(TypedDict):
    test_id: str
    harness: str
    passed: bool
    actual: str
    expected: str
    error: str
    blame: str  # none | code | test | unknown
    repaired: bool


class EvalReport(TypedDict):
    passed: bool
    passed_count: int
    total: int
    feedback: str
    error_class: str  # none | logical | mechanical | empty
    blame: str  # none | code | test (dominant blame across failures)


class AgentState(TypedDict, total=False):
    run_id: str
    requirement: str
    gather_mode: str  # interactive | quiet
    status: str  # ok | assumed | aborted | infra_error
    rejection_class: str  # scope | language | incomplete | under_specified | ... | infra
    workability: dict  # quiet-mode verdict (verdict, assumptions, blocking_ambiguities, ...)
    out_dir: str  # output root (batch runs override to output/<batch_id>)
    clarified_intent: str
    intent_iterations: int
    intent_clear: bool
    confirmed: bool
    assumed: bool
    plan_steps: list[str]
    function_signature: str
    arg_spec: list[dict[str, str]]
    returns_spec: str
    test_cases: list[dict[str, Any]]
    holdout_tests: list[dict[str, Any]]  # evaluator-only; hidden from the coder
    code: str
    coder_smoke: dict[str, str]
    guardrail_report: dict
    execution_results: list[ExecutionResult]  # replace semantics: latest run only
    eval: EvalReport
    failure_report: dict  # {culprit, reason, failing[]} when the last eval failed
    next_action: str  # router hint: replan_code | fix_tests | "" (fresh run)
    feedback: str
    iterations: int
    fallback_used: bool
    rejected: bool
    rejection_reason: str
    critic_overruled: bool
    output_path: str
