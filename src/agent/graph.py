from __future__ import annotations

from langgraph.graph import END, StateGraph

from .config import get_config
from .nodes import (
    MAX_INTENT_ITERATIONS,
    coder,
    evaluator,
    planner,
    requirement_gatherer,
    run_reporter,
    test_case_gen,
)
from .state import AgentState

# The coder no longer self-loops: it produces one solution and hands off to the
# evaluator. Failures route back through the planner (code at fault) or the
# tests node (tests at fault) with a single retry.
MAX_PLANNER_RETRIES = get_config().agent.max_planner_retries
MAX_ITERATIONS = MAX_PLANNER_RETRIES + 1  # 1 original run + 1 retry run


def _route_gather(state: AgentState) -> str:
    if state.get("rejected"):
        return "reporter"
    if state.get("gather_mode") == "quiet":
        # Quiet mode is a single pass: never re-ask the user, always move on.
        return "planner"
    if state.get("intent_clear") or state.get("assumed"):
        return "planner"
    if state.get("intent_iterations", 0) >= MAX_INTENT_ITERATIONS:
        return "planner"
    return "gather"


def _route_planner(state: AgentState) -> str:
    """After a code-blame retry, keep the tests and go straight to the coder."""
    if state.get("next_action") == "replan_code":
        return "coder"
    return "tests"


def _route_tests(state: AgentState) -> str:
    """After a test-blame retry, fix the tests but keep the existing code."""
    if state.get("next_action") == "fix_tests":
        return "evaluator"
    return "coder"


def _route_eval(state: AgentState) -> str:
    """Route failures back to the producer of the blame (tests vs code)."""
    ev = state.get("eval", {})
    if ev.get("passed"):
        return "reporter"
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "reporter"
    if ev.get("error_class") == "empty" or ev.get("blame") == "test":
        return "tests"
    return "planner"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("gather", requirement_gatherer)
    g.add_node("planner", planner)
    g.add_node("tests", test_case_gen)
    g.add_node("coder", coder)
    g.add_node("evaluator", evaluator)
    g.add_node("reporter", run_reporter)
    g.set_entry_point("gather")
    g.add_conditional_edges(
        "gather", _route_gather,
        {"gather": "gather", "planner": "planner", "reporter": "reporter"},
    )
    # TDD ordering: tests derive from the planner contract, then drive the coder.
    # On a code-blame retry the planner skips test generation (tests are retained).
    g.add_conditional_edges("planner", _route_planner, {"tests": "tests", "coder": "coder"})
    # On a test-blame retry the tests node skips the coder (code is retained).
    g.add_conditional_edges("tests", _route_tests, {"coder": "coder", "evaluator": "evaluator"})
    g.add_edge("coder", "evaluator")
    g.add_conditional_edges(
        "evaluator", _route_eval,
        {"tests": "tests", "planner": "planner", "reporter": "reporter"},
    )
    g.add_edge("reporter", END)
    return g.compile()
