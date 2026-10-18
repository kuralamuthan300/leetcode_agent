from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    MAX_CODER_RETRIES,
    MAX_INTENT_ITERATIONS,
    coder,
    evaluator,
    planner,
    requirement_gatherer,
    run_reporter,
    test_case_gen,
)
from .state import AgentState

MAX_ITERATIONS = 3


def _route_gather(state: AgentState) -> str:
    if state.get("rejected"):
        return "reporter"
    if state.get("intent_clear") or state.get("assumed"):
        return "planner"
    if state.get("intent_iterations", 0) >= MAX_INTENT_ITERATIONS:
        return "planner"
    return "gather"


def _route_coder(state: AgentState) -> str:
    """Inner self-loop: retry while the coder has budget, else hand to evaluator."""
    if state.get("coder_status") == "retry" and state.get("coder_iterations", 0) <= MAX_CODER_RETRIES:
        return "coder"
    return "evaluator"


def _route_eval(state: AgentState) -> str:
    """Route failures back to the producer of the blame (test vs code)."""
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
    g.add_edge("planner", "tests")
    g.add_edge("tests", "coder")
    g.add_conditional_edges("coder", _route_coder, {"coder": "coder", "evaluator": "evaluator"})
    g.add_conditional_edges(
        "evaluator", _route_eval,
        {"tests": "tests", "planner": "planner", "reporter": "reporter"},
    )
    g.add_edge("reporter", END)
    return g.compile()
