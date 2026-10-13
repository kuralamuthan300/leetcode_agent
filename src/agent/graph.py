from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import MAX_INTENT_ITERATIONS, coder, evaluator, intent_clarifier, planner, run_reporter, test_case_gen
from .state import AgentState

MAX_ITERATIONS = 3


def _route_intent(state: AgentState) -> str:
    if state.get("rejected"):
        return "reporter"
    if state.get("intent_clear"):
        return "planner"
    if state.get("intent_iterations", 0) >= MAX_INTENT_ITERATIONS:
        return "planner"
    return "intent"


def _route_eval(state: AgentState) -> str:
    ev = state.get("eval", {})
    if ev.get("passed"):
        return "reporter"
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return "reporter"
    return "planner"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("intent", intent_clarifier)
    g.add_node("planner", planner)
    g.add_node("tests", test_case_gen)
    g.add_node("coder", coder)
    g.add_node("evaluator", evaluator)
    g.add_node("reporter", run_reporter)
    g.set_entry_point("intent")
    g.add_conditional_edges(
        "intent", _route_intent,
        {"intent": "intent", "planner": "planner", "reporter": "reporter"},
    )
    g.add_edge("planner", "tests")
    g.add_edge("planner", "coder")
    g.add_edge("tests", "evaluator")
    g.add_edge("coder", "evaluator")
    g.add_conditional_edges("evaluator", _route_eval, {"planner": "planner", "reporter": "reporter"})
    g.add_edge("reporter", END)
    return g.compile()
