"""Step 6: minimal top-level router graph.

Flow: START -> router -> creator | solver -> END
Router rule (deterministic, no LLM): if a problem spec is present
(or mode == "solver"), go solver; otherwise go creator.
Stub nodes only mark which branch ran; real subgraphs land in Steps 7-8.
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.state import AgentState

Route = Literal["creator", "solver"]


def router_node(state: AgentState) -> dict:
    """Pass-through node; branching is decided by route_after_router."""
    return {}


def route_after_router(state: AgentState) -> Route:
    """Light routing: problem spec present -> solver, else creator."""
    if state.get("problem_spec"):
        return "solver"
    if (state.get("mode") or "").strip().lower() == "solver":
        return "solver"
    return "creator"


def creator_node(state: AgentState) -> dict:
    """Stub for creator_graph (Step 7)."""
    errors = list(state.get("errors") or [])
    errors.append("creator stub reached")
    return {"mode": "creator", "errors": errors}


def solver_node(state: AgentState) -> dict:
    """Stub for solver_graph (Step 8)."""
    errors = list(state.get("errors") or [])
    errors.append("solver stub reached")
    return {"mode": "solver", "errors": errors}


def build_graph() -> object:
    """Build and compile START->router->creator/solver->END with MemorySaver."""
    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("creator", creator_node)
    builder.add_node("solver", solver_node)
    builder.add_edge(START, "router")
    builder.add_conditional_edges("router", route_after_router, {"creator": "creator", "solver": "solver"})
    builder.add_edge("creator", END)
    builder.add_edge("solver", END)
    return builder.compile(checkpointer=MemorySaver())


app = build_graph()


def get_mermaid() -> str:
    """Return mermaid diagram text for verify step."""
    return app.get_graph().draw_mermaid()


if __name__ == "__main__":
    print(get_mermaid())
