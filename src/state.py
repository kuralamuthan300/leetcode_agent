"""Step 6: shared AgentState for top-level router graph."""

from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    """Clipboard passed between graph nodes.

    Nullable fields use None when absent. Lists default to [] at invoke time.
    """

    job_id: str
    mode: str  # "creator" | "solver" | "" (unset lets router decide)
    workdir: str
    problem_spec: dict | None
    solution_code: str | None
    test_results: list
    timings_ms: list
    attempt: int
    errors: list
    safety_flags: list
    needs_human: bool
