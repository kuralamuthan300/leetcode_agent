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
    # Step 10: human-approval gates (opt-in). None = legacy auto-approve
    # (keeps Steps 6-9 tests green); explicit False = block for review.
    human_approved_promote: bool | None
    human_approved_opt: bool | None
    pending_approval: str | None
    audit_path: str | None
    # Step 12: HITL understanding gate + mid-run clarify.
    # pending_approval values: understanding | clarify | promote | optimize.
    understanding_summary: str | None
    human_question: str | None
    human_answer: str | None
    # Step 12 control flags (runtime schema channels so invoke payloads
    # and update_state values survive graph I/O):
    # require_understanding=True enforces the HITL-1 pause,
    # allow_clarify=True enables the mid-run clarify detour,
    # understanding_confirmed/clarify_asked track gate progress.
    require_understanding: bool | None
    allow_clarify: bool | None
    understanding_confirmed: bool | None
    clarify_asked: bool | None
