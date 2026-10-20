import pytest


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Force deterministic fallback (no model gateway) for hermetic tests."""
    import agent.llms as llms

    monkeypatch.setattr(llms, "get_small", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_heavy", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_critic", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_testgen", lambda: (_ for _ in ()).throw(RuntimeError("offline")))



def test_truncated_assumed_offline(tmp_path, monkeypatch):
    """Non-TTY vague input proceeds flagged assumed (Bug 1 class)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    from agent.graph import build_graph

    g = build_graph()
    final = g.invoke({"run_id": "run_id_trunc", "requirement": "write a code to rotate an array of"})
    assert final["assumed"] is True
    assert final["intent_clear"] is False
    assert final.get("output_path")


def test_clear_intent_confirmed_fast(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.graph import build_graph

    g = build_graph()
    req = "Given an array of ints, return the sorted array. Example: [3,1] -> [1,3]."
    final = g.invoke({"run_id": "run_id_gather_clear", "requirement": req})
    assert final["intent_iterations"] == 1
    assert final["intent_clear"] is True
    assert final["confirmed"] is True
    assert final["assumed"] is False


def test_interactive_confirm_ok(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "ok")
    from agent.nodes import requirement_gatherer

    out = requirement_gatherer({"requirement": "write a code to rotate an array of"})
    assert out["confirmed"] is True
    assert out["assumed"] is False


def test_interactive_correction_folded_in(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["a 1-D array of ints rotated right by k", "ok"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    from agent.nodes import requirement_gatherer

    out = requirement_gatherer({"requirement": "write a code to rotate an array of"})
    assert "1-D array" in out["requirement"]
    assert out["confirmed"] is True


# --- quiet gather mode (Increment 4) ---------------------------------------

_CLEAR_REQ = "Given an array of ints, return the maximum element."


def _patch_verdict(monkeypatch, **over):
    """Patch the gatherer's verdict call to return a fixed workability JSON."""
    import json

    from agent import prompts

    payload = {
        "clarified_intent": _CLEAR_REQ,
        "workable": True,
        "reason": "clear",
        "assumptions": [],
        "blocking_ambiguities": [],
        "confidence": "high",
        "rejection_class": None,
    }
    payload.update(over)
    monkeypatch.setattr(
        prompts, "try_invoke", lambda kind, user, system="", retries=1: json.dumps(payload)
    )


def test_quiet_workable_ok(monkeypatch):
    from agent.nodes import requirement_gatherer

    _patch_verdict(monkeypatch)
    out = requirement_gatherer({"requirement": _CLEAR_REQ, "gather_mode": "quiet"})
    assert out["status"] == "ok"
    assert out["rejected"] is False and out["assumed"] is False
    assert out["intent_clear"] is True and out["confirmed"] is False
    assert out["workability"]["rubric_version"] == "1"


def test_quiet_assumptions_mark_assumed(monkeypatch):
    from agent.nodes import requirement_gatherer

    _patch_verdict(monkeypatch, assumptions=["input may be empty"])
    out = requirement_gatherer({"requirement": _CLEAR_REQ, "gather_mode": "quiet"})
    assert out["status"] == "assumed"
    assert out["assumed"] is True and out["intent_clear"] is False
    assert out["workability"]["assumptions"] == ["input may be empty"]


def test_quiet_blocking_ambiguity_aborts(monkeypatch):
    """workable=true but a blocking ambiguity still aborts (strict)."""
    from agent.nodes import requirement_gatherer

    _patch_verdict(monkeypatch, blocking_ambiguities=["1-D vs 2-D input"])
    out = requirement_gatherer({"requirement": _CLEAR_REQ, "gather_mode": "quiet"})
    assert out["rejected"] is True
    assert out["status"] == "aborted"
    assert out["rejection_class"] == "under_specified"


def test_quiet_not_workable_keeps_class(monkeypatch):
    from agent.nodes import requirement_gatherer

    _patch_verdict(monkeypatch, workable=False, rejection_class="nondeterministic")
    out = requirement_gatherer({"requirement": _CLEAR_REQ, "gather_mode": "quiet"})
    assert out["rejected"] is True and out["status"] == "aborted"
    assert out["rejection_class"] == "nondeterministic"


def test_quiet_bad_class_falls_back_to_under_specified(monkeypatch):
    from agent.nodes import requirement_gatherer

    _patch_verdict(monkeypatch, workable=False, rejection_class="banana")
    out = requirement_gatherer({"requirement": _CLEAR_REQ, "gather_mode": "quiet"})
    assert out["rejection_class"] == "under_specified"


def test_quiet_truncated_aborts_without_asking(monkeypatch):
    """Truncated input is an `incomplete` abort; quiet mode never prompts."""
    monkeypatch.setattr("builtins.input", lambda _prompt="": (_ for _ in ()).throw(AssertionError("no prompt")))
    from agent.nodes import requirement_gatherer

    out = requirement_gatherer({"requirement": "write a code to rotate an array of", "gather_mode": "quiet"})
    assert out["rejected"] is True
    assert out["status"] == "aborted" and out["rejection_class"] == "incomplete"


def test_quiet_llm_unavailable_is_infra_error(monkeypatch):
    """Offline/no model => infra_error (excluded from the dataset), not a rejection."""
    from agent.nodes import requirement_gatherer

    out = requirement_gatherer({"requirement": _CLEAR_REQ, "gather_mode": "quiet"})
    assert out["status"] == "infra_error"
    assert out["rejection_class"] == "infra"
    assert out["workability"]["verdict"] == "unavailable"


def test_quiet_scope_gate_aborts_non_dsa():
    from agent.nodes import requirement_gatherer

    out = requirement_gatherer({"requirement": "Write me a poem about the sea.", "gather_mode": "quiet"})
    assert out["rejected"] is True and out["rejection_class"] == "scope"


def test_quiet_routing_never_reasks():
    from agent.graph import _route_gather

    assert _route_gather({"gather_mode": "quiet"}) == "planner"
    assert _route_gather({"gather_mode": "quiet", "rejected": True}) == "reporter"
    assert _route_gather({"gather_mode": "interactive", "intent_clear": False, "intent_iterations": 1}) == "gather"
