import pytest


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Force deterministic fallback (no Ollama) for hermetic tests."""
    import agent.llms as llms

    monkeypatch.setattr(llms, "get_small", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_heavy", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_critic", lambda: (_ for _ in ()).throw(RuntimeError("offline")))



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
