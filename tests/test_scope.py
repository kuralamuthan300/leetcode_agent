import pytest


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    import agent.llms as llms

    monkeypatch.setattr(llms, "get_small", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_heavy", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_critic", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_testgen", lambda: (_ for _ in ()).throw(RuntimeError("offline")))


def test_noncoding_rejected_fast():
    from agent.tools.guardrail import check_scope

    assert check_scope("Write an essay on climate change.").get("allowed") is False
    assert check_scope("Summarize this article for me.").get("allowed") is False
    assert check_scope("What is the weather today?").get("allowed") is False


def test_dsa_allowed_fast():
    from agent.tools.guardrail import check_scope

    assert check_scope("Reverse a linked list.").get("allowed") is True
    assert check_scope("Binary search in a sorted array.").get("allowed") is True


def test_gray_zone_needs_llm():
    from agent.tools.guardrail import check_scope

    assert check_scope("Add two numbers.").get("allowed") is None


def test_gray_allow_via_llm(monkeypatch):
    import agent.prompts as prompts

    monkeypatch.setattr(prompts, "try_invoke", lambda *a, **k: "ALLOW")
    from agent.nodes import requirement_gatherer

    out = requirement_gatherer({"requirement": "Add two numbers and return the result please."})
    assert out.get("rejected") is not True


def test_gray_reject_and_offline_fail_closed(monkeypatch):
    import agent.prompts as prompts

    from agent.nodes import requirement_gatherer

    monkeypatch.setattr(prompts, "try_invoke", lambda *a, **k: "REJECT: not a coding task")
    out = requirement_gatherer({"requirement": "Add two numbers and return the result please."})
    assert out["rejected"] is True
    assert "DSA coding tasks only" in out["rejection_reason"]

    monkeypatch.setattr(prompts, "try_invoke", lambda *a, **k: None)
    out = requirement_gatherer({"requirement": "Add two numbers and return the result please."})
    assert out["rejected"] is True


def test_fast_reject_short_circuits(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.graph import build_graph

    g = build_graph()
    final = g.invoke({"run_id": "run_id_essay", "requirement": "Write a poem about the sea."})
    assert final["rejected"] is True
    assert final.get("plan_steps", []) == []
    assert final.get("output_path")
