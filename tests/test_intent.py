def test_unclear_intent_caps_at_three(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.graph import build_graph

    g = build_graph()
    final = g.invoke({"run_id": "run_id_intent_cap", "requirement": "solve sorting"})
    assert final["intent_iterations"] == 3
    assert final["intent_clear"] is False
    assert final.get("output_path")


def test_clear_intent_single_pass(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.graph import build_graph

    g = build_graph()
    req = "Given an array of ints, return the sorted array. Example: [3,1] -> [1,3]."
    final = g.invoke({"run_id": "run_id_intent_clear", "requirement": req})
    assert final["intent_iterations"] == 1
    assert final["intent_clear"] is True
