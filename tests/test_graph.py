def test_graph_runs_to_reporter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.graph import build_graph

    g = build_graph()
    final = g.invoke({"run_id": "run_id_test_001", "requirement": "Return sum of two ints."})
    assert (tmp_path / "output" / "run_id_test_001.json").exists()
    assert final.get("output_path")
