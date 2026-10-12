import json


def test_toolbox_runs_coder_code():
    from agent.tools.sandbox import python_toolbox, run_python

    code = "def add(a, b):\n    return a + b\n"
    r = run_python(code, harness="print(add(19, 23))")
    assert r["error"] == ""
    assert r["stdout"].strip() == "42"

    # LangChain tool wrapper returns JSON the coder LLM can read
    assert python_toolbox is not None
    assert python_toolbox.name == "python_sandbox"
    out = json.loads(python_toolbox.func(code, "print(add(1, 2))"))
    assert out["stdout"].strip() == "3"


def test_toolbox_captures_errors_and_timeout():
    from agent.tools.sandbox import run_python

    bad = run_python("raise ValueError('boom')")
    assert "ValueError" in bad["error"]

    slow = run_python("while True: pass", timeout_s=1)
    assert "timeout" in slow["error"]


def test_coder_node_uses_toolbox(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.graph import build_graph

    g = build_graph()
    final = g.invoke({"run_id": "run_id_coder_smoke", "requirement": "solve sorting"})
    assert "smoke" in final["coder_smoke"]["stdout"]
    assert final["coder_smoke"]["error"] == ""
