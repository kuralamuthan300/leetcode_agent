def test_sql_rejected():
    from agent.tools.guardrail import check_language

    assert check_language("Write a SQL query to find duplicate emails. SELECT email FROM users GROUP BY email.").get("allowed") is False
    assert check_language("Join orders and customers, order by total.").get("allowed") is False


def test_java_rejected():
    from agent.tools.guardrail import check_language

    assert check_language("Solve Two Sum in Java with public class Solution.").get("allowed") is False
    assert check_language("Write it in Go.").get("allowed") is False


def test_python_and_neutral_allowed():
    from agent.tools.guardrail import check_language

    assert check_language("Given an array of ints, return the sorted array. Example: [3,1] -> [1,3].").get("allowed") is True
    assert check_language("Solve Two Sum in Python using a dict.").get("allowed") is True
    assert check_language("Given a scalar value, return its square.").get("allowed") is True


def test_rejected_short_circuits_to_reporter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import agent.llms as llms

    monkeypatch.setattr(llms, "get_small", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_heavy", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_critic", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_testgen", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    from agent.graph import build_graph

    g = build_graph()
    final = g.invoke({"run_id": "run_id_reject", "requirement": "Write a SQL query listing all users."})
    assert final["rejected"] is True
    assert final.get("plan_steps", []) == []
    assert final.get("code", "") == ""
    assert "Python only" in final["rejection_reason"]
    assert final.get("output_path")
