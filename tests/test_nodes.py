import pytest


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    import agent.llms as llms

    monkeypatch.setattr(llms, "get_small", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_heavy", lambda: (_ for _ in ()).throw(RuntimeError("offline")))


def test_prompt_parsers():
    from agent.prompts import extract_code, extract_json_array, strip_think

    assert strip_think("<think>hmm</think>answer") == "answer"
    assert "def solve" in extract_code("```python\ndef solve(): pass\n```")
    assert extract_json_array('[{"id": "t1"}]') == [{"id": "t1"}]
    assert extract_json_array("not json") is None


def test_planner_fallback():
    from agent.nodes import planner

    out = planner({"clarified_intent": "sort stuff", "feedback": "too slow"})
    assert out["plan_steps"][0].startswith("Solve:")
    assert "too slow" in out["plan_steps"][1]


def test_testgen_fallback_empty():
    from agent.nodes import test_case_gen

    assert test_case_gen({"clarified_intent": "x", "plan_steps": []}) == {"test_cases": []}


def test_coder_fallback_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.nodes import coder

    out = coder({"clarified_intent": "x", "plan_steps": []})
    assert out["code"].startswith("# TODO")
    assert out["coder_smoke"]["error"] == ""
    assert out["guardrail_report"]["allowed"] is True


def test_evaluator_matrix_and_iterations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.nodes import evaluator

    code = "def solve(a, b):\n    return a + b\n"
    tests = [
        {"id": "t1", "harness": "print(solve(1, 2))", "expected": "3"},
        {"id": "t2", "harness": "print(solve(2, 2))", "expected": "5"},
    ]
    out = evaluator({"code": code, "test_cases": tests})
    assert out["iterations"] == 1
    assert out["eval"]["passed"] is False
    assert out["eval"]["passed_count"] == 1
    assert "t2" in out["eval"]["feedback"]
    assert "5" in out["eval"]["feedback"] and "4" in out["eval"]["feedback"]


def test_heavy_fallback_flag(monkeypatch):
    import agent.prompts as prompts

    calls = []

    def fake(kind, user, system, retries=1):
        calls.append(kind)
        return None if kind == "heavy" else "1. do thing"

    monkeypatch.setattr(prompts, "try_invoke", fake)
    from agent.nodes import planner

    out = planner({"clarified_intent": "sort an array of ints, return sorted"})
    assert out["fallback_used"] is True
    assert calls[0] == "heavy" and calls[-1] == "small"
    assert out["plan_steps"]
