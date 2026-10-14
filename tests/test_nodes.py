import pytest


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    import agent.llms as llms

    monkeypatch.setattr(llms, "get_small", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_heavy", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_critic", lambda: (_ for _ in ()).throw(RuntimeError("offline")))


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


def test_signature_helpers():
    from agent.prompts import (
        code_defines_signature,
        extract_signature,
        harness_matches,
        signature_params,
    )

    assert extract_signature("plan\nSIGNATURE: def solve(arr, k)") == "def solve(arr, k)"
    assert extract_signature("no sig here") is None
    assert signature_params("def solve(arr, k)") == ["arr", "k"]
    assert code_defines_signature("def solve(arr, k):\n    return arr", "def solve(arr, k)") is True
    assert code_defines_signature("def solve(n, arr, k):\n    return arr", "def solve(arr, k)") is False
    assert harness_matches("print(solve([1], 2))", "def solve(arr, k)") is True
    assert harness_matches("print(solve([1]))", "def solve(arr, k)") is False


def test_valid_test_enforces_arity():
    from agent.nodes import _valid_test

    ok = {"id": "t1", "harness": "print(solve([1, 2], 1))", "expected": "x"}
    assert _valid_test(ok, "def solve(arr, k)") is not None
    assert _valid_test(ok, "def solve(arr)") is None
    assert _valid_test(ok) is not None  # no contract: skip check


def test_evaluator_replaces_results():
    from agent.nodes import evaluator

    code = "def solve(a):\n    return a\n"
    tests = [{"id": "t1", "harness": "print(solve(1))", "expected": "1"}]
    first = evaluator({"code": code, "test_cases": tests})
    assert len(first["execution_results"]) == 1
    merged = {}
    merged.update(first)
    second = evaluator({"code": code, "test_cases": tests, "execution_results": merged["execution_results"]})
    merged.update(second)
    assert len(merged["execution_results"]) == 1  # replace, not append (Bug 3)


def test_critique_parsing(monkeypatch):
    import agent.llms as llms
    import agent.prompts as prompts

    class Msg:
        def __init__(self, content):
            self.content = content

    monkeypatch.setattr(llms, "get_critic", lambda: type("M", (), {"invoke": lambda self, _: Msg("PASS")})())
    assert prompts.critique("plan", "x") == {"passed": True, "issues": ""}

    monkeypatch.setattr(llms, "get_critic", lambda: type("M", (), {"invoke": lambda self, _: Msg("ISSUES: no signature")})())
    assert prompts.critique("plan", "x") == {"passed": False, "issues": "no signature"}

    monkeypatch.setattr(llms, "get_critic", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    assert prompts.critique("plan", "x") is None


def test_planner_critic_regen_and_overruled(monkeypatch):
    import agent.nodes as nodes

    calls = []

    def fake_gen(user, system):
        calls.append(user)
        if len(calls) == 1:
            return ("1. approach X", False)
        return ("1. approach Y\nSIGNATURE: def solve(a)", False)

    verdicts = iter([{"passed": False, "issues": "missing signature"}, {"passed": True, "issues": ""}])
    monkeypatch.setattr(nodes, "try_invoke_with_fallback", fake_gen)
    monkeypatch.setattr(nodes, "critique", lambda *a, **k: next(verdicts))
    out = nodes.planner({"clarified_intent": "sum two ints, return total"})
    assert out.get("function_signature") == "def solve(a)"
    assert "critic_overruled" not in out
    assert len(calls) == 2

    verdicts = iter([{"passed": False, "issues": "bad"}, {"passed": False, "issues": "still bad"}])
    monkeypatch.setattr(nodes, "critique", lambda *a, **k: next(verdicts))
    out = nodes.planner({"clarified_intent": "sum two ints, return total"})
    assert out["critic_overruled"] is True


def test_coder_critic_regen(monkeypatch):
    import agent.nodes as nodes

    v1 = "def solve(a):\n    return a\n"
    v2 = "def solve(a):\n    return a if a else 0\n"
    gen_calls = []

    def fake_gen(user, system):
        gen_calls.append(system)
        return (v1 if len(gen_calls) == 1 else v2, False)

    verdicts = iter([{"passed": False, "issues": "no edge handling"}, {"passed": True, "issues": ""}])
    monkeypatch.setattr(nodes, "try_invoke_with_fallback", fake_gen)
    monkeypatch.setattr(nodes, "critique", lambda *a, **k: next(verdicts))
    out = nodes.coder({"clarified_intent": "return a", "plan_steps": [], "function_signature": "def solve(a)"})
    assert "if a else" in out["code"]
    assert "critic_overruled" not in out


def test_testgen_critic_regen(monkeypatch):
    import agent.nodes as nodes

    good = '[{"id": "t1", "harness": "print(solve(1))", "expected": "1"}, {"id": "t2", "harness": "print(solve(2))", "expected": "2"}, {"id": "t3", "harness": "print(solve(3))", "expected": "3"}]'
    monkeypatch.setattr(nodes, "try_invoke", lambda *a, **k: good)
    verdicts = iter([{"passed": False, "issues": "thin"}, {"passed": True, "issues": ""}])
    monkeypatch.setattr(nodes, "critique", lambda *a, **k: next(verdicts))
    out = nodes.test_case_gen({"clarified_intent": "identity", "plan_steps": []})
    assert len(out["test_cases"]) == 3
    assert "critic_overruled" not in out


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
