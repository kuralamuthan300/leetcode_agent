import pytest


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    import agent.llms as llms

    monkeypatch.setattr(llms, "get_small", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_heavy", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_critic", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_testgen", lambda: (_ for _ in ()).throw(RuntimeError("offline")))


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

    assert test_case_gen({"clarified_intent": "x", "plan_steps": []}) == {
        "test_cases": [],
        "holdout_tests": [],
        "coder_iterations": 0,
        "coder_status": "",
    }


def test_coder_fallback_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agent.nodes import coder

    out = coder({"clarified_intent": "x", "plan_steps": []})
    assert out["code"].startswith("# TODO")
    assert out["coder_smoke"]["error"] == ""
    assert out["guardrail_report"]["allowed"] is True
    assert out["coder_checks"] == []


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
        extract_contract,
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


def test_extract_contract_block_legacy_absent():
    from agent.prompts import extract_contract

    block = (
        "1. Hash map\n"
        "CONTRACT:\n"
        "```json\n"
        '{"signature": "def solve(arr, k)", '
        '"args": [{"name": "arr", "type": "List[int]", "range": "len 1..5"}, '
        '{"name": "k", "type": "int", "range": "0..10"}], '
        '"returns": "List[int]"}\n'
        "```\n"
    )
    c = extract_contract(block)
    assert c is not None
    assert c["signature"] == "def solve(arr, k)"
    assert c["args"] == [
        {"name": "arr", "type": "List[int]", "range": "len 1..5"},
        {"name": "k", "type": "int", "range": "0..10"},
    ]
    assert c["returns"] == "List[int]"

    legacy = extract_contract("plan stuff\nSIGNATURE: def solve(a, b)")
    assert legacy == {"signature": "def solve(a, b)", "args": [], "returns": ""}

    assert extract_contract("just prose, no contract") is None


def test_planner_stores_contract(monkeypatch):
    import agent.nodes as nodes

    plan_text = (
        "1. Use a hash map\n"
        "CONTRACT:\n"
        "```json\n"
        '{"signature": "def solve(arr, k)", "args": [{"name": "arr", "type": "List[int]", "range": ""}], "returns": "List[int]"}\n'
        "```\n"
    )
    monkeypatch.setattr(nodes, "try_invoke_with_fallback", lambda *a, **k: (plan_text, False))
    monkeypatch.setattr(nodes, "critique", lambda *a, **k: {"passed": True, "issues": ""})
    out = nodes.planner({"clarified_intent": "rotate array"})
    assert out["function_signature"] == "def solve(arr, k)"
    assert out["arg_spec"] == [{"name": "arr", "type": "List[int]", "range": ""}]
    assert out["returns_spec"] == "List[int]"
    assert not any("CONTRACT" in s or s.startswith("{") for s in out["plan_steps"])


def test_testgen_prompt_carries_contract(monkeypatch):
    import agent.nodes as nodes

    seen = {}

    def fake_gen(*a, **k):
        seen["user"] = a[1]
        return '[{"id": "t1", "harness": "print(solve([1], 0))", "expected": "x"}]'

    monkeypatch.setattr(nodes, "try_invoke", fake_gen)
    out = nodes.test_case_gen({
        "clarified_intent": "rotate",
        "plan_steps": [],
        "function_signature": "def solve(arr, k)",
        "arg_spec": [{"name": "k", "type": "int", "range": "0..10"}],
        "code": "",
    })
    assert "Contract: def solve(arr, k)" in seen["user"]
    assert "strictly within these ranges" in seen["user"]
    assert "boundary" in seen["user"]
    assert len(out["test_cases"]) == 1


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
    assert len(out["test_cases"]) == 2
    assert len(out["holdout_tests"]) == 1
    assert "critic_overruled" not in out


def test_evaluator_short_circuits_on_smoke_failure():
    from agent.nodes import evaluator

    code = "def solve(a):\n    return a\n"
    tests = [{"id": "t1", "harness": "print(solve(1))", "expected": "1"}]
    out = evaluator({
        "code": code,
        "test_cases": tests,
        "coder_smoke": {"error": "NameError('x')"},
    })
    assert out["execution_results"] == []  # no matrix run
    assert out["eval"]["passed"] is False
    assert out["eval"]["error_class"] == "mechanical"
    assert "smoke failed" in out["eval"]["feedback"]
    assert out["iterations"] == 1


def test_evaluator_error_classes():
    from agent.nodes import evaluator

    code = "def solve(a):\n    return a\n"
    good = [{"id": "t1", "harness": "print(solve(1))", "expected": "1"}]
    assert evaluator({"code": code, "test_cases": good})["eval"]["error_class"] == "none"
    bad = [{"id": "t1", "harness": "print(solve(1))", "expected": "2"}]
    assert evaluator({"code": code, "test_cases": bad})["eval"]["error_class"] == "logical"
    assert evaluator({"code": code, "test_cases": []})["eval"]["error_class"] == "empty"


def test_graph_order_planner_coder_tests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import agent.nodes as nodes

    order = []
    real = {}

    for name in ("planner", "coder", "test_case_gen", "evaluator"):
        real[name] = getattr(nodes, name)

    def wrap(name, fn):
        def inner(state):
            order.append(name)
            return fn(state)
        return inner

    import agent.graph as graph_mod

    monkeypatch.setattr(graph_mod, "planner", wrap("planner", real["planner"]))
    monkeypatch.setattr(graph_mod, "coder", wrap("coder", real["coder"]))
    monkeypatch.setattr(graph_mod, "test_case_gen", wrap("tests", real["test_case_gen"]))
    monkeypatch.setattr(graph_mod, "evaluator", wrap("evaluator", real["evaluator"]))
    import agent.llms as llms

    monkeypatch.setattr(llms, "get_small", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_heavy", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_critic", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_testgen", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    g = graph_mod.build_graph()
    g.invoke({"run_id": "run_id_order", "requirement": "Given an array of ints, return the sorted array. Example: [3,1] -> [1,3]."})
    seq = [n for n in order if n in ("planner", "coder", "tests", "evaluator")]
    assert seq[:4] == ["planner", "tests", "coder", "evaluator"], order
    # testgen derives from the planner contract, and every tests run is followed
    # by the coder; the coder always follows a tests run (TDD chain).
    for i, n in enumerate(seq):
        if n == "tests":
            assert seq[i - 1] in ("planner", "evaluator"), order
        if n == "coder":
            assert seq[i - 1] == "tests", order


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


def test_testgen_ignores_code_reference(monkeypatch):
    """TDD ordering: testgen must derive from the contract, never the code."""
    import agent.nodes as nodes

    seen = {}

    def fake_gen(*a, **k):
        seen["user"] = a[1]
        return '[{"id": "t1", "harness": "print(solve([1], 0))", "expected": "x"}]'

    monkeypatch.setattr(nodes, "try_invoke", fake_gen)
    nodes.test_case_gen({
        "clarified_intent": "rotate",
        "plan_steps": [],
        "function_signature": "def solve(arr, k)",
        "code": "def solve(arr, k):\n    return 'SECRET_MARKER'\n",
    })
    assert "SECRET_MARKER" not in seen["user"]
    assert "Reference implementation" not in seen["user"]


def test_coder_inner_loop_retries_on_code_failure(monkeypatch):
    import agent.nodes as nodes

    monkeypatch.setattr(nodes, "try_invoke_with_fallback", lambda *a, **k: ("def solve(a):\n    return a + 1\n", False))
    monkeypatch.setattr(nodes, "critique", lambda *a, **k: {"passed": True, "issues": ""})
    out = nodes.coder({
        "clarified_intent": "identity",
        "plan_steps": [],
        "function_signature": "def solve(a)",
        "test_cases": [{"id": "t1", "harness": "print(solve(1))", "expected": "1"}],
        "coder_iterations": 0,
    })
    assert out["coder_status"] == "retry"
    assert out["coder_iterations"] == 1
    assert "t1" in out["feedback"]
    assert out["coder_checks"][0]["test_id"] == "t1"
    assert out["coder_checks"][0]["harness"] == "print(solve(1))"


def test_coder_escalates_when_all_failures_blame_test(monkeypatch):
    import agent.nodes as nodes

    monkeypatch.setattr(nodes, "try_invoke_with_fallback", lambda *a, **k: ("def solve(a):\n    return a\n", False))
    monkeypatch.setattr(nodes, "critique", lambda *a, **k: {"passed": True, "issues": ""})
    out = nodes.coder({
        "clarified_intent": "identity",
        "plan_steps": [],
        "function_signature": "def solve(a)",
        # harness calls solve with the wrong arity -> deterministic test blame
        "test_cases": [{"id": "t1", "harness": "print(solve(1, 2))", "expected": "1"}],
    })
    assert out["coder_status"] == "escalate"
    assert "coder_iterations" not in out


def test_evaluator_dominant_blame():
    from agent.nodes import evaluator

    code = "def solve(a):\n    return a\n"
    test_side = evaluator({
        "code": code,
        "test_cases": [{"id": "t1", "harness": "print(solve(1, 2))", "expected": "1"}],
        "function_signature": "def solve(a)",
    })
    assert test_side["eval"]["blame"] == "test"
    code_side = evaluator({
        "code": code,
        "test_cases": [{"id": "t1", "harness": "print(solve(1))", "expected": "2"}],
    })
    assert code_side["eval"]["blame"] == "code"


def test_evaluator_runs_holdout_tests():
    from agent.nodes import evaluator

    code = "def solve(a):\n    return a\n"
    out = evaluator({
        "code": code,
        "test_cases": [{"id": "t1", "harness": "print(solve(1))", "expected": "1"}],
        "holdout_tests": [{"id": "h1", "harness": "print(solve(2))", "expected": "2"}],
    })
    assert {r["test_id"] for r in out["execution_results"]} == {"t1", "h1"}
    assert out["eval"]["total"] == 2


def test_route_coder_inner_loop():
    from agent.graph import _route_coder

    assert _route_coder({"coder_status": "retry", "coder_iterations": 1}) == "coder"
    assert _route_coder({"coder_status": "retry", "coder_iterations": 4}) == "evaluator"
    assert _route_coder({"coder_status": "escalate"}) == "evaluator"
    assert _route_coder({"coder_status": "pass"}) == "evaluator"


def test_route_eval_blame_aware():
    from agent.graph import _route_eval

    assert _route_eval({"eval": {"passed": True}}) == "reporter"
    assert _route_eval({"eval": {"passed": False, "error_class": "empty"}, "iterations": 1}) == "tests"
    assert _route_eval({"eval": {"passed": False, "blame": "test"}, "iterations": 1}) == "tests"
    assert _route_eval({"eval": {"passed": False, "blame": "code"}, "iterations": 1}) == "planner"
    assert _route_eval({"eval": {"passed": False, "blame": "code"}, "iterations": 3}) == "reporter"

