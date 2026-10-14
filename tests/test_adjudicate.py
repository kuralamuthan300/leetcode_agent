import pytest


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    import agent.llms as llms

    monkeypatch.setattr(llms, "get_small", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_heavy", lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(llms, "get_critic", lambda: (_ for _ in ()).throw(RuntimeError("offline")))


def _t(harness, expected="x", actual="", error=""):
    return {"test_id": "t1", "harness": harness, "expected": expected, "actual": actual, "error": error}


def test_adjudicate_contract_violations_no_llm(monkeypatch):
    """Arity/range violations blame the test without touching the LLM."""
    import agent.prompts as prompts

    monkeypatch.setattr(prompts, "try_invoke", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM must not be called")))
    code = "def solve(arr, k):\n    return arr\n"
    out = prompts.adjudicate("p", "def solve(arr, k)", [], code, _t("print(solve([1]))"))
    assert out["blame"] == "test"
    spec = [{"name": "k", "type": "int", "range": "0..10"}]
    out = prompts.adjudicate("p", "", spec, "def solve(k):\n    return k", _t("print(solve(99))"))
    assert out["blame"] == "test"


def test_adjudicate_judge_verdicts(monkeypatch):
    import agent.prompts as prompts

    monkeypatch.setattr(prompts, "try_invoke", lambda *a, **k: "CODE_WRONG: off by one")
    out = prompts.adjudicate("p", "", [], "code", _t("print(solve(1))", expected="2", actual="1"))
    assert out == {"blame": "code", "reason": "off by one"}

    monkeypatch.setattr(prompts, "try_invoke", lambda *a, **k: "TEST_WRONG: bad expectation")
    out = prompts.adjudicate("p", "", [], "code", _t("print(solve(1))"))
    assert out["blame"] == "test"

    monkeypatch.setattr(prompts, "try_invoke", lambda *a, **k: None)
    assert prompts.adjudicate("p", "", [], "code", _t("x"))["blame"] == "unknown"
    monkeypatch.setattr(prompts, "try_invoke", lambda *a, **k: "mumble")
    assert prompts.adjudicate("p", "", [], "code", _t("x"))["blame"] == "unknown"


def test_evaluator_records_blame():
    from agent.nodes import evaluator

    code = "def solve(a, b):\n    return a + b\n"
    tests = [
        {"id": "t1", "harness": "print(solve(1, 2))", "expected": "3"},
        {"id": "t2", "harness": "print(solve(2, 2))", "expected": "5"},
    ]
    out = evaluator({"code": code, "test_cases": tests})
    by_id = {r["test_id"]: r for r in out["execution_results"]}
    assert by_id["t1"]["blame"] == "none"
    assert by_id["t2"]["blame"] in ("code", "unknown")  # offline judge unavailable


def test_repair_test_only_path(monkeypatch):
    import agent.nodes as nodes

    fixed_json = '[{"id": "t1", "harness": "print(solve(1, 2))", "expected": "3"}]'
    monkeypatch.setattr(nodes, "try_invoke", lambda *a, **k: fixed_json)
    code = "def solve(a, b):\n    return a + b\n"
    tests = [{"id": "t1", "harness": "print(solve([1]))", "expected": "3"}]
    out = nodes.evaluator({
        "code": code,
        "test_cases": tests,
        "function_signature": "def solve(a, b)",
        "clarified_intent": "add",
    })
    entry = out["execution_results"][0]
    assert entry["repaired"] is True
    assert entry["passed"] is True
    assert entry["blame"] == "none"
    assert out["eval"]["passed"] is True


def test_repair_invalid_keeps_failure(monkeypatch):
    import agent.nodes as nodes

    monkeypatch.setattr(nodes, "try_invoke", lambda *a, **k: "not json at all {{{")
    code = "def solve(a, b):\n    return a + b\n"
    tests = [{"id": "t1", "harness": "print(solve([1]))", "expected": "3"}]
    out = nodes.evaluator({
        "code": code,
        "test_cases": tests,
        "function_signature": "def solve(a, b)",
        "clarified_intent": "add",
    })
    entry = out["execution_results"][0]
    assert entry["passed"] is False
    assert entry["repaired"] is False
    assert entry["blame"] == "test"


def test_repair_still_failing_escalates(monkeypatch):
    import agent.nodes as nodes

    bad_json = '[{"id": "t1", "harness": "print(solve(1, 1))", "expected": "999"}]'
    monkeypatch.setattr(nodes, "try_invoke", lambda *a, **k: bad_json)
    code = "def solve(a, b):\n    return a + b\n"
    tests = [{"id": "t1", "harness": "print(solve([1]))", "expected": "3"}]
    out = nodes.evaluator({
        "code": code,
        "test_cases": tests,
        "function_signature": "def solve(a, b)",
        "clarified_intent": "add",
    })
    entry = out["execution_results"][0]
    assert entry["repaired"] is True
    assert entry["passed"] is False
    assert entry["blame"] == "unknown"
    assert "escalate" in entry["error"]
