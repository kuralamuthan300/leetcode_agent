def test_reporter_schema_complete(tmp_path, monkeypatch):
    import json

    monkeypatch.chdir(tmp_path)
    from agent.nodes import run_reporter

    state = {
        "run_id": "run_id_schema",
        "requirement": "r",
        "rejected": False,
        "rejection_reason": "",
        "clarified_intent": "c",
        "intent_iterations": 1,
        "intent_clear": True,
        "confirmed": True,
        "assumed": False,
        "plan_steps": ["p"],
        "function_signature": "def solve(a)",
        "test_cases": [{"id": "t", "harness": "print(solve(1))", "expected": "1"}],
        "code": "def solve(a):\n    return a\n",
        "coder_smoke": {"error": ""},
        "guardrail_report": {"allowed": True},
        "fallback_used": False,
        "critic_overruled": False,
        "session_notes": [],
        "execution_results": [],
        "eval": {"passed": True, "passed_count": 1, "total": 1, "feedback": "", "error_class": "none"},
    }
    out = run_reporter(state)
    assert out["output_path"] == "output/run_id_schema.json"
    doc = json.loads(open("output/run_id_schema.json").read())
    expected = set(state) | {"timestamp", "models"}
    assert set(doc) == expected, set(doc) ^ expected
    from agent.llms import HEAVY_MODEL, SMALL_MODEL

    assert doc["models"] == {"heavy": HEAVY_MODEL, "small": SMALL_MODEL}
