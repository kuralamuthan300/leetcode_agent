def test_prompt_reads_multiline(monkeypatch):
    import cli

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["Given nums and target.", "Return indices.", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert cli._prompt_requirement() == "Given nums and target.\nReturn indices.\n"


def test_prompt_eof_ends(monkeypatch):
    import cli

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    def fake_input(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli._prompt_requirement() == ""


def test_main_uses_prompt_when_no_args(tmp_path, monkeypatch):
    import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["Sort the array.", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    seen = {}

    class FakeGraph:
        def stream(self, state, stream_mode=None):
            seen.update(state)
            yield {"reporter": {"output_path": "output/x.json"}}

    monkeypatch.setattr(cli, "build_graph", lambda: FakeGraph())
    assert cli.main([]) == 0
    assert seen["requirement"].strip() == "Sort the array."


def test_main_empty_requirement_rejected(tmp_path, monkeypatch, capsys):
    import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    assert cli.main([]) == 2


def test_status_lines():
    import cli

    assert cli._line("intent", {"intent_clear": True, "intent_iterations": 1}) == "✓ intent clear (1 iter)"
    assert "retrying (1/3)" in cli._line("intent", {"intent_clear": False, "intent_iterations": 1})
    assert cli._line("planner", {"plan_steps": ["a", "b"]}) == "✓ plan (2 steps)"
    assert cli._line("tests", {"test_cases": [{}, {}]}) == "✓ tests (2)"
    assert cli._line("coder", {"coder_smoke": {"error": ""}}) == "✓ code (smoke ok)"
    assert cli._line("coder", {"coder_smoke": {"error": "blocked: open"}}).startswith("⚠ code blocked:")
    assert cli._line("evaluator", {"eval": {"passed": True, "passed_count": 5, "total": 5}}) == "✓ eval 5/5"
    assert "retry" in cli._line("evaluator", {"eval": {"passed": False, "passed_count": 3, "total": 5}, "iterations": 1})
    assert cli._line("reporter", {"output_path": "output/x.json"}) == "✓ wrote output/x.json"
    assert "(fallback)" in cli._line("planner", {"plan_steps": [], "fallback_used": True})


def test_run_graph_accumulates():
    import cli

    class FakeGraph:
        def stream(self, state, stream_mode=None):
            yield {"intent": {"intent_clear": True, "intent_iterations": 1}}
            yield {"reporter": {"output_path": "output/y.json"}}

    final = cli._run_graph(FakeGraph(), {"run_id": "r", "requirement": "q"})
    assert final["output_path"] == "output/y.json"
    assert final["intent_clear"] is True
