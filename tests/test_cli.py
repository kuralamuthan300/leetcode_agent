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
    answers = iter(["Sort the array.", "", "ok"])
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

    assert cli._line("gather", {"intent_clear": True, "intent_iterations": 1}) == "✓ requirements clear (1 iter)"
    assert "retrying (1/3)" in cli._line("gather", {"intent_clear": False, "intent_iterations": 1})
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


def test_gather_lines():
    import cli

    assert "(assumed)" in cli._line("gather", {"intent_clear": False, "intent_iterations": 1, "assumed": True})


def _capture(monkeypatch):
    import io

    from rich.console import Console

    import cli

    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=200, force_terminal=False))
    return buf


def _strip_ansi(text: str) -> str:
    """Drop SGR codes so plain-text assertions survive rich's highlighter."""
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_render_code_panel(monkeypatch):
    import cli

    buf = _capture(monkeypatch)
    cli._render_code("def solve(a):\n    return a\n", "code — attempt 1 (pass)")
    out = buf.getvalue()
    assert "attempt 1" in out
    assert "def solve" in out


def test_render_tests_table_labels_holdout(monkeypatch):
    import cli

    buf = _capture(monkeypatch)
    cli._render_tests(
        [{"id": "t1", "harness": "print(solve(1))", "expected": "1"}],
        [{"id": "h1", "harness": "print(solve(2))", "expected": "2"}],
    )
    out = buf.getvalue()
    assert "t1" in out and "h1" in out
    assert "holdout" in out and "visible" in out


def test_render_results_shows_io_and_result(monkeypatch):
    import cli

    buf = _capture(monkeypatch)
    cli._render_results([
        {"test_id": "t1", "harness": "print(solve(1))", "expected": "1", "actual": "1", "passed": True, "blame": "none", "error": ""},
        {"test_id": "t2", "harness": "print(solve(2))", "expected": "5", "actual": "4", "passed": False, "blame": "code", "error": ""},
    ])
    out = buf.getvalue()
    assert "print(solve(1))" in out and "print(solve(2))" in out
    assert "✓" in out and "✗" in out
    assert "code" in out
    assert cli._render_results([]) is None  # empty -> nothing printed


def test_render_diagnosis_shows_culprit_and_reason(monkeypatch):
    import cli

    buf = _capture(monkeypatch)
    cli._render_diagnosis({"culprit": "code", "reason": "off-by-one at the loop bound", "failing": ["t2"]})
    out = buf.getvalue()
    assert "code" in out and "off-by-one" in out and "t2" in out
    assert cli._render_diagnosis({}) is None  # empty -> nothing printed


def test_render_plan_panel(monkeypatch):
    import cli

    buf = _capture(monkeypatch)
    cli._render_plan(
        ["Use a hash map", "Walk once"],
        "def solve(nums, target)",
        [{"name": "nums", "type": "List[int]", "range": "len 1..1000"},
         {"name": "target", "type": "int", "range": "0..100"}],
        "List[int]",
    )
    out = buf.getvalue()
    assert "1. Use a hash map" in out and "2. Walk once" in out
    assert "Contract: def solve(nums, target)" in out
    assert "nums: List[int], range: len 1..1000" in out
    assert "Returns: List[int]" in out


def test_run_graph_show_toggle(monkeypatch):
    import cli

    calls = []
    monkeypatch.setattr(cli, "_render_plan", lambda *a, **k: calls.append("plan"))
    monkeypatch.setattr(cli, "_render_code", lambda *a, **k: calls.append("code"))
    monkeypatch.setattr(cli, "_render_tests", lambda *a, **k: calls.append("tests"))
    monkeypatch.setattr(cli, "_render_results", lambda *a, **k: calls.append("results"))

    class FakeGraph:
        def stream(self, state, stream_mode=None):
            yield {"planner": {"plan_steps": ["a"]}}
            yield {"tests": {"test_cases": [], "holdout_tests": []}}
            yield {"coder": {"code": "def solve():\n    pass\n"}}

    cli._run_graph(FakeGraph(), {}, show=False)
    assert calls == []
    cli._run_graph(FakeGraph(), {}, show=True)
    assert "plan" in calls and "tests" in calls and "code" in calls


def test_main_quiet_disables_panels(monkeypatch):
    import cli

    seen = {}

    def fake_run_graph(graph, state, show=True):
        seen["show"] = show
        return {"output_path": "output/x.json"}

    monkeypatch.setattr(cli, "build_graph", lambda: object())
    monkeypatch.setattr(cli, "_run_graph", fake_run_graph)
    assert cli.main(["--problem", "Return sum of two ints.", "--quiet"]) == 0
    assert seen["show"] is False
    assert cli.main(["--problem", "Return sum of two ints."]) == 0
    assert seen["show"] is True


def test_next_node_predicts_routing():
    import cli

    assert cli._next_node(None, {}) == "gather"
    assert cli._next_node("gather", {"intent_clear": True}) == "planner"
    assert cli._next_node("gather", {"intent_clear": False, "intent_iterations": 1}) == "gather"
    assert cli._next_node("gather", {"rejected": True}) == "reporter"
    assert cli._next_node("planner", {}) == "tests"
    assert cli._next_node("planner", {"next_action": "replan_code"}) == "coder"
    assert cli._next_node("tests", {}) == "coder"
    assert cli._next_node("tests", {"next_action": "fix_tests"}) == "evaluator"
    assert cli._next_node("coder", {}) == "evaluator"
    assert cli._next_node("evaluator", {"eval": {"passed": True}}) == "reporter"
    assert cli._next_node("evaluator", {"eval": {"passed": False, "blame": "test"}}) == "tests"
    assert cli._next_node("evaluator", {"eval": {"passed": False, "blame": "code"}}) == "planner"
    assert cli._next_node("evaluator", {"eval": {"passed": False}, "iterations": 3}) == "reporter"
    assert cli._next_node("reporter", {}) is None


def test_narrator_shows_seconds_and_rotates():
    import cli

    t0 = cli._narrator("planner", 0).plain
    assert "planner:" in t0 and "(0s)" in t0
    # The phrase is stable within the 3s interval, but the timer keeps counting.
    assert cli._NARRATION["planner"][0] in cli._narrator("planner", 2).plain
    t7 = cli._narrator("planner", 7).plain
    assert "(7s)" in t7  # the timer advances
    assert cli._NARRATION["planner"][2] in t7  # int(7//3)=2 -> 3rd phrase
    assert cli._NARRATION["planner"][0] not in t7  # ...and the phrase rotated away
    assert "coder:" in cli._narrator("coder", 0).plain  # node-aware
    assert cli._narrator(None, 0).plain.startswith(" working:")  # unknown/None -> generic


def test_run_graph_live_path_narrates(monkeypatch):
    import io
    import re

    from rich.console import Console

    import cli

    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=200, force_terminal=True))

    class FakeGraph:
        def stream(self, state, stream_mode=None):
            yield {"gather": {"intent_clear": True, "intent_iterations": 1}}
            yield {"planner": {"plan_steps": ["a"], "function_signature": "def solve()"}}
            yield {"reporter": {"output_path": "output/z.json"}}

    final = cli._run_graph(FakeGraph(), {"run_id": "r", "requirement": "q"})
    assert final["output_path"] == "output/z.json"
    assert final["intent_clear"] is True
    # Live may erase the transient spinner line, but the node status lines persist.
    assert "requirements clear" in buf.getvalue()
    assert "output/z.json" in buf.getvalue()
    assert re.search(r"·\s*\d+\.\d+s", _strip_ansi(buf.getvalue()))  # each node line carries its duration


def test_run_graph_direct_prints_duration(monkeypatch):
    import re

    import cli

    buf = _capture(monkeypatch)  # force_terminal=False -> direct driver

    class FakeGraph:
        def stream(self, state, stream_mode=None):
            yield {"planner": {"plan_steps": ["a"]}}
            yield {"reporter": {"output_path": "output/q.json"}}

    cli._run_graph(FakeGraph(), {"run_id": "r"})
    out = _strip_ansi(buf.getvalue())
    assert "✓ plan" in out
    assert re.search(r"·\s*\d+\.\d+s", out)


def test_run_graph_live_path_surfaces_stream_error(monkeypatch):
    import io

    from rich.console import Console

    import cli

    buf = io.StringIO()
    monkeypatch.setattr(cli, "console", Console(file=buf, width=200, force_terminal=True))
    monkeypatch.setattr(cli, "_NARRATION_INTERVAL", 0.01)

    class BoomGraph:
        def stream(self, state, stream_mode=None):
            yield {"gather": {"intent_clear": True, "intent_iterations": 1}}
            raise RuntimeError("kaboom")

    final = cli._run_graph(BoomGraph(), {"run_id": "r"})
    assert final["intent_clear"] is True  # updates before the error are kept
    assert "kaboom" in buf.getvalue()


# --- gather-mode resolution, exit codes, batch driver (Increment 4) --------


def test_gather_lines_quiet_and_aborted():
    import cli

    assert cli._line("gather", {"gather_mode": "quiet", "status": "ok"}) == "✓ requirement validated (ok)"
    assert "assumed" in cli._line("gather", {"gather_mode": "quiet", "status": "assumed", "assumed": True})
    assert "aborted" in cli._line("gather", {"rejected": True, "rejection_class": "incomplete", "rejection_reason": "cut off"})
    assert "incomplete" in cli._line("gather", {"rejected": True, "rejection_class": "incomplete", "rejection_reason": "cut off"})
    assert "infra_error" in cli._line("gather", {"status": "infra_error"})


def test_main_gather_mode_resolution(monkeypatch):
    import cli

    seen = {}

    def fake_run_graph(graph, state, show=True):
        seen.update(state)
        return {"output_path": "output/x.json", "status": "ok"}

    monkeypatch.setattr(cli, "build_graph", lambda: object())
    monkeypatch.setattr(cli, "_run_graph", fake_run_graph)

    # non-TTY default => quiet
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli.main(["--problem", "Return sum of two ints."]) == 0
    assert seen["gather_mode"] == "quiet"

    # explicit won't be overridden
    assert cli.main(["--problem", "Return sum of two ints.", "--gather-mode", "interactive"]) == 0
    assert seen["gather_mode"] == "interactive"

    # TTY default => interactive
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert cli.main(["--problem", "Return sum of two ints."]) == 0
    assert seen["gather_mode"] == "interactive"
    assert seen["out_dir"] == "output"


def test_main_infra_error_exit_code(monkeypatch):
    import cli

    monkeypatch.setattr(cli, "build_graph", lambda: object())
    monkeypatch.setattr(cli, "_run_graph", lambda g, s, show=True: {"status": "infra_error"})
    assert cli.main(["--problem", "Given an array, return max.", "--gather-mode", "quiet"]) == 1


def test_main_aborted_exit_code_zero(monkeypatch):
    """Aborts are legit outcomes (the reason lives in the JSON) -> exit 0."""
    import cli

    monkeypatch.setattr(cli, "build_graph", lambda: object())
    monkeypatch.setattr(cli, "_run_graph", lambda g, s, show=True: {"status": "aborted"})
    assert cli.main(["--problem", "Given an array, return max.", "--gather-mode", "quiet"]) == 0


def test_main_stream_exception_exit_code_one(monkeypatch):
    import cli

    def boom(graph, state, show=True):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(cli, "build_graph", lambda: object())
    monkeypatch.setattr(cli, "_run_graph", boom)
    assert cli.main(["--problem", "Given an array, return max."]) == 1


def test_batch_write_and_summary(tmp_path, monkeypatch):
    import glob
    import json
    from pathlib import Path

    import cli

    monkeypatch.chdir(tmp_path)
    problems = tmp_path / "problems.jsonl"
    problems.write_text(
        json.dumps({"id": "a", "requirement": "Given an array, return max."}) + "\n"
        + json.dumps({"id": "b", "requirement": "Given an array, return min."}) + "\n"
    )

    statuses = {"a": ("ok", ""), "b": ("aborted", "under_specified")}

    class FakeGraph:
        def stream(self, state, stream_mode=None):
            status, cls = statuses[state["run_id"]]
            yield {"gather": {"gather_mode": "quiet", "status": status, "rejection_class": cls}}
            yield {"reporter": {"output_path": "x"}}

    monkeypatch.setattr(cli, "build_graph", lambda: FakeGraph())
    assert cli.main(["--problems-file", str(problems)]) == 0

    summaries = glob.glob("output/batch_*/summary.json")
    assert len(summaries) == 1
    doc = json.loads(Path(summaries[0]).read_text())
    assert doc["total"] == 2 and doc["skipped"] == 0
    assert doc["status_counts"] == {"ok": 1, "aborted": 1}
    assert doc["rejection_class_counts"] == {"under_specified": 1}
    assert doc["yield"] == {"completed": 1, "aborted": 1, "infra": 0}
    assert doc["gather_mode"] == "quiet"


def test_batch_continues_after_infra_failure(tmp_path, monkeypatch):
    import glob
    import json
    from pathlib import Path

    import cli

    monkeypatch.chdir(tmp_path)
    problems = tmp_path / "problems.jsonl"
    problems.write_text(
        json.dumps({"id": "boom", "requirement": "Given an array, return max."}) + "\n"
        + "{ not valid json\n"
        + json.dumps({"id": "fine", "requirement": "Given an array, return min."}) + "\n"
    )

    class FakeGraph:
        def stream(self, state, stream_mode=None):
            if state["run_id"] == "boom":
                raise RuntimeError("kaboom")
            yield {"gather": {"gather_mode": "quiet", "status": "ok"}}
            yield {"reporter": {"output_path": "x"}}

    monkeypatch.setattr(cli, "build_graph", lambda: FakeGraph())
    assert cli.main(["--problems-file", str(problems), "--quiet"]) == 0

    doc = json.loads(Path(glob.glob("output/batch_*/summary.json")[0]).read_text())
    assert doc["total"] == 2 and doc["skipped"] == 1
    assert doc["status_counts"]["infra_error"] == 1
    assert doc["status_counts"]["ok"] == 1


def test_batch_requires_quiet_gathering(tmp_path, monkeypatch):
    import cli

    monkeypatch.chdir(tmp_path)
    problems = tmp_path / "problems.jsonl"
    problems.write_text('{"id": "a", "requirement": "x"}\n')
    monkeypatch.setattr(cli, "build_graph", lambda: object())
    assert cli.main(["--problems-file", str(problems), "--gather-mode", "interactive"]) == 2


def test_batch_missing_file_exit_two(tmp_path, monkeypatch, capsys):
    import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "build_graph", lambda: object())
    assert cli.main(["--problems-file", str(tmp_path / "nope.jsonl")]) == 2

