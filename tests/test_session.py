import pytest


@pytest.fixture(autouse=True)
def _tmp_session(tmp_path, monkeypatch):
    monkeypatch.setenv("LEETCODEAGENT_SESSION", str(tmp_path / "session.json"))


def test_session_roundtrip():
    from agent import session

    assert session.load() == []
    session.remember(["arrays are 1-D unless stated"])
    assert session.load() == ["arrays are 1-D unless stated"]
    session.clear()
    assert session.load() == []


def test_remembered_conventions_applied(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "ok")
    from agent import session
    from agent.nodes import requirement_gatherer

    session.remember(["arrays are 1-D unless stated"])
    out = requirement_gatherer({"requirement": "Rotate an array by k positions right."})
    assert out["confirmed"] is True
    assert any("1-D" in n for n in out["session_notes"])


def test_fresh_flag_clears(tmp_path, monkeypatch, capsys):
    import cli
    from agent import session

    monkeypatch.chdir(tmp_path)
    session.remember(["stale convention"])
    assert session.load() == ["stale convention"]
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    answers = iter(["Sort the array of ints.", "", "ok"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    seen = {}

    class FakeGraph:
        def stream(self, state, stream_mode=None):
            seen.update(state)
            yield {"reporter": {"output_path": "output/x.json"}}

    monkeypatch.setattr(cli, "build_graph", lambda: FakeGraph())
    assert cli.main(["--fresh"]) == 0
    assert "stale convention" not in session.load()
    assert "Session memory cleared" in capsys.readouterr().out
    assert seen["requirement"].strip() == "Sort the array of ints."
