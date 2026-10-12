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
        def invoke(self, state):
            seen.update(state)
            return {"output_path": "output/x.json"}

    monkeypatch.setattr(cli, "build_graph", lambda: FakeGraph())
    assert cli.main([]) == 0
    assert seen["requirement"].strip() == "Sort the array."


def test_main_empty_requirement_rejected(tmp_path, monkeypatch, capsys):
    import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    assert cli.main([]) == 2
