def test_classify_error():
    from agent.errors import classify_error

    assert classify_error("") == "none"
    assert classify_error("blocked: open") == "blocked"
    assert classify_error("timeout after 5s") == "timeout"
    assert classify_error("SyntaxError('x')") == "mechanical"
    assert classify_error("NameError(\"x\")") == "mechanical"
    assert classify_error("ImportError('os')") == "mechanical"
    assert classify_error("TypeError('bad')") == "logical"
    assert classify_error("AssertionError") == "logical"


def test_dummy_args():
    from agent.errors import dummy_args

    assert dummy_args("def solve(nums: List[int], target: int)") == ["[]", "0"]
    assert dummy_args("def solve(s: str)") == ['""']
    assert dummy_args("def solve(a)") == ["None"]
    assert dummy_args("not a def") == []


def test_smoke_call_catches_body_nameerror():
    from agent.nodes import _smoke

    code = "def solve(a):\n    return undefined_name + a\n"
    smoke, _ = _smoke(code, "def solve(a)")
    assert "NameError" in smoke["error"]


def test_smoke_call_ignores_dummy_type_errors():
    from agent.nodes import _smoke

    # enumerate(None) TypeErrors on dummy input: noted, not failed.
    code = "def solve(nums):\n    return [x for x in enumerate(nums)]\n"
    smoke, _ = _smoke(code, "def solve(nums)")
    assert smoke["error"] == ""
    assert "note" in smoke


def test_smoke_skips_call_without_signature():
    from agent.nodes import _smoke

    smoke, _ = _smoke("# TODO: implement solution\n", "")
    assert smoke["error"] == ""
    assert "note" not in smoke
