from __future__ import annotations

from agent.tools.guardrail import check
from evals.harness import build_harness, parse_harness_output


def test_harness_supports_positional_and_keyword_inputs():
    assert build_harness("two_sum", {"input": {"nums": [2, 7], "target": 9}}) == (
        "print('__RESULT__', two_sum(**{'nums': [2, 7], 'target': 9}))"
    )
    assert build_harness("merge", {"args": [[1], [2]]}) == (
        "print('__RESULT__', merge([1], [2]))"
    )


def test_harness_preserves_path_like_string_inputs():
    harness = build_harness("echo", {"input": "/a/b", "expected": "/a/b"})
    assert check("def echo(x):\n    return x", harness)["allowed"] is False
    assert check("def echo(x):\n    return x", harness, trusted_harness=True)["allowed"] is True
    assert check('def echo(x):\n    return "/a/b"', harness, trusted_harness=True)["allowed"] is False


def test_harness_output_parsing_uses_last_marker_and_literal_types():
    assert parse_harness_output("noise\n__RESULT__ [1, True, 'x']\n") == [1, True, "x"]
    assert parse_harness_output("__RESULT__ \n") == ""


def test_harness_output_parsing_rejects_missing_or_invalid_marker():
    import pytest

    with pytest.raises(ValueError, match="missing __RESULT__"):
        parse_harness_output("nothing here")
    with pytest.raises(ValueError, match="not a Python literal"):
        parse_harness_output("__RESULT__ object()")
