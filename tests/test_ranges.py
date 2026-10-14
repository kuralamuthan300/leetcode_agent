def test_parse_range():
    from agent.prompts import parse_range

    assert parse_range("0..10") == {"min": 0, "max": 10}
    assert parse_range("len 1..100000, values -1000000000..1000000000") == {
        "min_len": 1, "max_len": 100000, "min": -1000000000, "max": 1000000000}
    assert parse_range("0..10**9")["max"] == 10 ** 9
    assert parse_range("non-empty string")["min_len"] == 1
    assert parse_range("sorted array") == {}
    assert parse_range("") == {}


def test_args_in_range():
    from agent.prompts import args_in_range

    spec = [
        {"name": "arr", "type": "List[int]", "range": "len 1..5, values 0..10"},
        {"name": "k", "type": "int", "range": "0..10"},
    ]
    assert args_in_range(spec, "print(solve([1, 2], 3))") is True
    assert args_in_range(spec, "print(solve([1, 2], 12))") is False  # k out of range
    assert args_in_range(spec, "print(solve([], 3))") is False  # empty disallowed
    assert args_in_range(spec, "print(solve([1, 99], 3))") is False  # value out of range
    assert args_in_range(spec, "print(solve(make_arr(), k))") is True  # non-literal skips
    assert args_in_range([], "print(solve(1))") is True  # no spec skips


def test_valid_test_enforces_ranges():
    from agent.nodes import _valid_test

    spec = [{"name": "k", "type": "int", "range": "0..10"}]
    ok = {"id": "t1", "harness": "print(solve(5))", "expected": "5"}
    bad = {"id": "t2", "harness": "print(solve(99))", "expected": "99"}
    assert _valid_test(ok, None, spec) is not None
    assert _valid_test(bad, None, spec) is None
