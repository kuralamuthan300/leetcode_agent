from __future__ import annotations

from evals.mutants import generate_mutants
from evals.mutation import score_mutants


def test_mutants_are_deterministic_and_syntactically_valid():
    source = "def solve(x):\n    if x < 10:\n        return x + 1\n    return 0"
    first = generate_mutants(source, limit=5)
    second = generate_mutants(source, limit=5)
    assert first == second
    assert len(first) == 5


def test_reference_has_zero_mutation_kills_and_weak_tests_leave_survivors():
    reference = "def solve(x):\n    if x < 2:\n        return x + 1\n    return 0"
    assert len(generate_mutants(reference, limit=5)) == 5

    def run_harness(source, harness):
        if source == reference:
            return {"stdout": "__RESULT__ 0\n", "error": ""}
        return {"stdout": "__RESULT__ 1\n", "error": ""}

    # The unchanged reference matches its own expected answer; no mutants passed
    # to the test scorer can be confused with reference-equivalence.
    score = score_mutants(
        reference,
        [],
        [{"harness": "print(solve(5))", "expected": 0}],
        [{"id": "weak", "harness": "print(solve(5))", "expected": 0}],
        run_harness=run_harness,
    )
    assert score["killable"] == 0
    assert score["mutation_score"] is None


def test_killable_mutants_are_killed_by_behavioral_tests():
    reference = "def solve(x):\n    return x + 1"
    mutants = generate_mutants(reference, limit=5)

    def run_harness(source, harness):
        result = 2 if source == reference else 1
        return {"stdout": f"__RESULT__ {result}\n", "error": ""}

    score = score_mutants(
        reference,
        mutants,
        [{"harness": "print(solve(1))", "expected": 2}],
        [{"id": "case", "harness": "print(solve(1))", "expected": 2}],
        run_harness=run_harness,
    )
    assert score["killable"] > 0
    assert score["mutation_score"] == 1
