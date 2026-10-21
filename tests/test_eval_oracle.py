from __future__ import annotations

from evals.oracle import audit_blame, check_generated_tests, score_test_oracle


def _fake_runner(source, harness):
    value = 42 if "wrong" not in source else 0
    return {"stdout": f"__RESULT__ {value}\n", "error": ""}


def test_generated_test_oracle_and_holdout_rates():
    tests = [
        {"id": "valid", "harness": "print(solve())", "expected": 42},
        {"id": "wrong", "harness": "print(solve())", "expected": 7},
    ]
    scored = score_test_oracle(
        "def solve(): return 42",
        tests[:1],
        tests[1:],
        run_harness=_fake_runner,
    )
    assert scored["oracle_correctness"] == 0.5
    assert scored["oracle_correctness_agent_tests"] == 1
    assert scored["oracle_correctness_holdout_tests"] == 0


def test_repaired_test_is_deterministically_grounded_as_test_blame():
    oracle = check_generated_tests(
        "def solve(): return 42",
        [
            {"id": "wrong", "harness": "print(solve())", "expected": 7},
            {"id": "repaired", "harness": "print(solve())", "expected": 42, "repaired": True},
        ],
        run_harness=_fake_runner,
    )
    audit = audit_blame(
        [
            {"test_id": "wrong", "passed": False, "blame": "code"},
            {"test_id": "repaired", "passed": False, "blame": "test", "repaired": True},
        ],
        oracle["tests"],
        {"culprit": "test"},
    )
    assert audit["accuracy"] == 0.5
    assert audit["confusion_matrix"]["test_correct"] == 1
    assert audit["misblamed"] == [
        {
            "test_id": "wrong",
            "predicted": "code",
            "ground_truth": "test",
            "oracle_ok": False,
            "repaired": False,
        }
    ]
    assert audit["failure_report_correct"] is True
