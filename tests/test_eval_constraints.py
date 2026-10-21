from __future__ import annotations

from evals.complexity import estimate_complexity
from evals.constraints import check_constraints, load_constraint_rules


def test_constraint_rule_engine_forbids_and_requires_ast_patterns():
    rules = {
        "2": [
            {
                "rule_id": "no-division",
                "kind": "ast_forbid",
                "selector": "operator:Div,FloorDiv",
            }
        ],
        "25": [
            {
                "rule_id": "in-place-matrix",
                "kind": "ast_require",
                "selector": "matrix_write:matrix",
            }
        ],
    }
    assert not check_constraints("2", "def solve(x): return x // 2", rules)["adherent"]
    assert not check_constraints("2", "def solve(x): return x / 2", rules)["adherent"]
    assert check_constraints("2", "def solve(x): return x + 2", rules)["adherent"]
    assert check_constraints(
        "25",
        "def rotate(matrix):\n    matrix[0][0] = 1",
        rules,
    )["adherent"]
    assert not check_constraints("25", "def rotate(matrix): return matrix[:]", rules)["adherent"]


def test_constraint_config_loads_problem_keyed_rules(tmp_path):
    path = tmp_path / "constraints.yml"
    path.write_text(
        "2:\n"
        "  - rule_id: no-div\n"
        "    kind: ast_forbid\n"
        "    selector: operator:Div\n"
    )
    assert load_constraint_rules(path)["2"][0]["rule_id"] == "no-div"


def test_complexity_estimate_is_explicitly_advisory():
    linear = estimate_complexity("def solve(items):\n    for item in items: pass", "O(N)")
    quadratic = estimate_complexity(
        "def solve(items):\n    for x in items:\n        for y in items: pass",
        "O(N)",
    )
    unknown = estimate_complexity("not valid python !", "O(N)")
    assert linear["status"] == "match"
    assert linear["advisory"] is True
    assert quadratic["status"] == "worse"
    assert unknown["status"] == "unknown"
