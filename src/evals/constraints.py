"""Static, deterministic constraint checks configured by problem ID."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def load_constraint_rules(path: str | Path = "evals/constraints.yml") -> dict[str, list[dict[str, Any]]]:
    source = Path(path)
    if not source.exists():
        return {}
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(  # noqa: TRY004 - invalid configuration is a value error
            f"{source}: expected a mapping of problem IDs to rule lists"
        )
    normalized: dict[str, list[dict[str, Any]]] = {}
    for problem_id, rules in raw.items():
        if not isinstance(rules, list) or any(not isinstance(rule, dict) for rule in rules):
            raise ValueError(f"{source}: rules for problem {problem_id!r} must be a list of mappings")
        normalized[str(problem_id)] = rules
    return normalized


def _operator_names(tree: ast.AST) -> set[str]:
    return {type(node.op).__name__ for node in ast.walk(tree) if isinstance(node, ast.BinOp)}


def _matrix_write(tree: ast.AST, name: str) -> bool:
    def root_name(node: ast.AST) -> str | None:
        while isinstance(node, ast.Subscript):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if hasattr(node, "target") else node.targets
            for target in targets:
                if isinstance(target, ast.Subscript) and root_name(target) == name:
                    return True
    return False


def _matches(tree: ast.AST, rule: Mapping[str, Any]) -> bool:
    selector = str(rule.get("selector", ""))
    if selector.startswith("operator:"):
        operator_names = {name.strip() for name in selector.split(":", 1)[1].split(",")}
        return bool(operator_names & _operator_names(tree))
    if selector.startswith("matrix_write:"):
        return _matrix_write(tree, selector.split(":", 1)[1])
    if selector.startswith("name:"):
        name = selector.split(":", 1)[1]
        return any(isinstance(node, ast.Name) and node.id == name for node in ast.walk(tree))
    raise ValueError(f"unsupported constraint selector: {selector!r}")


def check_constraints(
    problem_id: str | int,
    code: str,
    rules: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {
            "problem_id": str(problem_id),
            "adherent": False,
            "results": [
                {
                    "rule_id": "syntax",
                    "passed": False,
                    "reason": f"solution syntax error: {exc.msg}",
                }
            ],
        }

    results = []
    for rule in rules.get(str(problem_id), []):
        kind = rule.get("kind")
        matches = _matches(tree, rule)
        if kind == "ast_forbid":
            passed = not matches
        elif kind == "ast_require":
            passed = matches
        else:
            raise ValueError(f"unsupported constraint rule kind: {kind!r}")
        results.append(
            {
                "rule_id": str(rule.get("rule_id", f"{problem_id}:{len(results) + 1}")),
                "passed": passed,
                "reason": (
                    str(rule.get("description") or f"{kind} {rule.get('selector')}")
                    if passed
                    else str(rule.get("failure_reason") or f"rule {kind} violated")
                ),
            }
        )
    return {
        "problem_id": str(problem_id),
        "adherent": all(result["passed"] for result in results),
        "applicable_rules": len(results),
        "passed_rules": sum(result["passed"] for result in results),
        "results": results,
    }
