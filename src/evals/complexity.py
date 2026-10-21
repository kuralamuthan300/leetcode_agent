"""Advisory static complexity estimate from Python syntax."""

from __future__ import annotations

import ast
import re
from typing import Any

_TARGET_RE = re.compile(r"O\s*\(\s*([^)]*)\)")


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.max_loop_depth = 0
        self.loop_depth = 0
        self.recursive = False
        self.function_names: set[str] = set()
        self.current_function: list[str] = []
        self.uses_sort = False
        self.uses_heap = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_names.add(node.name)
        self.current_function.append(node.name)
        self.generic_visit(node)
        self.current_function.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in {"sorted"}:
                self.uses_sort = True
            if self.current_function and node.func.id == self.current_function[-1]:
                self.recursive = True
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == "sort":
                self.uses_sort = True
            if node.func.attr.startswith("heapp"):
                self.uses_heap = True
        self.generic_visit(node)


def estimate_complexity(code: str, target: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"status": "unknown", "estimate": None, "target": target, "reason": exc.msg}
    visitor = _ComplexityVisitor()
    visitor.visit(tree)

    target_match = _TARGET_RE.search(target)
    target_expr = target_match.group(1).replace(" ", "").upper() if target_match else ""
    if not target_expr:
        estimate = "O(N)" if visitor.max_loop_depth == 1 else None
        return {
            "status": "unknown",
            "estimate": estimate,
            "target": target,
            "reason": "target complexity could not be parsed",
        }
    if visitor.max_loop_depth >= 3:
        estimate = "O(N^3)"
    elif visitor.max_loop_depth == 2:
        estimate = "O(N^2)"
    elif visitor.max_loop_depth == 1:
        estimate = "O(N)"
    else:
        estimate = "O(1)"
    if visitor.recursive:
        estimate = "O(log N)" if "MID" in ast.dump(tree).upper() else "unknown"
    if visitor.uses_sort or visitor.uses_heap:
        estimate = "O(N log N)"

    def rank(expr: str) -> int | None:
        if expr in {"1", "1.0"}:
            return 0
        if "LOG" in expr and "N" in expr:
            return 1
        if expr in {"N", "NLOGN"}:
            return 2 if expr == "N" else 3
        if re.fullmatch(r"N\^([2-9])", expr):
            return int(expr[-1]) + 2
        return None

    estimate_expr = estimate.removeprefix("O(").removesuffix(")").replace(" ", "")
    target_rank = rank(target_expr)
    estimate_rank = rank(estimate_expr)
    if target_rank is None or estimate_rank is None:
        status = "unknown"
    elif estimate_rank == target_rank:
        status = "match"
    elif estimate_rank < target_rank:
        status = "better"
    else:
        status = "worse"
    return {
        "status": status,
        "estimate": estimate,
        "target": target,
        "advisory": True,
        "loop_nesting_depth": visitor.max_loop_depth,
        "recursive": visitor.recursive,
        "uses_sort": visitor.uses_sort,
        "uses_heap": visitor.uses_heap,
    }
