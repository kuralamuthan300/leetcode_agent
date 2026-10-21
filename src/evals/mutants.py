"""Seeded deterministic AST mutations for reference implementations."""

from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass

_CMP_SWAP = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}
_BINOP_SWAP = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
}


@dataclass(frozen=True, slots=True)
class Mutant:
    mutant_id: str
    source: str
    mutation: str


def _mutation_for(node: ast.AST, parent: ast.AST | None) -> list[tuple[ast.AST, str]]:
    result: list[tuple[ast.AST, str]] = []
    if isinstance(node, ast.Compare):
        for index, operator in enumerate(node.ops):
            replacement = _CMP_SWAP.get(type(operator))
            if replacement:
                mutated = copy.deepcopy(node)
                mutated.ops[index] = replacement()
                result.append((mutated, f"comparison:{type(operator).__name__}"))
    elif isinstance(node, ast.BinOp):
        replacement = _BINOP_SWAP.get(type(node.op))
        if replacement:
            mutated = copy.deepcopy(node)
            mutated.op = replacement()
            result.append((mutated, f"arithmetic:{type(node.op).__name__}"))
    elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        mutated = copy.deepcopy(node)
        mutated.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        result.append((mutated, "boolean:and_or"))
    elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        if isinstance(parent, (ast.Compare, ast.Subscript)):
            mutated = copy.deepcopy(node)
            mutated.value = 1 if node.value == 0 else node.value - 1
            result.append((mutated, "off_by_one:constant"))
    elif isinstance(node, ast.Return) and node.value is not None:
        mutated = copy.deepcopy(node)
        mutated.value = ast.Constant(value=None)
        result.append((mutated, "return:none"))
        negated = copy.deepcopy(node)
        negated.value = ast.UnaryOp(op=ast.Not(), operand=negated.value)
        result.append((negated, "return:not"))
    elif isinstance(node, ast.Break):
        result.append((ast.Pass(), "break:removed"))
    return result


def generate_mutants(source: str, *, limit: int = 5) -> list[Mutant]:
    """Generate at most `limit` syntactically valid, stable mutants."""
    if limit < 0:
        raise ValueError("limit must not be negative")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"reference source is not valid Python: {exc}") from exc
    parents: dict[ast.AST, ast.AST] = {
        child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)
    }
    mutants: list[Mutant] = []
    seen: set[str] = set()
    for original in ast.walk(tree):
        parent = parents.get(original)
        for replacement, mutation in _mutation_for(original, parent):
            mutated = copy.deepcopy(tree)
            target_index = next(
                index for index, node in enumerate(ast.walk(tree)) if node is original
            )
            target = list(ast.walk(mutated))[target_index]
            if target is tree:
                continue
            parents_mutated = {
                child: parent for parent in ast.walk(mutated) for child in ast.iter_child_nodes(parent)
            }
            target_parent = parents_mutated.get(target)
            if target_parent is None:
                continue
            for field, value in ast.iter_fields(target_parent):
                if isinstance(value, list):
                    for index, item in enumerate(value):
                        if item is target:
                            value[index] = copy.deepcopy(replacement)
                            break
                elif value is target:
                    setattr(target_parent, field, copy.deepcopy(replacement))
                    break
            try:
                candidate = ast.unparse(ast.fix_missing_locations(mutated))
                compile(candidate, "<mutant>", "exec")
            except (SyntaxError, ValueError):
                continue
            digest = hashlib.sha256(candidate.encode()).hexdigest()[:12]
            if digest in seen:
                continue
            seen.add(digest)
            mutants.append(Mutant(f"mutant-{digest}", candidate, mutation))
            if len(mutants) >= limit:
                return mutants
    return mutants
