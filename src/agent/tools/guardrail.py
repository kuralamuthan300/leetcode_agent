"""Static security gate for coder-written code. Fail-closed.

`check()` must pass before any `run_python` exec. Blocks: malicious modules,
file/network/OS access, introspection escapes, and oversized payloads.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone

MAX_CODE_CHARS = 15_000
MAX_HARNESS_CHARS = 5_000

IMPORT_ALLOW = frozenset(
    {"math", "typing", "collections", "itertools", "functools", "heapq", "bisect"}
)

DENY_NAMES = frozenset(
    {
        # file / IO
        "open", "file", "input",
        # dynamic exec / import escapes
        "eval", "exec", "compile", "__import__", "importlib",
        # introspection escapes
        "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir",
        "breakpoint", "help", "exit", "quit",
        # dangerous modules (also blocked at import)
        "os", "sys", "subprocess", "socket", "pathlib", "shutil", "io",
        "codecs", "pickle", "inspect", "ctypes", "threading", "multiprocessing",
        "concurrent", "signal", "builtins",
    }
)


def _check_tree(tree: ast.AST) -> list[str]:
    reasons: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in IMPORT_ALLOW:
                    reasons.append(f"blocked import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in IMPORT_ALLOW:
                reasons.append(f"blocked import: {node.module or '(relative)'}")
        elif isinstance(node, ast.Name) and node.id in DENY_NAMES:
            reasons.append(f"blocked name: {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                reasons.append(f"blocked dunder attribute: {node.attr}")
            elif node.attr in DENY_NAMES:
                reasons.append(f"blocked attribute: {node.attr}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            if v.startswith("/") or v.startswith("~") or ".." in v:
                reasons.append(f"blocked path-like string: {v[:40]!r}")
    return reasons


def check(code: str, harness: str = "") -> dict:
    """Return {allowed, reasons}. Never raises; unparseable size violations fail closed."""
    reasons: list[str] = []
    if len(code) > MAX_CODE_CHARS:
        reasons.append(f"code exceeds {MAX_CODE_CHARS} chars")
    if len(harness) > MAX_HARNESS_CHARS:
        reasons.append(f"harness exceeds {MAX_HARNESS_CHARS} chars")
    for label, src in (("code", code), ("harness", harness)):
        if not src.strip():
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            reasons.append(f"{label} syntax error: {e}")
            continue
        reasons.extend(f"{label}: {r}" for r in _check_tree(tree))
    return {"allowed": not reasons, "reasons": reasons}


def report(code: str, harness: str = "") -> dict:
    """check() plus timestamp, for state/output (`guardrail_report`)."""
    result = check(code, harness)
    result["checked_at"] = datetime.now(timezone.utc).isoformat()
    return result
