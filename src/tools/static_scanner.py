"""Pre-run AST guard: block malicious code before Docker ever sees it (Step 3).

Policy (mirrors AGENTS.md security invariants + config/sandbox.yaml):
- Imports: only the allowlist is accepted. Banned modules get a
  `blocked-import:<mod>` flag, anything else outside the allowlist gets
  `non-allowlisted-import:<mod>`. Both fail closed (safe=False).
- Calls: eval, exec, __import__, compile, open, input, breakpointhook.
  All open() calls are flagged — solution code must be pure compute.
- Path strings: any string constant containing `..` or starting with `/`.
- Unparseable code (SyntaxError) is unsafe by default.
"""

from __future__ import annotations

import ast

ALLOWED_IMPORTS = frozenset(
    {"math", "heapq", "collections", "bisect", "itertools", "functools", "typing"}
)

BANNED_IMPORTS = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "shutil",
        "pathlib",
        "ctypes",
        "threading",
        "multiprocessing",
        "inspect",
    }
)

BANNED_CALLS = frozenset(
    {"eval", "exec", "__import__", "compile", "open", "input", "breakpointhook"}
)


def _top_level(module: str | None) -> str | None:
    if not module:
        return None
    return module.split(".", 1)[0]


def scan(code: str) -> dict:
    """Scan Python source without running it. Returns {safe: bool, flags: list}."""
    flags: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"safe": False, "flags": [f"syntax-error:{exc}"]}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = _top_level(alias.name)
                if top in BANNED_IMPORTS:
                    flags.append(f"blocked-import:{top}")
                elif top not in ALLOWED_IMPORTS:
                    flags.append(f"non-allowlisted-import:{top}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                flags.append("relative-import")
                continue
            top = _top_level(node.module)
            if top in BANNED_IMPORTS:
                flags.append(f"blocked-import:{top}")
            elif top not in ALLOWED_IMPORTS:
                flags.append(f"non-allowlisted-import:{top}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in BANNED_CALLS:
                flags.append(f"blocked-call:{func.id}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if ".." in node.value or node.value.startswith("/"):
                flags.append(f"path-escape:{node.value[:80]}")
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if (
                    isinstance(part, ast.Constant)
                    and isinstance(part.value, str)
                    and (".." in part.value or part.value.startswith("/"))
                ):
                    flags.append(f"path-escape:{part.value[:80]}")

    return {"safe": not flags, "flags": flags}
