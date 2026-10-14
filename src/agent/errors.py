"""Error classification + smoke-call dummy synthesis for Step 1.

Division of labor:
- compile/exec already surface SyntaxError / ImportError / top-level NameError.
- Signature AST checks catch arity mismatches without execution.
- The smoke-*call* exists to catch NameErrors hiding inside function bodies.
  Other call-time exceptions on synthetic dummies are recorded, not failed —
  dummy inputs are usually type-wrong, so failing on them would "repair" good code.
"""

from __future__ import annotations

MECHANICAL_MARKERS = ("SyntaxError", "IndentationError", "ImportError", "ModuleNotFoundError", "NameError")


def classify_error(error: str) -> str:
    """none | blocked | timeout | mechanical | logical."""
    if not error:
        return "none"
    if error.startswith("blocked:"):
        return "blocked"
    if error.startswith("timeout"):
        return "timeout"
    if any(m in error for m in MECHANICAL_MARKERS):
        return "mechanical"
    return "logical"


def is_call_failure(error: str) -> bool:
    """Only NameError/ImportError from a smoke-call fail the coder."""
    return "NameError" in error or "ImportError" in error or "ModuleNotFoundError" in error


def dummy_args(sig: str) -> list[str]:
    """Best-effort dummy arg sources from a `def solve(...)` signature line."""
    import ast

    try:
        tree = ast.parse(sig + ":\n    pass")
    except SyntaxError:
        return []
    fn = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)), None)
    if not fn:
        return []
    out = []
    for a in fn.args.args:
        ann = ast.unparse(a.annotation) if a.annotation else ""
        low = ann.lower()
        if "str" in low:
            out.append('""')
        elif "float" in low:
            out.append("0.0")
        elif "bool" in low:
            out.append("False")
        elif "dict" in low:
            out.append("{}")
        elif "list" in low or "tuple" in low or "set" in low:
            out.append("[]")
        elif "int" in low:
            out.append("0")
        else:
            out.append("None")
    return out
