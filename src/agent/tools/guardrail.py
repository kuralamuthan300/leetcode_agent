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

DENY_NAMES = frozenset(    {
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


# Strict Python-only policy: any other requested language is rejected.
EXPLICIT_LANGS = (
    "java", "javascript", "typescript", "sql", "golang", "rust", "kotlin",
    "swift", "scala", "php", "ruby",
)
# single strong markers of non-Python code
STRONG_MARKERS = (
    "public class", "public static void", "system.out", "jdbc",
    "console.log", "#include", "using namespace", "func main(",
    "fn main(", "c++", "c#",
)
# sql-ish keywords: needs 2+ distinct hits (each alone is common English)
SQL_KEYWORDS = (
    "select", "join", "group by", "order by", "insert into", "delete from",
    "create table", "drop table", "alter table", "union", "having",
)

PYTHON_HINTS = ("python", "def solve", "pip install")


# Pure-DSA scope policy: only data-structures & algorithms coding tasks pass.
DSA_MARKERS = (
    "array", "linked list", "binary tree", "binary search", "graph",
    "dynamic programming", "stack", "queue", "heap", "hash", "recursion",
    "iterative", "matrix", "matrices", "palindrome", "anagram",
    "sliding window", "two pointer", "bfs", "dfs", "backtracking", "greedy",
    "divide and conquer", "big-o", "complexity", "nodes", "edges",
    "indices", "leetcode", "sort", "search", "traverse", "traversal",
)

NONCODING_MARKERS = (
    "essay", "poem", "story", "email", "letter", "resume", "cover letter",
    "summarize", "summary", "translate", "weather", "news", "recipe",
    "workout", "diet", "travel plans", "itinerary", "joke", "lyrics",
    "blog post", "tweet", "caption", "grocery",
)


def check_scope(text: str) -> dict:
    """Return {allowed: True | False | None, reason}.

    True/False are heuristic fast paths; None means gray zone — caller
    must ask the small-LLM classifier (fail-closed: LLM failure rejects).
    """
    t = text.strip().lower()
    for marker in NONCODING_MARKERS:
        if marker in t:
            return {"allowed": False, "reason": f"not a DSA coding task ({marker})"}
    for marker in DSA_MARKERS:
        if marker in t:
            return {"allowed": True, "reason": ""}
    return {"allowed": None, "reason": "ambiguous scope"}


TRUNCATION_TAILS = frozenset(
    {"of", "in", "by", "to", "with", "for", "on", "at", "from", "as", "and", "or", "a", "an", "the"}
)


def is_truncated(text: str) -> bool:
    """Detect cut-off input: ends mid-phrase (preposition/article) or dangling comma."""
    import re
    import string

    s = text.strip()
    if not s:
        return True
    if s[-1] in ",;:" or s.endswith("..."):
        return True
    last = re.split(r"\s+", s)[-1].strip(string.punctuation).lower()
    return last in TRUNCATION_TAILS

def check_language(text: str) -> dict:
    """Return {allowed, reason}. Neutral DSA (no language cues) defaults to Python."""
    import re

    t = text.strip().lower()
    if any(h in t for h in PYTHON_HINTS):
        return {"allowed": True, "reason": ""}
    for lang in EXPLICIT_LANGS:
        if re.search(r"\b" + re.escape(lang) + r"\b", t):
            return {"allowed": False, "reason": f"non-Python language requested ({lang})"}
    if re.search(r"\b(in|using|with)\s+go\b", t):
        return {"allowed": False, "reason": "non-Python language requested (go)"}
    for marker in STRONG_MARKERS:
        if marker in t:
            return {"allowed": False, "reason": f"non-Python code requested ({marker})"}
    hits = [k for k in SQL_KEYWORDS if k in t]
    if len(hits) >= 2:
        return {"allowed": False, "reason": f"SQL query requested ({', '.join(hits[:3])})"}
    return {"allowed": True, "reason": ""}
