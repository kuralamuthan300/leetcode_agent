from __future__ import annotations

import json

from .guardrail import IMPORT_ALLOW, report as guardrail_report

TIMEOUT_S = 5
MAX_STDOUT_CHARS = 20_000
MEM_LIMIT_BYTES = 512 * 1024 * 1024

_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "pow", "print",
    "range", "reversed", "set", "sorted", "str", "sum", "super", "tuple",
    "zip", "Exception", "ValueError", "TypeError", "IndexError", "KeyError",
    "StopIteration", "__build_class__",
)


def _safe_import(name: str, *args, **kwargs):
    root = name.split(".")[0]
    if root not in IMPORT_ALLOW:
        raise ImportError(f"import blocked by guardrail: {name}")
    return __import__(name, *args, **kwargs)


def _exec_target(q, src: str, harness: str) -> None:
    import io
    import os
    import tempfile
    from contextlib import redirect_stderr, redirect_stdout

    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
    except Exception:
        pass
    try:
        os.chdir(tempfile.mkdtemp(prefix="sandbox_"))
    except Exception:
        pass

    import builtins as _builtins

    safe_ns = {n: getattr(_builtins, n) for n in _SAFE_BUILTIN_NAMES if hasattr(_builtins, n)}
    safe_ns["__import__"] = _safe_import

    out, err = io.StringIO(), io.StringIO()
    try:
        ns: dict = {"__builtins__": safe_ns}
        with redirect_stdout(out), redirect_stderr(err):
            exec(compile(src, "<solution>", "exec"), ns)
            if harness.strip():
                exec(compile(harness, "<harness>", "exec"), ns)
        q.put({"stdout": out.getvalue()[:MAX_STDOUT_CHARS], "stderr": err.getvalue()[:MAX_STDOUT_CHARS], "error": ""})
    except Exception as e:  # noqa: BLE001
        q.put({"stdout": out.getvalue()[:MAX_STDOUT_CHARS], "stderr": err.getvalue()[:MAX_STDOUT_CHARS], "error": repr(e)})


def run_python(code: str, harness: str = "", timeout_s: int = TIMEOUT_S) -> dict:
    """Run coder-written `code`, then `harness` in the same namespace.

    Guardrail-gated: blocked code never spawns a process; the violation is
    returned as `error`. Returns {stdout, stderr, error}.
    """
    import multiprocessing as mp

    gate = guardrail_report(code, harness)
    if not gate["allowed"]:
        return {"stdout": "", "stderr": "", "error": "blocked: " + "; ".join(gate["reasons"][:5])}

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_exec_target, args=(q, code, harness))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"stdout": "", "stderr": "", "error": f"timeout after {timeout_s}s"}
    return q.get() if not q.empty() else {"stdout": "", "stderr": "", "error": "no result"}


def _tool_impl(code: str, harness: str = "") -> str:
    """LangChain tool func: JSON string result so the coder LLM can read it."""
    return json.dumps(run_python(code, harness))


try:
    from langchain_core.tools import StructuredTool

    python_toolbox = StructuredTool.from_function(
        func=_tool_impl,
        name="python_sandbox",
        description=(
            "Run Python solution code with an optional harness. "
            "Args: code (solution source defining solve), harness (test snippet "
            "run after code, calls solve and prints results). "
            "A security guardrail blocks file/network/OS access. "
            "Returns JSON {stdout, stderr, error}."
        ),
    )
except Exception:  # langchain not installed in some contexts
    python_toolbox = None
