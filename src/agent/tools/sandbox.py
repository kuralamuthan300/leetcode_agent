from __future__ import annotations


def _exec_target(q, src: str, harness: str) -> None:
    import io
    import json
    from contextlib import redirect_stderr, redirect_stdout

    out, err = io.StringIO(), io.StringIO()
    try:
        ns: dict = {"__builtins__": __builtins__}
        with redirect_stdout(out), redirect_stderr(err):
            exec(compile(src, "<solution>", "exec"), ns)
            if harness.strip():
                exec(compile(harness, "<harness>", "exec"), ns)
        q.put({"stdout": out.getvalue(), "stderr": err.getvalue(), "error": ""})
    except Exception as e:  # noqa: BLE001
        q.put({"stdout": out.getvalue(), "stderr": err.getvalue(), "error": repr(e)})


def run_python(code: str, harness: str = "", timeout_s: int = 5) -> dict:
    """Run coder-written `code`, then optional `harness` in the same namespace.

    The harness can call functions/classes defined in `code`, e.g.:
        code="def add(a,b): return a+b"
        harness="print(add(1,2))"
    Returns {stdout, stderr, error}.
    """
    import multiprocessing as mp

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
    import json

    return json.dumps(run_python(code, harness))


try:
    from langchain_core.tools import StructuredTool

    python_toolbox = StructuredTool.from_function(
        func=_tool_impl,
        name="python_sandbox",
        description=(
            "Run Python solution code with an optional harness. "
            "Args: code (solution source), harness (test snippet run after code, "
            "can call functions from code and print results). "
            "Returns JSON {stdout, stderr, error}."
        ),
    )
except Exception:  # langchain not installed in some contexts
    python_toolbox = None
