from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone

from rich.console import Console

from agent.graph import build_graph

console = Console()


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"run_id_{ts}_{uuid.uuid4().hex[:6]}"


def _prompt_requirement() -> str:
    """Interactively read a (possibly multi-line) problem via input().

    Ends on a blank line or EOF. Used when --problem/--file are absent
    and stdin is a TTY.
    """
    console.print("[bold]Enter problem statement (blank line to finish):[/bold]")
    lines: list[str] = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def _line(node: str, patch: dict) -> str:
    """One minimal status line per node update."""
    tag = " (fallback)" if patch.get("fallback_used") else ""
    if node == "intent":
        n = patch.get("intent_iterations", "?")
        if patch.get("intent_clear"):
            return f"✓ intent clear ({n} iter){tag}"
        return f"↻ intent unclear, retrying ({n}/3){tag}"
    if node == "planner":
        return f"✓ plan ({len(patch.get('plan_steps', []))} steps){tag}"
    if node == "tests":
        return f"✓ tests ({len(patch.get('test_cases', []))}){tag}"
    if node == "coder":
        err = (patch.get("coder_smoke") or {}).get("error", "")
        if not err:
            return f"✓ code (smoke ok){tag}"
        if err.startswith("blocked:"):
            return f"⚠ code blocked: {err[len('blocked:'):].strip()[:100]}"
        return f"✗ code smoke failed: {err[:100]}{tag}"
    if node == "evaluator":
        ev = patch.get("eval", {})
        score = f"{ev.get('passed_count', 0)}/{ev.get('total', 0)}"
        if ev.get("passed"):
            return f"✓ eval {score}{tag}"
        return f"↻ eval {score} → retry ({patch.get('iterations', '?')}/3){tag}"
    if node == "reporter":
        return f"✓ wrote {patch.get('output_path', '')}"
    return f"· {node}"


def _run_graph(graph, state: dict) -> dict:
    """Stream node updates, printing one status line each; return final state."""
    final = dict(state)
    for chunk in graph.stream(state, stream_mode="updates"):
        for node, patch in chunk.items():
            final.update(patch or {})
            console.print(_line(node, patch or {}))
    return final
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve a DSA problem with LangGraph + Ollama")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--problem", help="Problem statement inline")
    src.add_argument("--file", help="Path to file with problem statement")
    ap.add_argument("--run-id", help="Override run_id")
    ap.add_argument("--verbose", action="store_true", help="Print code and tests too")
    args = ap.parse_args(argv)

    if args.file:
        requirement = open(args.file).read()
    elif args.problem:
        requirement = args.problem
    elif not sys.stdin.isatty():
        requirement = sys.stdin.read()
    else:
        requirement = _prompt_requirement()

    if not requirement.strip():
        console.print("[red]Empty requirement.[/red]")
        return 2

    run_id = args.run_id or _new_run_id()
    graph = build_graph()
    final = _run_graph(graph, {"run_id": run_id, "requirement": requirement})
    if args.verbose:
        console.print("[bold]Code:[/bold]\n" + final.get("code", ""))
        for t in final.get("test_cases", []):
            console.print(f"[bold]{t['id']}:[/bold] {t['harness']} → {t['expected']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
