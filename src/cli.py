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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve a DSA problem with LangGraph + Ollama")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--problem", help="Problem statement inline")
    src.add_argument("--file", help="Path to file with problem statement")
    ap.add_argument("--run-id", help="Override run_id")
    args = ap.parse_args(argv)

    if args.file:
        requirement = open(args.file).read()
    elif args.problem:
        requirement = args.problem
    else:
        console.print("[bold]Enter problem statement (Ctrl-D to finish):[/bold]")
        requirement = sys.stdin.read()

    if not requirement.strip():
        console.print("[red]Empty requirement.[/red]")
        return 2

    run_id = args.run_id or _new_run_id()
    graph = build_graph()
    final = graph.invoke({"run_id": run_id, "requirement": requirement})
    console.print(f"[green]Done.[/green] Output: {final.get('output_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
