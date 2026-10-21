from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent.graph import (
    MAX_ITERATIONS,
    _route_eval,
    _route_gather,
    _route_planner,
    _route_tests,
    build_graph,
)
from agent.tracing import init_tracing, last_trace_id, trace_url

console = Console()

_CLIP = 120
_CODE_LINES = 80

# While a node runs (LLM calls block for 90-180s), keep the screen alive with a
# transient spinner + rotating, node-aware status phrases. The rotation tick
# advances every _NARRATION_INTERVAL seconds spent waiting on the producer.
_NARRATION_INTERVAL = 3.0

_NARRATION: dict[str, list[str]] = {
    "gather": [
        "reading your problem…",
        "checking scope and language…",
        "untangling the requirements…",
    ],
    "planner": [
        "thinking…",
        "sketching an approach…",
        "weighing time & space complexity…",
    ],
    "tests": [
        "working on it…",
        "designing edge cases…",
        "computing expected outputs…",
    ],
    "coder": [
        "writing the solution…",
        "honing the algorithm…",
        "running a quick smoke check…",
    ],
    "evaluator": [
        "testing against the harness…",
        "checking every case…",
        "adjudicating failures…",
    ],
    "reporter": [
        "writing the report…",
        "saving the run…",
    ],
}


def _clip(value: object, limit: int = _CLIP) -> str:
    """Trim long values so terminal tables stay readable."""
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _render_code(code: str, title: str) -> None:
    body = code.rstrip("\n")
    lines = body.splitlines()
    if len(lines) > _CODE_LINES:
        body = "\n".join(lines[:_CODE_LINES]) + f"\n# … (+{len(lines) - _CODE_LINES} more lines)"
    console.print(Panel(Syntax(body, "python", line_numbers=True), title=title, border_style="cyan"))


def _render_plan(plan_steps: list, signature: str = "", arg_spec: list | None = None, returns_spec: str = "") -> None:
    """Print the planner's steps + CONTRACT (signature / args / returns)."""
    if not plan_steps and not signature:
        return
    body = Text()
    for i, step in enumerate(plan_steps, 1):
        body.append(f"{i}. {step}\n")
    if signature:
        body.append("\nContract: ", style="bold")
        body.append(f"{signature}\n")
    for a in arg_spec or []:
        if isinstance(a, dict) and a.get("name"):
            typ = f": {a['type']}" if a.get("type") else ""
            rng = f", range: {a['range']}" if a.get("range") else ""
            body.append(f"  - {a['name']}{typ}{rng}\n")
    if returns_spec:
        body.append(f"  Returns: {returns_spec}")
    console.print(Panel(body, title="Plan", border_style="magenta"))


def _render_tests(test_cases: list, holdout_tests: list) -> None:
    rows = [(t, "visible") for t in test_cases] + [(t, "holdout") for t in holdout_tests]
    if not rows:
        console.print("[dim]no test cases[/dim]")
        return
    table = Table(title="Test cases")
    table.add_column("id", style="bold")
    table.add_column("category")
    table.add_column("input (harness)")
    table.add_column("expected")
    table.add_column("kind")
    for t, kind in rows:
        table.add_row(
            str(t.get("id", "?")),
            str(t.get("category", "?")),
            _clip(t.get("harness", "")),
            _clip(t.get("expected", "")),
            kind,
        )
    console.print(table)


def _render_results(results: list, title: str = "Test results") -> None:
    if not results:
        return
    table = Table(title=title)
    table.add_column("id", style="bold")
    table.add_column("input (harness)")
    table.add_column("expected")
    table.add_column("actual")
    table.add_column("result")
    table.add_column("blame")
    for r in results:
        ok = r.get("passed")
        table.add_row(
            str(r.get("test_id", "?")),
            _clip(r.get("harness", "")),
            _clip(r.get("expected", "")),
            _clip(r.get("actual", "")),
            "[green]✓[/green]" if ok else "[red]✗[/red]",
            "" if ok else str(r.get("blame", "")),
        )
    console.print(table)


def _render_diagnosis(report: dict) -> None:
    """Show the evaluator's failure verdict (code vs tests) plus its reason."""
    if not report or not report.get("culprit"):
        return
    body = Text()
    body.append("culprit: ", style="bold")
    body.append(f"{report.get('culprit')}\n")
    body.append(str(report.get("reason", "")))
    failing = report.get("failing") or []
    if failing:
        body.append(f"\n\nfailed: {', '.join(str(t) for t in failing)}")
    console.print(Panel(body, title="Evaluator diagnosis", border_style="red"))


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
    if patch.get("assumed"):
        tag += " (assumed)"
    if node == "gather":
        if patch.get("status") == "infra_error":
            return "⚠ workability check unavailable (infra_error)"
        if patch.get("rejected"):
            cls = patch.get("rejection_class") or "rejected"
            return f"⛔ aborted [{cls}]: {patch.get('rejection_reason', '')[:100]}"
        if patch.get("gather_mode") == "quiet":
            return f"✓ requirement validated ({patch.get('status', 'ok')})"
        n = patch.get("intent_iterations", "?")
        if patch.get("intent_clear"):
            return f"✓ requirements clear ({n} iter){tag}"
        return f"↻ requirements unclear, retrying ({n}/3){tag}"
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
        return f"↻ eval {score} → retry ({patch.get('iterations', '?')}/{MAX_ITERATIONS}){tag}"
    if node == "reporter":
        return f"✓ wrote {patch.get('output_path', '')}"
    return f"· {node}"


def _next_node(node: str | None, state: dict) -> str | None:
    """Predict the node that runs next, for the during-node narrator.

    Mirrors the graph wiring: entry -> gather; gather/planner/tests/coder/evaluator
    consult the same routers; reporter (or anything unknown) ends the run.
    """
    if node is None:
        return "gather"
    if node == "gather":
        return _route_gather(state)
    if node == "planner":
        return _route_planner(state)
    if node == "tests":
        return _route_tests(state)
    if node == "coder":
        return "evaluator"
    if node == "evaluator":
        return _route_eval(state)
    return None


def _narrator(node: str | None, elapsed: float) -> Text:
    """Status text for the running node: label, rotating phrase, elapsed seconds.

    The phrase advances every `_NARRATION_INTERVAL` seconds; `elapsed` is how long
    the current node has been running.
    """
    phrases = _NARRATION.get(node or "", ["working…"])
    phrase = phrases[int(elapsed // _NARRATION_INTERVAL) % len(phrases)]
    return Text(f" {node or 'working'}: {phrase}  ({int(elapsed)}s)", style="cyan")


def _print_trace_link(trace_id: str | None) -> None:
    """Print a deep link to the run's MLflow trace (no-op when tracing is off)."""
    url = trace_url(trace_id)
    if url:
        console.print(f"[dim]trace: {url}[/dim]", soft_wrap=True)


def _produce(graph, state: dict, updates: queue.Queue) -> None:
    """Background producer: stream node updates into the queue, always ending with a sentinel."""
    try:
        for chunk in graph.stream(state, stream_mode="updates"):
            updates.put(("chunk", chunk))
    except Exception as exc:  # noqa: BLE001 - surface the error on the main thread
        updates.put(("error", exc))
    finally:
        updates.put(("trace", last_trace_id()))
        updates.put(("done", None))


def _render_patch(node: str, patch: dict, show: bool, attempt: int, elapsed: float) -> int:
    """Print one node's status line (+ panels when `show`); return the code attempt count."""
    console.print(_line(node, patch) + f" [dim]· {elapsed:.1f}s[/dim]")
    if show:
        if node == "planner":
            _render_plan(
                patch.get("plan_steps", []),
                patch.get("function_signature", ""),
                patch.get("arg_spec", []),
                patch.get("returns_spec", ""),
            )
        elif node == "tests":
            _render_tests(patch.get("test_cases", []), patch.get("holdout_tests", []))
        elif node == "coder" and patch.get("code"):
            attempt += 1
            _render_code(patch["code"], f"code — attempt {attempt}")
        elif node == "evaluator":
            _render_results(patch.get("execution_results", []), title="Test results")
            _render_diagnosis(patch.get("failure_report", {}))
    return attempt


def _run_graph(graph, state: dict, show: bool = True) -> dict:
    """Stream node updates, printing one status line each; return final state.

    With `show` (default), also render the plan, generated code, test cases, and
    per-test input/output results as each node emits them. Every status line ends
    with how long that node took. When stdout is an interactive terminal, a
    transient spinner with rotating node-aware phrases and a live seconds counter
    is shown while each node runs (the stream blocks for the length of an LLM call).
    """
    final = dict(state)
    attempt = 0
    started = time.monotonic()

    if not (show and console.is_terminal):
        for chunk in graph.stream(state, stream_mode="updates"):
            for node, patch in chunk.items():
                patch = patch or {}
                elapsed = time.monotonic() - started
                started = time.monotonic()
                attempt = _render_patch(node, patch, show, attempt, elapsed)
                final.update(patch)
        _print_trace_link(last_trace_id())
        return final

    # Interactive: drain the graph from a worker thread so the main thread can
    # animate a spinner + timer while waiting (keeps Rich's Live on one thread).
    # `get_renderable` recomputes the timer on Live's refresh thread, so no manual
    # ticking is needed here.
    updates: queue.Queue = queue.Queue()
    threading.Thread(target=_produce, args=(graph, state, updates), daemon=True).start()
    current: list[str | None] = ["gather"]
    node_started = [time.monotonic()]
    trace_id: list[str | None] = [None]
    spinner = Spinner("dots")  # persistent: a fresh Spinner each render would freeze

    def tick():
        spinner.text = _narrator(current[0], time.monotonic() - node_started[0])
        return spinner

    with Live(console=console, transient=True, refresh_per_second=8, get_renderable=tick) as live:
        while True:
            kind, payload = updates.get()
            if kind == "done":
                break
            if kind == "trace":
                trace_id[0] = payload
                continue
            if kind == "error":
                console.print(f"[red]✗ stream error: {payload!r}[/red]")
                break
            for name, patch in payload.items():
                patch = patch or {}
                elapsed = time.monotonic() - node_started[0]
                attempt = _render_patch(name, patch, show, attempt, elapsed)
                final.update(patch)
                current[0] = _next_node(name, final)
                node_started[0] = time.monotonic()
            live.refresh()
    _print_trace_link(trace_id[0])
    return final


def _run_batch(problems_file: str, out_root: str, show: bool = True) -> int:
    """Headless batch: one JSONL line per problem -> per-problem JSON + summary.json.

    Each row runs the graph in quiet gather mode (never prompts). A row that
    raises is recorded as `infra_error` and the batch continues.
    """
    import io
    import json
    import time
    from datetime import UTC, datetime
    from pathlib import Path

    global console

    try:
        raw = Path(problems_file).read_text()
    except OSError as exc:
        console.print(f"[red]cannot read problems file: {exc}[/red]")
        return 2

    real = console
    batch_id = datetime.now(UTC).strftime("batch_%Y%m%d_%H%M%S")
    out_dir = f"{out_root.rstrip('/')}/{batch_id}"
    graph = build_graph()
    status_counts: dict[str, int] = {}
    class_counts: dict[str, int] = {}
    total = 0
    skipped = 0
    run_records: list[dict] = []
    real.print(f"[bold]batch {batch_id}[/bold] → {out_dir}/")

    for i, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            rid = str(obj.get("id") or f"row{i}")
            requirement = str(obj.get("requirement") or "")
        except (json.JSONDecodeError, AttributeError):
            skipped += 1
            real.print(f"[red]  line {i}: malformed JSON (skipped)[/red]")
            continue

        total += 1
        state = {"run_id": rid, "requirement": requirement, "gather_mode": "quiet", "out_dir": out_dir}
        # Without --verbose, swallow the per-node status lines for batch rows.
        console = real if show else Console(file=io.StringIO(), width=120)
        started_at = datetime.now(UTC)
        started_clock = time.monotonic()
        output_path = ""
        try:
            final = _run_graph(graph, state, show=show)
            status = str(final.get("status") or "ok")
            cls = str(final.get("rejection_class") or "")
            output_path = str(final.get("output_path") or "")
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill the batch
            status, cls = "infra_error", "infra"
            real.print(f"[red]  {rid}: infra failure {exc!r}[/red]")
        finally:
            console = real
        ended_at = datetime.now(UTC)
        run_records.append(
            {
                "run_id": rid,
                "status": status,
                "rejection_class": cls,
                "output_path": output_path,
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "duration_s": time.monotonic() - started_clock,
            }
        )

        status_counts[status] = status_counts.get(status, 0) + 1
        if cls:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        real.print(f"  {rid}: {status}" + (f" [{cls}]" if cls and cls != "infra" else ""))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "batch_id": batch_id,
        "problems_file": problems_file,
        "finished_at": datetime.now(UTC).isoformat(),
        "total": total,
        "skipped": skipped,
        "status_counts": status_counts,
        "rejection_class_counts": class_counts,
        "runs": run_records,
        "yield": {
            "completed": sum(v for k, v in status_counts.items() if k in ("ok", "assumed")),
            "aborted": status_counts.get("aborted", 0),
            "infra": status_counts.get("infra_error", 0),
        },
        "gather_mode": "quiet",
    }
    (out_path / "summary.json").write_text(json.dumps(summary, indent=2))
    tally = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items())) or "none"
    real.print(f"[bold]batch {batch_id}:[/bold] {total} problems · {tally} · summary → {out_path}/summary.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Solve a DSA problem with LangGraph + MLflow")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--problem", help="Problem statement inline")
    src.add_argument("--file", help="Path to file with problem statement")
    src.add_argument("--problems-file", help="Batch: JSONL of {id, requirement} objects (implies quiet gathering)")
    ap.add_argument("--run-id", help="Override run_id")
    ap.add_argument("--out-dir", default="output", help="Output root directory (default: output)")
    ap.add_argument(
        "--gather-mode", choices=["interactive", "quiet"], default=None,
        help="Requirement-gatherer mode (default: interactive on a TTY, else quiet)",
    )
    ap.add_argument("--quiet", action="store_true", help="Status lines only (no code/test panels)")
    ap.add_argument("--verbose", action="store_true", help="Also print the final code and tests")
    ap.add_argument("--no-tracing", action="store_true", help="Disable MLflow tracing for this run")
    args = ap.parse_args(argv)

    if args.no_tracing:
        os.environ["MLFLOW_TRACING_DISABLED"] = "1"
    if init_tracing():
        console.print("[dim]MLflow tracing enabled[/dim]")

    if args.problems_file:
        if args.gather_mode == "interactive":
            console.print("[red]--problems-file needs quiet gathering (no TTY to prompt).[/red]")
            return 2
        return _run_batch(args.problems_file, args.out_dir, show=not args.quiet)

    mode = args.gather_mode or ("interactive" if sys.stdin.isatty() else "quiet")

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
    state = {"run_id": run_id, "requirement": requirement, "gather_mode": mode, "out_dir": args.out_dir}
    try:
        final = _run_graph(graph, state, show=not args.quiet)
    except Exception as exc:  # noqa: BLE001 - infra failure, not a legit outcome
        console.print(f"[red]infra failure: {exc!r}[/red]")
        return 1
    if args.verbose:
        console.print("[bold]Code:[/bold]\n" + final.get("code", ""))
        for t in final.get("test_cases", []):
            console.print(
                f"[bold]{t['id']}[/bold] [dim]({t.get('category', '?')})[/dim]: "
                f"{t['harness']} → {t['expected']}"
            )
        for t in final.get("holdout_tests", []):
            console.print(
                f"[bold]{t['id']} (holdout)[/bold] [dim]({t.get('category', '?')})[/dim]: "
                f"{t['harness']} → {t['expected']}"
            )
    return 1 if final.get("status") == "infra_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
