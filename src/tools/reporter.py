"""Step 9: optimizer report builder (deterministic, no LLM).

Produces report.md with before/after timing table + Big-O + what changed.
All inputs are plain dicts/lists so solver_graph nodes stay thin.
File I/O is done by the caller via access_broker.scoped_write;
build_report() here is pure markdown rendering for easy unit testing.
"""

from __future__ import annotations


def _total(timings: list[float] | None) -> float:
    total = 0.0
    for t in timings or []:
        try:
            total += float(t)
        except (TypeError, ValueError):
            continue
    return total


def _table(rows: list[dict] | None) -> str:
    lines = ["| test_id | pass | mean_ms |", "| --- | --- | --- |"]
    for r in rows or []:
        tid = r.get("test_id", "?")
        ok = "true" if r.get("passed") else "false"
        try:
            ms = f"{float(r.get('time_ms', 0.0)):.3f}"
        except (TypeError, ValueError):
            ms = str(r.get("time_ms"))
        lines.append(f"| {tid} | {ok} | {ms} |")
    return "\n".join(lines)


def _speedup(before_total: float, after_total: float) -> str:
    if after_total <= 0 or before_total <= 0:
        return "n/a"
    ratio = before_total / after_total
    pct = (before_total - after_total) / before_total * 100
    return f"{ratio:.2f}x ({pct:+.1f}%)"


def build_report(
    baseline_results: list[dict] | None,
    optimized_results: list[dict] | None,
    baseline_timings: list[float] | None,
    optimized_timings: list[float] | None,
    complexity_before: str = "unknown",
    complexity_after: str = "unknown",
    what_changed: str = "",
    opt_status: str = "rolled-back",
    opt_reason: str = "",
) -> str:
    """Render report.md markdown. Pure function (no I/O)."""
    before_total = _total(baseline_timings)
    after_total = (
        _total(optimized_timings)
        if optimized_results is not None
        else before_total
    )
    lines = [
        "# Optimizer Report",
        "",
        f"Status: `{opt_status}`" + (f" — {opt_reason}" if opt_reason else ""),
        "",
        "## Complexity",
        "",
        f"- Before: `{complexity_before or 'unknown'}`",
        f"- After: `{complexity_after or 'unknown'}`",
        "",
        "## Timings (before)",
        "",
        _table(baseline_results),
        "",
        f"Total before: `{before_total:.3f} ms`",
        "",
        "## Timings (after)",
        "",
        _table(optimized_results)
        if optimized_results is not None
        else "_No optimized run accepted — showing baseline._",
        "",
        f"Total after: `{after_total:.3f} ms`",
        f"Speedup: `{_speedup(before_total, after_total)}`",
        "",
        "## What changed",
        "",
        what_changed or "_No change (rolled back to baseline)._",
        "",
    ]
    return "\n".join(lines)
