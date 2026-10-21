"""Infrastructure, repeatability, latency, and cost metrics."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return float(ordered[rank])


def _load_output(record: Mapping[str, Any]) -> dict[str, Any]:
    path = record.get("output_path")
    if not path:
        return {}
    try:
        return json.loads(Path(str(path)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _token_counts(trace: Mapping[str, Any]) -> tuple[int, int]:
    info = trace.get("info", trace)
    usage = info.get("token_usage") or {}
    inputs = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    outputs = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    return int(inputs), int(outputs)


def _fallback_cost(
    trace: Mapping[str, Any],
    pricing: Mapping[str, Mapping[str, float]],
) -> float | None:
    spans = trace.get("spans") or []
    total = 0.0
    found = False
    for span in spans:
        name = str(span.get("name", ""))
        role = name.split(":", 1)[1] if name.startswith("model:") else str(
            (span.get("attributes") or {}).get("model", "")
        )
        price = pricing.get(role)
        if price is None:
            continue
        usage = span.get("attributes", {}).get("token_usage", {}) or {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total += (
            input_tokens * float(price.get("input_per_1k", 0))
            + output_tokens * float(price.get("output_per_1k", 0))
        ) / 1000
        found = True
    return total if found else None


def compute_ops_metrics(
    run_index: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    strict_verdicts: Mapping[str, Sequence[bool]] | None = None,
    traces: Mapping[str, Mapping[str, Any]] | None = None,
    pricing: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    traces = traces or {}
    pricing = pricing or {}
    records = [record for repeated in run_index.values() for record in repeated]
    durations = [
        float(record["duration_s"])
        for record in records
        if record.get("duration_s") is not None
    ]
    infra = sum(record.get("status") == "infra_error" for record in records)
    repeated = {
        problem_id: verdicts
        for problem_id, verdicts in (strict_verdicts or {}).items()
        if len(verdicts) > 1
    }
    flaky = sum(1 for verdicts in repeated.values() if len(set(verdicts)) > 1)
    repeated_problems = len(repeated)

    total_input_tokens = total_output_tokens = 0
    total_cost = 0.0
    cost_known = False
    trace_latencies: list[float] = []
    unavailable_trace_ids: list[str] = []
    for record in records:
        trace_id = record.get("trace_id")
        trace = traces.get(str(trace_id)) if trace_id else None
        if trace is None:
            if trace_id:
                unavailable_trace_ids.append(str(trace_id))
            continue
        info = trace.get("info", trace)
        execution_ms = info.get("execution_time_ms")
        if execution_ms is not None:
            trace_latencies.append(float(execution_ms) / 1000)
        input_tokens, output_tokens = _token_counts(trace)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        trace_cost = info.get("cost")
        if trace_cost is not None:
            total_cost += float(trace_cost)
            cost_known = True
        else:
            estimated = _fallback_cost(trace, pricing)
            if estimated is not None:
                total_cost += estimated
                cost_known = True

    solved = sum(
        bool(verdicts[-1])
        for verdicts in (strict_verdicts or {}).values()
        if verdicts
    )
    return {
        "infra_error_rate": infra / len(records) if records else None,
        "infra_errors": infra,
        "runs": len(records),
        "flaky_rate": flaky / repeated_problems if repeated_problems else None,
        "flaky_problems": flaky,
        "repeated_problems": repeated_problems,
        "latency_p50": _percentile(durations, 0.50),
        "latency_p95": _percentile(durations, 0.95),
        "trace_latency_p50": _percentile(trace_latencies, 0.50),
        "trace_latency_p95": _percentile(trace_latencies, 0.95),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_cost": total_cost if cost_known else None,
        "cost_per_solved": total_cost / solved if cost_known and solved else None,
        "cost_status": "available" if cost_known else "unavailable",
        "fallback_used": sum(bool(_load_output(record).get("fallback_used")) for record in records),
        "critic_overruled": sum(bool(_load_output(record).get("critic_overruled")) for record in records),
        "mean_iterations": (
            sum(
                float((_load_output(record).get("eval") or {}).get("iterations", 0))
                for record in records
            )
            / len(records)
            if records
            else None
        ),
        "unavailable_trace_ids": unavailable_trace_ids,
    }
