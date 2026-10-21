from __future__ import annotations

from evals.ops import compute_ops_metrics


def test_ops_metrics_compute_latency_cost_infra_and_flaky_rates():
    index = {
        "1": [
            {"repeat": 1, "status": "ok", "duration_s": 2, "trace_id": "t1"},
            {"repeat": 2, "status": "ok", "duration_s": 4, "trace_id": "t2"},
        ],
        "2": [{"repeat": 1, "status": "infra_error", "duration_s": 10, "trace_id": None}],
    }
    traces = {
        "t1": {
            "info": {
                "execution_time_ms": 1900,
                "token_usage": {"input_tokens": 100, "output_tokens": 50},
                "cost": 0.02,
            }
        },
        "t2": {
            "info": {
                "execution_time_ms": 3900,
                "token_usage": {"input_tokens": 200, "output_tokens": 100},
                "cost": 0.03,
            }
        },
    }
    result = compute_ops_metrics(
        index,
        strict_verdicts={"1": [True, False], "2": [True]},
        traces=traces,
    )
    assert result["infra_error_rate"] == 1 / 3
    assert result["flaky_rate"] == 1.0
    assert result["repeated_problems"] == 1
    assert result["latency_p50"] == 4
    assert result["latency_p95"] == 10
    assert result["trace_latency_p95"] == 3.9
    assert result["total_tokens"] == 450
    assert result["total_cost"] == 0.05
    assert result["cost_per_solved"] == 0.05


def test_ops_degrades_to_unavailable_cost_without_trace_server():
    result = compute_ops_metrics(
        {"1": [{"status": "ok", "duration_s": 3, "trace_id": "unavailable"}]},
        strict_verdicts={"1": [True]},
        traces={},
    )
    assert result["cost_status"] == "unavailable"
    assert result["total_cost"] is None
    assert result["latency_p95"] == 3
    assert result["unavailable_trace_ids"] == ["unavailable"]


def test_single_repeat_reports_flaky_rate_as_unavailable():
    result = compute_ops_metrics(
        {"1": [{"status": "ok", "duration_s": 3}], "2": [{"status": "ok", "duration_s": 4}]},
        strict_verdicts={"1": [True], "2": [False]},
    )
    assert result["flaky_rate"] is None
    assert result["repeated_problems"] == 0


def test_token_pricing_fallback_uses_model_spans():
    result = compute_ops_metrics(
        {"1": [{"status": "ok", "duration_s": 1, "trace_id": "t1"}]},
        strict_verdicts={"1": [True]},
        traces={
            "t1": {
                "info": {"token_usage": {"input_tokens": 100, "output_tokens": 20}},
                "spans": [
                    {
                        "name": "model:heavy",
                        "attributes": {
                            "token_usage": {"input_tokens": 1000, "output_tokens": 500}
                        },
                    }
                ],
            }
        },
        pricing={"heavy": {"input_per_1k": 0.01, "output_per_1k": 0.02}},
    )
    assert result["total_cost"] == 0.02
    assert result["cost_status"] == "available"
