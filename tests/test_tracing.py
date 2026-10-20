import mlflow.langchain as mlflow_langchain
import pytest

from agent import tracing
from agent.config import get_config


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_DISABLED", "1")
    assert tracing.tracing_enabled() is False
    assert tracing.init_tracing() is False


def test_init_configures_tracking_and_autolog(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACING_DISABLED", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    calls = {}

    monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: calls.setdefault("uri", uri))
    monkeypatch.setattr(
        "mlflow.set_experiment", lambda name: calls.setdefault("experiment", name)
    )
    monkeypatch.setattr(
        mlflow_langchain,
        "autolog",
        lambda **kwargs: calls.setdefault("autolog", kwargs),
    )

    assert tracing.init_tracing() is True
    config = get_config()
    assert calls["uri"] == config.gateway.tracking_uri
    assert calls["experiment"] == config.gateway.experiment
    assert calls["autolog"] == {"log_traces": True}


def test_experiment_id_skips_set_experiment(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACING_DISABLED", raising=False)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "42")
    called = {"set_experiment": False}

    def boom(_name):
        called["set_experiment"] = True

    monkeypatch.setattr("mlflow.set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr("mlflow.set_experiment", boom)
    monkeypatch.setattr(mlflow_langchain, "autolog", lambda **_kwargs: None)

    assert tracing.init_tracing() is True
    assert called["set_experiment"] is False


def test_failures_are_swallowed(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACING_DISABLED", raising=False)

    def boom(_uri):
        raise RuntimeError("tracking server down")

    monkeypatch.setattr("mlflow.set_tracking_uri", boom)
    assert tracing.init_tracing() is False


def test_tag_run_names_and_tags_trace(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("MLFLOW_TRACING_DISABLED", raising=False)
    calls = {}
    monkeypatch.setattr("mlflow.get_current_active_span", lambda: SimpleNamespace())
    monkeypatch.setattr("mlflow.update_current_trace", lambda **kwargs: calls.update(kwargs))

    tracing.tag_run(
        {
            "run_id": "r1",
            "requirement": "  sum\n of ints ",
            "status": "ok",
            "gather_mode": "quiet",
            "eval": {"passed_count": 3, "total": 4},
            "failure_report": {"culprit": "code"},
        }
    )

    tags = calls["tags"]
    assert tags["run_id"] == "r1"
    assert tags["status"] == "ok"
    assert tags["gather_mode"] == "quiet"
    assert tags["eval"] == "3/4"
    assert tags["mlflow.traceName"] == "r1 · sum of ints"
    assert "sum of ints" in calls["request_preview"]
    assert "culprit=code" in calls["response_preview"]


def test_tag_run_noops_without_active_span(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACING_DISABLED", raising=False)
    monkeypatch.setattr("mlflow.get_current_active_span", lambda: None)
    monkeypatch.setattr(
        "mlflow.update_current_trace",
        lambda **_kwargs: pytest.fail("must not tag without an active span"),
    )
    tracing.tag_run({"run_id": "r1", "requirement": "q"})


def test_tag_run_noops_when_disabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_DISABLED", "1")
    monkeypatch.setattr(
        "mlflow.update_current_trace",
        lambda **_kwargs: pytest.fail("must not tag when tracing is disabled"),
    )
    tracing.tag_run({"run_id": "r1", "requirement": "q"})


def test_last_trace_id_guarded(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACING_DISABLED", "1")
    assert tracing.last_trace_id() is None
    monkeypatch.delenv("MLFLOW_TRACING_DISABLED", raising=False)
    monkeypatch.setattr("mlflow.get_last_active_trace_id", lambda: "tr-123")
    assert tracing.last_trace_id() == "tr-123"


def test_trace_url_uses_tracking_uri_and_experiment(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("MLFLOW_TRACING_DISABLED", raising=False)
    monkeypatch.setattr(tracing, "_experiment_id", None)
    monkeypatch.setattr(
        "mlflow.get_experiment_by_name", lambda _name: SimpleNamespace(experiment_id="7")
    )
    url = tracing.trace_url("tr-123")
    assert url == "http://127.0.0.1:5000/#/experiments/7/traces/tr-123"


def test_trace_url_none_without_id(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACING_DISABLED", raising=False)
    assert tracing.trace_url(None) is None


def test_init_tracing_caches_experiment_id(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.delenv("MLFLOW_TRACING_DISABLED", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_ID", raising=False)
    monkeypatch.setattr(tracing, "_experiment_id", None)
    monkeypatch.setattr("mlflow.set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(
        "mlflow.set_experiment", lambda _name: SimpleNamespace(experiment_id="11")
    )
    monkeypatch.setattr(mlflow_langchain, "autolog", lambda **_kwargs: None)
    assert tracing.init_tracing() is True
    assert tracing._experiment_id == "11"
