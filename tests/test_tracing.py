import mlflow.langchain as mlflow_langchain

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
