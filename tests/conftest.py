import pytest


@pytest.fixture(autouse=True)
def _disable_tracing(monkeypatch):
    """Keep every test hermetic: never talk to the MLflow tracking server."""
    monkeypatch.setenv("MLFLOW_TRACING_DISABLED", "1")


@pytest.fixture(autouse=True)
def _reset_config_cache():
    """Env-driven config tests must not leak a stale singleton into others."""
    from agent.config import reload_config

    reload_config()
    yield
    reload_config()
