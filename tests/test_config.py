import pytest

from agent import config as cfg
from agent.config import AgentConfig, load_config, reload_config


def test_defaults_when_file_missing(tmp_path):
    loaded = load_config(tmp_path / "missing.yml")
    assert isinstance(loaded, AgentConfig)
    assert loaded.gateway.base_url.startswith("http://127.0.0.1:5000/gateway/")
    assert set(loaded.models) == set(cfg.MODEL_ROLES)
    assert loaded.models["evaluator"].name == "nemotron-3-ultra"
    assert loaded.agent.max_tests == 10


def test_repo_config_parses():
    loaded = load_config()
    assert loaded.models["heavy"].name == "nemotron-3-super"
    assert loaded.models["small"].name == "gemma4"
    assert loaded.models["critic"].max_tokens >= 1024
    assert loaded.models["evaluator"].timeout_s == 240
    assert loaded.gateway.experiment == "leetcodeagent"


def test_yaml_overrides_partial_roles(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_HEAVY", raising=False)
    path = tmp_path / "config.yml"
    path.write_text(
        "models:\n"
        "  heavy:\n"
        "    name: custom-heavy\n"
        "    temperature: 0.7\n"
        "    max_tokens: 2048\n"
        "    timeout_s: 42\n"
        "gateway:\n"
        "  experiment: unit-test\n",
        encoding="utf-8",
    )
    loaded = load_config(path)
    assert loaded.models["heavy"].name == "custom-heavy"
    assert loaded.models["heavy"].temperature == 0.7
    assert loaded.models["heavy"].max_tokens == 2048
    assert loaded.models["heavy"].timeout_s == 42
    # Roles absent from the YAML keep their defaults.
    assert loaded.models["small"].name == "gemma4"
    assert loaded.gateway.experiment == "unit-test"


def test_env_overrides_file(tmp_path, monkeypatch):
    path = tmp_path / "config.yml"
    path.write_text("models:\n  heavy:\n    name: from-file\n", encoding="utf-8")
    monkeypatch.setenv("MLFLOW_HEAVY", "from-env")
    monkeypatch.setenv("MLFLOW_HEAVY_TEMPERATURE", "0.9")
    monkeypatch.setenv("MLFLOW_HEAVY_MAX_TOKENS", "1234")
    monkeypatch.setenv("MLFLOW_HEAVY_TIMEOUT", "77")
    monkeypatch.setenv("MLFLOW_GATEWAY_URL", "http://gateway.test/v1")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "env-experiment")
    loaded = load_config(path)
    heavy = loaded.models["heavy"]
    assert heavy.name == "from-env"
    assert heavy.temperature == 0.9
    assert heavy.max_tokens == 1234
    assert heavy.timeout_s == 77
    assert loaded.gateway.base_url == "http://gateway.test/v1"
    assert loaded.gateway.experiment == "env-experiment"


def test_unknown_role_rejected(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("models:\n  tiny:\n    name: nope\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown model role"):
        load_config(path)


def test_invalid_yaml_shape_rejected(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(path)


def test_reload_config_refreshes_singleton(monkeypatch):
    monkeypatch.setenv("MLFLOW_SMALL", "gemma4-refreshed")
    loaded = reload_config()
    assert loaded.models["small"].name == "gemma4-refreshed"
    assert cfg.get_config() is loaded
