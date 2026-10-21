"""Typed configuration for the agent (models, gateway, tracing, graph knobs).

Values are layered: built-in defaults -> config.yml -> environment variables.
The file location defaults to the repo-root `config.yml` and can be overridden
with `MLFLOW_AGENT_CONFIG`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Repo root: src/agent/config.py -> src/agent -> src -> <repo>
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yml"
CONFIG_PATH_ENV = "MLFLOW_AGENT_CONFIG"

MODEL_ROLES = ("heavy", "small", "critic", "testgen", "evaluator")


class ModelSettings(BaseModel):
    """One model role (endpoint name + sampling/timeout knobs)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_s: int = 180


class GatewayConfig(BaseModel):
    """MLflow AI gateway + tracking server connection."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = "http://127.0.0.1:5000/gateway/mlflow/v1"
    api_key: str = ""
    tracking_uri: str = "http://127.0.0.1:5000"
    experiment: str = "leetcodeagent"


class AgentKnobs(BaseModel):
    """Graph-wide limits."""

    model_config = ConfigDict(extra="forbid")

    max_intent_iterations: int = 3
    max_tests: int = 10  # overall safety cap
    tests_per_category: int = 2  # corner/easy/medium/hard
    holdout_min_tests: int = 3
    max_planner_retries: int = 1


class EvalsConfig(BaseModel):
    """Evaluation dataset and reporting settings."""

    model_config = ConfigDict(extra="forbid")

    experiment: str = "leetcodeagent-evals"
    dataset_name: str = "leetcode-golden-v1"
    log_feedback: bool = True
    llm_judge: bool = False
    baseline_run_id: str = ""


class AgentConfig(BaseModel):
    """Full config tree."""

    model_config = ConfigDict(extra="forbid")

    gateway: GatewayConfig
    models: dict[str, ModelSettings]
    agent: AgentKnobs
    evals: EvalsConfig = Field(default_factory=EvalsConfig)


def default_config() -> AgentConfig:
    """Built-in defaults used when config.yml is missing or partial."""
    return AgentConfig(
        gateway=GatewayConfig(),
        models={
            "heavy": ModelSettings(name="nemotron-3-super", temperature=0.2, max_tokens=4096, timeout_s=180),
            "small": ModelSettings(name="gemma4", temperature=0.0, max_tokens=4096, timeout_s=90),
            "critic": ModelSettings(name="nemotron-3-super", temperature=0.0, max_tokens=1024, timeout_s=60),
            "testgen": ModelSettings(name="nemotron-3-super", temperature=0.0, max_tokens=8192, timeout_s=180),
            "evaluator": ModelSettings(name="nemotron-3-ultra", temperature=0.0, max_tokens=4096, timeout_s=240),
        },
        agent=AgentKnobs(),
        evals=EvalsConfig(),
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _env(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_int(name: str, current: int) -> int:
    raw = _env(name)
    if raw is None:
        return current
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, current: float) -> float:
    raw = _env(name)
    if raw is None:
        return current
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, current: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return current
    normalized = raw.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _apply_env(config: AgentConfig) -> AgentConfig:
    """Environment overrides (highest precedence)."""
    gw = config.gateway.model_copy(
        update={
            "base_url": _env("MLFLOW_GATEWAY_URL") or config.gateway.base_url,
            # Empty stays empty; llms.py sends a placeholder key.
            "api_key": _env("MLFLOW_GATEWAY_API_KEY") or config.gateway.api_key,
            "tracking_uri": _env("MLFLOW_TRACKING_URI") or config.gateway.tracking_uri,
            "experiment": _env("MLFLOW_EXPERIMENT_NAME") or config.gateway.experiment,
        }
    )
    models = dict(config.models)
    for role in MODEL_ROLES:
        settings = models[role]
        prefix = f"MLFLOW_{role.upper()}"
        models[role] = settings.model_copy(
            update={
                "name": _env(prefix) or settings.name,
                "temperature": _env_float(f"{prefix}_TEMPERATURE", settings.temperature),
                "max_tokens": _env_int(f"{prefix}_MAX_TOKENS", settings.max_tokens),
                "timeout_s": _env_int(f"{prefix}_TIMEOUT", settings.timeout_s),
            }
        )
    knobs = config.agent.model_copy(
        update={
            "max_intent_iterations": _env_int(
                "MLFLOW_MAX_INTENT_ITERATIONS", config.agent.max_intent_iterations
            ),
            "max_tests": _env_int("MLFLOW_MAX_TESTS", config.agent.max_tests),
            "tests_per_category": _env_int(
                "MLFLOW_TESTS_PER_CATEGORY", config.agent.tests_per_category
            ),
            "holdout_min_tests": _env_int(
                "MLFLOW_HOLDOUT_MIN_TESTS", config.agent.holdout_min_tests
            ),
            "max_planner_retries": _env_int(
                "MLFLOW_MAX_PLANNER_RETRIES", config.agent.max_planner_retries
            ),
        }
    )
    evals = config.evals.model_copy(
        update={
            "experiment": _env("MLFLOW_EVAL_EXPERIMENT") or config.evals.experiment,
            "dataset_name": _env("MLFLOW_EVAL_DATASET") or config.evals.dataset_name,
            "log_feedback": _env_bool("MLFLOW_EVAL_LOG_FEEDBACK", config.evals.log_feedback),
            "llm_judge": _env_bool("MLFLOW_EVAL_LLM_JUDGE", config.evals.llm_judge),
            "baseline_run_id": _env("MLFLOW_EVAL_BASELINE_RUN_ID")
            or config.evals.baseline_run_id,
        }
    )
    return AgentConfig(gateway=gw, models=models, agent=knobs, evals=evals)


def _validate_roles(config: AgentConfig) -> AgentConfig:
    unknown = sorted(set(config.models) - set(MODEL_ROLES))
    if unknown:
        raise ValueError(
            f"unknown model role(s) in config: {', '.join(unknown)}; "
            f"expected one of: {', '.join(MODEL_ROLES)}"
        )
    missing = sorted(set(MODEL_ROLES) - set(config.models))
    if missing:
        raise ValueError(f"missing model role(s) in config: {', '.join(missing)}")
    return config


def load_config(path: str | Path | None = None) -> AgentConfig:
    """Load defaults -> YAML -> env. Missing file falls back to defaults."""
    config_path = Path(
        path or _env(CONFIG_PATH_ENV) or DEFAULT_CONFIG_PATH
    ).expanduser()

    config = default_config()
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path} must contain a YAML mapping at the top level")
        merged = _deep_merge(config.model_dump(), raw)
        try:
            config = AgentConfig.model_validate(merged)
        except ValidationError as exc:
            raise ValueError(f"invalid config in {config_path}:\n{exc}") from exc

    return _validate_roles(_apply_env(config))


@lru_cache(maxsize=1)
def get_config() -> AgentConfig:
    """Process-wide config singleton."""
    return load_config()


def reload_config(path: str | Path | None = None) -> AgentConfig:
    """Drop the cached config (tests, or after changing env at runtime)."""
    get_config.cache_clear()
    return get_config() if path is None else load_config(path)
