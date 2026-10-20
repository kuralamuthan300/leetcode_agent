"""Model factory: all calls go through the MLflow AI gateway.

The gateway is OpenAI-compatible, so every role is a `ChatOpenAI` pointed at
`gateway.base_url` with the endpoint name as the model. Endpoints, timeouts,
temperatures and token budgets live in `config.yml` (see `agent.config`).
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import ModelSettings, get_config

# Role model names, kept as module constants for the run reporter and tests.
HEAVY_MODEL = get_config().models["heavy"].name
SMALL_MODEL = get_config().models["small"].name
CRITIC_MODEL = get_config().models["critic"].name
TESTGEN_MODEL = get_config().models["testgen"].name
EVALUATOR_MODEL = get_config().models["evaluator"].name


def _make(role: str) -> ChatOpenAI:
    config = get_config()
    gateway = config.gateway
    settings: ModelSettings = config.models[role]
    return ChatOpenAI(
        model=settings.name,
        base_url=gateway.base_url,
        # The gateway needs no key; OpenAI's client rejects None.
        api_key=gateway.api_key or "not-needed",
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_s,
    )


def get_small() -> ChatOpenAI:
    """Clarification, scope gate, quiet gather."""
    return _make("small")


def get_heavy() -> ChatOpenAI:
    """Planner + coder."""
    return _make("heavy")


def get_critic() -> ChatOpenAI:
    """Advisory reviewer: verdicts, not essays."""
    return _make("critic")


def get_testgen() -> ChatOpenAI:
    """Test generation with a generous budget (expected outputs are load-bearing)."""
    return _make("testgen")


def get_evaluator() -> ChatOpenAI:
    """Evaluator-only judge: blame + root-cause diagnosis."""
    return _make("evaluator")
