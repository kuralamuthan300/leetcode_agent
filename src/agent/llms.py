from __future__ import annotations

import os

from langchain_ollama import ChatOllama

HEAVY_MODEL = os.getenv("OLLAMA_HEAVY", "nemotron-3-super:cloud")
SMALL_MODEL = os.getenv("OLLAMA_SMALL", "gemma4:31b-cloud")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Wall-clock guardrails for inference (Step 7 tunes these).
SMALL_TIMEOUT_S = int(os.getenv("OLLAMA_SMALL_TIMEOUT", "90"))
HEAVY_TIMEOUT_S = int(os.getenv("OLLAMA_HEAVY_TIMEOUT", "180"))
CRITIC_MODEL = os.getenv("OLLAMA_CRITIC", "nemotron-3-super:cloud")

# Test generation is promoted to the heavy model with reasoning enabled so the
# expected outputs are computed deliberately (TDD ordering: tests precede code).
TESTGEN_MODEL = os.getenv("OLLAMA_TESTGEN", HEAVY_MODEL)
TESTGEN_TIMEOUT_S = int(os.getenv("OLLAMA_TESTGEN_TIMEOUT", str(HEAVY_TIMEOUT_S)))
TESTGEN_THINK = os.getenv("OLLAMA_TESTGEN_THINK", "1").strip().lower() not in ("0", "false", "no", "")

# The evaluator judges failures (blame + diagnosis) and is deliberately the
# strongest model in the stack. It is used ONLY by the evaluator node.
EVALUATOR_MODEL = os.getenv("OLLAMA_EVALUATOR", "nemotron-3-ultra:cloud")
EVALUATOR_TIMEOUT_S = int(os.getenv("OLLAMA_EVALUATOR_TIMEOUT", "240"))
EVALUATOR_THINK = os.getenv("OLLAMA_EVALUATOR_THINK", "1").strip().lower() not in ("0", "false", "no", "")


def get_small() -> ChatOllama:
    return ChatOllama(
        model=SMALL_MODEL,
        base_url=BASE_URL,
        temperature=0,
        num_predict=4096,  # reasoning traces consume budget; content needs room
        client_kwargs={"timeout": SMALL_TIMEOUT_S},
    )


def get_heavy() -> ChatOllama:
    # Cloud models (e.g. gemma4:31b-cloud) route through the local Ollama
    # daemon which holds the cloud auth; same base_url works.
    return ChatOllama(
        model=HEAVY_MODEL,
        base_url=BASE_URL,
        temperature=0.2,
        client_kwargs={"timeout": HEAVY_TIMEOUT_S},
    )


def get_critic() -> ChatOllama:
    """Reviewer model (nemotron by default; override via OLLAMA_CRITIC).
    Small budget: verdicts, not essays."""
    return ChatOllama(
        model=CRITIC_MODEL,
        base_url=BASE_URL,
        temperature=0,
        num_predict=512,
        client_kwargs={"timeout": 60},
    )


def get_testgen() -> ChatOllama:
    """Heavy test-generation model with reasoning enabled.

    TDD ordering makes expected outputs load-bearing, so testgen runs on the
    heavy model with `reasoning=True` (override OLLAMA_TESTGEN / *_THINK).
    """
    return ChatOllama(
        model=TESTGEN_MODEL,
        base_url=BASE_URL,
        temperature=0,
        num_predict=4096,
        reasoning=TESTGEN_THINK,
        client_kwargs={"timeout": TESTGEN_TIMEOUT_S},
    )


def get_evaluator() -> ChatOllama:
    """Evaluator-only judge model (nemotron-3-ultra by default).

    Used exclusively by the evaluator node to blame failures and diagnose the
    root cause. Reasoning is on so the verdict is deliberate; the timeout is
    generous because the ultra model is stronger (and slower) than the rest.
    """
    return ChatOllama(
        model=EVALUATOR_MODEL,
        base_url=BASE_URL,
        temperature=0,
        num_predict=4096,
        reasoning=EVALUATOR_THINK,
        client_kwargs={"timeout": EVALUATOR_TIMEOUT_S},
    )
