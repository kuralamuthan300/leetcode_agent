from __future__ import annotations

import os

from langchain_ollama import ChatOllama

HEAVY_MODEL = os.getenv("OLLAMA_HEAVY", "nemotron-3-super:cloud")
SMALL_MODEL = os.getenv("OLLAMA_SMALL", "gemma4:31b-cloud")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Wall-clock guardrails for inference (Step 7 tunes these).
SMALL_TIMEOUT_S = int(os.getenv("OLLAMA_SMALL_TIMEOUT", "90"))
HEAVY_TIMEOUT_S = int(os.getenv("OLLAMA_HEAVY_TIMEOUT", "180"))
CRITIC_MODEL = os.getenv("OLLAMA_CRITIC", "gemma4:31b-cloud")


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
    """Reviewer model (same family by default; override via OLLAMA_CRITIC).
    Small budget: verdicts, not essays."""
    return ChatOllama(
        model=CRITIC_MODEL,
        base_url=BASE_URL,
        temperature=0,
        num_predict=512,
        client_kwargs={"timeout": 60},
    )
