from __future__ import annotations

import os

from langchain_ollama import ChatOllama

HEAVY_MODEL = os.getenv("OLLAMA_HEAVY", "gemma4:31b-cloud")
SMALL_MODEL = os.getenv("OLLAMA_SMALL", "deepseek-r1:1.5b")
BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def get_small() -> ChatOllama:
    return ChatOllama(model=SMALL_MODEL, base_url=BASE_URL, temperature=0)


def get_heavy() -> ChatOllama:
    # Cloud models (e.g. gemma4:31b-cloud) route through the local Ollama
    # daemon which holds the cloud auth; same base_url works.
    return ChatOllama(model=HEAVY_MODEL, base_url=BASE_URL, temperature=0.2)
