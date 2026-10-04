"""LLM router: single entry for all LLM calls (heavy/light + JSON enforcement)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from langchain_ollama import ChatOllama

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

_DEFAULTS = {
    "heavy": {"model": "gemma4:31b-cloud", "base_url": "http://localhost:11434"},
    "light": {"model": "deepseek-r1:1.5b", "base_url": "http://localhost:11434"},
    "routing": {"light_tasks": ["classify", "extract", "format", "summarize"],
                "escalate_to_heavy_on_retry": True},
}


def load_config(path: str | Path | None = None) -> dict:
    cfg = {k: dict(v) for k, v in _DEFAULTS.items()}
    p = Path(path) if path else _CONFIG_PATH
    try:
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for section in ("heavy", "light"):
            if section in data and isinstance(data[section], dict):
                cfg[section].update(data[section])
        if "routing" in data and isinstance(data["routing"], dict):
            cfg["routing"].update(data["routing"])
    except FileNotFoundError:
        pass
    return cfg


_CONFIG = load_config()
LIGHT_TASKS: list[str] = list(_CONFIG["routing"].get("light_tasks", []))
ESCALATE_ON_RETRY: bool = bool(_CONFIG["routing"].get("escalate_to_heavy_on_retry", True))

_clients: dict[str, ChatOllama] = {}


def pick_model(task: str, retry_count: int = 0) -> str:
    """Return 'light' or 'heavy'. Light tasks escalate to heavy on retry."""
    if retry_count >= 1 and ESCALATE_ON_RETRY:
        return "heavy"
    t = (task or "").lower()
    for keyword in LIGHT_TASKS:
        if keyword.lower() in t:
            return "light"
    return "heavy"


def get_llm(which: str, temperature: float | None = None) -> ChatOllama:
    cfg = _CONFIG.get(which, {})
    model = cfg.get("model", _DEFAULTS[which]["model"])
    base_url = cfg.get("base_url", "http://localhost:11434")
    key = f"{which}:{temperature}"
    if key not in _clients:
        kwargs: dict[str, Any] = {"model": model, "base_url": base_url}
        if temperature is not None:
            kwargs["temperature"] = temperature
        elif "temperature_default" in cfg:
            kwargs["temperature"] = cfg["temperature_default"]
        _clients[key] = ChatOllama(**kwargs)
    return _clients[key]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _extract_json(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("no JSON found", text, 0)


def generate(
    task: str,
    prompt: str,
    json_mode: bool = False,
    retry_count: int = 0,
    schema: Any | None = None,
    temperature: float | None = None,
    max_json_retries: int = 2,
) -> Any:
    """Route to light/heavy, call Ollama, optionally enforce valid JSON.

    - light for classify|extract|format|summarize else heavy
    - escalate to heavy when retry_count >= 1
    - json_mode: retry up to max_json_retries parse fixes, validate with
      Pydantic schema when given, else raise after retries.
    """
    which = pick_model(task, retry_count)
    llm = get_llm(which, temperature)
    effective_prompt = prompt + ("\n\nReturn valid JSON only." if json_mode else "")
    attempts = (max_json_retries + 1) if json_mode else 1
    last_err: Exception | None = None
    for attempt in range(attempts):
        if attempt > 0:
            # Escalate mid-loop and ask for a fix explicitly.
            llm = get_llm("heavy" if ESCALATE_ON_RETRY else which, temperature)
            effective_prompt = (
                prompt + "\n\nPrevious output was not valid JSON. Return valid JSON only."
            )
        resp = llm.invoke(effective_prompt)
        text = _content_to_text(getattr(resp, "content", resp))
        if not json_mode:
            return text
        try:
            parsed = _extract_json(text)
            if schema is not None:
                if hasattr(schema, "model_validate"):
                    schema.model_validate(parsed)
                elif isinstance(schema, dict):
                    pass
            return parsed
        except Exception as exc:  # JSON error or Pydantic ValidationError
            last_err = exc
            continue
    raise ValueError(f"generate(json_mode=True) failed after {attempts} attempts: {last_err}")


def healthcheck() -> int:
    """Check Ollama reachable + both tags listed. Returns exit code."""
    import urllib.request

    ok = True
    base_url = _CONFIG["heavy"].get("base_url", "http://localhost:11434")
    try:
        with urllib.request.urlopen(base_url, timeout=5) as r:
            print(f"ollama reachable: {base_url} -> HTTP {r.status}")
    except Exception as exc:
        print(f"ollama NOT reachable at {base_url}: {exc}")
        ok = False
    for which in ("light", "heavy"):
        try:
            llm = get_llm(which)
            resp = llm.invoke("ping (reply with pong)")
            print(f"{which} ({llm.model}): {_content_to_text(getattr(resp, 'content', resp))[:80]}")
        except Exception as exc:
            print(f"{which} healthcheck FAILED: {exc}")
            ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args(argv)
    if args.healthcheck:
        return healthcheck()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
