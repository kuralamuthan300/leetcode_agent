"""Mocked tests for src.llm_router (no real Ollama needed)."""

import json

import pytest
from pydantic import BaseModel

import src.llm_router as router


class _Msg:
    def __init__(self, content):
        self.content = content


class _FakeLLM:
    instances: list = []

    def __init__(self, model, base_url=None, **kwargs):
        self.model = model
        self.base_url = base_url
        self.kwargs = kwargs
        self.calls: list[str] = []
        _FakeLLM.instances.append(self)
        self.script: list[str] = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if self.script:
            return _Msg(self.script.pop(0))
        return _Msg('{"ok": true}')


@pytest.fixture(autouse=True)
def _patch_chat(monkeypatch):
    _FakeLLM.instances.clear()
    router._clients.clear()
    monkeypatch.setattr(router, "ChatOllama", _FakeLLM)
    yield


def _last_instance():
    return _FakeLLM.instances[-1]


def test_light_tasks_route_to_light():
    for task in ["classify", "extract", "format", "summarize", "extract_requirements"]:
        router._clients.clear()
        _FakeLLM.instances.clear()
        router.generate(task, "hello")
        assert _last_instance().model == "deepseek-r1:1.5b", task


def test_heavy_tasks_route_to_heavy():
    for task in ["draft_problem", "solve", "optimize", "reason"]:
        router._clients.clear()
        _FakeLLM.instances.clear()
        router.generate(task, "hello")
        assert _last_instance().model == "gemma4:31b-cloud", task


def test_escalate_light_to_heavy_on_retry():
    router._clients.clear()
    _FakeLLM.instances.clear()
    router.generate("classify", "hello", retry_count=1)
    assert _last_instance().model == "gemma4:31b-cloud"


def test_pick_model_unit():
    assert router.pick_model("classify") == "light"
    assert router.pick_model("format_code") == "light"
    assert router.pick_model("draft_problem") == "heavy"
    assert router.pick_model("classify", retry_count=1) == "heavy"


def test_json_mode_parses_direct():
    out = router.generate("solve", "p", json_mode=True)
    assert out == {"ok": True}


def test_json_mode_retries_then_succeeds(monkeypatch):
    llm = router.get_llm("heavy")
    llm.script = ["not json at all", '{"a": 1}']
    out = router.generate("solve", "p", json_mode=True)
    assert out == {"a": 1}
    assert len(llm.calls) == 2


def test_json_mode_raises_after_retries():
    llm = router.get_llm("heavy")
    llm.script = ["nope", "still nope", "never json"]
    with pytest.raises(ValueError, match="failed after"):
        router.generate("solve", "p", json_mode=True)


def test_json_mode_with_pydantic_schema_retries():
    class Item(BaseModel):
        name: str
        qty: int

    llm = router.get_llm("heavy")
    llm.script = ['{"name": "x"}', '{"name": "x", "qty": 2}']
    out = router.generate("solve", "p", json_mode=True, schema=Item)
    assert out == {"name": "x", "qty": 2}


def test_config_values_used():
    router._clients.clear()
    _FakeLLM.instances.clear()
    router.generate("solve", "hi")
    inst = _last_instance()
    assert inst.base_url == "http://localhost:11434"
