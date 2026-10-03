# LeetCode Agent (LangGraph + Ollama + Secure Sandbox)

See `docs/architecture/LEETCODE_AGENT_PLAN.md` for architecture and `docs/build/BUILD_PLAN_STEP_BY_STEP.md` for build order.

## Prerequisites

- Python 3.11+ (pinned via `.python-version`, managed by `uv`)
- `uv` (install: https://docs.astral.sh/uv/getting-started/installation/)
- Docker (for sandbox)
- Ollama running at `http://localhost:11434`

## Setup

```bash
uv sync
ollama pull gemma4:31b-cloud
ollama pull deepseek-r1:1.5b
ollama list  # verify both tags exist
docker build -f Dockerfile.sandbox -t sandbox-img .  # added in Step 4
```

## Verify Step 0

```bash
uv sync
uv run pytest -q
```

Expected: 0 tests collected, configs load.
