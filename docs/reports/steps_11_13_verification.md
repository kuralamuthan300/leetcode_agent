# Verification — Steps 11–13 (UI + HITL), 2026-10-04

All runs mocked where noted (no Ollama/Docker). Full suite: `uv run pytest -q`.

## Step 11 — Streamlit UI shell

- `uv run pytest tests/test_ui_smoke.py -v` → **4 passed**
  (status mapping, stepper order 9 creator / 13 solver, mocked client wrapper,
  tab data).
- Full suite at merge → **79 passed, 1 skipped**.
- `uv run python -c "import ui.agent_client, ui.app"` → imports ok.
- `streamlit 1.65.0` (`streamlit>=1.32` in `pyproject.toml` + `uv.lock`).
- Manual: `uv run streamlit run ui/app.py`.

## Step 12 — HITL understanding + clarify + approve/decline

- `uv run pytest tests/test_hitl_understanding.py -v` → **9 passed**
  - understanding blocks heavy work: `pending_approval="understanding"`,
    `needs_human`, zero `draft_problem` calls, zero executor calls.
  - resume via `update_state({human_answer})` + re-invoke completes;
    answer present in draft/initial prompts.
  - legacy default runs skip the gate (headless compat).
  - promote/opt decline → repair (`attempt>=1`) + `declined` in `audit.jsonl`;
    no-budget decline → human review.
  - clarify pauses at `attempt>=2` with question + audit entry; resume retries.
- Full suite at merge → **88 passed, 1 skipped**.

## Step 13 — UI E2E + docs

- `uv run pytest tests/test_ui_e2e_step13.py -v` → **3 passed**
  - creator: requirements(`num_tests=5`) → understanding confirm →
    streamed stage order → 5-row all-pass table → `pass_rate 5/5`;
    `audit.jsonl` has `understanding_confirmed` + `promoted`.
  - decline path: `understanding_confirmed` + `declined` (reason) + `promoted`.
  - solver: understanding confirm → all-pass → `understanding_confirmed` +
    `accepted`/`rolled-back` in audit.
- Full suite → **91 passed, 1 skipped** (Steps 0–12 intact).
- README documents `uv run streamlit run ui/app.py` (modes, `num_tests`,
  HITL-1/HITL-2 buttons, clarify popup).
