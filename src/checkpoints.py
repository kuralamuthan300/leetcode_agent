"""Step 10: SqliteSaver checkpointer factory (persistent, thread-safe).

Default is in-memory SQLite (no cross-run pollution for tests).
Set LEETCODE_CHECKPOINT_DB to a file path for prod persistence
(e.g. workspace/checkpoints.sqlite).
Falls back to MemorySaver if sqlite is unavailable.
"""

from __future__ import annotations

import os
import sqlite3


def get_checkpointer():
    """Return a SqliteSaver (or MemorySaver fallback). Caller owns lifetime."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        db = os.environ.get("LEETCODE_CHECKPOINT_DB", ":memory:")
        if db != ":memory:":
            from pathlib import Path

            Path(db).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db, check_same_thread=False)
        return SqliteSaver(conn)
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
