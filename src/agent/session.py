"""Per-session memory for the RequirementGatherer.

Stores confirmed conventions (e.g. "arrays are 1-D unless stated") so repeat
choices aren't re-asked within a session. Explicitly short-lived: default file
lives in /tmp (OS-cleared on reboot); non-TTY runs ignore it entirely.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MAX_CONVENTIONS = 20


def path() -> Path:
    override = os.getenv("LEETCODEAGENT_SESSION", "").strip()
    return Path(override) if override else Path("/tmp/leetcodeagent_session.json")


def load() -> list[str]:
    try:
        data = json.loads(path().read_text())
        items = data.get("conventions", [])
        return [str(c)[:200] for c in items if str(c).strip()][:MAX_CONVENTIONS]
    except (OSError, ValueError, AttributeError):
        return []


def save(conventions: list[str]) -> None:
    seen: list[str] = []
    for c in conventions:
        c = str(c).strip()[:200]
        if c and c not in seen:
            seen.append(c)
    try:
        path().write_text(json.dumps({"conventions": seen[:MAX_CONVENTIONS]}, indent=2))
    except OSError:
        pass


def remember(new_items: list[str]) -> list[str]:
    """Merge new conventions into memory; return the full list."""
    merged = load()
    for item in new_items:
        item = str(item).strip()[:200]
        if item and item not in merged:
            merged.append(item)
    save(merged)
    return merged


def clear() -> None:
    try:
        path().unlink()
    except OSError:
        pass
