"""Step 10: audit.jsonl logging (deterministic, no LLM).

Each graph node appends one JSON line per security-relevant event:
code_hash (sha256[:16]), safety flags, timings, decisions.
Best-effort: never raises — audit must not break the graph.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


def hash_code(code: str | None) -> str:
    return hashlib.sha256((code or "").encode()).hexdigest()[:16]


def append_audit(workdir: str | Path | None, event: dict) -> Path | None:
    """Append one JSON line to <workdir>/audit.jsonl. Returns path or None."""
    if not workdir:
        return None
    try:
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        entry = {"ts": time.time(), **event}
        path = wd / "audit.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        return path
    except Exception:
        return None


def log_scan(workdir: str | Path | None, *, stage: str, code: str | None, flags: list) -> Path | None:
    return append_audit(workdir, {
        "stage": stage,
        "code_hash": hash_code(code),
        "flags": list(flags or []),
    })


def log_run(
    workdir: str | Path | None,
    *,
    stage: str,
    code: str | None,
    results: list | None,
    timings: list | None,
) -> Path | None:
    passed = None
    try:
        passed = [bool(r.get("passed")) for r in (results or [])] if results else []
    except Exception:
        passed = []
    return append_audit(workdir, {
        "stage": stage,
        "code_hash": hash_code(code),
        "passed": passed,
        "timings_ms": list(timings or []),
    })


def log_decision(workdir: str | Path | None, *, stage: str, decision: str, reason: str = "") -> Path | None:
    return append_audit(workdir, {"stage": stage, "decision": decision, "reason": reason})
