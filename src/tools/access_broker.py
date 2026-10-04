"""File jail: agent sees only workspace/jobs/<job_id>/ with scoped reads/writes (Step 2)."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

_JOBS_ROOT = Path(__file__).resolve().parent.parent.parent / "workspace" / "jobs"
_PROBLEMS_ROOT = Path(__file__).resolve().parent.parent.parent / "workspace" / "problems"


def _resolve_workdir(workdir: str | Path) -> Path:
    wd = Path(workdir).resolve()
    root = _JOBS_ROOT.resolve()
    if wd != root and root not in wd.parents:
        raise PermissionError(f"workdir escapes jobs root: {workdir!r}")
    return wd


def _scoped_path(workdir: str | Path, path: str | Path) -> Path:
    if isinstance(path, Path):
        raw = path
    else:
        raw = Path(path)
    if raw.is_absolute():
        raise PermissionError(f"absolute paths are blocked: {path!r}")
    wd = _resolve_workdir(workdir)
    resolved = (wd / raw).resolve()
    if resolved != wd and wd not in resolved.parents:
        raise PermissionError(f"path escapes job dir: {path!r}")
    return resolved


def create_job(source_path: str | Path) -> Path:
    """Create workspace/jobs/<uuid8>/ and copy the ONE source file into it."""
    src = Path(source_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"source file not found: {source_path!r}")
    job_id = uuid.uuid4().hex[:8]
    workdir = _JOBS_ROOT.resolve() / job_id
    workdir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(src, workdir / src.name)
    return workdir


def scoped_read(workdir: str | Path, path: str | Path) -> str:
    """Read a file inside workdir. Anything else raises PermissionError."""
    target = _scoped_path(workdir, path)
    if not target.is_file():
        raise FileNotFoundError(f"file not found in job dir: {path!r}")
    return target.read_text()


def scoped_write(workdir: str | Path, path: str | Path, content: str | bytes) -> Path:
    """Write a file inside workdir (subdirs allowed). Returns the resolved path."""
    target = _scoped_path(workdir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Re-check after mkdir in case a symlink component was created to escape.
    final = target.resolve()
    wd = _resolve_workdir(workdir)
    if final != wd and wd not in final.parents:
        raise PermissionError(f"path escapes job dir: {path!r}")
    if isinstance(content, bytes):
        final.write_bytes(content)
    else:
        final.write_text(content)
    return final


def promote_to_problems(workdir: str | Path, filename: str | Path) -> Path:
    """Copy a file from workdir to workspace/problems/ (plain copy, no validation)."""
    src = _scoped_path(workdir, filename)
    if not src.is_file():
        raise FileNotFoundError(f"file not found in job dir: {filename!r}")
    dest = _PROBLEMS_ROOT.resolve() / src.name
    shutil.copy2(src, dest)
    return dest
