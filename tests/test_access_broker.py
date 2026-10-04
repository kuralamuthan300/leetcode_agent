"""Step 2 tests: legit jail I/O works, all escapes raise PermissionError."""

import os
import shutil
from pathlib import Path

import pytest

from src.tools import access_broker
from src.tools.access_broker import create_job, promote_to_problems, scoped_read, scoped_write

REPO_DEMO_REQ = (
    Path(__file__).resolve().parent.parent
    / "workspace"
    / "jobs"
    / "examples"
    / "requirements_id_demo.json"
)


@pytest.fixture()
def workdir(tmp_path):
    src = tmp_path / "requirements_id_demo.json"
    src.write_text(REPO_DEMO_REQ.read_text())
    wd = create_job(src)
    yield wd
    shutil.rmtree(wd, ignore_errors=True)


def test_legit_write_read(workdir):
    scoped_write(workdir, "notes.txt", "hello jail")
    assert scoped_read(workdir, "notes.txt") == "hello jail"
    scoped_write(workdir, "sub/dir/data.json", '{"a": 1}')
    assert scoped_read(workdir, "sub/dir/data.json") == '{"a": 1}'


def test_create_job_copies_single_file(tmp_path):
    src = tmp_path / "requirements_id_demo.json"
    src.write_text(REPO_DEMO_REQ.read_text())
    wd = create_job(src)
    try:
        files = [p for p in wd.iterdir() if p.is_file()]
        assert len(files) == 1 and files[0].name == src.name
        assert files[0].read_text() == src.read_text()
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def test_dotdot_blocked(workdir):
    with pytest.raises(PermissionError):
        scoped_read(workdir, "../requirements_id_demo.json")
    with pytest.raises(PermissionError):
        scoped_write(workdir, "../escape.txt", "nope")
    with pytest.raises(PermissionError):
        scoped_write(workdir, "sub/../../escape.txt", "nope")


def test_absolute_blocked(workdir):
    with pytest.raises(PermissionError):
        scoped_read(workdir, "/etc/passwd")
    with pytest.raises(PermissionError):
        scoped_write(workdir, "/etc/passwd", "nope")
    with pytest.raises(PermissionError):
        scoped_read(workdir, str(REPO_DEMO_REQ))


def test_symlink_escape_blocked(workdir, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret")
    link = workdir / "link.txt"
    try:
        os.symlink(secret, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not supported on this platform")
    with pytest.raises(PermissionError):
        scoped_read(workdir, "link.txt")


def test_promote_copies_file(workdir):
    scoped_write(workdir, "problem_id_demo.json", '{"id": "x"}')
    dest = promote_to_problems(workdir, "problem_id_demo.json")
    try:
        assert dest.is_file() and dest.read_text() == '{"id": "x"}'
    finally:
        dest.unlink(missing_ok=True)


def test_promote_blocks_escape(workdir):
    with pytest.raises(PermissionError):
        promote_to_problems(workdir, "../requirements_id_demo.json")
    with pytest.raises(PermissionError):
        promote_to_problems(workdir, "/etc/passwd")
