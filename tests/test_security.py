"""Step 10: security hardening — 8 attacks must block before Docker.

Each payload must: fail static scan (safe=False, flags non-empty) AND
return static-scan-blocked from safe_execute without touching docker
(subprocess.run patched to raise if reached).
"""

import json
from pathlib import Path

import pytest

from src.schemas import ProblemSpec
from src.tools.safe_executor import safe_execute
from src.tools.static_scanner import scan

ROOT = Path(__file__).resolve().parent.parent
PROBLEM_DEMO = ROOT / "workspace" / "problems" / "examples" / "problem_id_two_sum_demo.json"

ATTACKS = [
    ("os-remove", "def two_sum(nums, target):\n    import os\n    os.remove('/tmp/x')\n    return []\n"),
    ("shutil-rmtree", "def two_sum(nums, target):\n    import shutil\n    shutil.rmtree('/')\n    return []\n"),
    ("subprocess-run", "def two_sum(nums, target):\n    import subprocess\n    subprocess.run(['ls'])\n    return []\n"),
    ("socket-exfil", "def two_sum(nums, target):\n    import socket\n    s = socket.socket()\n    return []\n"),
    ("eval-call", "def two_sum(nums, target):\n    return eval('1+1')\n"),
    ("dunder-import", "def two_sum(nums, target):\n    m = __import__('os')\n    return []\n"),
    ("open-traversal", "def two_sum(nums, target):\n    open('../../etc/passwd').read()\n    return []\n"),
    ("pathlib-absolute", "def two_sum(nums, target):\n    from pathlib import Path\n    return str(Path('/etc/passwd'))\n"),
]


@pytest.mark.parametrize("name,code", ATTACKS)
def test_attack_blocked_by_scan(name, code):
    verdict = scan(code)
    assert verdict["safe"] is False, f"{name} passed scan"
    assert verdict["flags"], f"{name} has no flags"


@pytest.mark.parametrize("name,code", ATTACKS)
def test_attack_never_reaches_docker(name, code, tmp_path, monkeypatch):
    import subprocess as sp

    def _no_docker(*a, **k):
        raise AssertionError(f"{name} reached subprocess (docker/local)")

    monkeypatch.setattr(sp, "run", _no_docker)
    prob = ProblemSpec.model_validate(json.loads(PROBLEM_DEMO.read_text()))
    results = safe_execute(code, prob, tmp_path / name)
    assert results, f"{name} returned no results"
    assert all("static-scan-blocked" in (r.error or "") for r in results), (
        f"{name} not marked scan-blocked: {[r.error for r in results]}"
    )
    assert all(r.passed is False for r in results)
