import shutil
import subprocess

import pytest

from src.tools.safe_executor import build_docker_command, safe_execute

DEMO_PROBLEM = {
    "id": "two_sum_demo",
    "title": "Two Sum",
    "difficulty": "easy",
    "statement": "Return indices of two numbers adding to target.",
    "function_name": "twoSum",
    "signature": "def twoSum(nums: list[int], target: int) -> list[int]",
    "examples": [{"input": {"nums": [2, 7, 11, 15], "target": 9}, "output": [0, 1]}],
    "constraints": ["2 <= nums.length <= 10^4"],
    "testcases": [
        {"id": "t1", "input": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1], "is_hidden": False, "timeout_ms": 2000},
        {"id": "t2", "input": {"nums": [3, 2, 4], "target": 6}, "expected": [1, 2], "is_hidden": False, "timeout_ms": 2000},
    ],
    "oracle_solution_hash": None,
}

CLEAN = "def twoSum(nums, target):\n    seen = {}\n    for i, v in enumerate(nums):\n        if target - v in seen:\n            return [seen[target - v], i]\n        seen[v] = i\n"

MALICIOUS = "import os\ndef twoSum(nums, target):\n    os.remove('/tmp/x')\n    return [0, 1]\n"

LOOP = "def twoSum(nums, target):\n    while True:\n        pass\n"


def test_clean_passes_with_timings(tmp_path):
    results = safe_execute(CLEAN, DEMO_PROBLEM, tmp_path, use_docker=False)
    assert len(results) == 2
    assert all(r.passed for r in results)
    assert all(r.time_ms >= 0 for r in results)
    assert all(r.error is None for r in results)


def test_infinite_loop_times_out(tmp_path):
    results = safe_execute(LOOP, DEMO_PROBLEM, tmp_path, timeout_ms=1000, use_docker=False)
    assert len(results) == 2
    assert all(not r.passed for r in results)
    assert all("timeout" in (r.error or "") for r in results)


def test_malicious_blocked_never_reaches_docker(tmp_path, monkeypatch):
    def _fail(*a, **k):
        raise AssertionError("docker/subprocess must not run when scan blocks")
    monkeypatch.setattr(subprocess, "run", _fail)
    results = safe_execute(MALICIOUS, DEMO_PROBLEM, tmp_path, use_docker=True)
    assert all(not r.passed for r in results)
    assert all("static-scan-blocked" in (r.error or "") for r in results)


def test_container_sees_only_work(tmp_path):
    cmd = build_docker_command(tmp_path)
    assert "--network" in cmd and "none" in cmd
    assert "--read-only" in cmd
    assert "--memory" in cmd and "256m" in cmd
    assert "--cpus" in cmd and "--pids-limit" in cmd
    vols = [c for c in cmd if "/work" in c]
    assert len(vols) == 1 and str(tmp_path.resolve()) in vols[0]
    assert not any(":/workspace" in c or ":/app" in c or ":/src" in c for c in cmd)


@pytest.mark.skipif(shutil.which("docker") is None, reason="no docker")
def test_docker_clean_if_image_present(tmp_path):
    # Real docker only if image was built; else skip gracefully.
    chk = subprocess.run(["docker", "images", "-q", "sandbox-img"], capture_output=True, text=True)
    if not chk.stdout.strip():
        pytest.skip("sandbox-img not built")
    results = safe_execute(CLEAN, DEMO_PROBLEM, tmp_path, use_docker=True)
    assert all(r.passed for r in results)
