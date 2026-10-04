"""Secure sandbox executor: static_scan -> docker run -> RunResults.

Container sees only the job dir mounted at /work.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .static_scanner import scan

try:
    from ..schemas import ProblemSpec, RunResult
except ImportError:  # pragma: no cover - direct script use
    from src.schemas import ProblemSpec, RunResult

_HERE = Path(__file__).resolve().parent
_HARNESS_SRC = _HERE / "_harness.py"

_DEFAULTS = {
    "timeout_ms": 2000,
    "total_timeout_s": 30,
    "memory": "256m",
    "cpus": "1",
    "pids_limit": "64",
    "network": "none",
    "read_only_rootfs": True,
    "image": "sandbox-img",
    "work_mount": "/work",
}


def _load_sandbox_config() -> dict:
    cfg_path = Path(__file__).resolve().parents[2] / "config" / "sandbox.yaml"
    cfg = dict(_DEFAULTS)
    try:
        import yaml  # type: ignore

        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for k in _DEFAULTS:
            if k in data:
                cfg[k] = data[k]
    except Exception:
        pass
    return cfg


_DAEMON_MARKERS = (
    "cannot connect to the docker",
    "is the docker daemon running",
    "docker.sock",
    "connection refused",
    "context deadline exceeded",
)


def _daemon_unreachable(stderr: str) -> bool:
    """True when docker failed only because the daemon is down (not a job error)."""
    s = (stderr or "").lower()
    return any(m in s for m in _DAEMON_MARKERS)


def _sanitize(text: str | None, workdir: Path) -> str | None:
    if not text:
        return text
    s = str(text)
    try:
        s = s.replace(str(workdir.resolve()), "/work")
    except Exception:
        pass
    s = s.replace(str(workdir), "/work")
    home = os.path.expanduser("~")
    if home and home != "/":
        s = s.replace(home, "~")
    s = s.replace(os.getcwd(), ".")
    # Drop absolute host python frames, keep exception line + message.
    lines = [ln for ln in s.splitlines() if "/usr/local/lib" not in ln and "/opt/" not in ln]
    s = "\n".join(lines)
    if len(s) > 1500:
        s = s[:1500] + "...<truncated>"
    return s


def _normalize_problem(problem) -> ProblemSpec:
    if isinstance(problem, ProblemSpec):
        return problem
    if isinstance(problem, (str, Path)):
        with open(problem, encoding="utf-8") as f:
            return ProblemSpec.model_validate(json.load(f))
    if isinstance(problem, dict):
        return ProblemSpec.model_validate(problem)
    raise TypeError("problem must be ProblemSpec, dict, or path")


def _write_jobdir(workdir: Path, problem: ProblemSpec, solution_code: str) -> tuple[Path, Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    resolved = workdir.resolve()
    problem_path = resolved / "problem.json"
    solution_path = resolved / "solution.py"
    harness_path = resolved / "_harness.py"
    problem_path.write_text(problem.model_dump_json(indent=2), encoding="utf-8")
    solution_path.write_text(solution_code, encoding="utf-8")
    shutil.copyfile(_HARNESS_SRC, harness_path)
    return problem_path, solution_path


def _parse_harness_lines(stdout: str, testcases, workdir: Path) -> list[RunResult]:
    results: list[RunResult] = []
    by_id = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            by_id[json.loads(line)["test_id"]] = json.loads(line)
        except Exception:
            continue
    for tc in testcases:
        tid = tc.id
        if tid in by_id:
            d = by_id[tid]
            results.append(RunResult(
                test_id=tid,
                passed=bool(d.get("passed", False)),
                time_ms=float(d.get("time_ms", 0.0)),
                actual=d.get("actual"),
                error=_sanitize(d.get("error"), workdir),
                stdout=d.get("stdout"),
            ))
        else:
            results.append(RunResult(
                test_id=tid, passed=False, time_ms=0.0,
                actual=None, error="harness: missing output", stdout=None,
            ))
    return results


def _timeout_results(testcases, timeout_ms: float, msg: str) -> list[RunResult]:
    return [RunResult(test_id=tc.id, passed=False, time_ms=float(timeout_ms),
                      actual=None, error=msg, stdout=None) for tc in testcases]


def safe_execute(
    solution_code: str,
    problem,
    workdir: str | Path,
    timeout_ms: int | None = None,
    use_docker: bool = True,
) -> list[RunResult]:
    """Run solution against problem testcases. Static scan first; never touches docker if blocked."""
    cfg = _load_sandbox_config()
    per_test_ms = timeout_ms or int(cfg["timeout_ms"])
    workdir = Path(workdir)
    prob = _normalize_problem(problem)

    verdict = scan(solution_code)
    if not verdict["safe"]:
        msg = f"static-scan-blocked: {','.join(verdict['flags'])}"
        return _timeout_results(prob.testcases, 0.0, msg)

    _write_jobdir(workdir, prob, solution_code)
    resolved = workdir.resolve()
    n = max(1, len(prob.testcases))
    total_s = min(float(cfg["total_timeout_s"]), max(10.0, per_test_ms / 1000.0 * n + 5.0))

    if use_docker and shutil.which("docker") is not None:
        cmd = [
            "docker", "run", "--rm",
            "--network", str(cfg["network"]),
            "--memory", str(cfg["memory"]),
            "--cpus", str(cfg["cpus"]),
            "--pids-limit", str(cfg["pids_limit"]),
            "-v", f"{resolved}:{cfg['work_mount']}",
            str(cfg["image"]),
            "python3", "/work/_harness.py", "/work/problem.json", "/work/solution.py",
        ]
        if cfg.get("read_only_rootfs"):
            cmd.insert(-4, "--read-only")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=total_s)
        except subprocess.TimeoutExpired:
            return _timeout_results(prob.testcases, per_test_ms, f"timeout: exceeded {total_s:.0f}s (killed)")
        except FileNotFoundError:
            proc = None  # fall through to local
        else:
            if proc.returncode == 0 or proc.stdout.strip():
                return _parse_harness_lines(proc.stdout, prob.testcases, resolved)
            err = _sanitize(proc.stderr, resolved) or f"docker exit {proc.returncode}"
            if _daemon_unreachable(proc.stderr or ""):
                pass  # daemon down (env issue, not job fault) -> local fallback below
            else:
                return _timeout_results(prob.testcases, 0.0, f"docker-error: {err}")
        if proc is None:
            pass  # docker binary vanished; use local fallback below

    # Local fallback (no docker or use_docker=False): same harness, same timeout.
    cmd = [sys.executable, str(resolved / "_harness.py"),
           str(resolved / "problem.json"), str(resolved / "solution.py")]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=total_s, cwd=str(resolved))
    except subprocess.TimeoutExpired:
        return _timeout_results(prob.testcases, per_test_ms, f"timeout: exceeded {total_s:.0f}s (killed)")
    if proc.returncode != 0 and not proc.stdout.strip():
        err = _sanitize(proc.stderr, resolved) or f"local-harness exit {proc.returncode}"
        return _timeout_results(prob.testcases, 0.0, f"harness-error: {err}")
    return _parse_harness_lines(proc.stdout, prob.testcases, resolved)


def build_docker_command(workdir: str | Path) -> list[str]:
    """Expose exact docker flags for tests (container sees only /work)."""
    cfg = _load_sandbox_config()
    resolved = Path(workdir).resolve()
    cmd = ["docker", "run", "--rm", "--network", str(cfg["network"]),
           "--memory", str(cfg["memory"]), "--cpus", str(cfg["cpus"]),
           "--pids-limit", str(cfg["pids_limit"])]
    if cfg.get("read_only_rootfs"):
        cmd.append("--read-only")
    cmd += ["-v", f"{resolved}:{cfg['work_mount']}", str(cfg["image"])]
    return cmd
