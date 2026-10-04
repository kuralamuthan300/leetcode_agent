# Security Tests — Step 10 (8/8 blocked)

Date: 2026-10-04. Suite: `uv run pytest tests/test_security.py -v` (16 passed = 8 scan + 8 no-docker).

Policy: AST scan first (`src/tools/static_scanner.py`); `safe_execute` returns
`static-scan-blocked` without touching Docker when unsafe. Container flags
(`src/tools/safe_executor.py::build_docker_command`): `--network none`,
`--memory 256m`, `--cpus 1`, `--pids-limit 64`, `--read-only`, only
`<jobdir>:/work`.

| # | Attack | Signal | Blocked |
| - | ------ | ------ | ------- |
| 1 | `import os; os.remove('/tmp/x')` | `blocked-import:os` | ✅ |
| 2 | `import shutil; shutil.rmtree('/')` | `blocked-import:shutil` | ✅ |
| 3 | `import subprocess; subprocess.run(['ls'])` | `blocked-import:subprocess` | ✅ |
| 4 | `import socket; socket.socket()` | `blocked-import:socket` | ✅ |
| 5 | `eval('1+1')` | `blocked-call:eval` | ✅ |
| 6 | `__import__('os')` | `blocked-call:__import__` | ✅ |
| 7 | `open('../../etc/passwd')` | `blocked-call:open` + `path-escape` | ✅ |
| 8 | `from pathlib import Path; Path('/etc/passwd')` | `blocked-import:pathlib` + `path-escape` | ✅ |

Result: 8/8 blocked, 0 reached Docker (`subprocess.run` patched to raise in test).
Jail (`access_broker`) and E2E log (`docs/reports/e2e_step10.log`) verified separately.
