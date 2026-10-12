"""Guardrail corpus: malicious code must block, legit DSA must pass."""

BLOCKED = [
    "open('/etc/passwd').read()",
    "open('x.txt', 'w').write('hi')",
    "import os\nos.remove('f')",
    "import os\nprint(os.environ)",
    "import subprocess\nsubprocess.run(['ls'])",
    "import socket\nsocket.socket()",
    "from pathlib import Path\nPath('/tmp/x').write_text('hi')",
    "import shutil\nshutil.rmtree('/')",
    "eval('1+1')",
    "exec('x=1')",
    "__import__('os').system('ls')",
    "x = [].__class__.__subclasses__()",
    "import pickle\npickle.loads(b'x')",
    "print('/etc/passwd')",
    "import math\nimport os\nprint(math.pi)",
]


def test_blocked_corpus():
    from agent.tools.guardrail import check

    for src in BLOCKED:
        assert check(src)["allowed"] is False, src


def test_legit_code_allowed():
    from agent.tools.guardrail import check

    code = (
        "import math\nfrom typing import List\n\n"
        "def solve(nums: List[int]) -> List[int]:\n"
        "    return sorted(nums)\n"
    )
    harness = "print(solve([3, 1, 2]))"
    assert check(code, harness)["allowed"] is True


def test_blocked_never_executes():
    from agent.tools.sandbox import run_python

    r = run_python("open('/tmp/pwned_by_test', 'w').write('x')")
    assert r["error"].startswith("blocked:")
    import os

    assert not os.path.exists("/tmp/pwned_by_test")


def test_oversized_rejected():
    from agent.tools.guardrail import check

    assert check("x = 1\n" * 5000)["allowed"] is False


def test_report_has_timestamp():
    from agent.tools import guardrail_report

    r = guardrail_report("x = 1")
    assert r["allowed"] is True and "checked_at" in r
