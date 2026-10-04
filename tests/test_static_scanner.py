"""Step 3 tests: 8 malicious samples blocked, 2 clean samples pass."""

import pytest

from src.tools.static_scanner import scan

MALICIOUS = {
    "os_remove": "import os\nos.remove('/tmp/x')\n",
    "shutil_rmtree": "import shutil\nshutil.rmtree('/')\n",
    "socket_connect": "import socket\ns = socket.socket()\ns.connect(('evil.com', 4444))\n",
    "subprocess_run": "import subprocess\nsubprocess.run(['ls', '/'])\n",
    "pathlib_and_open_escape": (
        "from pathlib import Path\n"
        "p = Path('../../etc/passwd')\n"
        "data = open('../../etc/passwd', 'w')\n"
    ),
    "eval_dunder_import_compile": (
        "x = eval(\"__import__('os').getcwd()\")\n"
        "y = compile('1', '<s>', 'eval')\n"
    ),
    "exec_input_breakpointhook": (
        "exec('x = 1')\n"
        "name = input('name: ')\n"
        "breakpointhook()\n"
    ),
    "threading_and_friends": (
        "import threading\nimport multiprocessing\nimport ctypes\nimport inspect\nimport sys\n"
        "threading.Thread().start()\n"
    ),
}

CLEAN = {
    "two_sum_hashmap": (
        "from typing import List\n"
        "def two_sum(nums: List[int], target: int) -> List[int]:\n"
        "    seen = {}\n"
        "    for i, n in enumerate(nums):\n"
        "        if target - n in seen:\n"
        "            return [seen[target - n], i]\n"
        "        seen[n] = i\n"
        "    return []\n"
    ),
    "heapq_math": (
        "import heapq\nimport math\nfrom collections import Counter\n"
        "def top_k(nums, k):\n"
        "    c = Counter(nums)\n"
        "    return heapq.nlargest(k, c, key=c.get)\n"
    ),
}


@pytest.mark.parametrize("name", sorted(MALICIOUS))
def test_malicious_blocked(name):
    result = scan(MALICIOUS[name])
    assert result["safe"] is False, name
    assert len(result["flags"]) > 0, name


@pytest.mark.parametrize("name", sorted(CLEAN))
def test_clean_passes(name):
    result = scan(CLEAN[name])
    assert result == {"safe": True, "flags": []}, (name, result)


def test_syntax_error_is_unsafe():
    result = scan("def broken(:\n")
    assert result["safe"] is False
    assert any(f.startswith("syntax-error") for f in result["flags"])


def test_done_criteria_spot_checks():
    for code in (
        "import os\nos.remove('x')\n",
        "import shutil\nshutil.rmtree('x')\n",
        "import socket\n",
        "import subprocess\n",
        "open('../../x')\n",
    ):
        assert scan(code)["safe"] is False, code
