"""Prompts + lenient output parsing for Steps 1-4.

All LLM calls go through `try_invoke`, which returns None on any failure so
nodes can fall back to deterministic behavior (tests stay hermetic / offline).
"""

from __future__ import annotations

import json
import re

SOLVE_CONTRACT = (
    "Define plain top-level Python code. The entry point MUST be a top-level "
    "function named `solve`. Use only the standard library modules: math, "
    "typing, collections, itertools, functools, heapq, bisect. No file, "
    "network, OS, subprocess, or shell access. No input() — the harness calls "
    "`solve` directly."
)

INTENT_SYSTEM = (
    "You clarify data-structures & algorithms problem statements. "
    "Rewrite the user's requirement as 2-4 crisp sentences covering: goal, "
    "input format, output format, and any constraints/examples. "
    "If details are missing, state sensible defaults explicitly as "
    "'Assumption: ...'. Reply with ONLY the clarified statement."
)

PLANNER_SYSTEM = (
    "You plan solutions to DSA problems. Given a clarified problem (and optional "
    "feedback from a failed attempt), output a short plan: one approach, its "
    "time/space complexity, then numbered implementation steps. "
    "End your reply with a fenced JSON contract block describing inputs. "
    "The contracted function MUST be named `solve`, e.g.\n"
    "CONTRACT:\n"
    "```json\n"
    '{"signature": "def solve(arr, k)", '
    '"args": [{"name": "arr", "type": "List[int]", "range": "len 1..100000, values -1000000000..1000000000"}, '
    '{"name": "k", "type": "int", "range": "0..1000000000"}], '
    '"returns": "List[int]"}\n'
    "```\n"
    "Reply with ONLY the plan lines plus the contract block."
)

TESTGEN_SYSTEM = (
    "You write test cases for DSA problems. Given the problem and plan, output "
    "a JSON array of 3 to 5 test objects. Each object has exactly these keys: "
    "id (short string), harness (a Python snippet that calls `solve` and MUST "
    "use print to output the result), expected (the exact printed output as a "
    "string, never null or an array). Derive `expected` by mentally executing "
    "the harness step by step — never guess. Example:\n"
    '[{"id": "base", "harness": "print(solve([2, 7, 11, 15], 9))", '
    '"expected": "[0, 1]"}]\n'
    "Reply with ONLY the JSON array, no prose."
)

CODER_SYSTEM = (
    "You write Python solutions to DSA problems. " + SOLVE_CONTRACT + " "
    "Reply with ONLY the code (a python fenced block is fine)."
)

CODER_REPAIR_SYSTEM = (
    "You fix Python solutions. " + SOLVE_CONTRACT + " "
    "The previous code failed its smoke run; the error is given. "
    "Reply with ONLY the corrected code."
)

SCOPE_SYSTEM = (
    "You are a strict scope gate for a pure data-structures & algorithms "
    "Python solver. Reply with exactly one line: ALLOW if the request is a "
    "DSA coding task (arrays, strings, trees, graphs, sorting, searching, "
    "dynamic programming, etc.), else REJECT: <short reason>. Be strict."
)

# Quiet mode: validate + normalize in one call, without ever asking the user.
QUIET_GATHER_SYSTEM = (
    "You validate a data-structures & algorithms requirement WITHOUT asking the "
    "user anything. Rewrite it as 2-4 crisp sentences covering goal, input format, "
    "output format and constraints, then judge whether it is solvable as given. "
    "Reply with ONLY a JSON object with exactly these keys: "
    "clarified_intent (string), workable (boolean), reason (short string), "
    "assumptions (array of strings you had to make), blocking_ambiguities (array "
    "of strings — ONLY issues that would change the correct answer), confidence "
    '("low"|"medium"|"high"), rejection_class (null when workable, else one of: '
    "under_specified, nondeterministic, noncoding_task, undeliverable, "
    "contradictory, debug_unprovided, unsandboxable). "
    "Be strict: if an ambiguity changes the expected answer, set workable=false "
    "and list it in blocking_ambiguities. Missing constraints or examples are "
    "assumptions, NOT blocking. Reply with ONLY the JSON object."
)

# Rejection taxonomy. Hard gates own scope/language/incomplete; the rest come
# from the quiet-mode verdict. `infra` is reserved for transport/model failures.
REJECTION_CLASSES = (
    "scope", "language", "incomplete", "noncoding_task", "undeliverable",
    "under_specified", "nondeterministic", "contradictory", "debug_unprovided",
    "unsandboxable",
)
RUBRIC_VERSION = "1"

CRITIC_SYSTEM = (
    "You are a strict reviewer. Given the artifact and checklist, reply with "
    "exactly one line: PASS if it satisfies every checklist item, else "
    "ISSUES: <short comma-separated list>. Be strict but fair."
)

JUDGE_SYSTEM = (
    "You adjudicate a failed test. Given the problem, the function contract, "
    "the solution code, and one failing test (harness, expected, actual, error), "
    "decide who is wrong. Reply with exactly one line: "
    "CODE_WRONG: <reason> if the solution is buggy, "
    "TEST_WRONG: <reason> if the test's expectation or inputs are wrong, "
    "else UNKNOWN."
)

DIAGNOSE_SYSTEM = (
    "You are a strict judge for a DSA solver. Given the problem, the function "
    "contract, the candidate solution code, and the failing tests (harness, "
    "expected, actual, error), decide whether the CODE is at fault or the TESTS "
    "are at fault, and give a one-to-two sentence root-cause reason. "
    "Reply with ONLY a JSON object: "
    '{"culprit": "code" or "test", "reason": "<short explanation>"}. '
    'Use "code" when the solution logic is wrong; use "test" only when the '
    "expected output or harness inputs are wrong for the problem."
)

PLAN_CHECKLIST = (
    "1. States one approach. 2. States time/space complexity. "
    "3. Has numbered implementation steps. "
    "4. Ends with a CONTRACT json block whose signature is def solve(...) "
    "with typed args and ranges."
)

CODE_CHECKLIST = (
    "1. Defines the contracted solve function with exact parameters. "
    "2. Implements the planned approach (no placeholder logic). "
    "3. Handles edge cases (empty input, single element). "
    "4. Imports only allowed stdlib modules."
)

TESTS_CHECKLIST = (
    "1. At least 3 tests covering base and edge cases. "
    "2. Every harness calls solve and prints the result. "
    "3. Expected outputs are plausible for the problem."
)


def adjudicate(problem: str, sig: str, arg_spec: list, code: str, test: dict) -> dict:
    """Blame a failing test: {"blame": code|test|unknown, "reason": str}.

    Deterministic contract violations (arity, out-of-range args) blame the
    test without any LLM call; the rest goes to one evaluator-model verdict
    (small-model fallback when the evaluator is unavailable).
    """
    harness = test.get("harness", "")
    if sig and not harness_matches(harness, sig):
        return {"blame": "test", "reason": "harness arity violates contract"}
    if arg_spec and not args_in_range(arg_spec, harness):
        return {"blame": "test", "reason": "harness args out of contracted range"}
    contract = _contract_brief_fallback(sig, arg_spec)
    user = (
        f"Problem:\n{problem[:2000]}\n\n{contract}\n\nCode:\n{code[:3000]}\n\n"
        f"Failing test {test.get('test_id')}: harness={harness[:500]} "
        f"expected={test.get('expected', '')[:500]!r} actual={test.get('actual', '')[:500]!r} "
        f"error={test.get('error', '')[:500]!r}"
    )
    verdict = try_invoke("evaluator", user, JUDGE_SYSTEM)
    if not verdict:
        # Fall back to the small model only if the judge model is unavailable.
        verdict = try_invoke("small", user, JUDGE_SYSTEM)
    if not verdict:
        return {"blame": "unknown", "reason": "judge unavailable"}
    up = verdict.strip().upper()
    if up.startswith("CODE_WRONG:"):
        return {"blame": "code", "reason": verdict[len("CODE_WRONG:"):].strip()[:300]}
    if up.startswith("TEST_WRONG:"):
        return {"blame": "test", "reason": verdict[len("TEST_WRONG:"):].strip()[:300]}
    return {"blame": "unknown", "reason": verdict[:300]}


def diagnose(problem: str, sig: str, arg_spec: list, code: str, failing: list, default_blame: str = "code") -> dict:
    """Holistic failure verdict: {"culprit": code|test, "reason": str}.

    Reads the code and every failing test. Falls back to `default_blame` (the
    evaluator's dominant per-test blame) when the judge model is unavailable or
    the reply is unparseable, so the run never stalls on a diagnosis.
    """
    contract = _contract_brief_fallback(sig, arg_spec)
    lines = []
    for r in failing[:5]:
        lines.append(
            f"- {r.get('test_id')}: harness={str(r.get('harness', ''))[:300]} "
            f"expected={str(r.get('expected', ''))[:200]!r} actual={str(r.get('actual', ''))[:200]!r} "
            f"error={str(r.get('error', ''))[:200]!r}"
        )
    user = (
        f"Problem:\n{problem[:2000]}\n\n{contract}\n\nCode:\n{code[:3000]}\n\n"
        "Failing tests:\n" + ("\n".join(lines) or "- (none provided)")
    )
    text = try_invoke("evaluator", user, DIAGNOSE_SYSTEM)
    data = extract_json_object(text) if text else None
    if isinstance(data, dict):
        culprit = str(data.get("culprit", "")).strip().lower()
        if culprit in ("code", "test"):
            reason = str(data.get("reason", "")).strip()[:500]
            return {"culprit": culprit, "reason": reason or "no reason given"}
    return {"culprit": default_blame, "reason": "diagnosis unavailable; defaulting to per-test blame"}


def _contract_brief_fallback(sig: str, arg_spec: list) -> str:
    lines = [f"Contract: {sig or 'def solve(...)'}"]
    for a in arg_spec or []:
        if isinstance(a, dict) and a.get("name"):
            lines.append(f"- {a.get('name')}: {a.get('type', '')} range {a.get('range', '')}")
    return "\n".join(lines)


def critique(kind: str, artifact: str, context: str = "") -> dict | None:
    """Local-model review. Returns {passed, issues} or None if critic down.

    The critic is advisory and fail-open: None means proceed unflagged.
    """
    from . import llms

    checklists = {"plan": PLAN_CHECKLIST, "code": CODE_CHECKLIST, "tests": TESTS_CHECKLIST}
    user = f"Checklist:\n{checklists[kind]}\n\nContext:\n{context[:2000]}\n\nArtifact:\n{artifact[:4000]}"
    try:
        msg = llms.get_critic().invoke([("system", CRITIC_SYSTEM), ("human", user)])
        content = msg.content if isinstance(msg.content, str) else ""
        verdict = strip_think(content).strip()
    except Exception:
        return None
    if verdict.upper().startswith("PASS"):
        return {"passed": True, "issues": ""}
    if verdict.upper().startswith("ISSUES:"):
        return {"passed": False, "issues": verdict[len("ISSUES:"):].strip()[:500]}
    return None


def strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning traces (some models inline them)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_code(text: str) -> str:
    """Pull code out of a fenced block if present, else return text as-is."""
    text = strip_think(text)
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    return (m.group(1) if m else text).strip() + "\n"


def extract_json_array(text: str) -> list | None:
    """Parse a JSON array from raw or fenced LLM output; None if unparseable."""
    text = strip_think(text)
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.DOTALL)
    candidate = m.group(1) if m else text.strip()
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, list) else None


def extract_json_object(text: str) -> dict | None:
    """Parse a JSON object from raw or fenced LLM output; None if unparseable."""
    text = strip_think(text)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def try_invoke(kind: str, user: str, system: str, retries: int = 1) -> str | None:
    """Invoke a model role (small / heavy / testgen); retry on empty; None on failure."""
    from . import llms

    getters = {"small": llms.get_small, "heavy": llms.get_heavy, "testgen": llms.get_testgen, "evaluator": llms.get_evaluator}
    getter = getters.get(kind, llms.get_small)
    for _ in range(retries + 1):
        try:
            model = getter()
            msg = model.invoke([("system", system), ("human", user)])
            content = msg.content if isinstance(msg.content, str) else ""
            if strip_think(content):
                return strip_think(content)
        except Exception:
            return None
    return None


def try_invoke_with_fallback(user: str, system: str) -> tuple[str | None, bool]:
    """Heavy first, small on failure. Returns (text, fell_back)."""
    text = try_invoke("heavy", user, system)
    if text:
        return text, False
    text = try_invoke("small", user, system)
    return text, text is not None


def extract_signature(text: str) -> str | None:
    """Pull `SIGNATURE: def solve(...)` contract line; None if absent/invalid."""
    m = re.search(r"^SIGNATURE:\s*(def\s+solve\s*\(.*?\))\s*$", text, flags=re.MULTILINE)
    return m.group(1).strip() if m else None


def extract_contract(text: str) -> dict | None:
    """Pull fenced JSON CONTRACT block; else legacy SIGNATURE line; else None.

    Returns {"signature": str, "args": [...], "returns": str} with args
    normalized to [{name, type, range}]. Legacy path yields empty args.
    """
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        try:
            data = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        sig = data.get("signature", "")
        if isinstance(sig, str) and re.match(r"def\s+solve\s*\(.*\)", sig.strip()):
            args = []
            for a in data.get("args", []) or []:
                if isinstance(a, dict) and a.get("name"):
                    args.append({
                        "name": str(a["name"]),
                        "type": str(a.get("type", "")),
                        "range": str(a.get("range", "")),
                    })
            return {"signature": sig.strip(), "args": args, "returns": str(data.get("returns", ""))}
    sig = extract_signature(text)
    if sig:
        return {"signature": sig, "args": [], "returns": ""}
    return None


def _def_args(code: str, name: str) -> list[str] | None:
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    return None


def signature_params(sig: str) -> list[str] | None:
    """Param names of a `def ...` signature line; None if unparseable."""
    try:
        return _def_args(sig + ":\n    pass", "solve")
    except Exception:
        return None


def code_defines_signature(code: str, sig: str) -> bool:
    want = signature_params(sig)
    got = _def_args(code, "solve")
    return want is not None and got == want


def harness_matches(harness: str, sig: str) -> bool:
    """Harness must call solve() with the contracted arity."""
    import ast

    want = signature_params(sig)
    if want is None:
        return True
    try:
        tree = ast.parse(harness)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "solve":
            given = len(node.args) + len(node.keywords)
            if given == len(want):
                return True
    return False


def safe_num(token: str) -> float | None:
    """Evaluate a numeric token with only safe operators (**, *, /, +, -, e-notation)."""
    import ast
    import operator

    token = token.strip().replace(",", "").replace("^", "**")
    if not token or not re.fullmatch(r"[\d\s.eE+\-*/()]+", token):
        return None
    try:
        tree = ast.parse(token, mode="eval")
    except SyntaxError:
        return None
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod,
    )
    if any(not isinstance(n, allowed) or (isinstance(n, ast.Constant) and not isinstance(n.value, (int, float))) for n in ast.walk(tree)):
        return None
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.UnaryOp):
            return (1 if isinstance(n.op, ast.UAdd) else -1) * ev(n.operand)
        if isinstance(n, ast.BinOp):
            return ops[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("unsafe")

    try:
        return ev(tree)
    except Exception:
        return None


def parse_range(range_str: str) -> dict:
    """Parse free-form planner ranges into checkable bounds.

    Understands `A..B` pairs (value bounds, or len bounds when preceded by
    len/length/size), `non-empty`, and `one of: a, b` enums. Unknown text
    yields {} (no constraints, never rejects).
    """
    spec: dict = {}
    low = (range_str or "").lower()
    pairs = []
    for m in re.finditer(r"(.{0,12}?)\.\.(.{0,24}?)($|[,;])", low):
        left = re.findall(r"-?[\d][\d,._eE*^ +\-/()]*$", m.group(1))
        right = re.findall(r"^ *\(? *-?[\d][\d,._eE*^ +\-/()]*", m.group(2))
        if not left or not right:
            continue
        lo, hi = safe_num(left[-1]), safe_num(right[0])
        if lo is None or hi is None:
            continue
        pairs.append((m.group(1), lo, hi))
    for prefix, lo, hi in pairs:
        pre = re.sub(r"[\d\s.,()]+$", "", prefix)
        if re.search(r"(len|length|size)", pre):
            spec["min_len"] = lo if spec.get("min_len") is None else max(spec["min_len"], lo)
            spec["max_len"] = hi if spec.get("max_len") is None else min(spec["max_len"], hi)
        elif "min" not in spec:
            spec["min"], spec["max"] = lo, hi
    if re.search(r"non-?empty|at least 1|1 or more", low):
        spec["min_len"] = max(1, spec.get("min_len") or 0)
    m = re.search(r"one of\s*:(.+)", low)
    if m:
        spec["enum"] = [v.strip() for v in m.group(1).split(",") if v.strip()][:10]
    return spec


def _in_bounds(value, spec: dict) -> bool:
    import numbers

    if not spec:
        return True
    if isinstance(value, bool):
        return True
    if isinstance(value, numbers.Real):
        if spec.get("min") is not None and value < spec["min"]:
            return False
        if spec.get("max") is not None and value > spec["max"]:
            return False
        return True
    if isinstance(value, (str, bytes, list, tuple, set, dict)):
        n = len(value)
        if spec.get("min_len") is not None and n < spec["min_len"]:
            return False
        if spec.get("max_len") is not None and n > spec["max_len"]:
            return False
        if isinstance(value, (list, tuple)) and (spec.get("min") is not None or spec.get("max") is not None):
            elems = [e for e in value if isinstance(e, numbers.Real) and not isinstance(e, bool)]
            if elems and len(elems) == len(value):
                return all(_in_bounds(e, {"min": spec.get("min"), "max": spec.get("max")}) for e in elems)
        return True
    if spec.get("enum") and value not in spec["enum"]:
        return False
    return True


def solve_call_args(harness: str) -> list | None:
    """Literal arg values of the first solve() call; None if unparseable."""
    import ast

    try:
        tree = ast.parse(harness)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "solve":
            vals = []
            for a in node.args:
                try:
                    vals.append(ast.literal_eval(a))
                except (ValueError, SyntaxError):
                    return None
            for kw in node.keywords:
                try:
                    vals.append(ast.literal_eval(kw.value))
                except (ValueError, SyntaxError):
                    return None
            return vals
    return None


def args_in_range(arg_spec: list, harness: str) -> bool:
    """Every literal harness arg satisfies its contracted range.

    Non-literal args or missing specs skip the check (never false-drop).
    """
    if not arg_spec:
        return True
    vals = solve_call_args(harness)
    if not vals:
        return True
    specs = {a.get("name"): parse_range(a.get("range", "")) for a in arg_spec if isinstance(a, dict)}
    names = [a.get("name") for a in arg_spec if isinstance(a, dict)]
    for i, v in enumerate(vals):
        if i >= len(names):
            break
        if not _in_bounds(v, specs.get(names[i], {})):
            return False
    return True
