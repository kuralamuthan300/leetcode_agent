"""Input/output adaptation and comparison for golden test payloads."""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Adapter:
    input_transform: Callable[[Any], Any]
    output_transform: Callable[[Any], Any]


def _identity(value: Any) -> Any:
    return value


_ADAPTERS: dict[str | int, Adapter] = {}
_FLOAT_TOLERANCE_PROBLEMS = {"find_median_sorted_arrays"}


def register_adapter(
    problem_id: str | int,
    *,
    input_transform: Callable[[Any], Any] = _identity,
    output_transform: Callable[[Any], Any] = _identity,
) -> None:
    """Register a problem-specific encoding override."""
    _ADAPTERS[problem_id] = Adapter(input_transform, output_transform)


def adapter_for(problem_id: str | int) -> Adapter:
    return _ADAPTERS.get(problem_id, Adapter(_identity, _identity))


def adapt_input(problem_id: str | int, value: Any) -> Any:
    return adapter_for(problem_id).input_transform(value)


def adapt_output(problem_id: str | int, value: Any) -> Any:
    return adapter_for(problem_id).output_transform(value)


def parse_output(value: Any) -> Any:
    """Parse Python/JSON literal output while preserving ordinary text output."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text


def outputs_equal(actual: Any, expected: Any, *, function_name: str = "") -> bool:
    """Compare parsed outputs, with tolerance only for the median problem."""
    parsed_actual = parse_output(actual)
    parsed_expected = parse_output(expected)
    if isinstance(parsed_actual, bool) or isinstance(parsed_expected, bool):
        return (
            isinstance(parsed_actual, bool)
            and isinstance(parsed_expected, bool)
            and parsed_actual is parsed_expected
        )
    if (
        function_name in _FLOAT_TOLERANCE_PROBLEMS
        and isinstance(parsed_actual, (int, float))
        and isinstance(parsed_expected, (int, float))
    ):
        return math.isclose(parsed_actual, parsed_expected, rel_tol=1e-9, abs_tol=1e-9)
    return parsed_actual == parsed_expected
