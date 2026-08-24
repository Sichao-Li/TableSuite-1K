"""Structured parsing and scoring for runtime-generated gold answers."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from tablesuite._util import normalize_value
from tablesuite.evaluation.contracts import EvaluationGold, ScoreResult, ScoringSpec


def score_response(
    response: Any,
    gold: EvaluationGold,
    scoring: ScoringSpec,
) -> ScoreResult:
    """Parse and compare one response under the plan's scoring contract."""

    try:
        parsed = _parse_response(response, scoring.answer_type)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return ScoreResult(
            item_id=gold.item_id,
            correct=False,
            exact_match=False,
            numeric_within_tolerance=None,
            parse_error=str(error),
        )

    expected = normalize_value(gold.answer)
    if scoring.answer_type in {"integer", "float"}:
        expected_number = _number(expected)
        parsed_number = _number(parsed)
        exact = parsed_number == expected_number
        within_tolerance = math.isclose(
            parsed_number,
            expected_number,
            rel_tol=scoring.relative_tolerance,
            abs_tol=scoring.absolute_tolerance,
        )
        return ScoreResult(
            item_id=gold.item_id,
            correct=within_tolerance,
            exact_match=exact,
            numeric_within_tolerance=within_tolerance,
            parsed_answer=parsed,
        )

    if scoring.answer_type == "string":
        parsed_text = _normalize_text(str(parsed), scoring.case_sensitive)
        expected_text = _normalize_text(str(expected), scoring.case_sensitive)
        exact = parsed_text == expected_text
    else:
        exact = normalize_value(parsed) == expected
    return ScoreResult(
        item_id=gold.item_id,
        correct=exact,
        exact_match=exact,
        numeric_within_tolerance=None,
        parsed_answer=parsed,
    )


def _parse_response(response: Any, answer_type: str) -> Any:
    if answer_type == "json":
        if isinstance(response, dict | list):
            return normalize_value(response)
        return json.loads(_strip_fence(str(response)))
    text = str(response).strip()
    if answer_type == "string":
        return text
    if answer_type == "boolean":
        lowered = text.casefold()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
        raise ValueError(f"cannot parse boolean response: {response!r}")
    number = float(text.replace(",", ""))
    if not math.isfinite(number):
        raise ValueError("numeric response must be finite")
    if answer_type == "integer":
        if not number.is_integer():
            raise ValueError(f"cannot parse integer response: {response!r}")
        return int(number)
    return number


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"gold answer is not numeric: {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("gold answer must be finite")
    return number


def _normalize_text(value: str, case_sensitive: bool) -> str:
    collapsed = re.sub(r"\s+", " ", value.strip())
    return collapsed if case_sensitive else collapsed.casefold()


def _strip_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return text
