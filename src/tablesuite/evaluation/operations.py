"""Typed, deterministic execution of frozen semantic operations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

from tablesuite._util import normalize_value, stable_value_key
from tablesuite.evaluation.contracts import (
    CellReference,
    EvaluationPlan,
)
from tablesuite.types import MaterializedTableSlice

EXECUTOR_VERSION = "2.0"

_ARGUMENT_SCHEMAS: dict[str, frozenset[str]] = {
    "cell_lookup": frozenset({"column"}),
    "row_lookup": frozenset({"row_id"}),
    "column_values": frozenset({"column"}),
    "distinct_values": frozenset({"column"}),
    "value_counts": frozenset({"column"}),
    "aggregate": frozenset({"column", "aggregation"}),
    "argmax_lookup": frozenset({"maximize_column", "return_column"}),
    "filtered_argmax_lookup": frozenset(
        {
            "filter_column",
            "filter_value_row_id",
            "maximize_column",
            "return_column",
        }
    ),
    "prediction_lookup": frozenset({"query_row_id"}),
    "prediction_with_cell": frozenset(
        {"query_row_id", "value_row_id", "value_column"}
    ),
}


class PredictionResolver(Protocol):
    """Resolve a versioned prediction packet without coupling to a model."""

    def resolve(self, packet_id: str, row_id: str) -> Any:
        """Return the cached prediction for one packet and source row."""


class MappingPredictionResolver:
    """Small in-memory prediction resolver useful for local evaluation."""

    def __init__(self, packets: dict[str, dict[str, Any]]) -> None:
        self._packets = {
            str(packet_id): {
                str(row_id): normalize_value(value)
                for row_id, value in rows.items()
            }
            for packet_id, rows in packets.items()
        }

    def resolve(self, packet_id: str, row_id: str) -> Any:
        """Return one prediction or raise for an unresolved reference."""

        try:
            return normalize_value(self._packets[packet_id][row_id])
        except KeyError as error:
            raise KeyError(
                f"prediction packet {packet_id!r} has no row {row_id!r}"
            ) from error


@dataclass(frozen=True)
class OperationResult:
    """Private result passed from the executor to rendering and scoring."""

    answer: Any
    evidence: tuple[CellReference, ...]
    render_values: dict[str, Any]
    prediction_context: Any = None


def execute_operation(
    plan: EvaluationPlan,
    table: MaterializedTableSlice,
    *,
    prediction_resolver: PredictionResolver | None = None,
) -> OperationResult:
    """Execute one validated plan against its resolved source slice."""

    validate_operation_spec(plan)
    rows = dict(zip(table.source.row_ids, table.rows, strict=True))
    name = plan.operation.name
    arguments = plan.operation.arguments

    if name == "cell_lookup":
        if len(table.source.row_ids) != 1:
            raise ValueError("cell_lookup requires a one-row source slice")
        row_id = table.source.row_ids[0]
        column = arguments["column"]
        _require_columns(table, column)
        answer = _required_value(rows[row_id][column], column)
        return OperationResult(
            answer=answer,
            evidence=(_cell(plan, row_id, column),),
            render_values={"column": column},
        )

    if name == "row_lookup":
        row_id = arguments["row_id"]
        _require_row(rows, row_id)
        return OperationResult(
            answer={
                column: normalize_value(rows[row_id][column])
                for column in table.source.columns
            },
            evidence=tuple(_cell(plan, row_id, column) for column in table.source.columns),
            render_values={"row_id": row_id},
        )

    if name in {"column_values", "distinct_values", "value_counts"}:
        column = arguments["column"]
        _require_columns(table, column)
        values = [normalize_value(row[column]) for row in table.rows]
        evidence = tuple(
            _cell(plan, row_id, column) for row_id in table.source.row_ids
        )
        if name == "column_values":
            answer: Any = values
        else:
            keyed: dict[str, Any] = {}
            ordered_keys: list[str] = []
            counts: dict[str, int] = {}
            for value in values:
                key = stable_value_key(value)
                if key not in keyed:
                    keyed[key] = value
                    ordered_keys.append(key)
                    counts[key] = 0
                counts[key] += 1
            answer = (
                [keyed[key] for key in ordered_keys]
                if name == "distinct_values"
                else [
                    {"value": keyed[key], "count": counts[key]}
                    for key in ordered_keys
                ]
            )
        return OperationResult(
            answer=answer,
            evidence=evidence,
            render_values={"column": column},
        )

    if name == "aggregate":
        column = arguments["column"]
        _require_columns(table, column)
        aggregation = arguments["aggregation"]
        values = [_required_number(row[column], column) for row in table.rows]
        answer = _aggregate(values, aggregation)
        return OperationResult(
            answer=answer,
            evidence=tuple(_cell(plan, row_id, column) for row_id in table.source.row_ids),
            render_values={"column": column, "aggregation": aggregation},
        )

    if name == "argmax_lookup":
        maximize = arguments["maximize_column"]
        returned = arguments["return_column"]
        _require_columns(table, maximize, returned)
        row_id = _unique_argmax(rows, maximize)
        answer = _required_value(rows[row_id][returned], returned)
        return OperationResult(
            answer=answer,
            evidence=(
                _cell(plan, row_id, maximize),
                _cell(plan, row_id, returned),
            ),
            render_values={"maximize_column": maximize, "return_column": returned},
        )

    if name == "filtered_argmax_lookup":
        filter_column = arguments["filter_column"]
        filter_row_id = arguments["filter_value_row_id"]
        maximize = arguments["maximize_column"]
        returned = arguments["return_column"]
        _require_columns(table, filter_column, maximize, returned)
        _require_row(rows, filter_row_id)
        filter_value = _required_value(rows[filter_row_id][filter_column], filter_column)
        filtered = {
            row_id: row
            for row_id, row in rows.items()
            if stable_value_key(row[filter_column]) == stable_value_key(filter_value)
        }
        if not filtered:
            raise ValueError("filtered_argmax_lookup produced an empty row set")
        row_id = _unique_argmax(filtered, maximize)
        answer = _required_value(filtered[row_id][returned], returned)
        evidence = _deduplicate_cells(
            (
                _cell(plan, filter_row_id, filter_column),
                _cell(plan, row_id, filter_column),
                _cell(plan, row_id, maximize),
                _cell(plan, row_id, returned),
            )
        )
        return OperationResult(
            answer=answer,
            evidence=evidence,
            render_values={
                "filter_column": filter_column,
                "filter_value": filter_value,
                "maximize_column": maximize,
                "return_column": returned,
            },
        )

    packet_id = plan.prediction_packet_id
    if prediction_resolver is None or packet_id is None:
        raise ValueError(f"{name} requires a prediction resolver")
    query_row_id = arguments["query_row_id"]
    _require_row(rows, query_row_id)
    _require_single_query_row(table, query_row_id, name)
    prediction = prediction_resolver.resolve(packet_id, query_row_id)
    if prediction is None:
        raise ValueError("prediction packet resolved to a missing value")
    if name == "prediction_lookup":
        return OperationResult(
            answer=prediction,
            evidence=(),
            render_values={},
            prediction_context=prediction,
        )

    value_row_id = arguments["value_row_id"]
    value_column = arguments["value_column"]
    _require_row(rows, value_row_id)
    _require_columns(table, value_column)
    if value_row_id != query_row_id:
        raise ValueError("prediction_with_cell must use one shared query/source row")
    source_value = _required_value(rows[value_row_id][value_column], value_column)
    return OperationResult(
        answer={"prediction": prediction, "source_value": source_value},
        evidence=(_cell(plan, value_row_id, value_column),),
        render_values={"value_column": value_column},
        prediction_context=prediction,
    )


def validate_operation_spec(plan: EvaluationPlan) -> None:
    """Validate the operation schema without materializing source values."""

    expected = _ARGUMENT_SCHEMAS[plan.operation.name]
    actual = frozenset(plan.operation.arguments)
    if actual != expected:
        raise ValueError(
            f"{plan.operation.name} arguments must be {sorted(expected)}, "
            f"found {sorted(actual)}"
        )
    if plan.operation.name == "aggregate":
        aggregation = plan.operation.arguments["aggregation"]
        if aggregation not in {"count", "sum", "mean", "min", "max"}:
            raise ValueError(f"unsupported aggregation: {aggregation}")
    column_arguments = {
        value
        for key, value in plan.operation.arguments.items()
        if key == "column" or key.endswith("_column")
    }
    if outside := column_arguments - set(plan.source.columns):
        raise ValueError(
            f"operation columns are outside the source slice: {sorted(outside)}"
        )
    row_arguments = {
        value
        for key, value in plan.operation.arguments.items()
        if key == "row_id" or key.endswith("_row_id")
    }
    if outside := row_arguments - set(plan.source.row_ids):
        raise ValueError(
            f"operation rows are outside the source slice: {sorted(outside)}"
        )


def _aggregate(values: list[int | float], aggregation: str) -> int | float:
    if aggregation == "count":
        return len(values)
    if aggregation == "sum":
        return sum(values)
    if aggregation == "mean":
        return sum(values) / len(values)
    if aggregation == "min":
        return min(values)
    if aggregation == "max":
        return max(values)
    raise AssertionError("aggregation was validated before execution")


def _unique_argmax(rows: dict[str, dict[str, Any]], column: str) -> str:
    scored = [(row_id, _required_number(row[column], column)) for row_id, row in rows.items()]
    maximum = max(value for _, value in scored)
    winners = [row_id for row_id, value in scored if value == maximum]
    if len(winners) != 1:
        raise ValueError(f"argmax is ambiguous for column {column!r}")
    return winners[0]


def _required_number(value: Any, column: str) -> int | float:
    normalized = _required_value(value, column)
    if isinstance(normalized, bool) or not isinstance(normalized, int | float):
        raise TypeError(f"column {column!r} requires finite numeric values")
    if not math.isfinite(float(normalized)):
        raise ValueError(f"column {column!r} contains a non-finite value")
    return normalized


def _required_value(value: Any, column: str) -> Any:
    normalized = normalize_value(value)
    if normalized is None or stable_value_key(normalized) == "<missing>":
        raise ValueError(f"column {column!r} contains a missing value")
    return normalized


def _require_columns(table: MaterializedTableSlice, *columns: str) -> None:
    if missing := set(columns) - set(table.source.columns):
        raise ValueError(f"operation columns are outside the source slice: {sorted(missing)}")


def _require_row(rows: dict[str, dict[str, Any]], row_id: str) -> None:
    if row_id not in rows:
        raise ValueError(f"operation row {row_id!r} is outside the source slice")


def _require_single_query_row(
    table: MaterializedTableSlice,
    query_row_id: str,
    operation: str,
) -> None:
    if table.source.row_ids != (query_row_id,):
        raise ValueError(f"{operation} requires one source row matching query_row_id")


def _cell(plan: EvaluationPlan, row_id: str, column: str) -> CellReference:
    return CellReference(plan.source.dataset_id, row_id, column)


def _deduplicate_cells(cells: tuple[CellReference, ...]) -> tuple[CellReference, ...]:
    return tuple(dict.fromkeys(cells))
