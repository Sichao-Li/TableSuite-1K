"""Deterministic, model-agnostic views of TableSuite-1K benchmark requests."""

from __future__ import annotations

from typing import Any

from tablesuite._util import canonical_json, normalize_value
from tablesuite.types import (
    ICLPredictionRequest,
    MaterializedTableSlice,
    RenderedPrediction,
    SerializedTablePredictionRequest,
    TextView,
)

ICL_SERIALIZATION_VERSION = "icl_rows_v2"
TABLE_PREDICTION_SERIALIZATION_VERSION = "serialized_table_v2"


def render_cell_fact_views(column: str, value: Any) -> dict[str, str]:
    """Return deterministic text views of one source cell fact."""

    rendered = _display(normalize_value(value))
    return {
        "key_value": f"{column} = {rendered}",
        "json": canonical_json({"column": column, "value": normalize_value(value)}),
        "markdown": _render_rows([{column: value}], [column], "markdown"),
        "natural_language": f"The value of {column} is {rendered}.",
    }


def render_icl_prediction(
    request: ICLPredictionRequest,
) -> RenderedPrediction:
    """Render zero/few-shot prediction as row examples followed by queries."""

    query_aliases = {
        f"q{index}": source_id
        for index, source_id in enumerate(request.query.source.row_ids)
    }
    sections = [f"Target: {request.target_column}"]
    if request.demonstrations is not None:
        examples = [
            (
                f"Row {_alphabetic_label(index)}: "
                f"{_render_feature_row(row, request.feature_columns)} -> "
                f"{_display(normalize_value(row[request.target_column]))}"
            )
            for index, row in enumerate(request.demonstrations.rows)
        ]
        sections.append("Examples:\n" + "\n".join(examples))
    queries = [
        f"Query q{index}: {_render_feature_row(row, request.feature_columns)} -> ?"
        for index, row in enumerate(request.query.rows)
    ]
    sections.append("Queries:\n" + "\n".join(queries))
    sections.append("Return a JSON object mapping each query ID to its predicted value.")
    return RenderedPrediction(
        request_id=request.request_id,
        view="row_examples",
        serialization_version=ICL_SERIALIZATION_VERSION,
        input_text="\n\n".join(sections),
        query_aliases=query_aliases,
    )


def render_serialized_table_prediction(
    request: SerializedTablePredictionRequest,
    *,
    view: TextView = "markdown",
) -> RenderedPrediction:
    """Render zero-label or partially labelled serialized-table prediction."""

    source_aliases = {
        f"r{index}": source_id
        for index, source_id in enumerate(request.table.source.row_ids)
    }
    aliases_by_source = {source_id: alias for alias, source_id in source_aliases.items()}
    rows = _alias_rows(request.table, prefix="r")
    columns = ["row_id", *request.feature_columns]
    if request.visible_labels is None:
        instruction = f'Predict "{request.target_column}" for every row.'
    else:
        labels = {
            row_id: row[request.target_column]
            for row_id, row in zip(
                request.visible_labels.source.row_ids,
                request.visible_labels.rows,
                strict=True,
            )
        }
        for row, source_id in zip(
            rows, request.table.source.row_ids, strict=True
        ):
            row[request.target_column] = labels.get(source_id, "?")
        columns.append(request.target_column)
        instruction = (
            f'Predict "{request.target_column}" for rows where the target is masked.'
        )
    table = _render_rows(rows, columns, view)
    query_aliases = {
        aliases_by_source[source_id]: source_id for source_id in request.query_row_ids
    }
    return RenderedPrediction(
        request_id=request.request_id,
        view=view,
        serialization_version=TABLE_PREDICTION_SERIALIZATION_VERSION,
        input_text=f"{instruction}\n\n{table}",
        query_aliases=query_aliases,
    )


def render_table(
    table: MaterializedTableSlice,
    *,
    view: TextView,
    include_row_ids: bool = False,
) -> str:
    """Render one materialized row or subtable in a deterministic view."""

    if not include_row_ids:
        return _render_rows(list(table.rows), list(table.source.columns), view)
    row_column = _row_id_column(table.source.columns)
    rows = [
        {row_column: f"r{index}", **row}
        for index, row in enumerate(table.rows)
    ]
    return _render_rows(rows, [row_column, *table.source.columns], view)


def _alias_rows(
    table: MaterializedTableSlice,
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    return [
        {"row_id": f"{prefix}{index}", **row}
        for index, row in enumerate(table.rows)
    ]


def _render_feature_row(row: dict[str, Any], columns: tuple[str, ...]) -> str:
    return ", ".join(
        f"{column}={_display(normalize_value(row[column]))}" for column in columns
    )


def _alphabetic_label(index: int) -> str:
    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("A") + remainder) + label
        if value == 0:
            return label
        value -= 1


def _render_rows(rows: list[dict[str, Any]], columns: list[str], view: TextView) -> str:
    projected = [
        {column: normalize_value(row.get(column)) for column in columns}
        for row in rows
    ]
    if view == "json":
        return canonical_json(projected)
    if view == "key_value":
        return "\n".join(
            "; ".join(f"{column} = {_display(row[column])}" for column in columns)
            for row in projected
        )
    if view == "markdown":
        header = "| " + " | ".join(_escape(column) for column in columns) + " |"
        rule = "| " + " | ".join("---" for _ in columns) + " |"
        body = [
            "| "
            + " | ".join(_escape(_display(row[column])) for column in columns)
            + " |"
            for row in projected
        ]
        return "\n".join([header, rule, *body])
    raise ValueError(f"unsupported view: {view}")


def _display(value: Any) -> str:
    if value is None:
        return "[missing]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return str(value)
    return canonical_json(value)


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _row_id_column(columns: tuple[str, ...]) -> str:
    candidate = "row_id"
    while candidate in columns:
        candidate = f"_{candidate}"
    return candidate
