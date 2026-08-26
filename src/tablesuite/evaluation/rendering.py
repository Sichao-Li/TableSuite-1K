"""Deterministic wording for runtime evaluation items."""

from __future__ import annotations

import json
from typing import Any

from tablesuite._util import canonical_json, stable_order
from tablesuite.evaluation.contracts import EvaluationPlan, EvaluationRequest
from tablesuite.evaluation.operations import OperationResult
from tablesuite.rendering import render_table
from tablesuite.types import MaterializedTableSlice

GENERATOR_VERSION = "2.0"

_TEMPLATES: dict[str, dict[str, tuple[str, ...]]] = {
    "cell_lookup": {
        "train": (
            "What is the value of {column} in the displayed row?",
            "Report {column} for the row shown.",
            "Read the {column} entry from this row.",
            "Which value appears under {column}?",
            "Give the displayed row's {column} value.",
            "Return the cell value in column {column}.",
            "From this record, identify {column}.",
            "Look up {column} in the provided row.",
        ),
        "test": (
            "For the displayed row, what is its {column}?",
            "Read the {column} entry from the row in the table.",
            "Extract the value associated with {column}.",
            "What does this record list for {column}?",
        ),
    },
    "row_lookup": {
        "train": (
            "Return row {row_id} as a JSON object using the displayed columns.",
            "Read the complete record at {row_id} and return it as JSON.",
        ),
        "test": (
            "What is the full displayed record for {row_id}? Return JSON.",
            "Extract every shown field from {row_id} as one JSON object.",
        ),
    },
    "column_values": {
        "train": (
            "Return every value in column {column}, in displayed row order, as JSON.",
            "Read column {column} from top to bottom and return a JSON list.",
        ),
        "test": (
            "List the displayed values of {column} in row order as JSON.",
            "Extract the full visible {column} column as an ordered JSON list.",
        ),
    },
    "distinct_values": {
        "train": (
            "Return the distinct values in column {column}, in first-occurrence "
            "order, as a JSON list.",
            "Which unique values occur in the displayed {column} column? Return "
            "them in first-occurrence order as JSON.",
        ),
        "test": (
            "List each distinct visible value of {column} once, preserving "
            "first-occurrence order, as JSON.",
            "What unique values appear in {column}? Return them in first-occurrence "
            "order as JSON.",
        ),
    },
    "value_counts": {
        "train": (
            "Count each distinct value in column {column}; return a JSON list of "
            "objects with keys value and count, in first-occurrence order.",
            "Return the visible value frequencies for {column} as value/count "
            "objects in first-occurrence order.",
        ),
        "test": (
            "What are the value counts in the displayed {column} column? Return "
            "value/count objects in first-occurrence order.",
            "Tabulate each unique {column} value and its count as a JSON list in "
            "first-occurrence order.",
        ),
    },
    "aggregate": {
        "train": (
            "What is the {aggregation} of {column} across the displayed rows?",
            "Calculate the {aggregation} for the {column} column.",
            "Compute {aggregation} over all shown {column} values.",
            "Return the {aggregation} of the displayed {column} entries.",
            "Using this subtable, find {aggregation} for {column}.",
            "Aggregate {column} with the {aggregation} operation.",
            "Apply {aggregation} to every visible value in {column}.",
            "What result follows from taking {aggregation} of {column}?",
        ),
        "test": (
            "Using all shown rows, return the {aggregation} of {column}.",
            "Find the {aggregation} value for {column} in this table.",
            "Evaluate {aggregation} across the visible {column} column.",
            "What is obtained by applying {aggregation} to {column}?",
        ),
    },
    "argmax_lookup": {
        "train": (
            "What is {return_column} for the row with the highest {maximize_column}?",
            "Find the row maximizing {maximize_column} and report its {return_column}.",
            "Locate the greatest {maximize_column}; give that row's {return_column}.",
            "For the maximum {maximize_column} row, return {return_column}.",
            "Identify the row with largest {maximize_column} and read {return_column}.",
            "Select by maximum {maximize_column}, then answer with {return_column}.",
            "Which {return_column} occurs on the top-{maximize_column} row?",
            "Use {maximize_column} to find the maximum row and report {return_column}.",
        ),
        "test": (
            "Return {return_column} from the row whose {maximize_column} is largest.",
            "Which {return_column} belongs to the maximum-{maximize_column} row?",
            "At the highest {maximize_column}, what is the corresponding {return_column}?",
            "Read {return_column} after choosing the row that maximizes {maximize_column}.",
        ),
    },
    "filtered_argmax_lookup": {
        "train": (
            "Among rows where {filter_column} equals {filter_value}, what is "
            "{return_column} for the row with the highest {maximize_column}?",
            "Filter to {filter_column} = {filter_value}, maximize {maximize_column}, "
            "and report {return_column}.",
            "Within {filter_column} = {filter_value}, find the greatest "
            "{maximize_column} and return {return_column}.",
            "Restrict rows by {filter_column} = {filter_value}; which {return_column} "
            "is on the maximum-{maximize_column} row?",
            "After selecting {filter_column} = {filter_value}, report {return_column} "
            "for the row maximizing {maximize_column}.",
            "Use the {filter_column} value {filter_value}, then choose the largest "
            "{maximize_column} and read {return_column}.",
            "Among matching {filter_column} rows ({filter_value}), return "
            "{return_column} at the highest {maximize_column}.",
            "First filter {filter_column} to {filter_value}; next maximize "
            "{maximize_column}; finally answer {return_column}.",
        ),
        "test": (
            "For the {filter_column} value {filter_value}, return {return_column} "
            "from the row with greatest {maximize_column}.",
            "Considering only rows with {filter_column} equal to {filter_value}, "
            "which {return_column} is paired with the maximum {maximize_column}?",
            "In the {filter_column} = {filter_value} subset, what {return_column} "
            "corresponds to the largest {maximize_column}?",
            "Select rows matching {filter_value} in {filter_column}; at maximum "
            "{maximize_column}, return {return_column}.",
        ),
    },
    "prediction_lookup": {
        "train": (
            "What output does the frozen predictor assign to the displayed row?",
            "Report the prediction-channel value for the row shown.",
        ),
        "test": (
            "Read the frozen predictor output for this row.",
            "What prediction is carried by the prediction channel?",
        ),
    },
    "prediction_with_cell": {
        "train": (
            "Return JSON with prediction from the frozen predictor and source_value "
            "from column {value_column}.",
            "Combine the prediction-channel output with the exact {value_column} "
            "source value as JSON.",
        ),
        "test": (
            "Provide a JSON object containing prediction and the exact "
            "{value_column} source_value.",
            "Read both channels: return prediction plus {value_column} as "
            "source_value in JSON.",
        ),
    },
}


def render_evaluation_request(
    plan: EvaluationPlan,
    table: MaterializedTableSlice,
    result: OperationResult,
) -> EvaluationRequest:
    """Render one input-only request and record the selected template ID."""

    templates = _TEMPLATES[plan.rendering.template_family][plan.rendering.template_split]
    template_index = int(stable_order(plan.item_id, plan.rendering.render_seed), 16) % len(
        templates
    )
    template_id = (
        f"{plan.rendering.template_family}:{plan.rendering.template_split}:{template_index}"
    )
    values = {
        key: _render_argument(plan, table, key, value)
        for key, value in result.render_values.items()
    }
    question = templates[template_index].format(**values)
    if plan.scoring.answer_type == "json":
        question += " Represent every displayed [missing] value as JSON null."
    table_text = render_table(
        table,
        view=plan.rendering.view,
        include_row_ids=True,
    )
    sections = ["Table:", table_text]
    if result.prediction_context is not None:
        sections.extend(
            ["Frozen prediction channel:", _display(result.prediction_context)]
        )
    sections.extend(["Question:", question])
    return EvaluationRequest(
        item_id=plan.item_id,
        task=plan.task,
        input_text="\n\n".join(sections),
        question=question,
        table_text=table_text,
        template_id=template_id,
    )


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


def _render_argument(
    plan: EvaluationPlan,
    table: MaterializedTableSlice,
    key: str,
    value: Any,
) -> str:
    if key == "column" or key.endswith("_column"):
        return json.dumps(str(value), ensure_ascii=False)
    if key == "row_id" or key.endswith("_row_id"):
        try:
            return f"r{table.source.row_ids.index(str(value))}"
        except ValueError as error:
            raise ValueError(
                f"rendered row reference {value!r} is outside item {plan.item_id!r}"
            ) from error
    return _display(value)
