"""Compact public records for official TableSuite task configurations."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from tablesuite.catalog import Catalog
from tablesuite.evaluation.contracts import (
    BENCHMARK_VERSION,
    PLAN_SCHEMA_VERSION,
    EvaluationPlan,
    OperationSpec,
    PlanRegistry,
    RenderingSpec,
    ScoringSpec,
)
from tablesuite.evaluation.operations import EXECUTOR_VERSION, validate_operation_spec
from tablesuite.evaluation.rendering import GENERATOR_VERSION
from tablesuite.types import DatasetSpec, TableSlice

TaskConfig = Literal["cell_grounding", "table_question_answering"]
TASK_RECORD_SCHEMA_VERSION = "1.0"

_TASK_NAMES = {
    "cell_grounding": "grounding",
    "table_question_answering": "qa",
}
_QA_ARGUMENTS = (
    "aggregation",
    "column",
    "filter_column",
    "filter_value_row_id",
    "maximize_column",
    "return_column",
)


def public_task_record(plan: EvaluationPlan) -> dict[str, Any]:
    """Project one internal plan into its concise public task schema."""

    common = {
        "item_id": plan.item_id,
        "dataset_id": plan.source.dataset_id,
        "evaluation_split": plan.evaluation_split,
        "render_seed": plan.rendering.render_seed,
    }
    scoring = {
        "answer_type": plan.scoring.answer_type,
        "absolute_tolerance": plan.scoring.absolute_tolerance,
        "relative_tolerance": plan.scoring.relative_tolerance,
    }
    if plan.task == "grounding":
        if len(plan.source.row_ids) != 1 or plan.operation.name != "cell_lookup":
            raise ValueError("cell grounding requires one source row and cell_lookup")
        return {
            **common,
            "source_row_id": plan.source.row_ids[0],
            "context_columns": list(plan.source.columns),
            "answer_column": plan.operation.arguments["column"],
            **scoring,
            "template_split": plan.rendering.template_split,
        }
    if plan.task != "qa":
        raise ValueError(f"unsupported public task plan: {plan.task!r}")
    arguments = plan.operation.arguments
    return {
        **common,
        "source_row_ids": list(plan.source.row_ids),
        "source_columns": list(plan.source.columns),
        "operation": plan.operation.name,
        "operation_arguments": {
            name: arguments.get(name) for name in _QA_ARGUMENTS
        },
        **scoring,
        "template_split": plan.rendering.template_split,
    }


def task_registry(
    records: Iterable[dict[str, Any]],
    *,
    catalog: Catalog,
    name: TaskConfig,
    split: str,
) -> PlanRegistry:
    """Restore public or legacy records as validated internal plans."""

    if name not in _TASK_NAMES:
        raise ValueError(f"unsupported task configuration: {name!r}")
    rows = tuple(dict(record) for record in records)
    datasets = {dataset.dataset_id: dataset for dataset in catalog.datasets}
    plans = tuple(
        _task_plan(
            record,
            datasets=datasets,
            reference_id=catalog.reference_id,
            name=name,
            split=split,
        )
        for record in rows
    )
    return PlanRegistry(plans)


def legacy_reference_id(records: Iterable[dict[str, Any]]) -> str | None:
    """Return the embedded reference ID used by the legacy nested schema."""

    values = {
        str(record["reference_id"])
        for record in records
        if record.get("reference_id")
    }
    if len(values) > 1:
        raise ValueError("task records contain multiple reference IDs")
    return next(iter(values), None)


def read_task_records(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read task records from a JSONL file, Parquet file, or shard directory."""

    source = Path(path)
    if source.is_dir() or source.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError(
                "install tablesuite[local] to load Parquet task files"
            ) from error
        shards = sorted(source.rglob("*.parquet")) if source.is_dir() else [source]
        if not shards:
            raise FileNotFoundError(f"no Parquet task files under {source}")
        return tuple(
            dict(row)
            for shard in shards
            for row in pq.read_table(shard).to_pylist()
        )
    return tuple(
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def load_huggingface_task_records(
    repository: str,
    *,
    name: TaskConfig,
    split: str,
    revision: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Load one public task configuration and split from Hugging Face."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "install tablesuite[hf] to load a task from Hugging Face"
        ) from error
    rows = load_dataset(repository, name, split=split, revision=revision)
    return tuple(dict(row) for row in rows)


def task_arrow_schema(name: TaskConfig) -> Any:
    """Return the explicit Arrow schema for one public task configuration."""

    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("install tablesuite[local] to author task files") from error
    common = [
        pa.field("item_id", pa.string(), nullable=False),
        pa.field("dataset_id", pa.string(), nullable=False),
        pa.field("evaluation_split", pa.string(), nullable=False),
        pa.field("render_seed", pa.int64(), nullable=False),
    ]
    scoring = [
        pa.field("answer_type", pa.string(), nullable=False),
        pa.field("absolute_tolerance", pa.float64(), nullable=False),
        pa.field("relative_tolerance", pa.float64(), nullable=False),
        pa.field("template_split", pa.string(), nullable=False),
    ]
    if name == "cell_grounding":
        return pa.schema(
            [
                *common,
                pa.field("source_row_id", pa.string(), nullable=False),
                pa.field("context_columns", pa.list_(pa.string()), nullable=False),
                pa.field("answer_column", pa.string(), nullable=False),
                *scoring,
            ]
        )
    if name != "table_question_answering":
        raise ValueError(f"unsupported task configuration: {name!r}")
    operation_arguments = pa.struct(
        [pa.field(field, pa.string()) for field in _QA_ARGUMENTS]
    )
    return pa.schema(
        [
            *common,
            pa.field("source_row_ids", pa.list_(pa.string()), nullable=False),
            pa.field("source_columns", pa.list_(pa.string()), nullable=False),
            pa.field("operation", pa.string(), nullable=False),
            pa.field("operation_arguments", operation_arguments, nullable=False),
            *scoring,
        ]
    )


def _task_plan(
    record: dict[str, Any],
    *,
    datasets: dict[str, DatasetSpec],
    reference_id: str,
    name: TaskConfig,
    split: str,
) -> EvaluationPlan:
    if "source" in record:
        plan = EvaluationPlan.from_record(record)
        if plan.task != _TASK_NAMES[name] or plan.evaluation_split != split:
            raise ValueError(
                f"legacy item {plan.item_id!r} does not match {name!r}/{split!r}"
            )
        return plan
    dataset_id = str(record["dataset_id"])
    try:
        dataset = datasets[dataset_id]
    except KeyError as error:
        raise KeyError(f"task item references unknown dataset {dataset_id!r}") from error
    evaluation_split = str(record["evaluation_split"])
    if evaluation_split != split:
        raise ValueError(
            f"item {record['item_id']!r} declares split {evaluation_split!r}, "
            f"loaded from {split!r}"
        )
    if name == "cell_grounding":
        source = TableSlice(
            dataset_id,
            (str(record["source_row_id"]),),
            tuple(str(value) for value in record["context_columns"]),
        )
        operation = OperationSpec(
            "cell_lookup",
            {"column": str(record["answer_column"])},
        )
    else:
        source = TableSlice(
            dataset_id,
            tuple(str(value) for value in record["source_row_ids"]),
            tuple(str(value) for value in record["source_columns"]),
        )
        raw_arguments = record.get("operation_arguments") or {}
        if not isinstance(raw_arguments, Mapping):
            raise TypeError("operation_arguments must be a mapping")
        operation = OperationSpec(
            str(record["operation"]),
            {
                str(key): str(value)
                for key, value in raw_arguments.items()
                if value is not None
            },
        )
    plan = EvaluationPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        benchmark_version=BENCHMARK_VERSION,
        reference_id=reference_id,
        item_id=str(record["item_id"]),
        task=_TASK_NAMES[name],
        evaluation_split=evaluation_split,
        dataset_split=dataset.dataset_split,
        dedup_cluster_id=dataset.dedup_cluster_id,
        source_id=dataset.source_id,
        source=source,
        operation=operation,
        rendering=RenderingSpec(
            template_family=operation.name,
            template_split=str(record["template_split"]),
            render_seed=int(record["render_seed"]),
        ),
        scoring=ScoringSpec(
            answer_type=str(record["answer_type"]),
            absolute_tolerance=float(record["absolute_tolerance"]),
            relative_tolerance=float(record["relative_tolerance"]),
        ),
        generator_version=GENERATOR_VERSION,
        executor_version=EXECUTOR_VERSION,
    )
    validate_operation_spec(plan)
    return plan


__all__ = [
    "TASK_RECORD_SCHEMA_VERSION",
    "TaskConfig",
    "legacy_reference_id",
    "load_huggingface_task_records",
    "public_task_record",
    "read_task_records",
    "task_arrow_schema",
    "task_registry",
]
