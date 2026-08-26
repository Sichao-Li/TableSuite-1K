"""Build and validate the publication-ready Hugging Face dataset directory."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tablesuite.authoring import (
    GROUNDING_SPLITS,
    QA_SPLITS,
    TaskGenerationConfig,
    generate_task_plans,
)
from tablesuite.benchmark import Benchmark
from tablesuite.catalog import CONFIGS, Catalog
from tablesuite.evaluation import (
    EvaluationPlan,
    PlanExecutor,
    PlanRegistry,
    audit_plans,
)
from tablesuite.source import ParquetSource
from tablesuite.task_records import (
    TASK_RECORD_SCHEMA_VERSION,
    public_task_record,
    read_task_records,
    task_arrow_schema,
    task_registry,
)
from tablesuite.types import Selection

TASK_CONFIGS = ("table_grounding", "table_question_answering")
RELEASE_CONFIGS = (*CONFIGS, *TASK_CONFIGS)
RELEASE_VERSION = "2.0.0"
REFERENCE_ID = "tablesuite-1k:2.0"
_GROUNDING_OPERATION_NAMES = (
    "cell_lookup",
    "row_lookup",
    "column_values",
    "distinct_values",
    "value_counts",
)
_QA_OPERATION_NAMES = (
    "count",
    "sum",
    "mean",
    "min",
    "max",
    "argmax_lookup",
    "filtered_argmax_lookup",
)


def build_huggingface_release(
    *,
    reference_root: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
    dataset_card: str | Path,
    config: TaskGenerationConfig | None = None,
) -> dict[str, Any]:
    """Create one audited six-configuration Hugging Face release directory.

    Source Parquet tables are read to validate operations and compute eligibility,
    but they are never copied into the release.
    """

    reference_path = Path(reference_root)
    source_path = Path(source_root)
    card_path = Path(dataset_card)
    destination = Path(output_dir)
    policy = config or TaskGenerationConfig()
    _validate_inputs(reference_path, source_path, card_path, destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        catalog = Catalog.from_path(reference_path)
        source = ParquetSource(source_path)
        generated = generate_task_plans(catalog, source, policy)
        audit = audit_plans(generated.plans)
        audit.require_passed()
        _validate_catalog_bindings(catalog, generated.plans)
        _require_release_coverage(generated.task_counts)
        _validate_value_free(generated.plans)
        task_matrix = _task_matrix(generated.plans)
        _validate_task_matrix(task_matrix)
        executed = _validate_execution(catalog, source, generated.plans)

        catalog_counts = _write_public_catalog(reference_path, staging)
        public_catalog = Catalog.from_path(staging)
        _validate_public_catalog(public_catalog)
        shutil.copy2(card_path, staging / "README.md")
        _write_task_configs(staging, generated.plans, policy.shard_size)
        public_plans = _load_release_plans(staging, public_catalog)
        _validate_task_roundtrip(generated.plans, public_plans)
        public_audit = audit_plans(public_plans)
        public_audit.require_passed()
        _validate_catalog_bindings(public_catalog, public_plans)
        summary = {
            "schema_version": "2.0",
            "release_version": RELEASE_VERSION,
            "release_kind": "value_free_huggingface_benchmark",
            "reference_id": public_catalog.reference_id,
            "passed": True,
            "configs": list(RELEASE_CONFIGS),
            "catalog_counts": catalog_counts,
            "source_tables_redistributed": False,
            "rendered_questions_stored": False,
            "gold_answers_stored": False,
            "task_plans_executed": executed,
            "task_counts": generated.task_counts,
            "task_matrix": task_matrix,
            "dataset_counts": generated.dataset_counts,
            "eligibility_skips": generated.skipped,
            "generation_config": policy.to_dict(),
            "plan_audit": {
                "plans": audit.plans,
                "datasets": audit.datasets,
                "dedup_clusters": audit.dedup_clusters,
                "errors": list(audit.errors),
            },
        }
        staging.replace(destination)
        return summary
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_huggingface_release(
    release_root: str | Path,
    *,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate release structure and optionally execute every official task plan."""

    root = Path(release_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    expected_paths = {
        **{name: root / name for name in CONFIGS},
        **{name: root / "tasks" / name for name in TASK_CONFIGS},
    }
    if missing := [name for name, path in expected_paths.items() if not path.is_dir()]:
        raise ValueError(f"release is missing configurations: {missing}")
    if not (root / "README.md").is_file():
        raise ValueError("release is missing its Hugging Face dataset card")
    catalog = Catalog.from_path(root)
    plans = _load_release_plans(root, catalog)
    audit = audit_plans(plans)
    audit.require_passed()
    _validate_public_catalog(catalog)
    _validate_catalog_bindings(catalog, plans)
    _validate_value_free(plans)
    executed = 0
    if source_root is not None:
        executed = _validate_execution(catalog, ParquetSource(source_root), plans)
    return {
        "passed": True,
        "configs": list(RELEASE_CONFIGS),
        "plans": len(plans),
        "datasets": audit.datasets,
        "dedup_clusters": audit.dedup_clusters,
        "task_plans_executed": executed,
    }


def _validate_inputs(
    reference: Path,
    source: Path,
    card: Path,
    destination: Path,
) -> None:
    if not reference.is_dir():
        raise FileNotFoundError(reference)
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not card.is_file():
        raise FileNotFoundError(card)
    card_text = card.read_text(encoding="utf-8")
    if missing_configs := [
        name
        for name in RELEASE_CONFIGS
        if f"config_name: {name}" not in card_text
    ]:
        raise ValueError(f"dataset card is missing configurations: {missing_configs}")
    for unreleased in ("tabular_prediction", "prediction_grounded_reasoning"):
        if f"config_name: {unreleased}" in card_text:
            raise ValueError(f"dataset card exposes unreleased config: {unreleased}")
    if destination.exists():
        raise FileExistsError(destination)
    if missing := [
        name for name in CONFIGS if not _reference_config(reference, name).is_dir()
    ]:
        raise ValueError(f"reference package is missing configurations: {missing}")


def _write_public_catalog(
    reference: Path,
    destination: Path,
) -> dict[str, int]:
    summaries = {
        "datasets": _rewrite_config(
            _reference_config(reference, "datasets"),
            destination / "datasets",
            _dataset_record,
            schema=_catalog_arrow_schema("datasets"),
        ),
        "table_prediction_tasks": _rewrite_config(
            _reference_config(reference, "table_prediction_tasks"),
            destination / "table_prediction_tasks",
            _table_prediction_record,
            schema=_catalog_arrow_schema("table_prediction_tasks"),
        ),
        "prediction_episodes": _rewrite_config(
            _reference_config(reference, "prediction_episodes"),
            destination / "prediction_episodes",
            _prediction_episode_record,
            schema=_catalog_arrow_schema("prediction_episodes"),
        ),
        "grounding_tasks": _rewrite_config(
            _reference_config(reference, "grounding_tasks"),
            destination / "grounding_tasks",
            _grounding_record,
            schema=_catalog_arrow_schema("grounding_tasks"),
        ),
    }
    summary = reference / "reference_summary.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    source_summary = json.loads(summary.read_text(encoding="utf-8"))
    if not isinstance(source_summary, dict):
        raise ValueError("reference summary must be a JSON object")
    reference_summary = {
        "schema_version": "2.0",
        "release_version": RELEASE_VERSION,
        "reference_id": REFERENCE_ID,
        "source_provider": "openml",
        "contains_source_values": False,
        "record_schemas": {
            "catalog": "1.0",
            "official_tasks": TASK_RECORD_SCHEMA_VERSION,
        },
        "configs": summaries,
    }
    _write_json(destination / summary.name, reference_summary)
    return {name: summary["records"] for name, summary in summaries.items()}


def _rewrite_config(
    source: Path,
    destination: Path,
    transform: Callable[[dict[str, Any]], dict[str, Any] | None],
    *,
    schema: Any,
) -> dict[str, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("install tablesuite[local] to author a release") from error
    files = sorted(source.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet shards under {source}")
    split_summaries: dict[str, dict[str, Any]] = {}
    total = 0
    for source_file in files:
        output_file = destination / source_file.relative_to(source)
        rows = [
            output
            for row in pq.read_table(source_file).to_pylist()
            if (output := transform(dict(row))) is not None
        ]
        if not rows:
            continue
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            output_file,
            compression="zstd",
        )
        split = output_file.parent.name
        split_summary = split_summaries.setdefault(
            split,
            {"records": 0, "shards": []},
        )
        split_summary["records"] += len(rows)
        split_summary["shards"].append(
            str(output_file.relative_to(destination))
        )
        total += len(rows)
    if total == 0:
        raise ValueError(f"public configuration would be empty: {source.name}")
    return {"records": total, "splits": split_summaries}


def _dataset_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": str(record["dataset_id"]),
        "dataset_split": str(record["dataset_split"]),
        "openml_data_id": str(
            _required_alias(record, "openml_data_id", "source_id")
        ),
        "openml_url": str(_required_alias(record, "openml_url", "source_url")),
        "dataset_name": str(record.get("dataset_name") or record["dataset_id"]),
        "task_type": str(record["task_type"]),
        "target_column": str(record["target_column"]),
        "feature_columns": [
            str(value) for value in record.get("feature_columns") or []
        ],
        "target_transform": str(record.get("target_transform") or "none"),
        "excluded_feature_columns": [
            str(value)
            for value in record.get("excluded_feature_columns") or []
        ],
        "source_adaptation_rationale": str(
            record.get("source_adaptation_rationale") or ""
        ),
        "n_rows": int(record["n_rows"]),
        "n_features": int(record.get("n_features") or 0),
        "n_classes": (
            int(record["n_classes"])
            if record.get("n_classes") is not None
            else None
        ),
        "dedup_cluster_id": str(
            record.get("dedup_cluster_id") or record["dataset_id"]
        ),
        "openml_license_claim": str(
            record.get("openml_license_claim") or record.get("license_claim") or ""
        ),
    }


def _required_alias(record: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise KeyError(f"record is missing required field aliases: {names}")


def _table_prediction_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if "feature_columns" in record and not record["feature_columns"]:
        return None
    return {
        "dataset_id": str(record["dataset_id"]),
        "primary_metrics": [str(value) for value in record["primary_metrics"]],
    }


def _prediction_episode_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if "feature_columns" in record and not record["feature_columns"]:
        return None
    support = [str(value) for value in record["support_row_ids"]]
    query = [str(value) for value in record["query_row_ids"]]
    return {
        "episode_id": str(record["episode_id"]),
        "dataset_id": str(record["dataset_id"]),
        "episode_split": str(record["episode_split"]),
        "support_row_ids": support,
        "query_row_ids": query,
        "shots": int(record.get("shots", record.get("k"))),
    }


def _grounding_record(record: dict[str, Any]) -> dict[str, Any] | None:
    eligible = [str(value) for value in record.get("eligible_columns") or []]
    if not eligible:
        return None
    return {
        "dataset_id": str(record["dataset_id"]),
        "eligible_columns": eligible,
        "excluded_identifier_columns": [
            str(value)
            for value in record.get("excluded_identifier_columns") or []
        ],
        "max_cells": int(record["max_cells"]),
    }


def _catalog_arrow_schema(name: str) -> Any:
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError("install tablesuite[local] to author a release") from error
    schemas = {
        "datasets": pa.schema(
            [
                pa.field("dataset_id", pa.string(), nullable=False),
                pa.field("dataset_split", pa.string(), nullable=False),
                pa.field("openml_data_id", pa.string(), nullable=False),
                pa.field("openml_url", pa.string(), nullable=False),
                pa.field("dataset_name", pa.string(), nullable=False),
                pa.field("task_type", pa.string(), nullable=False),
                pa.field("target_column", pa.string(), nullable=False),
                pa.field("feature_columns", pa.list_(pa.string()), nullable=False),
                pa.field("target_transform", pa.string(), nullable=False),
                pa.field(
                    "excluded_feature_columns",
                    pa.list_(pa.string()),
                    nullable=False,
                ),
                pa.field("source_adaptation_rationale", pa.string(), nullable=False),
                pa.field("n_rows", pa.int64(), nullable=False),
                pa.field("n_features", pa.int64(), nullable=False),
                pa.field("n_classes", pa.int64()),
                pa.field("dedup_cluster_id", pa.string(), nullable=False),
                pa.field("openml_license_claim", pa.string(), nullable=False),
            ]
        ),
        "table_prediction_tasks": pa.schema(
            [
                pa.field("dataset_id", pa.string(), nullable=False),
                pa.field("primary_metrics", pa.list_(pa.string()), nullable=False),
            ]
        ),
        "prediction_episodes": pa.schema(
            [
                pa.field("episode_id", pa.string(), nullable=False),
                pa.field("dataset_id", pa.string(), nullable=False),
                pa.field("episode_split", pa.string(), nullable=False),
                pa.field("support_row_ids", pa.list_(pa.string()), nullable=False),
                pa.field("query_row_ids", pa.list_(pa.string()), nullable=False),
                pa.field("shots", pa.int64(), nullable=False),
            ]
        ),
        "grounding_tasks": pa.schema(
            [
                pa.field("dataset_id", pa.string(), nullable=False),
                pa.field("eligible_columns", pa.list_(pa.string()), nullable=False),
                pa.field(
                    "excluded_identifier_columns",
                    pa.list_(pa.string()),
                    nullable=False,
                ),
                pa.field("max_cells", pa.int64(), nullable=False),
            ]
        ),
    }
    try:
        return schemas[name]
    except KeyError as error:
        raise ValueError(f"unknown public catalog configuration: {name!r}") from error


def _reference_config(reference: Path, name: str) -> Path:
    return reference / name


def _write_task_configs(
    root: Path,
    plans: tuple[EvaluationPlan, ...],
    shard_size: int,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("install tablesuite[local] to author a release") from error
    grouped: dict[tuple[str, str], list[EvaluationPlan]] = defaultdict(list)
    for plan in plans:
        task = "table_grounding" if plan.task == "grounding" else "table_question_answering"
        grouped[(task, plan.evaluation_split)].append(plan)
    for (task, split), split_plans in sorted(grouped.items()):
        destination = root / "tasks" / task / split
        destination.mkdir(parents=True, exist_ok=True)
        ordered = sorted(split_plans, key=lambda plan: plan.item_id)
        for shard_index, start in enumerate(range(0, len(ordered), shard_size)):
            records = [
                public_task_record(plan)
                for plan in ordered[start : start + shard_size]
            ]
            pq.write_table(
                pa.Table.from_pylist(records, schema=task_arrow_schema(task)),
                destination / f"part-{shard_index:05d}.parquet",
                compression="zstd",
            )


def _validate_execution(
    catalog: Catalog,
    source: ParquetSource,
    plans: tuple[EvaluationPlan, ...],
) -> int:
    registry = PlanRegistry(plans)
    executor = PlanExecutor(Benchmark(catalog, source), registry)
    ordered = sorted(
        plans,
        key=lambda plan: (plan.source.dataset_id, plan.item_id),
    )
    for plan in ordered:
        executor.materialize(plan)
    return len(ordered)


def _validate_catalog_bindings(
    catalog: Catalog,
    plans: tuple[EvaluationPlan, ...],
) -> None:
    datasets = {dataset.dataset_id: dataset for dataset in catalog.datasets}
    errors: list[str] = []
    for plan in plans:
        dataset = datasets.get(plan.source.dataset_id)
        if dataset is None:
            errors.append(f"item {plan.item_id!r} references an unknown dataset")
            continue
        if plan.source_id != dataset.source_id:
            errors.append(f"item {plan.item_id!r} has the wrong source ID")
        if plan.dataset_split != dataset.dataset_split:
            errors.append(f"item {plan.item_id!r} has the wrong dataset partition")
        if plan.dedup_cluster_id != dataset.dedup_cluster_id:
            errors.append(f"item {plan.item_id!r} has the wrong deduplication cluster")
        if dataset.target_column in plan.source.columns:
            errors.append(f"item {plan.item_id!r} exposes the target column")
        if unknown := set(plan.source.columns) - set(dataset.feature_columns):
            errors.append(
                f"item {plan.item_id!r} references unknown features: {sorted(unknown)}"
            )
        try:
            row_indices = [int(row_id) for row_id in plan.source.row_ids]
        except ValueError:
            errors.append(f"item {plan.item_id!r} has a non-integer row reference")
            continue
        if any(index < 0 or index >= dataset.n_rows for index in row_indices):
            errors.append(f"item {plan.item_id!r} has an out-of-range row reference")
    if errors:
        raise ValueError("invalid catalog bindings:\n- " + "\n- ".join(errors))


def _validate_public_catalog(catalog: Catalog) -> None:
    selected = catalog.select(
        Selection(
            tasks=(
                "zero_label_serialized_table",
                "few_shot_icl",
                "grounding",
            )
        )
    )
    datasets = {dataset.dataset_id: dataset for dataset in selected.datasets}
    errors: list[str] = []

    for dataset_id in sorted(selected.table_prediction_dataset_ids):
        dataset = datasets[dataset_id]
        if not dataset.feature_columns:
            errors.append(f"prediction task {dataset_id!r} has no feature columns")
        if dataset.target_column in dataset.feature_columns:
            errors.append(f"prediction task {dataset_id!r} exposes its target")

    for episode in selected.episodes:
        dataset_id = str(episode["dataset_id"])
        dataset = datasets.get(dataset_id)
        if dataset is None:
            errors.append(f"episode {episode['episode_id']!r} has no dataset")
            continue
        if dataset_id not in selected.table_prediction_dataset_ids:
            errors.append(
                f"episode {episode['episode_id']!r} has no prediction task"
            )
        support = tuple(str(value) for value in episode["support_row_ids"])
        query = tuple(str(value) for value in episode["query_row_ids"])
        shots = int(episode["shots"])
        if len(support) != shots:
            errors.append(f"episode {episode['episode_id']!r} has the wrong shot count")
        if not query:
            errors.append(f"episode {episode['episode_id']!r} has no query rows")
        if set(support) & set(query):
            errors.append(f"episode {episode['episode_id']!r} overlaps support and query")
        try:
            row_ids = [int(value) for value in (*support, *query)]
        except ValueError:
            errors.append(f"episode {episode['episode_id']!r} has non-integer row IDs")
            continue
        if row_ids and (min(row_ids) < 0 or max(row_ids) >= dataset.n_rows):
            errors.append(f"episode {episode['episode_id']!r} has invalid row IDs")

    for task in selected.grounding_tasks:
        dataset_id = str(task["dataset_id"])
        dataset = datasets[dataset_id]
        eligible = tuple(str(value) for value in task["eligible_columns"])
        if not eligible:
            errors.append(f"grounding task {dataset_id!r} has no eligible columns")
        if dataset.target_column in eligible:
            errors.append(f"grounding task {dataset_id!r} exposes its target")
        if unknown := set(eligible) - set(dataset.feature_columns):
            errors.append(
                f"grounding task {dataset_id!r} has unknown columns: {sorted(unknown)}"
            )

    if errors:
        raise ValueError("invalid public catalog:\n- " + "\n- ".join(errors))


def _validate_value_free(plans: tuple[EvaluationPlan, ...]) -> None:
    forbidden_keys = {
        "answer",
        "gold",
        "question",
        "rendered_question",
        "source_value",
        "target_value",
    }
    for plan in plans:
        record = plan.to_record()
        present = _nested_keys(record) & forbidden_keys
        if present:
            raise ValueError(
                f"item {plan.item_id!r} stores forbidden value fields: {sorted(present)}"
            )


def _nested_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key
            for nested in value.values()
            for key in _nested_keys(nested)
        }
    if isinstance(value, list | tuple):
        return {key for nested in value for key in _nested_keys(nested)}
    return set()


def _require_release_coverage(task_counts: dict[str, dict[str, int]]) -> None:
    expected = {
        "table_grounding": GROUNDING_SPLITS,
        "table_question_answering": QA_SPLITS,
    }
    missing = [
        f"{task}/{split}"
        for task, splits in expected.items()
        for split in splits
        if task_counts[task].get(split, 0) == 0
    ]
    if missing:
        raise ValueError(f"release authoring produced empty official splits: {missing}")


def _task_matrix(plans: tuple[EvaluationPlan, ...]) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for task, internal in (
        ("table_grounding", "grounding"),
        ("table_question_answering", "qa"),
    ):
        selected = [plan for plan in plans if plan.task == internal]
        by_operation: dict[str, int] = defaultdict(int)
        by_row_size: dict[str, int] = defaultdict(int)
        operation_row_sizes: dict[str, set[int]] = defaultdict(set)
        schema_languages: dict[str, int] = defaultdict(int)
        for plan in selected:
            operation = (
                plan.operation.arguments["aggregation"]
                if plan.operation.name == "aggregate"
                else plan.operation.name
            )
            row_size = len(plan.source.row_ids)
            by_operation[operation] += 1
            by_row_size[str(row_size)] += 1
            operation_row_sizes[operation].add(row_size)
            schema_languages[plan.rendering.schema_language] += 1
        matrix[task] = {
            "by_operation": dict(sorted(by_operation.items())),
            "by_row_size": dict(sorted(by_row_size.items(), key=lambda item: int(item[0]))),
            "operation_row_sizes": {
                operation: sorted(sizes)
                for operation, sizes in sorted(operation_row_sizes.items())
            },
            "schema_language": dict(sorted(schema_languages.items())),
        }
    return matrix


def _validate_task_matrix(matrix: dict[str, Any]) -> None:
    expected = {
        "table_grounding": set(_GROUNDING_OPERATION_NAMES),
        "table_question_answering": set(_QA_OPERATION_NAMES),
    }
    for task, operations in expected.items():
        actual = set(matrix[task]["by_operation"])
        if actual != operations:
            raise ValueError(
                f"{task} operation coverage mismatch: expected {sorted(operations)}, "
                f"found {sorted(actual)}"
            )
        if matrix[task]["schema_language"] != {
            "literal": sum(matrix[task]["by_operation"].values())
        }:
            raise ValueError(f"{task} contains unsupported schema-language conditions")
        for operation, row_sizes in matrix[task]["operation_row_sizes"].items():
            if operation == "cell_lookup":
                if row_sizes != [1]:
                    raise ValueError("cell_lookup must use exactly one visible row")
            elif len(row_sizes) < 2:
                raise ValueError(
                    f"{task}/{operation} is confounded with one table size: {row_sizes}"
                )


def _load_release_plans(
    root: Path,
    catalog: Catalog,
) -> tuple[EvaluationPlan, ...]:
    plans: list[EvaluationPlan] = []
    for task in TASK_CONFIGS:
        task_root = root / "tasks" / task
        if not task_root.is_dir():
            raise ValueError(f"release is missing task files for {task!r}")
        for split in sorted(path for path in task_root.iterdir() if path.is_dir()):
            plans.extend(
                task_registry(
                    read_task_records(split),
                    catalog=catalog,
                    name=task,
                    split=split.name,
                ).plans
            )
    return tuple(sorted(plans, key=lambda plan: plan.item_id))


def _validate_task_roundtrip(
    expected: tuple[EvaluationPlan, ...],
    restored: tuple[EvaluationPlan, ...],
) -> None:
    expected_records = sorted(
        (public_task_record(plan) for plan in expected),
        key=lambda record: record["item_id"],
    )
    restored_records = sorted(
        (public_task_record(plan) for plan in restored),
        key=lambda record: record["item_id"],
    )
    if expected_records != restored_records:
        raise ValueError("public task records failed their semantic round trip")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "RELEASE_CONFIGS",
    "TaskGenerationConfig",
    "build_huggingface_release",
    "validate_huggingface_release",
]
