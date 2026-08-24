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
    CELL_SPLITS,
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

TASK_CONFIGS = ("cell_grounding", "table_question_answering")
RELEASE_CONFIGS = (*CONFIGS, *TASK_CONFIGS)


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
        executed = _validate_execution(catalog, source, generated.plans)

        _copy_reference_configs(reference_path, staging)
        shutil.copy2(card_path, staging / "README.md")
        _write_task_configs(staging, generated.plans, policy.shard_size)
        summary = {
            "schema_version": "1.2",
            "release_kind": "value_free_huggingface_benchmark",
            "reference_id": catalog.reference_id,
            "passed": True,
            "configs": list(RELEASE_CONFIGS),
            "source_tables_redistributed": False,
            "rendered_questions_stored": False,
            "gold_answers_stored": False,
            "task_plans_executed": executed,
            "task_counts": generated.task_counts,
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
        _write_json(staging / "release_summary.json", summary)
        (staging / "release_summary.md").write_text(
            _summary_markdown(summary), encoding="utf-8"
        )
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
    plans = _load_release_plans(root)
    audit = audit_plans(plans)
    audit.require_passed()
    catalog = Catalog.from_path(root)
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


def _copy_reference_configs(reference: Path, destination: Path) -> None:
    for name in ("datasets", "grounding_tasks"):
        shutil.copytree(_reference_config(reference, name), destination / name)
    _rewrite_config(
        _reference_config(reference, "table_prediction_tasks"),
        destination / "table_prediction_tasks",
        _table_prediction_record,
    )
    _rewrite_config(
        _reference_config(reference, "prediction_episodes"),
        destination / "prediction_episodes",
        _prediction_episode_record,
    )
    summary = reference / "reference_summary.json"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    shutil.copy2(summary, destination / summary.name)
    for name in (
        "REFERENCE_POLICY.md",
        "reference_policy.json",
        "DATASET_LICENSES.md",
        "dataset_licenses.jsonl",
    ):
        candidate = reference / name
        if candidate.is_file():
            shutil.copy2(candidate, destination / name)


def _rewrite_config(
    source: Path,
    destination: Path,
    transform: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("install tablesuite[local] to author a release") from error
    files = sorted(source.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet shards under {source}")
    for source_file in files:
        output_file = destination / source_file.relative_to(source)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        rows = [transform(dict(row)) for row in pq.read_table(source_file).to_pylist()]
        pq.write_table(pa.Table.from_pylist(rows), output_file, compression="zstd")


def _table_prediction_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in record.items() if key != "fold_policy"}
    output.update(
        {
            "task_id": f"{record['dataset_id']}:zero_label_serialized_table",
            "protocol": "zero_label_serialized_table",
            "input_interface": "serialized_table",
            "parameter_updates": False,
            "visible_label_count": 0,
            "target_visibility": "private_evaluation_only",
        }
    )
    return output


def _prediction_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in record.items() if key != "k"}
    output.update(
        {
            "shots": int(record.get("shots", record.get("k"))),
            "protocols": [
                "zero_shot_icl",
                "few_shot_icl",
                "partially_labeled_serialized_table",
            ],
            "input_interfaces": ["row_examples", "serialized_table"],
            "serialized_table_query_scopes": ["full_table", "episode"],
            "parameter_updates": False,
            "support_target_visibility": "protocol_dependent",
            "query_target_visibility": "private_evaluation_only",
            "zero_shot_icl_derivable": True,
        }
    )
    return output


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
        task = (
            "cell_grounding"
            if plan.task == "grounding"
            else "table_question_answering"
        )
        grouped[(task, plan.evaluation_split)].append(plan)
    for (task, split), split_plans in sorted(grouped.items()):
        destination = root / "tasks" / task / split
        destination.mkdir(parents=True, exist_ok=True)
        ordered = sorted(split_plans, key=lambda plan: plan.item_id)
        for shard_index, start in enumerate(range(0, len(ordered), shard_size)):
            records = [plan.to_record() for plan in ordered[start : start + shard_size]]
            pq.write_table(
                pa.Table.from_pylist(records),
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
        "cell_grounding": CELL_SPLITS,
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


def _load_release_plans(root: Path) -> tuple[EvaluationPlan, ...]:
    plans: list[EvaluationPlan] = []
    for task in TASK_CONFIGS:
        task_root = root / "tasks" / task
        if not task_root.is_dir():
            raise ValueError(f"release is missing task files for {task!r}")
        for split in sorted(path for path in task_root.iterdir() if path.is_dir()):
            plans.extend(PlanRegistry.load(split).plans)
    return tuple(sorted(plans, key=lambda plan: plan.item_id))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Hugging Face Release Audit",
        "",
        f"- passed: `{summary['passed']}`",
        f"- reference: `{summary['reference_id']}`",
        f"- task plans executed: `{summary['task_plans_executed']}`",
        "- source tables redistributed: `False`",
        "- rendered questions stored: `False`",
        "- gold answers stored: `False`",
        "",
        "## Task Counts",
        "",
    ]
    for task, counts in summary["task_counts"].items():
        lines.append(f"### {task}")
        lines.append("")
        for split, count in counts.items():
            datasets = summary["dataset_counts"][task][split]
            lines.append(f"- {split}: `{count}` items across `{datasets}` datasets")
        lines.append("")
    lines.extend(["## Eligibility Skips", ""])
    if summary["eligibility_skips"]:
        for reason, count in summary["eligibility_skips"].items():
            lines.append(f"- {reason}: `{count}`")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


__all__ = [
    "RELEASE_CONFIGS",
    "TaskGenerationConfig",
    "build_huggingface_release",
    "validate_huggingface_release",
]
