"""Release-authoring tests for the official source-grounded task configs."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import get_dataset_config_names, load_dataset

from tablesuite import (
    Benchmark,
    Catalog,
    GeneratedTaskDataset,
    generate_task,
    load_generated_task,
    load_task,
)
from tablesuite._cli import main as cli_main
from tablesuite.evaluation import PlanExecutor, audit_plans
from tablesuite.release import TaskGenerationConfig, build_huggingface_release
from tablesuite.task_records import read_task_records, task_registry


def test_release_builder_is_deterministic_value_free_and_executable(
    tmp_path: Path,
) -> None:
    reference, source = _authoring_fixture(tmp_path)
    card = Path(__file__).parents[1] / "huggingface" / "README.md"
    config = TaskGenerationConfig(
        seed=17,
        cell_items_per_dataset=4,
        cell_transfer_items_per_dataset=2,
        qa_items_per_dataset=4,
        qa_transfer_items_per_dataset=2,
        max_cell_context_columns=4,
        max_qa_context_columns=4,
        qa_row_sizes=(4, 8),
        shard_size=3,
    )
    first = tmp_path / "release-first"
    second = tmp_path / "release-second"

    first_summary = build_huggingface_release(
        reference_root=reference,
        source_root=source,
        output_dir=first,
        dataset_card=card,
        config=config,
    )
    second_summary = build_huggingface_release(
        reference_root=reference,
        source_root=source,
        output_dir=second,
        dataset_card=card,
        config=config,
    )

    assert first_summary["passed"]
    assert first_summary["task_counts"] == second_summary["task_counts"]
    assert (first / "README.md").is_file()
    assert (first / "datasets" / "train" / "part-00000.parquet").is_file()
    assert not (first / "metadata").exists()
    assert not (first / "release_summary.json").exists()
    assert not (first / "release_summary.md").exists()
    assert first_summary["catalog_counts"] == {
        "datasets": 4,
        "grounding_tasks": 3,
        "prediction_episodes": 3,
        "table_prediction_tasks": 3,
    }
    assert first_summary["release_version"] == "1.3.0"
    reference_summary = json.loads(
        (first / "reference_summary.json").read_text(encoding="utf-8")
    )
    assert set(reference_summary) == {
        "configs",
        "contains_source_values",
        "reference_id",
        "record_schemas",
        "release_version",
        "schema_version",
        "source_provider",
    }
    assert reference_summary["release_version"] == "1.3.0"
    assert reference_summary["record_schemas"] == {
        "catalog": "1.0",
        "official_tasks": "1.0",
    }
    assert set(get_dataset_config_names(str(first))) == {
        "datasets",
        "table_prediction_tasks",
        "prediction_episodes",
        "grounding_tasks",
        "cell_grounding",
        "table_question_answering",
    }
    table_prediction_splits = load_dataset(
        str(first),
        "table_prediction_tasks",
        cache_dir=str(tmp_path / "hf-cache-prediction"),
    )
    assert sum(len(split) for split in table_prediction_splits.values()) == 3
    table_prediction = table_prediction_splits["train"]
    assert table_prediction.column_names == ["dataset_id", "primary_metrics"]
    assert {
        row["dataset_id"]
        for split in table_prediction_splits.values()
        for row in split
    } == {"openml_101", "openml_102", "openml_103"}

    prediction_episode_splits = load_dataset(
        str(first),
        "prediction_episodes",
        cache_dir=str(tmp_path / "hf-cache-icl"),
    )
    assert sum(len(split) for split in prediction_episode_splits.values()) == 3
    prediction_episodes = prediction_episode_splits["train"]
    assert prediction_episodes[0]["shots"] == 4
    assert prediction_episodes.column_names == [
        "episode_id",
        "dataset_id",
        "episode_split",
        "support_row_ids",
        "query_row_ids",
        "shots",
    ]
    assert {
        row["dataset_id"]
        for split in prediction_episode_splits.values()
        for row in split
    } == {"openml_101", "openml_102", "openml_103"}
    grounding_splits = load_dataset(
        str(first),
        "grounding_tasks",
        cache_dir=str(tmp_path / "hf-cache-grounding"),
    )
    assert sum(len(split) for split in grounding_splits.values()) == 3
    assert grounding_splits["train"].column_names == [
        "dataset_id",
        "eligible_columns",
        "excluded_identifier_columns",
        "max_cells",
    ]
    dataset_splits = load_dataset(
        str(first),
        "datasets",
        cache_dir=str(tmp_path / "hf-cache-datasets"),
    )
    assert sum(len(split) for split in dataset_splits.values()) == 4
    dataset_catalog = dataset_splits["train"]
    assert dataset_catalog.column_names == [
        "dataset_id",
        "dataset_split",
        "openml_data_id",
        "openml_url",
        "dataset_name",
        "task_type",
        "target_column",
        "feature_columns",
        "target_transform",
        "excluded_feature_columns",
        "source_adaptation_rationale",
        "n_rows",
        "n_features",
        "n_classes",
        "dedup_cluster_id",
        "openml_license_claim",
    ]
    hub_rows = load_dataset(
        str(first),
        "cell_grounding",
        split="dataset_test",
        cache_dir=str(tmp_path / "hf-cache"),
    )
    assert len(hub_rows) == first_summary["task_counts"]["cell_grounding"][
        "dataset_test"
    ]
    assert hub_rows.column_names == [
        "item_id",
        "dataset_id",
        "evaluation_split",
        "render_seed",
        "source_row_id",
        "context_columns",
        "answer_column",
        "answer_type",
        "absolute_tolerance",
        "relative_tolerance",
        "template_split",
    ]
    qa_rows = load_dataset(
        str(first),
        "table_question_answering",
        split="composition_test",
        cache_dir=str(tmp_path / "hf-cache-qa"),
    )
    assert qa_rows.column_names == [
        "item_id",
        "dataset_id",
        "evaluation_split",
        "render_seed",
        "source_row_ids",
        "source_columns",
        "operation",
        "operation_arguments",
        "answer_type",
        "absolute_tolerance",
        "relative_tolerance",
        "template_split",
    ]
    assert set(qa_rows[0]["operation_arguments"]) == {
        "aggregation",
        "column",
        "filter_column",
        "filter_value_row_id",
        "maximize_column",
        "return_column",
    }

    first_plans = _load_all_plans(first)
    second_plans = _load_all_plans(second)
    assert first_plans == second_plans
    assert audit_plans(first_plans).passed

    cell_plans = [plan for plan in first_plans if plan.task == "grounding"]
    qa_plans = [plan for plan in first_plans if plan.task == "qa"]
    assert cell_plans
    assert qa_plans
    assert all(len(plan.source.columns) > 1 for plan in cell_plans)
    assert all(plan.operation.arguments["column"] in plan.source.columns for plan in cell_plans)
    assert all("Target" not in plan.source.columns for plan in first_plans)
    assert {
        plan.evaluation_split for plan in cell_plans
    } == {"train", "validation", "episode_test", "dataset_test", "template_test"}
    assert {
        plan.evaluation_split for plan in qa_plans
    } == {
        "train",
        "validation",
        "episode_test",
        "dataset_test",
        "template_test",
        "composition_test",
    }
    composition = [
        plan for plan in qa_plans if plan.evaluation_split == "composition_test"
    ]
    assert composition
    assert all(plan.operation.name == "filtered_argmax_lookup" for plan in composition)

    records = json.dumps([plan.to_record() for plan in first_plans], sort_keys=True)
    assert '"answer"' not in records
    assert '"gold"' not in records
    assert '"question"' not in records
    assert '"prediction_packet_id"' not in records
    assert "north" not in records

    grounding = load_task(
        first,
        "cell_grounding",
        split="dataset_test",
        source=source,
    )
    grounding_registry = _load_registry(
        first,
        "cell_grounding",
        "dataset_test",
    )
    grounding_plan = grounding_registry.get_plan(grounding.ids[0])
    materialized = PlanExecutor(
        Benchmark.from_path(first, source), grounding_registry
    ).materialize(grounding_plan)
    assert grounding.score(grounding_plan.item_id, materialized.gold.answer).correct

    qa = load_task(
        first,
        "table_question_answering",
        split="composition_test",
        source=source,
    )
    qa_registry = _load_registry(
        first,
        "table_question_answering",
        "composition_test",
    )
    qa_plan = qa_registry.get_plan(qa.ids[0])
    qa_item = PlanExecutor(Benchmark.from_path(first, source), qa_registry).materialize(
        qa_plan
    )
    assert qa.score(qa_plan.item_id, qa_item.gold.answer).correct


def test_release_cli_builds_and_validates(tmp_path: Path, capsys) -> None:
    reference, source = _authoring_fixture(tmp_path)
    output = tmp_path / "release-cli"
    card = Path(__file__).parents[1] / "huggingface" / "README.md"

    cli_main(
        [
            "build-release",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--dataset-card",
            str(card),
            "--output",
            str(output),
            "--cell-items-per-dataset",
            "2",
            "--cell-transfer-items-per-dataset",
            "1",
            "--qa-items-per-dataset",
            "2",
            "--qa-transfer-items-per-dataset",
            "1",
            "--qa-row-size",
            "4",
        ]
    )
    assert json.loads(capsys.readouterr().out)["passed"]

    cli_main(
        [
            "validate-release",
            "--release",
            str(output),
            "--source",
            str(source),
        ]
    )
    validation = json.loads(capsys.readouterr().out)
    assert validation["passed"]
    assert validation["task_plans_executed"] == validation["plans"]


def test_generated_task_uses_the_standard_task_interface(tmp_path: Path) -> None:
    reference, source = _authoring_fixture(tmp_path)

    first = generate_task(
        reference,
        "cell_grounding",
        split="train",
        source=source,
        dataset_ids=("openml_101",),
        items_per_dataset=5,
        max_items=3,
        seed=17,
    )
    second = generate_task(
        reference,
        "cell_grounding",
        split="train",
        source=source,
        dataset_ids=("openml_101",),
        items_per_dataset=5,
        max_items=3,
        seed=17,
    )

    assert isinstance(first, GeneratedTaskDataset)
    assert len(first) == 3
    assert first.ids == second.ids
    assert first[0].prompt == second[0].prompt
    assert first.manifest.origin == "generated"
    assert first.manifest.task == "cell_grounding"
    assert first.manifest.dataset_ids == ("openml_101",)
    assert first.manifest.generated_items == 3
    assert first.manifest.plan_fingerprint == second.manifest.plan_fingerprint


def test_generated_task_bundle_round_trips_without_values(tmp_path: Path) -> None:
    reference, source = _authoring_fixture(tmp_path)
    generated = generate_task(
        reference,
        "table_question_answering",
        split="train",
        source=source,
        dataset_ids=("openml_101",),
        items_per_dataset=4,
        seed=9,
    )
    output = tmp_path / "generated-task"

    generated.save(output)
    restored = load_generated_task(output, source=source)

    assert restored.ids == generated.ids
    assert restored.manifest == generated.manifest
    assert restored[0].prompt == generated[0].prompt
    plan_text = (output / "plans.jsonl").read_text(encoding="utf-8")
    assert '"gold"' not in plan_text
    assert '"question"' not in plan_text
    assert "north" not in plan_text
    assert (output / "generation.json").is_file()


def test_generate_cli_writes_a_reusable_bundle(tmp_path: Path, capsys) -> None:
    reference, source = _authoring_fixture(tmp_path)
    output = tmp_path / "generated-cli"

    cli_main(
        [
            "generate",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--name",
            "table_question_answering",
            "--split",
            "train",
            "--dataset-id",
            "openml_101",
            "--items-per-dataset",
            "4",
            "--max-items",
            "2",
            "--seed",
            "11",
            "--output",
            str(output),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary["generated_items"] == 2
    assert summary["task"] == "table_question_answering"
    assert len(load_generated_task(output, source=source)) == 2


def _load_all_plans(root: Path) -> tuple:
    catalog = Catalog.from_path(root)
    plans = []
    for task in ("cell_grounding", "table_question_answering"):
        for split in sorted((root / "tasks" / task).iterdir()):
            plans.extend(
                task_registry(
                    read_task_records(split),
                    catalog=catalog,
                    name=task,
                    split=split.name,
                ).plans
            )
    return tuple(sorted(plans, key=lambda plan: plan.item_id))


def _load_registry(root: Path, task: str, split: str):
    return task_registry(
        read_task_records(root / "tasks" / task / split),
        catalog=Catalog.from_path(root),
        name=task,
        split=split,
    )


def _authoring_fixture(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "reference"
    source = tmp_path / "source"
    source.mkdir()
    datasets = [
        _dataset_record("openml_101", "101", "train", "cluster_train", 80),
        _dataset_record("openml_102", "102", "validation", "cluster_validation", 32),
        _dataset_record("openml_103", "103", "test", "cluster_test", 32),
    ]
    invalid = {
        **_dataset_record("openml_104", "104", "train", "cluster_invalid", 8),
        "feature_columns": [],
        "n_features": 0,
    }
    datasets.append(invalid)
    table_prediction_tasks = [
        {
            **dataset,
            "task_id": f"{dataset['dataset_id']}:zero_label_serialized_table",
            "fold_policy": "deterministic_within_dataset_up_to_10",
            "primary_metrics": ["accuracy", "balanced_accuracy", "macro_f1"],
        }
        for dataset in datasets
    ]
    episodes = [
        {
            **dataset,
            "episode_id": f"{dataset['dataset_id']}_k4_e0",
            "episode_split": "train",
            "k": 4,
            "support_row_ids": ["0", "1", "2", "3"],
            "query_row_ids": ["4", "5", "6", "7"],
            "query_size": 4,
        }
        for dataset in datasets
    ]
    grounding = [
        {
            **dataset,
            "task_id": f"{dataset['dataset_id']}:cell_fact_equivalence",
            "eligible_columns": (
                ["Age", "Income", "Group", "City"]
                if dataset["feature_columns"]
                else []
            ),
            "excluded_identifier_columns": [],
            "sampler": "column_balanced_v1",
            "sampler_seed": 0,
            "max_cells": 1000,
            "text_views": ["key_value", "json", "markdown", "natural_language"],
        }
        for dataset in datasets
    ]
    _write_config(reference, "datasets", datasets)
    _write_config(reference, "table_prediction_tasks", table_prediction_tasks)
    _write_config(reference, "prediction_episodes", episodes)
    _write_config(reference, "grounding_tasks", grounding)
    (reference / "reference_summary.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "reference_id": "openml-table-benchmark:v1.1",
                "datasets": 4,
                "episodes": 4,
            }
        ),
        encoding="utf-8",
    )
    for dataset in datasets:
        count = int(dataset["n_rows"])
        if not dataset["feature_columns"]:
            pq.write_table(
                pa.table({"Target": [index % 2 for index in range(count)]}),
                source / f"{dataset['source_id']}.parquet",
            )
            continue
        pq.write_table(
            pa.table(
                {
                    "Age": [18 + index for index in range(count)],
                    "Income": [1000 + 13 * index for index in range(count)],
                    "Group": ["a" if index % 3 else "b" for index in range(count)],
                    "City": ["north" if index % 2 else "south" for index in range(count)],
                    "Target": [index % 2 for index in range(count)],
                }
            ),
            source / f"{dataset['source_id']}.parquet",
        )
    return reference, source


def _dataset_record(
    dataset_id: str,
    source_id: str,
    split: str,
    cluster: str,
    rows: int,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "reference_policy": "openml_referenced",
        "dataset_id": dataset_id,
        "dataset_split": split,
        "source": "openml",
        "source_id": source_id,
        "source_url": f"https://www.openml.org/d/{source_id}",
        "task_type": "binary_classification",
        "target_column": "Target",
        "feature_columns": ["Age", "Income", "Group", "City"],
        "target_transform": "none",
        "excluded_feature_columns": [],
        "dataset_name": dataset_id,
        "n_rows": rows,
        "n_features": 4,
        "n_classes": 2,
        "dedup_cluster_id": cluster,
        "metadata_tier": "structural",
        "license_claim": "Public",
    }


def _write_config(root: Path, config: str, rows: list[dict[str, object]]) -> None:
    by_split: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_split.setdefault(str(row["dataset_split"]), []).append(row)
    for split, split_rows in by_split.items():
        destination = root / config / split
        destination.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(split_rows), destination / "part-00000.parquet")
