from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
def benchmark_fixture(tmp_path: Path) -> tuple[Path, Path]:
    reference = tmp_path / "reference"
    source = tmp_path / "source"
    source.mkdir()

    dataset = {
        "schema_version": "1.0",
        "reference_policy": "openml_referenced",
        "dataset_id": "openml_1",
        "dataset_split": "train",
        "source": "openml",
        "source_id": "1",
        "source_url": "https://www.openml.org/d/1",
        "task_type": "binary_classification",
        "target_column": "Default",
        "feature_columns": ["Age", "Income"],
        "target_transform": "none",
        "excluded_feature_columns": [],
        "dataset_name": "fixture",
        "n_rows": 8,
        "n_features": 2,
        "n_classes": 2,
        "dedup_cluster_id": "fixture",
        "metadata_tier": "structural",
        "license_claim": "",
    }
    second_dataset = {
        **dataset,
        "dataset_id": "openml_2",
        "dataset_split": "test",
        "source_id": "2",
        "source_url": "https://www.openml.org/d/2",
        "task_type": "regression",
        "target_column": "Score",
        "feature_columns": ["Height", "Group"],
        "dataset_name": "second_fixture",
        "n_rows": 4,
        "n_classes": None,
        "dedup_cluster_id": "second_fixture",
    }
    table_prediction = {
        **dataset,
        "task_id": "openml_1:zero_label_serialized_table",
        "input_interface": "serialized_table",
        "parameter_updates": False,
        "target_visibility": "private_evaluation_only",
        "primary_metrics": ["accuracy", "balanced_accuracy", "macro_f1"],
    }
    second_table_prediction = {
        **second_dataset,
        "task_id": "openml_2:zero_label_serialized_table",
        "input_interface": "serialized_table",
        "parameter_updates": False,
        "target_visibility": "private_evaluation_only",
        "primary_metrics": ["mae", "rmse", "r2"],
    }
    episodes = [
        {
            **dataset,
            "episode_id": "eligible_k4",
            "episode_split": "validation",
            "shots": 4,
            "input_interface": "row_examples",
            "parameter_updates": False,
            "support_row_ids": ["0", "1", "2", "3"],
            "query_row_ids": ["4", "5"],
            "query_size": 2,
        },
        {
            **dataset,
            "episode_id": "missing_class_k4",
            "episode_split": "validation",
            "shots": 4,
            "input_interface": "row_examples",
            "parameter_updates": False,
            "support_row_ids": ["0", "2", "4", "6"],
            "query_row_ids": ["1", "3"],
            "query_size": 2,
        },
    ]
    _write_config(reference, "datasets", [dataset, second_dataset])
    _write_config(
        reference,
        "table_prediction_tasks",
        [table_prediction, second_table_prediction],
    )
    _write_config(reference, "prediction_episodes", episodes)
    (reference / "reference_summary.json").write_text(
        json.dumps({"schema_version": "1.0", "datasets": 2, "episodes": 2}),
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "Age": [20, 21, 22, 23, 24, 25, 26, 27],
                "Income": [100, 200, 300, 400, 500, 600, 700, 800],
                "Default": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        ),
        source / "1.parquet",
        compression="gzip",
    )
    pq.write_table(
        pa.table(
            {
                "Height": [160, 170, 180, 190],
                "Group": ["a", "b", "a", "b"],
                "Score": [1.5, 2.5, 3.5, 4.5],
            }
        ),
        source / "2.parquet",
        compression="gzip",
    )
    return reference, source


def _write_config(root: Path, config: str, rows: list[dict[str, object]]) -> None:
    destination = root / config / "train"
    destination.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), destination / "part-00000.parquet")
