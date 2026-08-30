from __future__ import annotations

from pathlib import Path

import pytest

import tablesuite
from tablesuite import TableSlice, __version__
from tablesuite.benchmark import Benchmark, _derive_zero_shot_episodes
from tablesuite.catalog import Catalog
from tablesuite.rendering import (
    render_icl_prediction,
    render_serialized_table_prediction,
    render_table,
)
from tablesuite.types import Selection, SelectionManifest


def test_package_version_is_available() -> None:
    assert __version__ == "2.1.0"


def test_root_namespace_exposes_only_stable_user_interfaces() -> None:
    assert "TableSuite" in tablesuite.__all__
    assert "PredictionReport" in tablesuite.__all__
    assert "Benchmark" not in tablesuite.__all__
    assert "Catalog" not in tablesuite.__all__
    assert "Selection" not in tablesuite.__all__


def test_catalog_selection_and_manifest(
    benchmark_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    reference, source = benchmark_fixture
    catalog = Catalog.from_path(reference)

    assert catalog.summary()["datasets"] == 2
    benchmark = Benchmark.from_path(reference, source)
    subset = benchmark.select(
        Selection(
            tasks=("few_shot_icl",),
            dataset_ids=("openml_1",),
            shots=(4,),
            max_episodes_per_dataset_per_shot=1,
        )
    )

    assert subset.manifest.dataset_ids == ("openml_1",)
    assert subset.manifest.episode_ids == ("eligible_k4",)
    path = tmp_path / "selection.json"
    subset.manifest.save(path)
    assert SelectionManifest.load(path) == subset.manifest


def test_zero_label_serialized_table_is_multirow_and_target_free(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    benchmark = Benchmark.from_path(reference, source)

    example = next(
        benchmark.select(
            Selection(
                tasks=("zero_label_serialized_table",),
                dataset_ids=("openml_1",),
            )
        ).zero_label_serialized_table()
    )
    assert example.request.protocol == "zero_label_serialized_table"
    assert example.request.scope == "full_table"
    assert example.request.visible_labels is None
    assert example.request.class_labels == (0, 1)
    assert example.request.query_row_ids == tuple(str(index) for index in range(8))
    assert not hasattr(example.request, "query_targets")
    assert not hasattr(example, "fold")
    assert len(example.request.table.rows) == 8
    assert example.gold.query_targets == (0, 1, 0, 1, 0, 1, 0, 1)

    rendered = render_serialized_table_prediction(example.request, view="markdown")
    assert rendered.input_text.startswith(
        "Task family: classification\n"
        "Target: Default\n"
        "Allowed target labels: [0,1]\n"
        'Predict "Default" for every row.'
    )
    assert "| row_id | Age | Income |" in rendered.input_text
    assert "| r0 | 20 | 100 |" in rendered.input_text
    assert "| r7 | 27 | 800 |" in rendered.input_text
    assert "Default" not in rendered.input_text.split("\n\n", maxsplit=1)[1]
    assert rendered.query_aliases == {
        f"r{index}": str(index) for index in range(8)
    }


def test_partially_labeled_serialized_table_exposes_only_support_targets(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    example = next(
        Benchmark.from_path(reference, source)
        .select(
            Selection(
                tasks=("partially_labeled_serialized_table",),
                dataset_ids=("openml_1",),
                shots=(4,),
            )
        )
        .partially_labeled_serialized_table()
    )

    assert example.request.protocol == "partially_labeled_serialized_table"
    assert example.request.scope == "full_table"
    assert example.request.visible_labels is not None
    assert example.request.class_labels == (0, 1)
    assert example.request.visible_labels.source.row_ids == ("0", "1", "2", "3")
    assert example.request.query_row_ids == ("4", "5", "6", "7")
    assert example.gold.query_targets == (0, 1, 0, 1)

    rendered = render_serialized_table_prediction(example.request, view="markdown")
    assert rendered.input_text.startswith(
        "Task family: classification\n"
        "Target: Default\n"
        "Allowed target labels: [0,1]\n"
        'Predict "Default" for rows where the target is masked.'
    )
    assert "| row_id | Age | Income | Default |" in rendered.input_text
    assert "| r0 | 20 | 100 | 0 |" in rendered.input_text
    assert "| r3 | 23 | 400 | 1 |" in rendered.input_text
    assert "| r4 | 24 | 500 | ? |" in rendered.input_text
    assert "| r5 | 25 | 600 | ? |" in rendered.input_text
    assert "| r7 | 27 | 800 | ? |" in rendered.input_text
    assert rendered.query_aliases == {
        "r4": "4",
        "r5": "5",
        "r6": "6",
        "r7": "7",
    }


def test_few_shot_icl_uses_row_examples_not_table_serialization(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    benchmark = Benchmark.from_path(reference, source)

    example = next(
        benchmark.select(
            Selection(
                tasks=("few_shot_icl",),
                dataset_ids=("openml_1",),
                shots=(4,),
            )
        ).few_shot_icl()
    )
    assert example.request.protocol == "few_shot_icl"
    assert example.request.demonstrations is not None
    assert example.shots == 4
    assert len(example.request.demonstrations.rows) == 4
    assert example.request.class_labels == (0, 1)
    assert len(example.gold.query_targets) == 2

    rendered = render_icl_prediction(example.request)
    assert rendered.input_text.startswith(
        "Task family: classification\n"
        "Target: Default\n"
        "Allowed target labels: [0,1]"
    )
    assert "Row A: Age=20, Income=100 -> 0" in rendered.input_text
    assert "Row D: Age=23, Income=400 -> 1" in rendered.input_text
    assert "Query q0: Age=24, Income=500 -> ?" in rendered.input_text
    assert "Query q1: Age=25, Income=600 -> ?" in rendered.input_text
    assert "|" not in rendered.input_text


def test_zero_shot_icl_reuses_frozen_queries_without_demonstrations(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    example = next(
        Benchmark.from_path(reference, source)
        .select(
            Selection(
                tasks=("zero_shot_icl",),
                dataset_ids=("openml_1",),
            )
        )
        .zero_shot_icl()
    )

    assert example.request.protocol == "zero_shot_icl"
    assert example.shots == 0
    assert example.request.demonstrations is None
    assert example.request.class_labels == (0, 1)
    rendered = render_icl_prediction(example.request)
    assert "Row A:" not in rendered.input_text
    assert "Examples:" not in rendered.input_text
    assert "Query q0: Age=24, Income=500 -> ?" in rendered.input_text
    assert "Allowed target labels: [0,1]" in rendered.input_text


def test_regression_request_declares_family_without_class_labels(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    example = next(
        Benchmark.from_path(reference, source)
        .select(
            Selection(
                tasks=("zero_label_serialized_table",),
                dataset_ids=("openml_2",),
            )
        )
        .zero_label_serialized_table()
    )

    assert example.request.class_labels == ()
    rendered = render_serialized_table_prediction(example.request)
    assert rendered.input_text.startswith("Task family: regression\nTarget: Score\n")
    assert "Allowed target labels:" not in rendered.input_text


def test_zero_shot_episode_deduplication_is_dataset_local() -> None:
    candidates = [
        {
            "dataset_id": dataset_id,
            "episode_id": f"{dataset_id}:k4",
            "shots": 4,
            "support_row_ids": ["0", "1", "2", "3"],
            "query_row_ids": ["4", "5"],
        }
        for dataset_id in ("openml_1", "openml_2")
    ]

    assert {item["dataset_id"] for item in _derive_zero_shot_episodes(candidates)} == {
        "openml_1",
        "openml_2",
    }


def test_episode_limit_applies_to_each_selected_few_shot_protocol(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    examples = list(
        Benchmark.from_path(reference, source)
        .select(
            Selection(
                tasks=("few_shot_icl", "partially_labeled_serialized_table"),
                dataset_ids=("openml_1",),
                shots=(4,),
                max_episodes_per_dataset_per_shot=1,
            )
        )
        .few_shot_icl()
    )

    assert len(examples) == 1
    assert examples[0].shots == 4


def test_icl_and_partial_table_reuse_the_same_frozen_episode(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    subset = Benchmark.from_path(reference, source).select(
        Selection(
            tasks=("few_shot_icl", "partially_labeled_serialized_table"),
            dataset_ids=("openml_1",),
            shots=(4,),
        )
    )

    icl = next(subset.few_shot_icl())
    table = next(
        subset.partially_labeled_serialized_table(query_scope="episode")
    )

    assert icl.request.demonstrations is not None
    assert table.request.visible_labels is not None
    assert (
        icl.request.demonstrations.source.row_ids
        == table.request.visible_labels.source.row_ids
    )
    assert icl.request.query.source.row_ids == table.request.query_row_ids


def test_zero_label_episode_scope_reuses_zero_shot_queries(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    subset = Benchmark.from_path(reference, source).select(
        Selection(
            tasks=("zero_label_serialized_table", "zero_shot_icl"),
            dataset_ids=("openml_1",),
        )
    )

    icl = next(subset.zero_shot_icl())
    table = next(subset.zero_label_serialized_table(scope="episode"))

    assert table.request.scope == "episode"
    assert table.request.visible_labels is None
    assert table.request.query_row_ids == icl.request.query.source.row_ids
    assert table.gold.query_targets == icl.gold.query_targets


def test_serialized_table_prediction_can_be_deterministically_chunked(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    examples = list(
        Benchmark.from_path(reference, source)
        .select(
            Selection(
                tasks=("zero_label_serialized_table",),
                dataset_ids=("openml_1",),
            )
        )
        .zero_label_serialized_table(rows_per_table=3)
    )

    assert [len(example.request.table.rows) for example in examples] == [3, 3, 2]
    assert [example.request.table.source.row_ids for example in examples] == [
        ("0", "1", "2"),
        ("3", "4", "5"),
        ("6", "7"),
    ]


def test_partially_labeled_full_table_can_chunk_hidden_rows(
    benchmark_fixture: tuple[Path, Path]
) -> None:
    reference, source = benchmark_fixture
    examples = list(
        Benchmark.from_path(reference, source)
        .select(
            Selection(
                tasks=("partially_labeled_serialized_table",),
                dataset_ids=("openml_1",),
                shots=(4,),
            )
        )
        .partially_labeled_serialized_table(query_rows_per_table=2)
    )

    assert [example.request.query_row_ids for example in examples] == [
        ("4", "5"),
        ("6", "7"),
    ]
    assert all(len(example.request.visible_labels.rows) == 4 for example in examples)


def test_selection_is_deterministic(benchmark_fixture: tuple[Path, Path]) -> None:
    reference, source = benchmark_fixture
    benchmark = Benchmark.from_path(reference, source)
    selection = Selection(tasks=("few_shot_icl",), seed=7)

    first = benchmark.select(selection).manifest
    second = benchmark.select(selection).manifest

    assert first == second


def test_shots_are_only_valid_for_label_visible_protocols() -> None:
    with pytest.raises(ValueError, match="shots require"):
        Selection(tasks=("zero_shot_icl",), shots=(4,)).validate()
    with pytest.raises(ValueError, match="unsupported shot counts"):
        Selection(tasks=("few_shot_icl",), shots=(0,)).validate()


def test_rows_and_subtables_materialize_across_selected_datasets(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    subset = Benchmark.from_path(reference, source).select(
        Selection(tasks=(), dataset_ids=("openml_1", "openml_2"))
    )
    requested = (
        TableSlice(
            dataset_id="openml_1",
            row_ids=("1", "3"),
            columns=("Age", "Income"),
        ),
        TableSlice(
            dataset_id="openml_2",
            row_ids=("0",),
            columns=("Group",),
        ),
    )

    first, second = subset.materialize_many(requested)

    assert first.source == requested[0]
    assert first.rows == (
        {"Age": 21, "Income": 200},
        {"Age": 23, "Income": 400},
    )
    assert second.rows == ({"Group": "a"},)
    assert "| 21 | 200 |" in render_table(first, view="markdown")


def test_table_slice_rejects_implicit_or_invalid_source_access(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    subset = Benchmark.from_path(reference, source).select(
        Selection(tasks=(), dataset_ids=("openml_1",))
    )

    with pytest.raises(KeyError, match="not in this benchmark subset"):
        subset.materialize(
            TableSlice(dataset_id="openml_2", row_ids=("0",), columns=("Group",))
        )
    with pytest.raises(ValueError, match="unknown slice columns"):
        subset.materialize(
            TableSlice(dataset_id="openml_1", row_ids=("0",), columns=("unknown",))
        )
