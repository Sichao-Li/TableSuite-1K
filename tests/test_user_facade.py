from __future__ import annotations

from pathlib import Path

import pytest

from tablesuite import TableSuite, describe_task, list_tasks


def test_task_registry_exposes_three_public_families() -> None:
    tasks = list_tasks()

    assert tuple(task.name for task in tasks) == (
        "table_prediction",
        "table_grounding",
        "table_question_answering",
    )
    prediction, grounding, qa = tasks
    assert prediction.protocols == (
        "zero_shot_icl",
        "few_shot_icl",
        "zero_label_serialized_table",
        "partially_labeled_serialized_table",
    )
    assert prediction.generatable is False
    assert grounding.generatable is True
    assert qa.generatable is True
    assert describe_task("table_grounding") is grounding

    with pytest.raises(KeyError, match="unknown TableSuite task"):
        describe_task("not_a_task")


def test_tablesuite_facade_routes_catalog_and_prediction_selection(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    suite = TableSuite.open(reference, source=source)

    assert suite.tasks() == list_tasks()
    assert suite.describe("table_grounding").name == "table_grounding"
    assert suite.catalog_summary()["datasets"] == 2

    selected = suite.prediction(
        "few_shot_icl",
        dataset_ids=("openml_1",),
        shots=(4,),
        max_episodes_per_dataset_per_shot=1,
    )
    assert selected.manifest.selection.tasks == ("few_shot_icl",)
    assert selected.manifest.dataset_ids == ("openml_1",)
    assert selected.manifest.episode_ids == ("eligible_k4",)


def test_tablesuite_facade_rejects_wrong_mode_early(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    suite = TableSuite.open(reference, source=source)

    with pytest.raises(ValueError, match="official prediction protocols"):
        suite.official("table_prediction", split="dataset_test")
    with pytest.raises(ValueError, match="cannot be generated"):
        suite.generate("table_prediction")
    with pytest.raises(ValueError, match="unknown prediction protocol"):
        suite.prediction("not_a_protocol")
