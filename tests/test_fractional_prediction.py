from __future__ import annotations

from pathlib import Path

import pytest

from tablesuite import TableSuite, normalize_support_fractions


def test_scalar_and_one_element_sequence_are_equivalent() -> None:
    assert normalize_support_fractions(0.1) == (0.1,)
    assert normalize_support_fractions((0.1,)) == (0.1,)
    assert normalize_support_fractions((0.1, 0.3)) == (0.1, 0.3)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_support_fraction_validation(value: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        normalize_support_fractions(value)


def test_fractional_support_is_nested_and_keeps_queries_fixed(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    examples = list(
        TableSuite.open(reference, source=source).prediction(
            "icl",
            support=(0.0, 0.25, 0.5, 1.0),
            dataset_ids=("openml_1",),
            max_episodes_per_dataset=1,
        )
    )

    assert [example.shots for example in examples] == [0, 2, 3, 6]
    assert [example.support.count for example in examples if example.support] == [
        0,
        2,
        3,
        6,
    ]
    assert len({example.request.query.source.row_ids for example in examples}) == 1
    support_sets = [
        set(example.request.demonstrations.source.row_ids)
        if example.request.demonstrations
        else set()
        for example in examples
    ]
    assert all(
        left <= right
        for left, right in zip(support_sets, support_sets[1:], strict=False)
    )
    assert support_sets[-1].isdisjoint(examples[0].request.query.source.row_ids)


def test_icl_and_serialized_table_share_the_same_support_rows(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    suite = TableSuite.open(reference, source=source)
    common = {
        "support": (0.25, 0.5),
        "dataset_ids": ("openml_1",),
        "max_episodes_per_dataset": 1,
    }
    icl = list(suite.prediction("icl", **common))
    serialized = list(suite.prediction("serialized_table", **common))

    assert len(icl) == len(serialized) == 2
    for icl_example, table_example in zip(icl, serialized, strict=True):
        assert icl_example.request.demonstrations is not None
        assert table_example.request.visible_labels is not None
        assert (
            icl_example.request.demonstrations.source.row_ids
            == table_example.request.visible_labels.source.row_ids
        )
        assert (
            icl_example.request.query.source.row_ids
            == table_example.request.query_row_ids
        )
        assert icl_example.support == table_example.support


def test_zero_and_full_serialized_table_keep_query_targets_hidden(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    examples = list(
        TableSuite.open(reference, source=source).prediction(
            "serialized_table",
            support=(0.0, 1.0),
            dataset_ids=("openml_1",),
            max_episodes_per_dataset=1,
        )
    )
    zero, full = examples

    assert zero.request.visible_labels is None
    assert zero.request.query_row_ids == full.request.query_row_ids
    assert full.request.visible_labels is not None
    assert len(full.request.visible_labels.rows) == 6
    assert len(full.request.table.rows) == 8
    assert full.request.target_column not in full.request.table.source.columns
