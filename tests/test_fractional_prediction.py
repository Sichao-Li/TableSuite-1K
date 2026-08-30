from __future__ import annotations

from pathlib import Path

import pytest

from tablesuite import (
    TableSuite,
    render_icl_prediction,
)
from tablesuite.prediction import normalize_support_fractions
from tablesuite.prediction_evaluation import _cluster_macro_metric


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


def test_context_fit_is_exact_auditable_and_never_silently_truncates(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    suite = TableSuite.open(reference, source=source)
    zero = next(
        iter(
            suite.prediction(
                "icl",
                support=0.0,
                dataset_ids=("openml_1",),
                max_episodes_per_dataset=1,
            )
        )
    )
    def counter(text: str) -> int:
        return len(text.split())
    zero_tokens = counter(render_icl_prediction(zero.request).input_text)

    maximum = suite.prediction(
        "icl",
        support=1.0,
        dataset_ids=("openml_1",),
        max_episodes_per_dataset=1,
    )
    fitted = maximum.fit_context(
        max_prompt_tokens=zero_tokens,
        count_tokens=counter,
        tokenizer_id="fixture-whitespace",
    )

    assert len(fitted) == 1
    assert fitted[0].shots == 0
    assert fitted.report.coverage == 1.0
    assert fitted.report.decisions[0].prompt_tokens == zero_tokens
    assert fitted.report.decisions[0].support.count == 0

    excluded = maximum.fit_context(
        max_prompt_tokens=zero_tokens - 1,
        count_tokens=counter,
        tokenizer_id="fixture-whitespace",
    )
    assert len(excluded) == 0
    assert excluded.report.coverage == 0.0
    assert excluded.report.decisions[0].reason == (
        "zero_support_exceeds_prompt_budget"
    )


def test_context_fit_requires_one_support_cap(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    prediction = TableSuite.open(reference, source=source).prediction(
        "icl",
        support=(0.1, 0.3),
        dataset_ids=("openml_1",),
        max_episodes_per_dataset=1,
    )
    with pytest.raises(ValueError, match="exactly one support fraction"):
        prediction.fit_context(
            max_prompt_tokens=100,
            count_tokens=lambda text: len(text),
            tokenizer_id="fixture",
        )


def test_prediction_evaluation_reports_dataset_macro_metrics_and_coverage(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    prediction = TableSuite.open(reference, source=source).prediction(
        "icl",
        support=0.0,
        dataset_ids=("openml_1",),
        max_episodes_per_dataset=1,
    )
    example = next(iter(prediction))
    report = prediction.evaluate(
        {example.request.request_id: example.gold.query_targets}
    )
    classification = report.families[0]

    assert classification.task_family == "classification"
    assert classification.primary_metric == "dataset_macro_balanced_accuracy"
    assert classification.dataset_macro_balanced_accuracy == 1.0
    assert classification.dataset_macro_f1 == 1.0
    assert classification.coverage == 1.0

    invalid = prediction.evaluate(
        {example.request.request_id: ("not-a-label", None)}
    )
    assert invalid.families[0].coverage == 0.0
    assert invalid.families[0].dataset_macro_balanced_accuracy is None


def test_prediction_evaluation_rejects_missing_requests_by_default(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    prediction = TableSuite.open(reference, source=source).prediction(
        "icl",
        support=0.0,
        dataset_ids=("openml_1",),
        max_episodes_per_dataset=1,
    )
    with pytest.raises(ValueError, match="missing 1 requests"):
        prediction.evaluate({})

    partial = prediction.evaluate({}, allow_partial=True)
    assert partial.submitted_requests == 0
    assert partial.families[0].coverage == 0.0


def test_cluster_macro_metrics_preserve_dataset_local_scales() -> None:
    metrics = {
        "dataset_a": {"score": 1.0},
        "dataset_b": {"score": 0.5},
        "dataset_c": {"score": 0.0},
    }
    clusters = {
        "dataset_a": "duplicate_cluster",
        "dataset_b": "duplicate_cluster",
        "dataset_c": "independent_cluster",
    }

    assert _cluster_macro_metric(metrics, clusters, "score") == 0.375
