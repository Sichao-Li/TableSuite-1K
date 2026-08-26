"""Tests for official task specifications, execution, and public loading."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tablesuite import Benchmark, TableSlice, load_task
from tablesuite._cli import main as cli_main
from tablesuite.evaluation import (
    BENCHMARK_VERSION,
    PLAN_SCHEMA_VERSION,
    EvaluationPlan,
    MappingPredictionResolver,
    OperationSpec,
    PlanExecutor,
    PlanRegistry,
    RenderingSpec,
    ScoringSpec,
    audit_plans,
)
from tablesuite.evaluation.operations import EXECUTOR_VERSION, OperationResult
from tablesuite.evaluation.rendering import GENERATOR_VERSION, render_evaluation_request
from tablesuite.task_records import public_task_record
from tablesuite.types import MaterializedTableSlice


def test_plan_registry_round_trip_is_value_free(tmp_path: Path) -> None:
    plan = _plan(
        plan_id="qa_001",
        task="qa",
        source=TableSlice("openml_1", ("0", "1"), ("Income",)),
        operation=OperationSpec(
            "aggregate",
            {"column": "Income", "aggregation": "mean"},
        ),
        scoring=ScoringSpec("float"),
    )
    path = tmp_path / "plans.jsonl"

    PlanRegistry((plan,)).save(path)
    restored = PlanRegistry.load(path)

    assert restored.get_plan("qa_001") == plan
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["item_id"] == "qa_001"
    assert "plan_id" not in record
    assert "answer" not in record
    assert "question" not in record
    argument_names = {
        item["name"] for item in record["operation"]["arguments"]
    }
    assert "filter_value" not in argument_names
    with pytest.raises(TypeError):
        plan.operation.arguments["column"] = "Age"  # type: ignore[index]


def test_plan_registry_loads_mixed_operations_from_parquet(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    grounding = _plan(
        plan_id="grounding_parquet",
        task="grounding",
        source=TableSlice("openml_1", ("0",), ("Age",)),
        operation=OperationSpec("cell_lookup", {"column": "Age"}),
        scoring=ScoringSpec("integer"),
    )
    aggregate = _plan(
        plan_id="qa_parquet",
        task="qa",
        source=TableSlice("openml_1", ("0", "1"), ("Income",)),
        operation=OperationSpec(
            "aggregate",
            {"column": "Income", "aggregation": "mean"},
        ),
        scoring=ScoringSpec("float"),
    )
    path = tmp_path / "task_items.parquet"
    pq.write_table(
        pa.Table.from_pylist([grounding.to_record(), aggregate.to_record()]),
        path,
    )

    restored = PlanRegistry.load(path)

    assert restored.get_plan("grounding_parquet") == grounding
    assert restored.get_plan("qa_parquet") == aggregate


def test_grounding_plan_materializes_wording_gold_and_score(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    plan = _plan(
        plan_id="grounding_001",
        task="grounding",
        source=TableSlice("openml_1", ("1",), ("Age", "Income")),
        operation=OperationSpec("cell_lookup", {"column": "Age"}),
        scoring=ScoringSpec("integer"),
    )
    executor = PlanExecutor(
        Benchmark.from_path(reference, source), PlanRegistry((plan,))
    )

    first = executor.materialize("grounding_001")
    second = executor.materialize("grounding_001")

    assert first == second
    assert first.gold.answer == 21
    assert first.gold.evidence[0].row_id == "1"
    assert not hasattr(first.request, "answer")
    assert "Income" in first.request.table_text
    assert "What" in first.request.question or "Report" in first.request.question
    assert executor.score(plan, "21").correct
    assert not executor.score(plan, "22").correct


def test_qa_plan_executes_source_referenced_filter_without_plan_values(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    plan = _plan(
        plan_id="qa_filtered_001",
        task="qa",
        evaluation_split="dataset_test",
        dataset_split="test",
        dedup_cluster_id="second_fixture",
        source_id="2",
        source=TableSlice(
            "openml_2",
            ("0", "1", "2", "3"),
            ("Height", "Group"),
        ),
        operation=OperationSpec(
            "filtered_argmax_lookup",
            {
                "filter_column": "Group",
                "filter_value_row_id": "0",
                "maximize_column": "Height",
                "return_column": "Height",
            },
        ),
        scoring=ScoringSpec("integer"),
        template_split="train",
    )
    executor = PlanExecutor(
        Benchmark.from_path(reference, source), PlanRegistry((plan,))
    )

    item = executor.materialize(plan)

    assert item.gold.answer == 180
    assert "Group" in item.request.question
    assert "a" in item.request.question
    assert executor.score(plan, "180").correct


def test_qa_aggregate_and_argmax_operations(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    aggregate = _plan(
        plan_id="qa_sum_001",
        task="qa",
        source=TableSlice("openml_1", ("0", "1"), ("Income",)),
        operation=OperationSpec(
            "aggregate",
            {"column": "Income", "aggregation": "sum"},
        ),
        scoring=ScoringSpec("integer"),
    )
    argmax = _plan(
        plan_id="qa_argmax_001",
        task="qa",
        source=TableSlice("openml_1", ("0", "1"), ("Age", "Income")),
        operation=OperationSpec(
            "argmax_lookup",
            {"maximize_column": "Income", "return_column": "Age"},
        ),
        scoring=ScoringSpec("integer"),
    )
    executor = PlanExecutor(
        Benchmark.from_path(reference, source),
        PlanRegistry((aggregate, argmax)),
    )

    assert executor.materialize(aggregate).gold.answer == 300
    assert executor.materialize(argmax).gold.answer == 21


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (OperationSpec("row_lookup", {"row_id": "1"}), {"Age": 21, "Income": 200}),
        (OperationSpec("column_values", {"column": "Age"}), [20, 21]),
        (OperationSpec("distinct_values", {"column": "Age"}), [20, 21]),
        (
            OperationSpec("value_counts", {"column": "Age"}),
            [{"value": 20, "count": 1}, {"value": 21, "count": 1}],
        ),
    ],
)
def test_table_grounding_operations_are_closed_world(
    benchmark_fixture: tuple[Path, Path],
    operation: OperationSpec,
    expected: object,
) -> None:
    reference, source = benchmark_fixture
    plan = _plan(
        plan_id=f"grounding_{operation.name}",
        task="grounding",
        source=TableSlice("openml_1", ("0", "1"), ("Age", "Income")),
        operation=operation,
        scoring=ScoringSpec("json"),
    )
    executor = PlanExecutor(
        Benchmark.from_path(reference, source), PlanRegistry((plan,))
    )

    item = executor.materialize(plan)

    assert item.gold.answer == expected
    assert executor.score(plan, json.dumps(expected)).correct
    assert all(
        cell.row_id in plan.source.row_ids and cell.column in plan.source.columns
        for cell in item.gold.evidence
    )


def test_rendered_questions_quote_ambiguous_column_names() -> None:
    plan = _plan(
        plan_id="grounding_numeric_header",
        task="grounding",
        source=TableSlice("openml_1", ("0",), ("1",)),
        operation=OperationSpec("cell_lookup", {"column": "1"}),
        scoring=ScoringSpec("integer"),
    )
    table = MaterializedTableSlice(plan.source, ({"1": 42},))
    result = OperationResult(
        answer=42,
        evidence=(),
        render_values={"column": "1"},
    )

    request = render_evaluation_request(plan, table, result)

    assert '"1"' in request.question
    assert "| row_id | 1 |" in request.table_text


def test_prediction_and_integrated_plans_use_injected_packets(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    prediction = _plan(
        plan_id="prediction_001",
        task="prediction",
        source=TableSlice("openml_1", ("4",), ("Age", "Income")),
        operation=OperationSpec("prediction_lookup", {"query_row_id": "4"}),
        scoring=ScoringSpec("string"),
        prediction_packet_id="teacher_v1",
    )
    integrated = _plan(
        plan_id="reasoning_001",
        task="integrated_reasoning",
        source=TableSlice("openml_1", ("5",), ("Age",)),
        operation=OperationSpec(
            "prediction_with_cell",
            {
                "query_row_id": "5",
                "value_row_id": "5",
                "value_column": "Age",
            },
        ),
        scoring=ScoringSpec("json"),
        prediction_packet_id="teacher_v1",
    )
    resolver = MappingPredictionResolver(
        {"teacher_v1": {"4": "approve", "5": "decline"}}
    )
    executor = PlanExecutor(
        Benchmark.from_path(reference, source),
        PlanRegistry((prediction, integrated)),
        prediction_resolver=resolver,
    )

    prediction_item = executor.materialize(prediction)
    integrated_item = executor.materialize(integrated)

    assert prediction_item.gold.answer == "approve"
    assert "Frozen prediction channel" in prediction_item.request.input_text
    assert executor.score(prediction, "APPROVE").correct
    assert integrated_item.gold.answer == {
        "prediction": "decline",
        "source_value": 25,
    }
    response = '{"prediction":"decline","source_value":25}'
    assert executor.score(integrated, response).correct


def test_plan_audit_rejects_cluster_and_source_overlap() -> None:
    first = _plan(
        plan_id="train_001",
        task="grounding",
        source=TableSlice("openml_1", ("0",), ("Age",)),
        operation=OperationSpec("cell_lookup", {"column": "Age"}),
        scoring=ScoringSpec("integer"),
    )
    overlapping = replace(
        first,
        item_id="episode_001",
        evaluation_split="episode_test",
        rendering=replace(first.rendering, template_split="test"),
    )
    cross_partition = _plan(
        plan_id="dataset_001",
        task="grounding",
        evaluation_split="dataset_test",
        dataset_split="test",
        dedup_cluster_id="fixture",
        source_id="2",
        source=TableSlice("openml_2", ("0",), ("Height",)),
        operation=OperationSpec("cell_lookup", {"column": "Height"}),
        scoring=ScoringSpec("integer"),
        template_split="test",
    )

    audit = audit_plans((first, overlapping, cross_partition))

    assert not audit.passed
    assert any("source cell" in error for error in audit.errors)
    assert any("dedup cluster" in error for error in audit.errors)


def test_plan_audit_rejects_untyped_literal_arguments() -> None:
    plan = _plan(
        plan_id="qa_literal_filter",
        task="qa",
        source=TableSlice("openml_1", ("0", "1"), ("Age", "Income")),
        operation=OperationSpec(
            "filtered_argmax_lookup",
            {
                "filter_column": "Age",
                "filter_value": "20",
                "maximize_column": "Income",
                "return_column": "Age",
            },
        ),
        scoring=ScoringSpec("integer"),
    )

    audit = audit_plans((plan,))

    assert not audit.passed
    assert "filter_value_row_id" in audit.errors[0]


@pytest.mark.parametrize(
    "operation",
    [
        OperationSpec("cell_lookup", {"column": "Outside"}),
        OperationSpec("row_lookup", {"row_id": "outside"}),
    ],
)
def test_plan_audit_rejects_operation_references_outside_source(
    operation: OperationSpec,
) -> None:
    plan = _plan(
        plan_id=f"outside_{operation.name}",
        task="grounding",
        source=TableSlice("openml_1", ("0",), ("Age",)),
        operation=operation,
        scoring=ScoringSpec("json"),
    )

    audit = audit_plans((plan,))

    assert not audit.passed
    assert "outside the source slice" in audit.errors[0]


def test_executor_rejects_catalog_binding_mismatch(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    plan = _plan(
        plan_id="grounding_bad_source",
        task="grounding",
        source=TableSlice("openml_1", ("0",), ("Age",)),
        operation=OperationSpec("cell_lookup", {"column": "Age"}),
        scoring=ScoringSpec("integer"),
        source_id="999",
    )
    executor = PlanExecutor(
        Benchmark.from_path(reference, source), PlanRegistry((plan,))
    )

    with pytest.raises(ValueError, match="expects source"):
        executor.materialize(plan)


def test_executor_rejects_modified_plan_object(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    plan = _plan(
        plan_id="grounding_frozen",
        task="grounding",
        source=TableSlice("openml_1", ("0",), ("Age",)),
        operation=OperationSpec("cell_lookup", {"column": "Age"}),
        scoring=ScoringSpec("integer"),
    )
    executor = PlanExecutor(
        Benchmark.from_path(reference, source), PlanRegistry((plan,))
    )
    changed = replace(plan, rendering=replace(plan.rendering, render_seed=99))

    with pytest.raises(ValueError, match="differs from its frozen registry"):
        executor.materialize(changed)


def test_task_api_materializes_input_only_examples_and_reports_metrics(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    first = _plan(
        plan_id="grounding_task_1",
        task="grounding",
        source=TableSlice("openml_1", ("1",), ("Age",)),
        operation=OperationSpec("cell_lookup", {"column": "Age"}),
        scoring=ScoringSpec("integer"),
    )
    second = _plan(
        plan_id="grounding_task_2",
        task="grounding",
        source=TableSlice("openml_1", ("2",), ("Age",)),
        operation=OperationSpec("cell_lookup", {"column": "Age"}),
        scoring=ScoringSpec("integer"),
    )
    PlanRegistry((first, second)).save(
        reference / "tasks" / "table_grounding" / "train.jsonl"
    )
    task = load_task(
        reference,
        "table_grounding",
        split="train",
        source=source,
    )

    example = task["grounding_task_1"]
    report = task.evaluate({"grounding_task_1": "21", "grounding_task_2": "wrong"})

    assert len(task) == 2
    assert tuple(item.id for item in task.iter_by_dataset()) == (
        "grounding_task_1",
        "grounding_task_2",
    )
    assert example.id == "grounding_task_1"
    assert not hasattr(example, "answer")
    assert task.summary()["examples"] == 2
    assert task.score(example.id, "21").correct
    with pytest.raises(ValueError, match="missing 1 examples"):
        task.evaluate({"grounding_task_1": "21"})
    assert report.accuracy == 0.5
    assert report.dataset_macro_accuracy == 0.5
    assert report.by_operation[0].name == "cell_lookup"


def test_task_cli_previews_and_scores(
    benchmark_fixture: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference, source = benchmark_fixture
    plan = _plan(
        plan_id="grounding_cli",
        task="grounding",
        source=TableSlice("openml_1", ("2",), ("Age",)),
        operation=OperationSpec("cell_lookup", {"column": "Age"}),
        scoring=ScoringSpec("integer"),
    )
    plans_path = reference / "tasks" / "table_grounding" / "train.jsonl"
    PlanRegistry((plan,)).save(plans_path)

    cli_main(
        [
            "task",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--name",
            "table_grounding",
            "--split",
            "train",
            "--dataset-id",
            "openml_1",
            "--item-id",
            "grounding_cli",
            "--response",
            "22",
        ]
    )

    output = capsys.readouterr().out
    assert "Question:" in output
    assert '"correct": true' in output


def test_load_task_can_filter_a_bounded_dataset_subset(
    benchmark_fixture: tuple[Path, Path],
) -> None:
    reference, source = benchmark_fixture
    second = _plan(
        plan_id="grounding_openml_2",
        task="grounding",
        evaluation_split="dataset_test",
        dataset_split="test",
        dedup_cluster_id="second_fixture",
        source_id="2",
        source=TableSlice("openml_2", ("1",), ("Height",)),
        operation=OperationSpec("cell_lookup", {"column": "Height"}),
        scoring=ScoringSpec("integer"),
    )
    PlanRegistry((second,)).save(
        reference / "tasks" / "table_grounding" / "dataset_test.jsonl"
    )

    task = load_task(
        reference,
        "table_grounding",
        split="dataset_test",
        source=source,
        dataset_ids=("openml_2",),
    )

    assert len(task) == 1
    assert task[0].dataset_id == "openml_2"
    with pytest.raises(KeyError, match="unknown dataset IDs"):
        load_task(
            reference,
            "table_grounding",
            split="dataset_test",
            source=source,
            dataset_ids=("openml_missing",),
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_load_task_uses_huggingface_configuration_and_split(
    benchmark_fixture: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    legacy: bool,
) -> None:
    reference, source = benchmark_fixture
    import pyarrow.parquet as pq

    dataset_rows = [
        row
        for shard in sorted((reference / "datasets").rglob("*.parquet"))
        for row in pq.read_table(shard).to_pylist()
    ]
    plan = replace(
        _plan(
            plan_id="hf_grounding_1",
            task="grounding",
            source=TableSlice("openml_1", ("1",), ("Age",)),
            operation=OperationSpec("cell_lookup", {"column": "Age"}),
            scoring=ScoringSpec("integer"),
        ),
        reference_id="organization/openml-table-tasks",
    )
    calls: list[tuple[str, str, str | None]] = []

    def fake_load_dataset(
        repository: str,
        name: str,
        *,
        split: str | None = None,
        revision: str | None = None,
    ):
        del revision
        calls.append((repository, name, split))
        if name == "datasets":
            return {"train": dataset_rows}
        assert name == "table_grounding"
        assert split == "train"
        return [plan.to_record() if legacy else public_task_record(plan)]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=fake_load_dataset),
    )

    task = load_task(
        "organization/openml-table-tasks",
        "table_grounding",
        split="train",
        source=source,
    )

    assert task[0].id == "hf_grounding_1"
    assert calls == [
        ("organization/openml-table-tasks", "table_grounding", "train"),
        ("organization/openml-table-tasks", "datasets", None),
    ]


def _plan(
    *,
    plan_id: str,
    task: str,
    source: TableSlice,
    operation: OperationSpec,
    scoring: ScoringSpec,
    evaluation_split: str = "train",
    dataset_split: str = "train",
    dedup_cluster_id: str = "fixture",
    source_id: str = "1",
    template_split: str = "train",
    prediction_packet_id: str | None = None,
) -> EvaluationPlan:
    return EvaluationPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        benchmark_version=BENCHMARK_VERSION,
        reference_id="tablesuite-1k:1.0",
        item_id=plan_id,
        task=task,
        evaluation_split=evaluation_split,
        dataset_split=dataset_split,
        dedup_cluster_id=dedup_cluster_id,
        source_id=source_id,
        source=source,
        operation=operation,
        rendering=RenderingSpec(
            template_family=operation.name,
            template_split=template_split,
            render_seed=731,
            view="markdown",
        ),
        scoring=scoring,
        generator_version=GENERATOR_VERSION,
        executor_version=EXECUTOR_VERSION,
        prediction_packet_id=prediction_packet_id,
    )
