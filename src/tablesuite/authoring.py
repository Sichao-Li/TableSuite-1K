"""Deterministic authoring of value-free official evaluation plans."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from tablesuite._util import (
    canonical_json,
    normalize_value,
    stable_id,
    stable_order,
    stable_value_key,
)
from tablesuite.catalog import Catalog
from tablesuite.evaluation.contracts import (
    BENCHMARK_VERSION,
    PLAN_SCHEMA_VERSION,
    EvaluationPlan,
    EvaluationSplit,
    OperationSpec,
    RenderingSpec,
    ScoringSpec,
)
from tablesuite.evaluation.operations import EXECUTOR_VERSION
from tablesuite.evaluation.rendering import GENERATOR_VERSION
from tablesuite.source import ParquetSource
from tablesuite.types import DatasetSpec, Selection, TableSlice

CELL_SPLITS = (
    "train",
    "validation",
    "episode_test",
    "dataset_test",
    "template_test",
)
QA_SPLITS = (*CELL_SPLITS, "composition_test")
TASK_NAMES = ("cell_grounding", "table_question_answering")


@dataclass(frozen=True)
class TaskGenerationConfig:
    """Frozen authoring policy for the first public task release."""

    seed: int = 0
    cell_items_per_dataset: int = 32
    cell_transfer_items_per_dataset: int = 16
    qa_items_per_dataset: int = 12
    qa_transfer_items_per_dataset: int = 6
    min_cell_context_columns: int = 4
    max_cell_context_columns: int = 8
    min_qa_context_columns: int = 3
    max_qa_context_columns: int = 8
    qa_row_sizes: tuple[int, ...] = (4, 8, 16)
    shard_size: int = 50_000

    def __post_init__(self) -> None:
        positive = (
            "cell_items_per_dataset",
            "cell_transfer_items_per_dataset",
            "qa_items_per_dataset",
            "qa_transfer_items_per_dataset",
            "min_cell_context_columns",
            "max_cell_context_columns",
            "min_qa_context_columns",
            "max_qa_context_columns",
            "shard_size",
        )
        if invalid := [name for name in positive if getattr(self, name) <= 0]:
            raise ValueError(f"generation limits must be positive: {invalid}")
        if self.min_cell_context_columns > self.max_cell_context_columns:
            raise ValueError("minimum cell context exceeds its maximum")
        if self.min_qa_context_columns > self.max_qa_context_columns:
            raise ValueError("minimum QA context exceeds its maximum")
        if not self.qa_row_sizes or any(size < 2 for size in self.qa_row_sizes):
            raise ValueError("QA row sizes must contain integers of at least two")
        object.__setattr__(self, "qa_row_sizes", tuple(sorted(set(self.qa_row_sizes))))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible authoring configuration."""

        return asdict(self)


@dataclass(frozen=True)
class TaskGenerationReport:
    """Counts and eligibility notes from one deterministic authoring pass."""

    plans: tuple[EvaluationPlan, ...]
    task_counts: dict[str, dict[str, int]]
    dataset_counts: dict[str, dict[str, int]]
    skipped: dict[str, int]

    def summary(self) -> dict[str, Any]:
        """Return value-free counts suitable for release metadata."""

        return {
            "task_counts": self.task_counts,
            "dataset_counts": self.dataset_counts,
            "skipped": self.skipped,
        }


def generate_task_plans(
    catalog: Catalog,
    source: ParquetSource,
    config: TaskGenerationConfig,
    *,
    dataset_ids: tuple[str, ...] = (),
    tasks: tuple[str, ...] = TASK_NAMES,
    splits: tuple[EvaluationSplit, ...] = (),
) -> TaskGenerationReport:
    """Generate deterministic grounding and QA plans from a reference catalog.

    Optional filters let the public runtime generator reuse the same authoring
    path as the official release without constructing unrelated plans.
    """

    if unknown := set(tasks) - set(TASK_NAMES):
        raise ValueError(f"unsupported generation tasks: {sorted(unknown)}")
    if unknown := set(splits) - set(QA_SPLITS):
        raise ValueError(f"unsupported generation splits: {sorted(unknown)}")
    requested_tasks = set(tasks)
    requested_splits = set(splits)

    selected = catalog.select(
        Selection(
            tasks=("grounding",),
            dataset_ids=dataset_ids,
            seed=config.seed,
        )
    )
    grounding = {
        str(record["dataset_id"]): record for record in selected.grounding_tasks
    }
    plans: list[EvaluationPlan] = []
    task_counts: dict[str, dict[str, int]] = {
        "cell_grounding": defaultdict(int),
        "table_question_answering": defaultdict(int),
    }
    dataset_sets: dict[str, dict[str, set[str]]] = {
        "cell_grounding": defaultdict(set),
        "table_question_answering": defaultdict(set),
    }
    skipped: dict[str, int] = defaultdict(int)

    for dataset in sorted(selected.datasets, key=lambda item: item.dataset_id):
        grounding_contract = grounding.get(dataset.dataset_id)
        if grounding_contract is None:
            skipped["missing_grounding_contract"] += 1
            continue
        rows = source.rows(dataset)
        columns = _eligible_columns(dataset, grounding_contract)
        if len(columns) < config.min_qa_context_columns:
            skipped["too_few_eligible_columns"] += 1
            continue
        for split, row_ids in _evaluation_pools(dataset, len(rows), config.seed).items():
            if requested_splits and split not in requested_splits:
                continue
            cell_limit = _limit_for_split(
                split,
                config.cell_items_per_dataset,
                config.cell_transfer_items_per_dataset,
            )
            qa_limit = _limit_for_split(
                split,
                config.qa_items_per_dataset,
                config.qa_transfer_items_per_dataset,
            )
            if (
                "cell_grounding" in requested_tasks
                and split in CELL_SPLITS
                and len(columns) >= config.min_cell_context_columns
            ):
                generated = _cell_plans(
                    catalog.reference_id,
                    dataset,
                    rows,
                    columns,
                    row_ids,
                    split,
                    cell_limit,
                    config,
                )
                plans.extend(generated)
                task_counts["cell_grounding"][split] += len(generated)
                if generated:
                    dataset_sets["cell_grounding"][split].add(dataset.dataset_id)
                if len(generated) < cell_limit:
                    skipped[f"cell_shortfall:{split}"] += cell_limit - len(generated)
            elif "cell_grounding" in requested_tasks and split in CELL_SPLITS:
                skipped[f"cell_too_narrow:{split}"] += 1

            if "table_question_answering" in requested_tasks and split in QA_SPLITS:
                generated = _qa_plans(
                    catalog.reference_id,
                    dataset,
                    rows,
                    columns,
                    row_ids,
                    split,
                    qa_limit,
                    config,
                )
                plans.extend(generated)
                task_counts["table_question_answering"][split] += len(generated)
                if generated:
                    dataset_sets["table_question_answering"][split].add(
                        dataset.dataset_id
                    )
                if len(generated) < qa_limit:
                    skipped[f"qa_shortfall:{split}"] += qa_limit - len(generated)

    ordered = tuple(sorted(plans, key=lambda plan: plan.item_id))
    if len({plan.item_id for plan in ordered}) != len(ordered):
        raise ValueError("task authoring produced duplicate item IDs")
    normalized_counts = {
        task: {split: int(counts.get(split, 0)) for split in splits}
        for task, counts, splits in (
            ("cell_grounding", task_counts["cell_grounding"], CELL_SPLITS),
            (
                "table_question_answering",
                task_counts["table_question_answering"],
                QA_SPLITS,
            ),
        )
    }
    dataset_counts = {
        task: {split: len(dataset_sets[task].get(split, set())) for split in splits}
        for task, splits in (
            ("cell_grounding", CELL_SPLITS),
            ("table_question_answering", QA_SPLITS),
        )
    }
    return TaskGenerationReport(
        plans=ordered,
        task_counts=normalized_counts,
        dataset_counts=dataset_counts,
        skipped=dict(sorted(skipped.items())),
    )


def _evaluation_pools(
    dataset: DatasetSpec,
    row_count: int,
    seed: int,
) -> dict[EvaluationSplit, tuple[str, ...]]:
    row_ids = tuple(
        sorted(
            (str(index) for index in range(row_count)),
            key=lambda row_id: stable_order(
                f"{dataset.dataset_id}:row:{row_id}", seed
            ),
        )
    )
    if dataset.dataset_split == "validation":
        return {"validation": row_ids}
    if dataset.dataset_split == "test":
        return {"dataset_test": row_ids}
    train_end = int(row_count * 0.70)
    episode_end = train_end + int(row_count * 0.10)
    template_end = episode_end + int(row_count * 0.10)
    return {
        "train": row_ids[:train_end],
        "episode_test": row_ids[train_end:episode_end],
        "template_test": row_ids[episode_end:template_end],
        "composition_test": row_ids[template_end:],
    }


def _eligible_columns(
    dataset: DatasetSpec,
    task: dict[str, Any],
) -> tuple[str, ...]:
    declared = {str(value) for value in task.get("eligible_columns", ())}
    excluded = {
        dataset.target_column,
        *dataset.excluded_feature_columns,
        *(str(value) for value in task.get("excluded_identifier_columns", ())),
    }
    return tuple(
        column
        for column in dataset.feature_columns
        if column in declared and column not in excluded
    )


def _cell_plans(
    reference_id: str,
    dataset: DatasetSpec,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    row_ids: tuple[str, ...],
    split: EvaluationSplit,
    limit: int,
    config: TaskGenerationConfig,
) -> tuple[EvaluationPlan, ...]:
    by_column = {
        column: tuple(
            row_id
            for row_id in row_ids
            if stable_value_key(rows[int(row_id)][column]) != "<missing>"
        )
        for column in columns
    }
    column_order = sorted(
        columns,
        key=lambda column: stable_order(
            f"{dataset.dataset_id}:{split}:cell:{column}", config.seed
        ),
    )
    selected: list[EvaluationPlan] = []
    depth = 0
    while len(selected) < limit and any(depth < len(by_column[col]) for col in column_order):
        for column in column_order:
            if depth >= len(by_column[column]) or len(selected) >= limit:
                continue
            row_id = by_column[column][depth]
            context = _context_columns(
                dataset,
                columns,
                (column,),
                config.max_cell_context_columns,
                f"{split}:cell:{row_id}:{column}",
                config.seed,
            )
            if len(context) < config.min_cell_context_columns:
                continue
            value = rows[int(row_id)][column]
            selected.append(
                _plan(
                    reference_id=reference_id,
                    dataset=dataset,
                    split=split,
                    source=TableSlice(dataset.dataset_id, (row_id,), context),
                    operation=OperationSpec("cell_lookup", {"column": column}),
                    scoring=_scoring_for_value(value),
                    identity=("cell", split, dataset.dataset_id, row_id, column, context),
                    seed=config.seed,
                )
            )
        depth += 1
    return tuple(selected)


def _qa_plans(
    reference_id: str,
    dataset: DatasetSpec,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    row_ids: tuple[str, ...],
    split: EvaluationSplit,
    limit: int,
    config: TaskGenerationConfig,
) -> tuple[EvaluationPlan, ...]:
    if len(row_ids) < min(config.qa_row_sizes):
        return ()
    selected: list[EvaluationPlan] = []
    seen: set[str] = set()
    attempts = max(100, limit * 80)
    for attempt in range(attempts):
        if len(selected) >= limit:
            break
        available_sizes = tuple(size for size in config.qa_row_sizes if size <= len(row_ids))
        if not available_sizes:
            break
        row_size = available_sizes[attempt % len(available_sizes)]
        candidate_rows = tuple(
            sorted(
                row_ids,
                key=lambda row_id: stable_order(
                    f"{dataset.dataset_id}:{split}:qa:{attempt}:{row_id}", config.seed
                ),
            )[:row_size]
        )
        operation = (
            _filtered_argmax_operation(rows, columns, candidate_rows, attempt, config.seed)
            if split == "composition_test"
            else _base_qa_operation(rows, columns, candidate_rows, attempt, config.seed)
        )
        if operation is None:
            continue
        spec, scoring, required = operation
        context = _context_columns(
            dataset,
            columns,
            required,
            config.max_qa_context_columns,
            f"{split}:qa:{attempt}:{spec.name}",
            config.seed,
        )
        if len(context) < config.min_qa_context_columns:
            continue
        identity = canonical_json(
            [split, dataset.dataset_id, candidate_rows, context, spec.name, dict(spec.arguments)]
        )
        if identity in seen:
            continue
        seen.add(identity)
        selected.append(
            _plan(
                reference_id=reference_id,
                dataset=dataset,
                split=split,
                source=TableSlice(dataset.dataset_id, candidate_rows, context),
                operation=spec,
                scoring=scoring,
                identity=("qa", identity),
                seed=config.seed,
            )
        )
    return tuple(selected)


def _base_qa_operation(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    row_ids: tuple[str, ...],
    attempt: int,
    seed: int,
) -> tuple[OperationSpec, ScoringSpec, tuple[str, ...]] | None:
    numeric = _numeric_columns(rows, columns, row_ids)
    if not numeric:
        return None
    numeric = tuple(
        sorted(numeric, key=lambda column: stable_order(f"qa:{attempt}:{column}", seed))
    )
    if attempt % 2 == 0:
        aggregation = ("count", "sum", "mean", "min", "max")[(attempt // 2) % 5]
        column = numeric[attempt % len(numeric)]
        values = [rows[int(row_id)][column] for row_id in row_ids]
        answer = _aggregate_for_scoring(values, aggregation)
        return (
            OperationSpec(
                "aggregate", {"column": column, "aggregation": aggregation}
            ),
            _scoring_for_value(answer, force_float=aggregation == "mean"),
            (column,),
        )
    maximize = numeric[attempt % len(numeric)]
    winner = _unique_argmax_row(rows, row_ids, maximize)
    if winner is None:
        return None
    return_candidates = tuple(column for column in columns if column != maximize)
    return_candidates = tuple(
        sorted(
            return_candidates,
            key=lambda column: stable_order(f"return:{attempt}:{column}", seed),
        )
    )
    returned = next(
        (
            column
            for column in return_candidates
            if stable_value_key(rows[int(winner)][column]) != "<missing>"
        ),
        None,
    )
    if returned is None:
        return None
    return (
        OperationSpec(
            "argmax_lookup",
            {"maximize_column": maximize, "return_column": returned},
        ),
        _scoring_for_value(rows[int(winner)][returned]),
        (maximize, returned),
    )


def _filtered_argmax_operation(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    row_ids: tuple[str, ...],
    attempt: int,
    seed: int,
) -> tuple[OperationSpec, ScoringSpec, tuple[str, ...]] | None:
    numeric = _numeric_columns(rows, columns, row_ids)
    if not numeric:
        return None
    filter_columns = sorted(
        columns,
        key=lambda column: stable_order(f"filter:{attempt}:{column}", seed),
    )
    maximize_columns = sorted(
        numeric,
        key=lambda column: stable_order(f"maximize:{attempt}:{column}", seed),
    )
    for filter_column in filter_columns:
        groups: dict[str, list[str]] = defaultdict(list)
        for row_id in row_ids:
            value_key = stable_value_key(rows[int(row_id)][filter_column])
            if value_key != "<missing>":
                groups[value_key].append(row_id)
        eligible_groups = [group for group in groups.values() if len(group) >= 2]
        eligible_groups.sort(
            key=lambda group: stable_order(
                f"group:{attempt}:{filter_column}:{group[0]}", seed
            )
        )
        for group in eligible_groups:
            filter_row_id = group[0]
            for maximize in maximize_columns:
                if maximize == filter_column:
                    continue
                winner = _unique_argmax_row(rows, tuple(group), maximize)
                if winner is None:
                    continue
                return_candidates = sorted(
                    (
                        column
                        for column in columns
                        if column not in {filter_column, maximize}
                    ),
                    key=lambda column: stable_order(
                        f"filtered-return:{attempt}:{column}", seed
                    ),
                )
                returned = next(
                    (
                        column
                        for column in return_candidates
                        if stable_value_key(rows[int(winner)][column]) != "<missing>"
                    ),
                    None,
                )
                if returned is None:
                    continue
                return (
                    OperationSpec(
                        "filtered_argmax_lookup",
                        {
                            "filter_column": filter_column,
                            "filter_value_row_id": filter_row_id,
                            "maximize_column": maximize,
                            "return_column": returned,
                        },
                    ),
                    _scoring_for_value(rows[int(winner)][returned]),
                    (filter_column, maximize, returned),
                )
    return None


def _numeric_columns(
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
    row_ids: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        column
        for column in columns
        if all(_finite_number(rows[int(row_id)][column]) for row_id in row_ids)
    )


def _finite_number(value: Any) -> bool:
    normalized = normalize_value(value)
    if not isinstance(normalized, int | float) or isinstance(normalized, bool):
        return False
    try:
        return math.isfinite(float(normalized))
    except OverflowError:
        return False


def _unique_argmax_row(
    rows: list[dict[str, Any]],
    row_ids: tuple[str, ...],
    column: str,
) -> str | None:
    values = [(row_id, normalize_value(rows[int(row_id)][column])) for row_id in row_ids]
    maximum = max(value for _, value in values)
    winners = [row_id for row_id, value in values if value == maximum]
    return winners[0] if len(winners) == 1 else None


def _aggregate_for_scoring(values: list[Any], aggregation: str) -> int | float:
    numbers = [normalize_value(value) for value in values]
    if aggregation == "count":
        return len(numbers)
    if aggregation == "sum":
        return sum(numbers)
    if aggregation == "mean":
        return sum(numbers) / len(numbers)
    if aggregation == "min":
        return min(numbers)
    if aggregation == "max":
        return max(numbers)
    raise AssertionError("unsupported authoring aggregation")


def _context_columns(
    dataset: DatasetSpec,
    eligible: tuple[str, ...],
    required: tuple[str, ...],
    maximum: int,
    identity: str,
    seed: int,
) -> tuple[str, ...]:
    required_set = set(required)
    if not required_set <= set(eligible):
        raise ValueError("operation requires a column outside the eligible feature set")
    if len(required_set) > maximum:
        return ()
    distractors = sorted(
        (column for column in eligible if column not in required_set),
        key=lambda column: stable_order(
            f"{dataset.dataset_id}:{identity}:context:{column}", seed
        ),
    )
    selected = required_set | set(distractors[: maximum - len(required_set)])
    return tuple(column for column in dataset.feature_columns if column in selected)


def _scoring_for_value(value: Any, *, force_float: bool = False) -> ScoringSpec:
    normalized = normalize_value(value)
    if force_float:
        return ScoringSpec("float", absolute_tolerance=1e-6, relative_tolerance=1e-6)
    if isinstance(normalized, bool):
        return ScoringSpec("boolean")
    if isinstance(normalized, int):
        return ScoringSpec("integer")
    if isinstance(normalized, float):
        return ScoringSpec("float", absolute_tolerance=1e-6, relative_tolerance=1e-6)
    if isinstance(normalized, dict | list):
        return ScoringSpec("json")
    return ScoringSpec("string")


def _plan(
    *,
    reference_id: str,
    dataset: DatasetSpec,
    split: EvaluationSplit,
    source: TableSlice,
    operation: OperationSpec,
    scoring: ScoringSpec,
    identity: tuple[Any, ...],
    seed: int,
) -> EvaluationPlan:
    item_id = stable_id(operation.name, canonical_json((reference_id, *identity)))
    return EvaluationPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        benchmark_version=BENCHMARK_VERSION,
        reference_id=reference_id,
        item_id=item_id,
        task="grounding" if operation.name == "cell_lookup" else "qa",
        evaluation_split=split,
        dataset_split=dataset.dataset_split,
        dedup_cluster_id=dataset.dedup_cluster_id,
        source_id=dataset.source_id,
        source=source,
        operation=operation,
        rendering=RenderingSpec(
            template_family=operation.name,
            template_split="test" if split == "template_test" else "train",
            render_seed=seed,
            view="markdown",
        ),
        scoring=scoring,
        generator_version=GENERATOR_VERSION,
        executor_version=EXECUTOR_VERSION,
    )


def _limit_for_split(split: EvaluationSplit, main: int, transfer: int) -> int:
    return main if split == "train" else transfer
