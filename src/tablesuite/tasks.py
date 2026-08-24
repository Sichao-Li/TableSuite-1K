"""Hugging Face-native task loading and evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from tablesuite.benchmark import Benchmark
from tablesuite.catalog import Catalog
from tablesuite.evaluation.contracts import (
    DatasetPartition,
    EvaluationPlan,
    EvaluationSplit,
    EvaluationTask,
    OperationName,
    PlanRegistry,
)
from tablesuite.evaluation.executor import PlanExecutor
from tablesuite.source import ParquetSource
from tablesuite.types import TableSlice

TaskName = Literal[
    "cell_grounding",
    "table_question_answering",
]

_TASK_TYPES: dict[TaskName, EvaluationTask] = {
    "cell_grounding": "grounding",
    "table_question_answering": "qa",
}


@dataclass(frozen=True)
class TaskExample:
    """One input-only example ready to pass to a model."""

    id: str
    task: TaskName
    split: EvaluationSplit
    prompt: str
    question: str
    table: str
    dataset_id: str
    dataset_split: DatasetPartition
    dedup_cluster_id: str
    operation: OperationName
    template_id: str
    source: TableSlice


@dataclass(frozen=True)
class TaskScore:
    """Structured score for one submitted response."""

    id: str
    correct: bool
    exact_match: bool
    numeric_within_tolerance: bool | None
    parsed_answer: Any = None
    parse_error: str | None = None


@dataclass(frozen=True)
class GroupAccuracy:
    """Accuracy and support for one named evaluation group."""

    name: str
    examples: int
    accuracy: float


@dataclass(frozen=True)
class TaskReport:
    """Aggregate report for one task configuration and split."""

    task: TaskName
    split: EvaluationSplit
    total_examples: int
    submitted_examples: int
    correct_examples: int
    accuracy: float
    parse_failures: int
    parse_failure_rate: float
    dataset_macro_accuracy: float
    cluster_macro_accuracy: float
    by_operation: tuple[GroupAccuracy, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        payload = asdict(self)
        payload["by_operation"] = [asdict(item) for item in self.by_operation]
        return payload


class TaskDataset:
    """An indexable official task split with built-in deterministic scoring."""

    def __init__(
        self,
        *,
        name: TaskName,
        split: EvaluationSplit,
        benchmark: Benchmark,
        plans: PlanRegistry,
    ) -> None:
        internal_task = _TASK_TYPES[name]
        selected = plans.select(task=internal_task, split=split)
        if not selected.plans:
            raise ValueError(f"no {name!r} examples found for split {split!r}")
        self.name = name
        self.split = split
        self._plans = selected.plans
        self._by_id = {plan.item_id: plan for plan in self._plans}
        self._executor = PlanExecutor(benchmark, selected)

    def __len__(self) -> int:
        return len(self._plans)

    def __iter__(self) -> Iterator[TaskExample]:
        for plan in self._plans:
            yield self._example(plan)

    def __getitem__(self, key: int | str) -> TaskExample:
        """Materialize an example by stable ID or integer position."""

        if isinstance(key, int):
            plan = self._plans[key]
        else:
            try:
                plan = self._by_id[key]
            except KeyError as error:
                raise KeyError(f"unknown task example: {key}") from error
        return self._example(plan)

    @property
    def ids(self) -> tuple[str, ...]:
        """Return stable example IDs in deterministic order."""

        return tuple(plan.item_id for plan in self._plans)

    def score(self, example_id: str, response: Any) -> TaskScore:
        """Score one response without exposing its programmatic gold."""

        if example_id not in self._by_id:
            raise KeyError(f"unknown task example: {example_id}")
        result = self._executor.score(example_id, response)
        return TaskScore(
            id=result.item_id,
            correct=result.correct,
            exact_match=result.exact_match,
            numeric_within_tolerance=result.numeric_within_tolerance,
            parsed_answer=result.parsed_answer,
            parse_error=result.parse_error,
        )

    def evaluate(
        self,
        predictions: Mapping[str, Any],
        *,
        allow_partial: bool = False,
    ) -> TaskReport:
        """Evaluate responses keyed by example ID and return aggregate metrics."""

        unknown = sorted(set(predictions) - set(self._by_id))
        if unknown:
            raise KeyError(f"predictions contain unknown example IDs: {unknown[:3]}")
        missing = sorted(set(self._by_id) - set(predictions))
        if missing and not allow_partial:
            raise ValueError(
                f"predictions are missing {len(missing)} examples; "
                "pass allow_partial=True for a diagnostic subset"
            )
        selected = [plan for plan in self._plans if plan.item_id in predictions]
        if not selected:
            raise ValueError("predictions cannot be empty")
        scored = [
            (plan, self.score(plan.item_id, predictions[plan.item_id]))
            for plan in selected
        ]
        correct = sum(score.correct for _, score in scored)
        parse_failures = sum(score.parse_error is not None for _, score in scored)
        count = len(scored)
        return TaskReport(
            task=self.name,
            split=self.split,
            total_examples=len(self),
            submitted_examples=count,
            correct_examples=correct,
            accuracy=correct / count,
            parse_failures=parse_failures,
            parse_failure_rate=parse_failures / count,
            dataset_macro_accuracy=_macro_accuracy(
                scored, lambda plan: plan.source.dataset_id
            ),
            cluster_macro_accuracy=_macro_accuracy(
                scored, lambda plan: plan.dedup_cluster_id
            ),
            by_operation=_group_accuracy(scored),
        )

    def summary(self) -> dict[str, Any]:
        """Return compact split metadata without reading source table values."""

        return {
            "task": self.name,
            "split": self.split,
            "examples": len(self),
            "datasets": len({plan.source.dataset_id for plan in self._plans}),
            "dedup_clusters": len({plan.dedup_cluster_id for plan in self._plans}),
            "operations": dict(
                sorted(
                    _count(plan.operation.name for plan in self._plans).items()
                )
            ),
        }

    def _example(self, plan: EvaluationPlan) -> TaskExample:
        item = self._executor.materialize(plan)
        return TaskExample(
            id=plan.item_id,
            task=self.name,
            split=self.split,
            prompt=item.request.input_text,
            question=item.request.question,
            table=item.request.table_text,
            dataset_id=plan.source.dataset_id,
            dataset_split=plan.dataset_split,
            dedup_cluster_id=plan.dedup_cluster_id,
            operation=plan.operation.name,
            template_id=item.request.template_id,
            source=plan.source,
        )


def load_task(
    reference: str | Path,
    name: TaskName,
    *,
    split: EvaluationSplit,
    source: str | Path,
    revision: str | None = None,
) -> TaskDataset:
    """Load one official HF task configuration and evaluation split.

    ``reference`` may be a Hugging Face dataset repository ID or a downloaded
    reference directory. Local task files are discovered under
    ``tasks/<name>/<split>.{jsonl,parquet}``.
    """

    if name not in _TASK_TYPES:
        raise ValueError(f"unsupported task name: {name!r}")
    reference_path = Path(reference)
    if reference_path.is_dir():
        catalog = Catalog.from_path(reference_path)
        registry = _resolve_local_registry(reference_path, name, split)
    else:
        registry = PlanRegistry.from_huggingface(
            str(reference),
            name=name,
            split=split,
            revision=revision,
        )
        if not registry.plans:
            raise ValueError(f"HF task {name!r}/{split!r} contains no examples")
        catalog = Catalog.from_huggingface(
            str(reference),
            revision=revision,
            include_task_manifests=False,
            reference_id=registry.plans[0].reference_id,
        )
    return TaskDataset(
        name=name,
        split=split,
        benchmark=Benchmark(catalog, ParquetSource(source)),
        plans=registry,
    )


def _resolve_local_registry(
    reference: Path,
    name: TaskName,
    split: EvaluationSplit,
) -> PlanRegistry:
    root = reference / "tasks" / name
    candidates = (
        root / f"{split}.jsonl",
        root / f"{split}.parquet",
        root / split,
    )
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return PlanRegistry.load(candidate)
    raise FileNotFoundError(
        f"no local task split for {name!r}/{split!r} under {root}"
    )


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return counts


def _macro_accuracy(
    scored: list[tuple[EvaluationPlan, TaskScore]],
    key: Callable[[EvaluationPlan], str],
) -> float:
    groups: dict[str, list[bool]] = defaultdict(list)
    for plan, score in scored:
        groups[str(key(plan))].append(score.correct)
    return sum(sum(values) / len(values) for values in groups.values()) / len(groups)


def _group_accuracy(
    scored: list[tuple[EvaluationPlan, TaskScore]],
) -> tuple[GroupAccuracy, ...]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for plan, score in scored:
        groups[plan.operation.name].append(score.correct)
    return tuple(
        GroupAccuracy(name=name, examples=len(values), accuracy=sum(values) / len(values))
        for name, values in sorted(groups.items())
    )
