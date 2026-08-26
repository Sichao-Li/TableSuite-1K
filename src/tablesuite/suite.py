"""One user-facing entry point for exploring and running TableSuite."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tablesuite.benchmark import Benchmark, BenchmarkSubset
from tablesuite.catalog import Catalog
from tablesuite.evaluation.contracts import EvaluationSplit
from tablesuite.generation import GeneratedTaskDataset, generate_task
from tablesuite.registry import (
    PredictionProtocol,
    PublicTaskName,
    TaskDescriptor,
    describe_task,
    list_tasks,
)
from tablesuite.source import ParquetSource
from tablesuite.tasks import TaskDataset, TaskName, load_task
from tablesuite.types import Selection, TaskFamily


@dataclass(frozen=True)
class TableSuite:
    """A benchmark reference joined to explicitly materialized source tables.

    This facade only routes to the stable task, generation, and prediction
    APIs. Lower-level functions remain available for scripts that need them.
    """

    reference: str | Path
    source: Path
    revision: str | None = None

    @classmethod
    def open(
        cls,
        reference: str | Path,
        *,
        source: str | Path,
        revision: str | None = None,
    ) -> TableSuite:
        """Open a local or Hugging Face benchmark reference."""

        return cls(reference=reference, source=Path(source), revision=revision)

    def tasks(self) -> tuple[TaskDescriptor, ...]:
        """List the three user-facing task families."""

        return list_tasks()

    def describe(self, name: str) -> TaskDescriptor:
        """Describe one task family and its supported modes."""

        return describe_task(name)

    def catalog_summary(self) -> dict[str, object]:
        """Summarize datasets and task eligibility without reading table values."""

        return self._catalog().summary()

    def official(
        self,
        name: PublicTaskName,
        *,
        split: EvaluationSplit,
        dataset_ids: Iterable[str] = (),
    ) -> TaskDataset:
        """Load one frozen official table-grounding or table-QA split."""

        descriptor = describe_task(name)
        if descriptor.name == "table_prediction":
            raise ValueError(
                "official prediction protocols use TableSuite.prediction(); "
                "choose one protocol explicitly"
            )
        return load_task(
            self.reference,
            cast(TaskName, descriptor.name),
            split=split,
            source=self.source,
            revision=self.revision,
            dataset_ids=dataset_ids,
        )

    def generate(
        self,
        name: PublicTaskName,
        *,
        split: EvaluationSplit = "train",
        dataset_ids: Iterable[str] = (),
        task_families: Iterable[TaskFamily] = (),
        max_datasets: int | None = None,
        items_per_dataset: int | None = None,
        max_items: int | None = None,
        seed: int = 0,
    ) -> GeneratedTaskDataset:
        """Generate deterministic non-official grounding or QA plans."""

        descriptor = describe_task(name)
        if not descriptor.generatable:
            raise ValueError(f"task {name!r} cannot be generated")
        return generate_task(
            self.reference,
            cast(TaskName, descriptor.name),
            source=self.source,
            split=split,
            revision=self.revision,
            dataset_ids=dataset_ids,
            task_families=task_families,
            max_datasets=max_datasets,
            items_per_dataset=items_per_dataset,
            max_items=max_items,
            seed=seed,
        )

    def prediction(
        self,
        protocol: PredictionProtocol | str,
        *,
        dataset_ids: Iterable[str] = (),
        dataset_splits: Iterable[str] = (),
        task_families: Iterable[TaskFamily] = (),
        shots: Iterable[int] = (),
        max_datasets: int | None = None,
        max_episodes_per_dataset_per_shot: int | None = None,
        seed: int = 0,
    ) -> BenchmarkSubset:
        """Select one frozen inference-only table-prediction protocol."""

        descriptor = describe_task("table_prediction")
        if protocol not in descriptor.protocols:
            choices = ", ".join(descriptor.protocols)
            raise ValueError(
                f"unknown prediction protocol {protocol!r}; choose one of: {choices}"
            )
        selection = Selection(
            tasks=(cast(PredictionProtocol, protocol),),
            dataset_ids=tuple(str(value) for value in dataset_ids),
            dataset_splits=tuple(str(value) for value in dataset_splits),
            task_families=tuple(task_families),
            shots=tuple(int(value) for value in shots),
            max_datasets=max_datasets,
            max_episodes_per_dataset_per_shot=max_episodes_per_dataset_per_shot,
            seed=seed,
        )
        return Benchmark(self._catalog(), ParquetSource(self.source)).select(selection)

    def _catalog(self) -> Catalog:
        path = Path(self.reference)
        if path.is_dir():
            return Catalog.from_path(path)
        return Catalog.from_huggingface(str(self.reference), revision=self.revision)
