"""Public deterministic generation of additional TableSuite task plans."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from tablesuite._util import canonical_json, stable_id, stable_order
from tablesuite.authoring import TaskGenerationConfig, generate_task_plans
from tablesuite.benchmark import Benchmark
from tablesuite.catalog import Catalog
from tablesuite.evaluation.contracts import (
    BENCHMARK_VERSION,
    EvaluationPlan,
    EvaluationSplit,
    PlanRegistry,
)
from tablesuite.evaluation.rendering import GENERATOR_VERSION
from tablesuite.source import ParquetSource
from tablesuite.tasks import TaskDataset, TaskName
from tablesuite.types import Selection, TaskFamily

GENERATION_SCHEMA_VERSION = "1.0"

_TASK_DEFAULTS: dict[TaskName, int] = {
    "table_grounding": 30,
    "table_question_answering": 12,
}
_INTERNAL_TASKS = {
    "table_grounding": "grounding",
    "table_question_answering": "qa",
}
_SPLITS = {
    "train",
    "validation",
    "episode_test",
    "dataset_test",
    "template_test",
    "composition_test",
}


@dataclass(frozen=True)
class GenerationManifest:
    """Reproducibility record for one generated task bundle."""

    schema_version: str
    origin: Literal["generated"]
    benchmark_version: str
    reference: str
    revision: str | None
    reference_id: str
    task: TaskName
    split: EvaluationSplit
    seed: int
    items_per_dataset: int
    max_items: int | None
    dataset_ids: tuple[str, ...]
    datasets_with_items: int
    generated_items: int
    eligibility_skips: dict[str, int]
    generator_version: str
    config_fingerprint: str
    plan_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest."""

        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write the manifest as deterministic JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> GenerationManifest:
        """Load a generated-task manifest."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["dataset_ids"] = tuple(payload["dataset_ids"])
        manifest = cls(**payload)
        if manifest.schema_version != GENERATION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported generation schema: {manifest.schema_version!r}"
            )
        if manifest.origin != "generated":
            raise ValueError(f"unsupported task origin: {manifest.origin!r}")
        if manifest.task not in _TASK_DEFAULTS:
            raise ValueError(f"unsupported generated task: {manifest.task!r}")
        if manifest.split not in _SPLITS:
            raise ValueError(f"unsupported generated split: {manifest.split!r}")
        if manifest.benchmark_version != BENCHMARK_VERSION:
            raise ValueError(
                f"unsupported benchmark version: {manifest.benchmark_version!r}"
            )
        if manifest.generator_version != GENERATOR_VERSION:
            raise ValueError(
                f"unsupported generator version: {manifest.generator_version!r}"
            )
        return manifest


class GeneratedTaskDataset(TaskDataset):
    """A generated task split with the standard TableSuite scoring interface."""

    def __init__(
        self,
        *,
        name: TaskName,
        split: EvaluationSplit,
        benchmark: Benchmark,
        plans: PlanRegistry,
        manifest: GenerationManifest,
    ) -> None:
        super().__init__(name=name, split=split, benchmark=benchmark, plans=plans)
        self.manifest = manifest

    def save(self, output: str | Path) -> Path:
        """Save a value-free plan bundle for deterministic replay."""

        destination = Path(output)
        if destination.exists() and (
            not destination.is_dir() or any(destination.iterdir())
        ):
            raise FileExistsError(
                f"generated task output is not empty: {destination}"
            )
        destination.mkdir(parents=True, exist_ok=True)
        PlanRegistry(self._plans).save(destination / "plans.jsonl")
        self.manifest.save(destination / "generation.json")
        return destination


def generate_task(
    reference: str | Path,
    name: TaskName,
    *,
    source: str | Path,
    split: EvaluationSplit = "train",
    revision: str | None = None,
    dataset_ids: Iterable[str] = (),
    task_families: Iterable[TaskFamily] = (),
    max_datasets: int | None = None,
    items_per_dataset: int | None = None,
    max_items: int | None = None,
    seed: int = 0,
) -> GeneratedTaskDataset:
    """Generate a deterministic task split from locally materialized sources.

    The task semantics remain fixed to the official TableSuite recipe. Users
    control source selection and scale; wording and gold are still produced at
    access time from the generated value-free plans.
    """

    if name not in _TASK_DEFAULTS:
        raise ValueError(f"unsupported generated task: {name!r}")
    if name == "table_grounding" and split == "composition_test":
        raise ValueError("composition_test is only defined for table question answering")
    resolved_items = _TASK_DEFAULTS[name] if items_per_dataset is None else items_per_dataset
    if resolved_items <= 0:
        raise ValueError("items_per_dataset must be positive")
    if max_items is not None and max_items <= 0:
        raise ValueError("max_items must be positive")

    catalog = _load_catalog(reference, revision)
    selected = catalog.select(
        Selection(
            tasks=(),
            dataset_ids=tuple(str(value) for value in dataset_ids),
            dataset_splits=(_dataset_partition(split),),
            task_families=tuple(task_families),
            max_datasets=max_datasets,
            seed=seed,
        )
    )
    selected_ids = tuple(dataset.dataset_id for dataset in selected.datasets)
    if not selected_ids:
        raise ValueError(f"no datasets are eligible for generated split {split!r}")

    source_store = ParquetSource(source)
    report = generate_task_plans(
        catalog,
        source_store,
        TaskGenerationConfig(
            seed=seed,
            grounding_items_per_dataset=resolved_items,
            grounding_transfer_items_per_dataset=resolved_items,
            qa_items_per_dataset=resolved_items,
            qa_transfer_items_per_dataset=resolved_items,
        ),
        dataset_ids=selected_ids,
        tasks=(name,),
        splits=(split,),
    )
    internal_task = _INTERNAL_TASKS[name]
    plans = tuple(
        plan
        for plan in report.plans
        if plan.task == internal_task and plan.evaluation_split == split
    )
    if max_items is not None:
        plans = _balanced_limit(plans, max_items, seed)
    if not plans:
        raise ValueError(
            f"no {name!r} plans could be generated for split {split!r}; "
            "check source availability and task eligibility"
        )

    registry = PlanRegistry(plans)
    manifest = _manifest(
        reference=reference,
        revision=revision,
        catalog=catalog,
        name=name,
        split=split,
        seed=seed,
        items_per_dataset=resolved_items,
        max_items=max_items,
        dataset_ids=selected_ids,
        eligibility_skips=report.skipped,
        plans=registry.plans,
    )
    return GeneratedTaskDataset(
        name=name,
        split=split,
        benchmark=Benchmark(catalog, source_store),
        plans=registry,
        manifest=manifest,
    )


def load_generated_task(
    bundle: str | Path,
    *,
    source: str | Path,
    reference: str | Path | None = None,
    revision: str | None = None,
) -> GeneratedTaskDataset:
    """Load a value-free bundle previously written by ``GeneratedTaskDataset``."""

    root = Path(bundle)
    manifest = GenerationManifest.load(root / "generation.json")
    if manifest.config_fingerprint != _config_fingerprint(
        reference_id=manifest.reference_id,
        name=manifest.task,
        split=manifest.split,
        seed=manifest.seed,
        items_per_dataset=manifest.items_per_dataset,
        max_items=manifest.max_items,
        dataset_ids=manifest.dataset_ids,
    ):
        raise ValueError("generated task configuration fingerprint is invalid")
    resolved_reference = manifest.reference if reference is None else reference
    resolved_revision = manifest.revision if revision is None else revision
    catalog = _load_catalog(resolved_reference, resolved_revision)
    if catalog.reference_id != manifest.reference_id:
        raise ValueError(
            "generated task reference does not match the loaded catalog: "
            f"{manifest.reference_id!r} != {catalog.reference_id!r}"
        )
    registry = PlanRegistry.load(root / "plans.jsonl")
    if len(registry.plans) != manifest.generated_items:
        raise ValueError("generated task plan count does not match its manifest")
    if _plan_fingerprint(registry.plans) != manifest.plan_fingerprint:
        raise ValueError("generated task plan fingerprint does not match its manifest")
    expected_task = _INTERNAL_TASKS[manifest.task]
    if any(
        plan.task != expected_task
        or plan.evaluation_split != manifest.split
        or plan.reference_id != manifest.reference_id
        or plan.source.dataset_id not in manifest.dataset_ids
        for plan in registry.plans
    ):
        raise ValueError("generated task plans do not match their manifest")
    return GeneratedTaskDataset(
        name=manifest.task,
        split=manifest.split,
        benchmark=Benchmark(catalog, ParquetSource(source)),
        plans=registry,
        manifest=manifest,
    )


def _load_catalog(reference: str | Path, revision: str | None) -> Catalog:
    path = Path(reference)
    return (
        Catalog.from_path(path)
        if path.is_dir()
        else Catalog.from_huggingface(str(reference), revision=revision)
    )


def _dataset_partition(split: EvaluationSplit) -> str:
    if split == "validation":
        return "validation"
    if split == "dataset_test":
        return "test"
    return "train"


def _balanced_limit(
    plans: tuple[EvaluationPlan, ...],
    limit: int,
    seed: int,
) -> tuple[EvaluationPlan, ...]:
    if len(plans) <= limit:
        return plans
    grouped: dict[str, list[EvaluationPlan]] = defaultdict(list)
    for plan in plans:
        grouped[plan.source.dataset_id].append(plan)
    datasets = sorted(grouped, key=lambda value: stable_order(value, seed))
    for dataset_id in datasets:
        grouped[dataset_id].sort(key=lambda plan: stable_order(plan.item_id, seed))
    selected: list[EvaluationPlan] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for dataset_id in datasets:
            if depth >= len(grouped[dataset_id]):
                continue
            selected.append(grouped[dataset_id][depth])
            added = True
            if len(selected) == limit:
                break
        if not added:
            break
        depth += 1
    return tuple(selected)


def _manifest(
    *,
    reference: str | Path,
    revision: str | None,
    catalog: Catalog,
    name: TaskName,
    split: EvaluationSplit,
    seed: int,
    items_per_dataset: int,
    max_items: int | None,
    dataset_ids: tuple[str, ...],
    eligibility_skips: dict[str, int],
    plans: tuple[EvaluationPlan, ...],
) -> GenerationManifest:
    return GenerationManifest(
        schema_version=GENERATION_SCHEMA_VERSION,
        origin="generated",
        benchmark_version=BENCHMARK_VERSION,
        reference=str(reference),
        revision=revision,
        reference_id=catalog.reference_id,
        task=name,
        split=split,
        seed=seed,
        items_per_dataset=items_per_dataset,
        max_items=max_items,
        dataset_ids=dataset_ids,
        datasets_with_items=len({plan.source.dataset_id for plan in plans}),
        generated_items=len(plans),
        eligibility_skips=dict(sorted(eligibility_skips.items())),
        generator_version=GENERATOR_VERSION,
        config_fingerprint=_config_fingerprint(
            reference_id=catalog.reference_id,
            name=name,
            split=split,
            seed=seed,
            items_per_dataset=items_per_dataset,
            max_items=max_items,
            dataset_ids=dataset_ids,
        ),
        plan_fingerprint=_plan_fingerprint(plans),
    )


def _config_fingerprint(
    *,
    reference_id: str,
    name: TaskName,
    split: EvaluationSplit,
    seed: int,
    items_per_dataset: int,
    max_items: int | None,
    dataset_ids: tuple[str, ...],
) -> str:
    return stable_id(
        "config",
        canonical_json(
            {
                "reference_id": reference_id,
                "task": name,
                "split": split,
                "seed": seed,
                "items_per_dataset": items_per_dataset,
                "max_items": max_items,
                "dataset_ids": dataset_ids,
                "generator_version": GENERATOR_VERSION,
            }
        ),
    )


def _plan_fingerprint(plans: tuple[EvaluationPlan, ...]) -> str:
    return stable_id(
        "plans",
        canonical_json(
            [plan.to_record() for plan in sorted(plans, key=lambda item: item.item_id)]
        ),
    )
