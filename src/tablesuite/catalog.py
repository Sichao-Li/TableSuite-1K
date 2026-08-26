"""Load and filter the compact TableSuite-1K reference catalog."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tablesuite._util import stable_order
from tablesuite.types import DatasetSpec, Selection, SelectionManifest

CONFIGS = (
    "datasets",
    "table_prediction_tasks",
    "prediction_episodes",
    "grounding_tasks",
)
EPISODE_PROTOCOLS = {
    "zero_shot_icl",
    "few_shot_icl",
    "zero_label_serialized_table",
    "partially_labeled_serialized_table",
}


@dataclass(frozen=True)
class CatalogSelection:
    """Internal catalog records selected by one reproducible manifest."""

    manifest: SelectionManifest
    datasets: tuple[DatasetSpec, ...]
    episodes: tuple[dict[str, Any], ...]
    table_prediction_dataset_ids: frozenset[str]
    grounding_tasks: tuple[dict[str, Any], ...]


class Catalog:
    """The value-free TableSuite-1K dataset and task catalog."""

    def __init__(
        self,
        *,
        reference_id: str,
        dataset_records: list[dict[str, Any]],
        table_prediction_records: list[dict[str, Any]],
        episode_records: list[dict[str, Any]],
        grounding_records: list[dict[str, Any]],
    ) -> None:
        self.reference_id = reference_id
        self._datasets = tuple(DatasetSpec.from_record(row) for row in dataset_records)
        self._table_prediction_ids = {
            str(row["dataset_id"]) for row in table_prediction_records
        }
        self._episodes = tuple(episode_records)
        self._grounding = tuple(grounding_records)

    @classmethod
    def from_path(cls, root: str | Path) -> Catalog:
        """Load a downloaded Hugging Face reference package."""

        path = Path(root)
        if not path.is_dir():
            raise FileNotFoundError(path)
        summary_path = path / "reference_summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        records = {
            config: _read_local_config(path / config)
            for config in CONFIGS
        }
        reference_id = str(
            summary.get("reference_id")
            or f"tablesuite-1k:{summary.get('schema_version', 'unknown')}"
        )
        return cls(
            reference_id=reference_id,
            dataset_records=records["datasets"],
            table_prediction_records=records["table_prediction_tasks"],
            episode_records=records["prediction_episodes"],
            grounding_records=records["grounding_tasks"],
        )

    @classmethod
    def from_huggingface(
        cls,
        repository: str,
        *,
        revision: str | None = None,
        include_task_manifests: bool = True,
        reference_id: str | None = None,
    ) -> Catalog:
        """Load reference metadata from Hugging Face Datasets.

        Task execution only needs the ``datasets`` configuration. The legacy
        prediction/grounding iterators can request all four catalog manifests.
        """

        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError("install tablesuite[hf] to load a remote catalog") from error
        configs = CONFIGS if include_task_manifests else ("datasets",)
        records: dict[str, list[dict[str, Any]]] = {
            config: [] for config in CONFIGS
        }
        for config in configs:
            loaded = load_dataset(repository, config, revision=revision)
            records[config] = [dict(row) for split in loaded.values() for row in split]
        resolved_reference_id = reference_id or (
            repository if revision is None else f"{repository}@{revision}"
        )
        return cls(
            reference_id=resolved_reference_id,
            dataset_records=records["datasets"],
            table_prediction_records=records["table_prediction_tasks"],
            episode_records=records["prediction_episodes"],
            grounding_records=records["grounding_tasks"],
        )

    @property
    def datasets(self) -> tuple[DatasetSpec, ...]:
        """Return all catalogued dataset specifications."""

        return self._datasets

    def summary(self) -> dict[str, Any]:
        """Return compact reference counts for display and smoke tests."""

        families: dict[str, int] = defaultdict(int)
        splits: dict[str, int] = defaultdict(int)
        for dataset in self._datasets:
            families[dataset.task_family] += 1
            splits[dataset.dataset_split] += 1
        return {
            "reference_id": self.reference_id,
            "datasets": len(self._datasets),
            "table_prediction_tasks": len(self._table_prediction_ids),
            "prediction_episode_candidates": len(self._episodes),
            "grounding_tasks": len(self._grounding),
            "task_families": dict(sorted(families.items())),
            "dataset_splits": dict(sorted(splits.items())),
        }

    def select(self, selection: Selection) -> CatalogSelection:
        """Resolve dataset and candidate-episode identities deterministically."""

        selection.validate()
        requested_ids = set(selection.dataset_ids)
        known_ids = {dataset.dataset_id for dataset in self._datasets}
        if unknown := requested_ids - known_ids:
            raise KeyError(f"unknown dataset IDs: {sorted(unknown)}")
        datasets = [
            dataset
            for dataset in self._datasets
            if (not requested_ids or dataset.dataset_id in requested_ids)
            and (
                not selection.dataset_splits
                or dataset.dataset_split in selection.dataset_splits
            )
            and (
                not selection.task_families
                or dataset.task_family in selection.task_families
            )
        ]
        datasets.sort(key=lambda item: stable_order(item.dataset_id, selection.seed))
        if selection.max_datasets is not None:
            datasets = datasets[: selection.max_datasets]
        selected_ids = {dataset.dataset_id for dataset in datasets}

        episodes: list[dict[str, Any]] = []
        if set(selection.tasks) & EPISODE_PROTOCOLS:
            by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for episode in self._episodes:
                if str(episode["dataset_id"]) not in selected_ids:
                    continue
                normalized = dict(episode)
                normalized["shots"] = _episode_shots(episode)
                normalized.pop("k", None)
                if int(normalized["shots"]) <= 0:
                    continue
                by_dataset[str(episode["dataset_id"])].append(normalized)
            for dataset_id in sorted(by_dataset):
                items = sorted(
                    by_dataset[dataset_id],
                    key=lambda item: stable_order(str(item["episode_id"]), selection.seed),
                )
                requested = set(selection.shots)
                selected_episodes = (
                    items
                    if not requested
                    else [
                        item
                        for item in items
                        if int(item["shots"]) in requested
                    ]
                )
                episodes.extend(selected_episodes)

        grounding = (
            tuple(
                row
                for row in self._grounding
                if str(row["dataset_id"]) in selected_ids
            )
            if "grounding" in selection.tasks
            else ()
        )
        manifest = SelectionManifest(
            schema_version="1.2",
            reference_id=self.reference_id,
            selection=selection,
            dataset_ids=tuple(dataset.dataset_id for dataset in datasets),
            episode_ids=tuple(str(row["episode_id"]) for row in episodes),
        )
        return CatalogSelection(
            manifest=manifest,
            datasets=tuple(datasets),
            episodes=tuple(episodes),
            table_prediction_dataset_ids=frozenset(
                selected_ids & self._table_prediction_ids
            ),
            grounding_tasks=grounding,
        )


def _episode_shots(record: dict[str, Any]) -> int:
    value = record.get("shots", record.get("k"))
    if value is None:
        raise ValueError(f"episode {record.get('episode_id')!r} has no shot count")
    return int(value)


def _read_local_config(root: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("install tablesuite[local] to load local Parquet") from error
    shards = sorted(root.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no Parquet shards under {root}")
    return [row for shard in shards for row in pq.read_table(shard).to_pylist()]
