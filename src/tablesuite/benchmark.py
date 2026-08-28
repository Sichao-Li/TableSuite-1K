"""Public façade for selecting and iterating TableSuite-1K tasks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from tablesuite._util import (
    canonical_json,
    normalize_identifier,
    normalize_value,
    stable_id,
    stable_order,
    stable_value_key,
    valid_target,
)
from tablesuite.catalog import Catalog, CatalogSelection
from tablesuite.prediction import PredictionDataset, PredictionInterface
from tablesuite.rendering import render_cell_fact_views
from tablesuite.source import ParquetSource
from tablesuite.types import (
    CellFact,
    DatasetSpec,
    ICLPredictionExample,
    ICLPredictionRequest,
    ICLProtocol,
    MaterializedTableSlice,
    PredictionGold,
    Selection,
    SelectionManifest,
    SerializedTablePredictionExample,
    SerializedTablePredictionRequest,
    SerializedTableScope,
    TableSlice,
)


class Benchmark:
    """A compact TableSuite-1K catalog connected to local OpenML source tables."""

    def __init__(self, catalog: Catalog, source: ParquetSource) -> None:
        self.catalog = catalog
        self.source = source

    @classmethod
    def from_path(
        cls,
        reference_root: str | Path,
        source_root: str | Path,
    ) -> Benchmark:
        """Open a downloaded reference package and local source directory."""

        return cls(Catalog.from_path(reference_root), ParquetSource(source_root))

    @classmethod
    def from_huggingface(
        cls,
        repository: str,
        source_root: str | Path,
        *,
        revision: str | None = None,
    ) -> Benchmark:
        """Open a Hugging Face reference package and local source directory."""

        return cls(
            Catalog.from_huggingface(repository, revision=revision),
            ParquetSource(source_root),
        )

    def select(self, selection: Selection) -> BenchmarkSubset:
        """Resolve a deterministic subset and its executable ICL episodes."""

        catalog_selection = self.catalog.select(selection)
        episodes = self._eligible_episodes(catalog_selection)
        manifest = SelectionManifest(
            schema_version="1.2",
            reference_id=self.catalog.reference_id,
            selection=selection,
            dataset_ids=catalog_selection.manifest.dataset_ids,
            episode_ids=tuple(str(item["episode_id"]) for item in episodes),
        )
        return BenchmarkSubset(
            source=self.source,
            selection=catalog_selection,
            manifest=manifest,
            eligible_episodes=episodes,
        )

    def _eligible_episodes(
        self,
        selected: CatalogSelection,
    ) -> tuple[dict[str, Any], ...]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        datasets = {item.dataset_id: item for item in selected.datasets}
        for episode in selected.episodes:
            grouped[str(episode["dataset_id"])].append(episode)
        eligible: list[dict[str, Any]] = []
        limit = selected.manifest.selection.max_episodes_per_dataset_per_shot
        requested = set(selected.manifest.selection.shots)
        for dataset_id in sorted(grouped):
            dataset = datasets[dataset_id]
            rows = self.source.rows(dataset)
            valid = [
                episode
                for episode in grouped[dataset_id]
                if _episode_is_eligible(dataset, rows, episode)
            ]
            items = (
                valid
                if not requested
                else [
                    episode
                    for episode in valid
                    if int(episode["shots"]) in requested
                ]
            )
            if limit is not None:
                counts: dict[int, int] = defaultdict(int)
                limited: list[dict[str, Any]] = []
                for episode in items:
                    shots = int(episode["shots"])
                    if counts[shots] >= limit:
                        continue
                    counts[shots] += 1
                    limited.append(episode)
                items = limited
            eligible.extend(items)
        return tuple(eligible)


class BenchmarkSubset:
    """An exact benchmark subset with task iterators and a saveable manifest."""

    def __init__(
        self,
        *,
        source: ParquetSource,
        selection: CatalogSelection,
        manifest: SelectionManifest,
        eligible_episodes: tuple[dict[str, Any], ...],
    ) -> None:
        self.source = source
        self._selection = selection
        self.manifest = manifest
        self._episodes = eligible_episodes
        self._datasets = {dataset.dataset_id: dataset for dataset in selection.datasets}
        self._class_labels: dict[str, tuple[Any, ...]] = {}

    @property
    def datasets(self) -> tuple[DatasetSpec, ...]:
        """Return the selected dataset specifications in manifest order."""

        return self._selection.datasets

    def materialize(self, source_slice: TableSlice) -> MaterializedTableSlice:
        """Resolve one selected row or subtable slice from its source table."""

        try:
            dataset = self._datasets[source_slice.dataset_id]
        except KeyError as error:
            raise KeyError(
                f"dataset {source_slice.dataset_id!r} is not in this benchmark subset"
            ) from error
        return self.source.materialize(dataset, source_slice)

    def materialize_many(
        self,
        source_slices: Iterable[TableSlice],
    ) -> tuple[MaterializedTableSlice, ...]:
        """Resolve an ordered collection of slices from one or more datasets."""

        return tuple(self.materialize(source_slice) for source_slice in source_slices)

    def prediction(
        self,
        protocol: PredictionInterface,
        *,
        support: float | Iterable[float],
        max_episodes_per_dataset: int | None = None,
        seed: int | None = None,
    ) -> PredictionDataset:
        """Build nested percentage-controlled requests from frozen query rows."""

        return PredictionDataset(
            source=self.source,
            datasets=self.datasets,
            episodes=self._episodes,
            protocol=protocol,
            support=support,
            max_episodes_per_dataset=max_episodes_per_dataset,
            seed=self.manifest.selection.seed if seed is None else seed,
        )

    def zero_label_serialized_table(
        self,
        *,
        scope: SerializedTableScope = "full_table",
        rows_per_table: int | None = None,
    ) -> Iterator[SerializedTablePredictionExample]:
        """Yield feature-only tables with separate per-row evaluation targets.

        By default, each request contains every source row with a valid target.
        ``scope="episode"`` uses the same frozen query rows as zero-shot ICL for
        a matched, bounded interface comparison. ``rows_per_table`` chunks the
        selected scope without sampling or dropping its rows.
        """

        self._require_task("zero_label_serialized_table")
        if scope not in {"full_table", "episode"}:
            raise ValueError("scope must be 'full_table' or 'episode'")
        if rows_per_table is not None and rows_per_table <= 0:
            raise ValueError("rows_per_table must be positive")
        if scope == "episode":
            sources = (
                (
                    self._datasets[str(episode["dataset_id"])],
                    tuple(str(value) for value in episode["query_row_ids"]),
                    str(episode["episode_id"]),
                )
                for episode in _derive_zero_shot_episodes(list(self._episodes))
            )
        else:
            sources = (
                (
                    dataset,
                    tuple(
                        str(row_id)
                        for row_id, row in enumerate(self.source.rows(dataset))
                        if valid_target(
                            row[dataset.target_column], dataset.task_family
                        )
                    ),
                    dataset.dataset_id,
                )
                for dataset in self._selection.datasets
                if dataset.dataset_id
                in self._selection.table_prediction_dataset_ids
            )
        for dataset, row_ids, source_id in sources:
            if not row_ids:
                continue
            chunk_size = rows_per_table or len(row_ids)
            rows = self.source.rows(dataset)
            for chunk_index, start in enumerate(range(0, len(row_ids), chunk_size)):
                query_ids = row_ids[start : start + chunk_size]
                request_id = (
                    f"{source_id}:zero_label_serialized_table:{scope}:{chunk_index}"
                )
                table = self.materialize(
                    TableSlice(
                        dataset_id=dataset.dataset_id,
                        row_ids=query_ids,
                        columns=dataset.feature_columns,
                    )
                )
                yield SerializedTablePredictionExample(
                    request=SerializedTablePredictionRequest(
                        request_id=request_id,
                        protocol="zero_label_serialized_table",
                        scope=scope,
                        dataset_split=dataset.dataset_split,
                        task_type=dataset.task_type,
                        task_family=dataset.task_family,
                        target_column=dataset.target_column,
                        table=table,
                        class_labels=self._classification_labels(dataset),
                    ),
                    gold=PredictionGold(
                        request_id=request_id,
                        query_targets=tuple(
                            rows[int(row_id)][dataset.target_column]
                            for row_id in query_ids
                        ),
                    ),
                )

    def partially_labeled_serialized_table(
        self,
        *,
        query_scope: SerializedTableScope = "full_table",
        query_rows_per_table: int | None = None,
    ) -> Iterator[SerializedTablePredictionExample]:
        """Yield tables with frozen support labels and target-hidden rows.

        The default exposes every eligible non-support row in the source table.
        ``query_scope="episode"`` restricts queries to the frozen ICL episode for
        a controlled interface comparison. Bounded consumers may chunk query
        rows; visible support rows are repeated in every chunk.
        """

        self._require_task("partially_labeled_serialized_table")
        if query_scope not in {"full_table", "episode"}:
            raise ValueError("query_scope must be 'full_table' or 'episode'")
        if query_rows_per_table is not None and query_rows_per_table <= 0:
            raise ValueError("query_rows_per_table must be positive")
        for episode in self._episodes:
            dataset = self._datasets[str(episode["dataset_id"])]
            support_ids = tuple(str(value) for value in episode["support_row_ids"])
            if query_scope == "episode":
                query_ids = tuple(
                    str(value) for value in episode["query_row_ids"]
                )
            else:
                support_set = set(support_ids)
                query_ids = tuple(
                    str(row_id)
                    for row_id, row in enumerate(self.source.rows(dataset))
                    if str(row_id) not in support_set
                    and valid_target(
                        row[dataset.target_column], dataset.task_family
                    )
                )
            chunk_size = query_rows_per_table or len(query_ids)
            if chunk_size == 0:
                continue
            visible_labels = self.materialize(
                TableSlice(
                    dataset_id=dataset.dataset_id,
                    row_ids=support_ids,
                    columns=(dataset.target_column,),
                )
            )
            for chunk_index, start in enumerate(range(0, len(query_ids), chunk_size)):
                query_chunk = query_ids[start : start + chunk_size]
                table_ids = tuple(
                    sorted({*support_ids, *query_chunk}, key=lambda value: int(value))
                )
                table = self.materialize(
                    TableSlice(
                        dataset_id=dataset.dataset_id,
                        row_ids=table_ids,
                        columns=dataset.feature_columns,
                    )
                )
                request_id = (
                    f"{episode['episode_id']}:partially_labeled_serialized_table:"
                    f"{query_scope}:{chunk_index}"
                )
                request = SerializedTablePredictionRequest(
                    request_id=request_id,
                    protocol="partially_labeled_serialized_table",
                    scope=query_scope,
                    dataset_split=dataset.dataset_split,
                    task_type=dataset.task_type,
                    task_family=dataset.task_family,
                    target_column=dataset.target_column,
                    table=table,
                    class_labels=self._classification_labels(dataset),
                    visible_labels=visible_labels,
                )
                query_targets = tuple(
                    row[dataset.target_column]
                    for row in self.source.rows(dataset, request.query_row_ids)
                )
                yield SerializedTablePredictionExample(
                    request=request,
                    gold=PredictionGold(
                        request_id=request_id,
                        query_targets=query_targets,
                    ),
                )

    def zero_shot_icl(self) -> Iterator[ICLPredictionExample]:
        """Yield target-hidden row queries with no visible demonstrations."""

        self._require_task("zero_shot_icl")
        for episode in _derive_zero_shot_episodes(list(self._episodes)):
            yield self._icl_example(episode, protocol="zero_shot_icl")

    def few_shot_icl(self) -> Iterator[ICLPredictionExample]:
        """Yield frozen labelled row demonstrations followed by queries."""

        self._require_task("few_shot_icl")
        for episode in self._episodes:
            yield self._icl_example(episode, protocol="few_shot_icl")

    def _icl_example(
        self,
        episode: dict[str, Any],
        *,
        protocol: ICLProtocol,
    ) -> ICLPredictionExample:
        dataset = self._datasets[str(episode["dataset_id"])]
        shots = int(episode["shots"])
        support_ids = tuple(str(value) for value in episode["support_row_ids"])
        query_ids = tuple(str(value) for value in episode["query_row_ids"])
        demonstrations = (
            self.materialize(
                TableSlice(
                    dataset_id=dataset.dataset_id,
                    row_ids=support_ids,
                    columns=(*dataset.feature_columns, dataset.target_column),
                )
            )
            if shots
            else None
        )
        query = self.materialize(
            TableSlice(
                dataset_id=dataset.dataset_id,
                row_ids=query_ids,
                columns=dataset.feature_columns,
            )
        )
        request_id = str(episode["episode_id"])
        query_targets = tuple(
            row[dataset.target_column]
            for row in self.source.rows(dataset, query_ids)
        )
        return ICLPredictionExample(
            request=ICLPredictionRequest(
                request_id=request_id,
                protocol=protocol,
                dataset_split=dataset.dataset_split,
                task_type=dataset.task_type,
                task_family=dataset.task_family,
                target_column=dataset.target_column,
                query=query,
                shots=shots,
                class_labels=self._classification_labels(dataset),
                demonstrations=demonstrations,
            ),
            gold=PredictionGold(
                request_id=request_id,
                query_targets=query_targets,
            ),
            episode_id=request_id,
            shots=shots,
        )

    def _classification_labels(self, dataset: DatasetSpec) -> tuple[Any, ...]:
        if dataset.task_family == "regression":
            return ()
        cached = self._class_labels.get(dataset.dataset_id)
        if cached is not None:
            return cached
        labels = {
            canonical_json(normalize_value(row[dataset.target_column])): normalize_value(
                row[dataset.target_column]
            )
            for row in self.source.rows(dataset)
            if valid_target(row[dataset.target_column], dataset.task_family)
        }
        ordered = tuple(labels[key] for key in sorted(labels))
        if not ordered:
            raise ValueError(f"{dataset.dataset_id}: classification label space is empty")
        self._class_labels[dataset.dataset_id] = ordered
        return ordered

    def grounding(self) -> Iterator[CellFact]:
        """Yield deterministic, column-balanced non-target cell facts."""

        self._require_task("grounding")
        task_by_dataset = {
            str(task["dataset_id"]): task for task in self._selection.grounding_tasks
        }
        selected_limit = self.manifest.selection.max_grounding_facts_per_dataset
        for dataset in self._selection.datasets:
            task = task_by_dataset.get(dataset.dataset_id)
            if task is None:
                continue
            rows = self.source.rows(dataset)
            columns = tuple(str(value) for value in task["eligible_columns"])
            task_limit = int(task["max_cells"])
            limit = task_limit if selected_limit is None else min(selected_limit, task_limit)
            for column, row_id in _balanced_coordinates(
                dataset.dataset_id, columns, len(rows), limit
            ):
                value = rows[row_id][column]
                value_key = stable_value_key(value)
                if value_key == "<missing>":
                    continue
                fact_key = canonical_json([dataset.dataset_id, str(row_id), column])
                equivalence_key = canonical_json(
                    [dataset.dataset_id, normalize_identifier(column), value_key]
                )
                yield CellFact(
                    fact_id=stable_id("cell", fact_key),
                    equivalence_id=stable_id("fact", equivalence_key),
                    source=TableSlice(
                        dataset_id=dataset.dataset_id,
                        row_ids=(str(row_id),),
                        columns=(column,),
                    ),
                    dataset_split=dataset.dataset_split,
                    value=value,
                    stable_value_key=value_key,
                    text_views=render_cell_fact_views(column, value),
                )

    def _require_task(self, task: str) -> None:
        if task not in self.manifest.selection.tasks:
            raise ValueError(f"task {task!r} was not selected")


def _episode_is_eligible(
    dataset: DatasetSpec,
    rows: list[dict[str, Any]],
    episode: dict[str, Any],
) -> bool:
    shots = int(episode["shots"])
    support_ids = [int(value) for value in episode["support_row_ids"]]
    query_ids = [int(value) for value in episode["query_row_ids"]]
    if set(support_ids) & set(query_ids):
        return False
    if not query_ids or len(support_ids) != shots:
        return False
    all_ids = [*support_ids, *query_ids]
    if min(all_ids) < 0 or max(all_ids) >= len(rows):
        return False
    support = [rows[index][dataset.target_column] for index in support_ids]
    query = [rows[index][dataset.target_column] for index in query_ids]
    if not all(valid_target(value, dataset.task_family) for value in [*support, *query]):
        return False
    if shots == 0:
        return True
    if dataset.task_family == "classification":
        support_classes = {canonical_json(value) for value in support}
        query_classes = {canonical_json(value) for value in query}
        return query_classes <= support_classes
    return True


def _derive_zero_shot_episodes(
    valid_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reuse query sets from source-validated few-shot episodes."""

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for candidate in valid_candidates:
        if int(candidate["shots"]) == 0:
            continue
        query_ids = tuple(str(value) for value in candidate["query_row_ids"])
        key = (str(candidate["dataset_id"]), query_ids)
        if key in seen:
            continue
        seen.add(key)
        record = dict(candidate)
        record["episode_id"] = f"{candidate['episode_id']}:shots0"
        record["base_episode_id"] = str(candidate["episode_id"])
        record["shots"] = 0
        record["support_row_ids"] = []
        output.append(record)
    return output


def _balanced_coordinates(
    dataset_id: str,
    columns: tuple[str, ...],
    n_rows: int,
    limit: int,
) -> Iterator[tuple[str, int]]:
    if not columns or n_rows <= 0:
        return
    ranked = sorted(columns, key=lambda column: stable_order(f"{dataset_id}:{column}:quota"))
    base, remainder = divmod(limit, len(ranked))
    for rank, column in enumerate(ranked):
        quota = min(n_rows, base + int(rank < remainder))
        if quota <= 0:
            continue
        start = int(stable_order(f"{dataset_id}:{column}:start"), 16) % n_rows
        step = int(stable_order(f"{dataset_id}:{column}:step"), 16) % max(n_rows - 1, 1) + 1
        while math.gcd(step, n_rows) != 1:
            step = step % n_rows + 1
        for offset in range(quota):
            yield column, (start + offset * step) % n_rows
