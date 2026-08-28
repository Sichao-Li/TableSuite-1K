"""Deterministic percentage-controlled prediction interfaces."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal

from tablesuite._util import canonical_json, stable_id, stable_order, valid_target
from tablesuite.source import ParquetSource
from tablesuite.types import (
    DatasetSpec,
    ICLPredictionExample,
    ICLPredictionRequest,
    PredictionGold,
    SerializedTablePredictionExample,
    SerializedTablePredictionRequest,
    SupportLevel,
    TableSlice,
)

PredictionInterface = Literal["icl", "serialized_table"]
OFFICIAL_SUPPORT_LEVELS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
PREDICTION_PLAN_VERSION = "fractional_support_v1"


@dataclass(frozen=True)
class PredictionManifest:
    """Exact runtime prediction selection without source values or labels."""

    protocol: PredictionInterface
    support_fractions: tuple[float, ...]
    dataset_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    plan_version: str = PREDICTION_PLAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible manifest."""

        return asdict(self)


@dataclass(frozen=True)
class _PredictionPlan:
    plan_id: str
    dataset_id: str
    query_row_ids: tuple[str, ...]


class PredictionDataset:
    """Lazy prediction requests over one nested support-fraction schedule."""

    def __init__(
        self,
        *,
        source: ParquetSource,
        datasets: Iterable[DatasetSpec],
        episodes: Iterable[dict[str, Any]],
        protocol: PredictionInterface,
        support: float | Iterable[float],
        max_episodes_per_dataset: int | None = None,
        seed: int = 0,
    ) -> None:
        if protocol not in {"icl", "serialized_table"}:
            raise ValueError("protocol must be 'icl' or 'serialized_table'")
        if max_episodes_per_dataset is not None and max_episodes_per_dataset <= 0:
            raise ValueError("max_episodes_per_dataset must be positive")
        self.protocol = protocol
        self.support_fractions = normalize_support_fractions(support)
        self.source = source
        self._datasets = {dataset.dataset_id: dataset for dataset in datasets}
        self._plans = _prediction_plans(
            episodes,
            max_episodes_per_dataset=max_episodes_per_dataset,
            seed=seed,
        )
        self._seed = seed
        if not self._plans:
            raise ValueError("no eligible frozen prediction queries were selected")
        self.manifest = PredictionManifest(
            protocol=protocol,
            support_fractions=self.support_fractions,
            dataset_ids=tuple(sorted({plan.dataset_id for plan in self._plans})),
            episode_ids=tuple(plan.plan_id for plan in self._plans),
        )

    def __len__(self) -> int:
        return len(self._plans) * len(self.support_fractions)

    def __iter__(
        self,
    ) -> Iterator[ICLPredictionExample | SerializedTablePredictionExample]:
        for plan in self._plans:
            dataset = self._datasets[plan.dataset_id]
            rows = self.source.rows(dataset)
            support_pool = _ordered_support_rows(
                dataset,
                rows,
                query_row_ids=plan.query_row_ids,
                seed=self._seed,
            )
            for fraction in self.support_fractions:
                level = resolve_support_level(fraction, len(support_pool))
                support_row_ids = support_pool[: level.count]
                if self.protocol == "icl":
                    yield self._icl_example(
                        plan, dataset, support_row_ids, level
                    )
                else:
                    yield self._serialized_example(
                        plan, dataset, support_row_ids, level
                    )

    def summary(self) -> dict[str, Any]:
        """Return compact plan metadata without materializing source tables."""

        return {
            "protocol": self.protocol,
            "support_fractions": self.support_fractions,
            "datasets": len(self.manifest.dataset_ids),
            "episodes": len(self._plans),
            "requests": len(self),
            "plan_version": self.manifest.plan_version,
        }

    def _icl_example(
        self,
        plan: _PredictionPlan,
        dataset: DatasetSpec,
        support_row_ids: tuple[str, ...],
        level: SupportLevel,
    ) -> ICLPredictionExample:
        protocol = "few_shot_icl" if level.count else "zero_shot_icl"
        request_id = _request_id(plan.plan_id, self.protocol, level.requested_fraction)
        demonstrations = (
            self.source.materialize(
                dataset,
                TableSlice(
                    dataset.dataset_id,
                    support_row_ids,
                    (*dataset.feature_columns, dataset.target_column),
                ),
            )
            if support_row_ids
            else None
        )
        query = self.source.materialize(
            dataset,
            TableSlice(
                dataset.dataset_id,
                plan.query_row_ids,
                dataset.feature_columns,
            ),
        )
        request = ICLPredictionRequest(
            request_id=request_id,
            protocol=protocol,
            dataset_split=dataset.dataset_split,
            task_type=dataset.task_type,
            task_family=dataset.task_family,
            target_column=dataset.target_column,
            query=query,
            shots=level.count,
            demonstrations=demonstrations,
        )
        targets = tuple(
            row[dataset.target_column]
            for row in self.source.rows(dataset, plan.query_row_ids)
        )
        return ICLPredictionExample(
            request=request,
            gold=PredictionGold(request_id, targets),
            episode_id=plan.plan_id,
            shots=level.count,
            support=level,
        )

    def _serialized_example(
        self,
        plan: _PredictionPlan,
        dataset: DatasetSpec,
        support_row_ids: tuple[str, ...],
        level: SupportLevel,
    ) -> SerializedTablePredictionExample:
        protocol = (
            "partially_labeled_serialized_table"
            if level.count
            else "zero_label_serialized_table"
        )
        request_id = _request_id(plan.plan_id, self.protocol, level.requested_fraction)
        table_row_ids = (*support_row_ids, *plan.query_row_ids)
        table = self.source.materialize(
            dataset,
            TableSlice(dataset.dataset_id, table_row_ids, dataset.feature_columns),
        )
        visible_labels = (
            self.source.materialize(
                dataset,
                TableSlice(
                    dataset.dataset_id,
                    support_row_ids,
                    (dataset.target_column,),
                ),
            )
            if support_row_ids
            else None
        )
        request = SerializedTablePredictionRequest(
            request_id=request_id,
            protocol=protocol,
            scope="episode",
            dataset_split=dataset.dataset_split,
            task_type=dataset.task_type,
            task_family=dataset.task_family,
            target_column=dataset.target_column,
            table=table,
            visible_labels=visible_labels,
        )
        targets = tuple(
            row[dataset.target_column]
            for row in self.source.rows(dataset, request.query_row_ids)
        )
        return SerializedTablePredictionExample(
            request=request,
            gold=PredictionGold(request_id, targets),
            support=level,
        )


def normalize_support_fractions(
    support: float | Iterable[float],
) -> tuple[float, ...]:
    """Normalize one fraction or an ordered fraction sequence."""

    if isinstance(support, Real) and not isinstance(support, bool):
        values = (float(support),)
    else:
        if isinstance(support, str | bytes):
            raise TypeError("support must be a number or an iterable of numbers")
        try:
            values = tuple(float(value) for value in support)
        except (TypeError, ValueError) as error:
            raise TypeError(
                "support must be a number or an iterable of numbers"
            ) from error
    if not values:
        raise ValueError("support cannot be empty")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("support fractions must be finite values between 0 and 1")
    if len(values) != len(set(values)):
        raise ValueError("support fractions must be unique")
    return values


def resolve_support_level(fraction: float, pool_size: int) -> SupportLevel:
    """Resolve a requested fraction to a deterministic labelled-row count."""

    if pool_size < 0:
        raise ValueError("support pool size cannot be negative")
    count = 0 if fraction == 0.0 else min(pool_size, max(1, math.ceil(fraction * pool_size)))
    return SupportLevel(float(fraction), pool_size, count)


def _prediction_plans(
    episodes: Iterable[dict[str, Any]],
    *,
    max_episodes_per_dataset: int | None,
    seed: int,
) -> tuple[_PredictionPlan, ...]:
    unique: dict[tuple[str, tuple[str, ...]], _PredictionPlan] = {}
    for episode in episodes:
        dataset_id = str(episode["dataset_id"])
        query_row_ids = tuple(str(value) for value in episode["query_row_ids"])
        key = (dataset_id, query_row_ids)
        unique.setdefault(
            key,
            _PredictionPlan(
                plan_id=stable_id(
                    "prediction",
                    canonical_json([dataset_id, list(query_row_ids)]),
                ),
                dataset_id=dataset_id,
                query_row_ids=query_row_ids,
            ),
        )
    grouped: dict[str, list[_PredictionPlan]] = defaultdict(list)
    for plan in unique.values():
        grouped[plan.dataset_id].append(plan)
    selected: list[_PredictionPlan] = []
    for dataset_id in sorted(grouped):
        plans = sorted(
            grouped[dataset_id],
            key=lambda plan: stable_order(plan.plan_id, seed),
        )
        if max_episodes_per_dataset is not None:
            plans = plans[:max_episodes_per_dataset]
        selected.extend(plans)
    return tuple(selected)


def _ordered_support_rows(
    dataset: DatasetSpec,
    rows: list[dict[str, Any]],
    *,
    query_row_ids: tuple[str, ...],
    seed: int,
) -> tuple[str, ...]:
    query = set(query_row_ids)
    eligible = [
        index
        for index, row in enumerate(rows)
        if str(index) not in query
        and valid_target(row[dataset.target_column], dataset.task_family)
    ]
    if dataset.task_family == "classification":
        groups: dict[str, list[int]] = defaultdict(list)
        for index in eligible:
            groups[canonical_json(rows[index][dataset.target_column])].append(index)
    else:
        ranked = sorted(eligible, key=lambda index: float(rows[index][dataset.target_column]))
        bins = min(10, len(ranked))
        groups = defaultdict(list)
        for rank, index in enumerate(ranked):
            groups[str(rank * bins // max(len(ranked), 1))].append(index)
    ordered_groups = [
        sorted(
            values,
            key=lambda index: stable_order(f"{dataset.dataset_id}:{index}", seed),
        )
        for _, values in sorted(
            groups.items(),
            key=lambda item: stable_order(f"{dataset.dataset_id}:{item[0]}", seed),
        )
    ]
    output: list[str] = []
    for offset in range(max((len(group) for group in ordered_groups), default=0)):
        output.extend(str(group[offset]) for group in ordered_groups if offset < len(group))
    return tuple(output)


def _request_id(plan_id: str, protocol: str, fraction: float) -> str:
    level = format(fraction, ".6f").rstrip("0").rstrip(".").replace(".", "p")
    return f"{plan_id}:{protocol}:support_{level or '0'}"


__all__ = [
    "OFFICIAL_SUPPORT_LEVELS",
    "PREDICTION_PLAN_VERSION",
    "PredictionDataset",
    "PredictionInterface",
    "PredictionManifest",
    "normalize_support_fractions",
    "resolve_support_level",
]
