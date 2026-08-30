"""Dependency-free evaluation for structured table predictions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from tablesuite._util import canonical_json
from tablesuite.types import (
    DatasetSpec,
    ICLPredictionExample,
    SerializedTablePredictionExample,
    TaskFamily,
)

PredictionExample = ICLPredictionExample | SerializedTablePredictionExample
PredictionValues = Sequence[Any] | Mapping[str, Any] | None


@dataclass(frozen=True)
class PredictionFamilyReport:
    """Dataset-macro metrics for one prediction task family."""

    task_family: TaskFamily
    primary_metric: str
    records: int
    scored_records: int
    coverage: float
    requested_datasets: int
    scored_datasets: int
    per_dataset: dict[str, dict[str, float | None]]
    dataset_macro_accuracy: float | None = None
    dataset_macro_balanced_accuracy: float | None = None
    dataset_macro_f1: float | None = None
    cluster_macro_balanced_accuracy: float | None = None
    dataset_macro_mae: float | None = None
    dataset_macro_rmse: float | None = None
    dataset_macro_normalized_mae: float | None = None
    dataset_macro_normalized_rmse: float | None = None
    dataset_macro_r2: float | None = None
    cluster_macro_normalized_mae: float | None = None


@dataclass(frozen=True)
class PredictionReport:
    """Complete prediction report across classification and regression."""

    requests: int
    submitted_requests: int
    families: tuple[PredictionFamilyReport, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        return {
            "requests": self.requests,
            "submitted_requests": self.submitted_requests,
            "families": {
                report.task_family: asdict(report) for report in self.families
            },
        }


@dataclass(frozen=True)
class _Record:
    dataset_id: str
    dedup_cluster_id: str
    task_family: TaskFamily
    target: Any
    prediction: Any | None


def evaluate_predictions(
    examples: Iterable[PredictionExample],
    predictions: Mapping[str, PredictionValues],
    *,
    datasets: Iterable[DatasetSpec],
    allow_partial: bool = False,
) -> PredictionReport:
    """Evaluate structured predictions in each request's query-row order."""

    items = tuple(examples)
    if not items:
        raise ValueError("prediction examples cannot be empty")
    by_id = {item.request.request_id: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("prediction request IDs must be unique")
    unknown = sorted(set(predictions) - set(by_id))
    if unknown:
        raise KeyError(f"predictions contain unknown request IDs: {unknown[:3]}")
    missing = sorted(set(by_id) - set(predictions))
    if missing and not allow_partial:
        raise ValueError(
            f"predictions are missing {len(missing)} requests; "
            "pass allow_partial=True for diagnostic coverage"
        )

    specifications = {dataset.dataset_id: dataset for dataset in datasets}
    records: list[_Record] = []
    for item in items:
        request = item.request
        try:
            dataset = specifications[request.dataset_id]
        except KeyError as error:
            raise KeyError(
                f"request {request.request_id!r} references an unknown dataset"
            ) from error
        query_ids = _query_row_ids(item)
        values = _prediction_values(
            predictions.get(request.request_id),
            query_ids,
            provided=request.request_id in predictions,
        )
        if len(values) != len(item.gold.query_targets):
            raise ValueError(
                f"request {request.request_id!r} has {len(values)} predictions for "
                f"{len(item.gold.query_targets)} query rows"
            )
        for target, prediction in zip(item.gold.query_targets, values, strict=True):
            records.append(
                _Record(
                    dataset_id=dataset.dataset_id,
                    dedup_cluster_id=dataset.dedup_cluster_id,
                    task_family=dataset.task_family,
                    target=target,
                    prediction=_valid_prediction(prediction, request),
                )
            )

    families = tuple(
        _family_report(family, [record for record in records if record.task_family == family])
        for family in ("classification", "regression")
        if any(record.task_family == family for record in records)
    )
    return PredictionReport(
        requests=len(items),
        submitted_requests=len(set(predictions) & set(by_id)),
        families=families,
    )


def _prediction_values(
    submitted: PredictionValues,
    query_ids: tuple[str, ...],
    *,
    provided: bool,
) -> tuple[Any | None, ...]:
    if not provided or submitted is None:
        return (None,) * len(query_ids)
    if isinstance(submitted, Mapping):
        unknown = sorted(set(submitted) - set(query_ids))
        if unknown:
            raise KeyError(f"prediction contains unknown query row IDs: {unknown[:3]}")
        return tuple(submitted.get(row_id) for row_id in query_ids)
    if isinstance(submitted, str | bytes) or not isinstance(submitted, Sequence):
        raise TypeError("each prediction must be a sequence or query-row mapping")
    return tuple(submitted)


def _query_row_ids(item: PredictionExample) -> tuple[str, ...]:
    if isinstance(item, ICLPredictionExample):
        return item.request.query.source.row_ids
    return item.request.query_row_ids


def _valid_prediction(prediction: Any, request: Any) -> Any | None:
    if prediction is None:
        return None
    if request.task_family == "classification":
        key = canonical_json(prediction)
        allowed = {canonical_json(value): value for value in request.class_labels}
        return allowed.get(key)
    try:
        value = float(prediction)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _family_report(family: TaskFamily, records: list[_Record]) -> PredictionFamilyReport:
    valid = [record for record in records if record.prediction is not None]
    requested_datasets = len({record.dataset_id for record in records})
    by_dataset = _group(valid, "dataset_id")
    metric = _classification_metrics if family == "classification" else _regression_metrics
    dataset_metrics = {key: metric(group) for key, group in by_dataset.items()}
    dataset_clusters = {
        dataset_id: group[0].dedup_cluster_id
        for dataset_id, group in by_dataset.items()
    }
    common = {
        "task_family": family,
        "records": len(records),
        "scored_records": len(valid),
        "coverage": len(valid) / len(records),
        "requested_datasets": requested_datasets,
        "scored_datasets": len(by_dataset),
        "per_dataset": dataset_metrics,
    }
    if family == "classification":
        return PredictionFamilyReport(
            primary_metric="dataset_macro_balanced_accuracy",
            dataset_macro_accuracy=_metric_mean(dataset_metrics, "accuracy"),
            dataset_macro_balanced_accuracy=_metric_mean(
                dataset_metrics, "balanced_accuracy"
            ),
            dataset_macro_f1=_metric_mean(dataset_metrics, "macro_f1"),
            cluster_macro_balanced_accuracy=_cluster_macro_metric(
                dataset_metrics,
                dataset_clusters,
                "balanced_accuracy",
            ),
            **common,
        )
    return PredictionFamilyReport(
        primary_metric="dataset_macro_normalized_mae",
        dataset_macro_mae=_metric_mean(dataset_metrics, "mae"),
        dataset_macro_rmse=_metric_mean(dataset_metrics, "rmse"),
        dataset_macro_normalized_mae=_metric_mean(dataset_metrics, "normalized_mae"),
        dataset_macro_normalized_rmse=_metric_mean(
            dataset_metrics, "normalized_rmse"
        ),
        dataset_macro_r2=_metric_mean(dataset_metrics, "r2"),
        cluster_macro_normalized_mae=_cluster_macro_metric(
            dataset_metrics,
            dataset_clusters,
            "normalized_mae",
        ),
        **common,
    )


def _group(records: list[_Record], field: str) -> dict[str, list[_Record]]:
    grouped: dict[str, list[_Record]] = defaultdict(list)
    for record in records:
        grouped[str(getattr(record, field))].append(record)
    return dict(sorted(grouped.items()))


def _classification_metrics(records: list[_Record]) -> dict[str, float]:
    labels = _unique(
        [record.target for record in records]
        + [record.prediction for record in records]
    )
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        true_positive = sum(
            _same(record.target, label) and _same(record.prediction, label)
            for record in records
        )
        false_positive = sum(
            not _same(record.target, label) and _same(record.prediction, label)
            for record in records
        )
        false_negative = sum(
            _same(record.target, label) and not _same(record.prediction, label)
            for record in records
        )
        support = true_positive + false_negative
        if support:
            recalls.append(true_positive / support)
        denominator = 2 * true_positive + false_positive + false_negative
        f1s.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return {
        "accuracy": _mean(_same(record.target, record.prediction) for record in records),
        "balanced_accuracy": _mean(recalls),
        "macro_f1": _mean(f1s),
    }


def _regression_metrics(records: list[_Record]) -> dict[str, float | None]:
    targets = [float(record.target) for record in records]
    predictions = [float(record.prediction) for record in records]
    errors = [target - prediction for target, prediction in zip(targets, predictions, strict=True)]
    mae = _mean(abs(error) for error in errors)
    mse = _mean(error * error for error in errors)
    center = _mean(targets)
    variance = _mean((target - center) ** 2 for target in targets)
    baseline_mae = _mean(abs(target - center) for target in targets)
    return {
        "mae": mae,
        "rmse": math.sqrt(mse),
        "normalized_mae": mae / baseline_mae if baseline_mae else None,
        "normalized_rmse": math.sqrt(mse / variance) if variance else None,
        "r2": 1.0 - mse / variance if variance else None,
    }


def _metric_mean(
    metrics: Mapping[str, Mapping[str, float | None]],
    name: str,
) -> float | None:
    values = [metric[name] for metric in metrics.values() if metric[name] is not None]
    return _mean(values) if values else None


def _cluster_macro_metric(
    metrics: Mapping[str, Mapping[str, float | None]],
    dataset_clusters: Mapping[str, str],
    name: str,
) -> float | None:
    """Macro-average dataset-local metrics without mixing label or value spaces."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for dataset_id, values in metrics.items():
        value = values[name]
        if value is not None:
            grouped[dataset_clusters[dataset_id]].append(value)
    return _mean(_mean(values) for values in grouped.values()) if grouped else None


def _unique(values: Iterable[Any]) -> list[Any]:
    output: dict[str, Any] = {}
    for value in values:
        output.setdefault(canonical_json(value), value)
    return list(output.values())


def _same(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _mean(values: Iterable[float | int | bool]) -> float:
    items = [float(value) for value in values]
    if not items:
        raise ValueError("cannot average an empty collection")
    return sum(items) / len(items)


__all__ = [
    "PredictionFamilyReport",
    "PredictionReport",
    "PredictionValues",
    "evaluate_predictions",
]
