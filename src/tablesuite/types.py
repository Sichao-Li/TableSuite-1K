"""Small, stable public contracts for the TableSuite-1K benchmark."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from tablesuite._util import canonical_json, normalize_value

TaskName = Literal[
    "zero_shot_icl",
    "few_shot_icl",
    "zero_label_serialized_table",
    "partially_labeled_serialized_table",
]
TaskFamily = Literal["classification", "regression"]
TextView = Literal["json", "key_value", "markdown"]
PredictionView = TextView | Literal["row_examples"]
ICLProtocol = Literal["zero_shot_icl", "few_shot_icl"]
SerializedTableProtocol = Literal[
    "zero_label_serialized_table",
    "partially_labeled_serialized_table",
]
SerializedTableScope = Literal["full_table", "episode"]


@dataclass(frozen=True)
class SupportLevel:
    """Resolved labelled support for one inference-only prediction request."""

    requested_fraction: float
    pool_size: int
    count: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.requested_fraction <= 1.0:
            raise ValueError("requested support fraction must be between 0 and 1")
        if self.pool_size < 0:
            raise ValueError("support pool size cannot be negative")
        if not 0 <= self.count <= self.pool_size:
            raise ValueError("resolved support count is outside the support pool")

    @property
    def realized_fraction(self) -> float:
        """Return the fraction of eligible support rows actually exposed."""

        return self.count / self.pool_size if self.pool_size else 0.0


@dataclass(frozen=True)
class DatasetSpec:
    """One OpenML-referenced dataset and its benchmark contract."""

    dataset_id: str
    dataset_split: str
    source_id: str
    source_url: str
    task_type: str
    target_column: str
    feature_columns: tuple[str, ...]
    n_rows: int
    dedup_cluster_id: str = ""
    target_transform: str = "none"
    excluded_feature_columns: tuple[str, ...] = ()
    semantic_columns: tuple[str, ...] = ()
    dataset_name: str = ""
    license_claim: str = ""
    source_license_url: str = ""

    @property
    def task_family(self) -> TaskFamily:
        """Return the shared classification/regression family."""

        return "regression" if self.task_type == "regression" else "classification"

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> DatasetSpec:
        """Create a typed dataset specification from a catalog record."""

        source_id = record.get("openml_data_id", record.get("source_id"))
        if source_id is None:
            raise KeyError("dataset record has no OpenML data ID")
        feature_columns = tuple(str(value) for value in record["feature_columns"])
        target_column = str(record["target_column"])
        excluded = tuple(
            str(value) for value in record.get("excluded_feature_columns", [])
        )
        declared_semantic = record.get("semantic_columns")
        semantic_columns = (
            tuple(str(value) for value in declared_semantic)
            if declared_semantic is not None
            else tuple(
                column
                for column in feature_columns
                if column != target_column and column not in set(excluded)
            )
        )
        return cls(
            dataset_id=str(record["dataset_id"]),
            dataset_split=str(record["dataset_split"]),
            source_id=str(source_id),
            source_url=str(record.get("openml_url", record.get("source_url", ""))),
            task_type=str(record["task_type"]),
            target_column=target_column,
            feature_columns=feature_columns,
            n_rows=int(record["n_rows"]),
            dedup_cluster_id=str(record.get("dedup_cluster_id") or record["dataset_id"]),
            target_transform=str(record.get("target_transform") or "none"),
            excluded_feature_columns=excluded,
            semantic_columns=semantic_columns,
            dataset_name=str(record.get("dataset_name") or ""),
            license_claim=str(
                record.get("openml_license_claim", record.get("license_claim")) or ""
            ),
            source_license_url=str(
                record.get("openml_license_url", record.get("source_license_url"))
                or ""
            ),
        )


@dataclass(frozen=True)
class Selection:
    """A deterministic request for datasets and benchmark tasks."""

    tasks: tuple[TaskName, ...]
    dataset_ids: tuple[str, ...] = ()
    dataset_splits: tuple[str, ...] = ()
    task_families: tuple[TaskFamily, ...] = ()
    shots: tuple[int, ...] = ()
    max_datasets: int | None = None
    max_episodes_per_dataset_per_shot: int | None = None
    seed: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Selection:
        """Load a selection from its JSON representation."""

        values = dict(payload)
        for name in ("tasks", "dataset_ids", "dataset_splits", "task_families", "shots"):
            if name in values:
                values[name] = tuple(values[name])
        selection = cls(**values)
        selection.validate()
        return selection

    def validate(self) -> None:
        """Raise when a selection is internally inconsistent."""

        if unknown := set(self.tasks) - {
            "zero_shot_icl",
            "few_shot_icl",
            "zero_label_serialized_table",
            "partially_labeled_serialized_table",
        }:
            raise ValueError(f"unknown tasks: {sorted(unknown)}")
        if len(self.dataset_ids) != len(set(self.dataset_ids)):
            raise ValueError("dataset IDs must be unique")
        if unknown := set(self.dataset_splits) - {"train", "validation", "test"}:
            raise ValueError(f"unknown dataset splits: {sorted(unknown)}")
        if unknown := set(self.task_families) - {"classification", "regression"}:
            raise ValueError(f"unknown task families: {sorted(unknown)}")
        if unknown := set(self.shots) - {4, 16, 32}:
            raise ValueError(f"unsupported shot counts: {sorted(unknown)}")
        label_visible = {"few_shot_icl", "partially_labeled_serialized_table"}
        if self.shots and not set(self.tasks) & label_visible:
            raise ValueError("shots require a label-visible prediction protocol")
        for name in (
            "max_datasets",
            "max_episodes_per_dataset_per_shot",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SelectionManifest:
    """The exact dataset and episode identities resolved from a selection."""

    schema_version: str
    reference_id: str
    selection: Selection
    dataset_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]

    def save(self, path: str | Path) -> None:
        """Write this manifest as deterministic JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> SelectionManifest:
        """Load a previously resolved selection manifest."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["selection"] = Selection.from_dict(payload["selection"])
        payload["dataset_ids"] = tuple(payload["dataset_ids"])
        payload["episode_ids"] = tuple(payload["episode_ids"])
        return cls(**payload)


@dataclass(frozen=True)
class TableSlice:
    """A source reference to selected rows and columns from one table.

    A row is a one-row slice. A subtable is a multi-row, multi-column slice.
    Collections of slices may span datasets, but a single slice never mixes
    schemas.
    """

    dataset_id: str
    row_ids: tuple[str, ...]
    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_id", str(self.dataset_id))
        object.__setattr__(self, "row_ids", tuple(str(value) for value in self.row_ids))
        object.__setattr__(self, "columns", tuple(str(value) for value in self.columns))
        if not self.dataset_id:
            raise ValueError("table slice requires a dataset ID")
        if not self.row_ids:
            raise ValueError("table slice requires at least one row")
        if not self.columns:
            raise ValueError("table slice requires at least one column")
        if len(self.row_ids) != len(set(self.row_ids)):
            raise ValueError("table slice row IDs must be unique")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError("table slice columns must be unique")


@dataclass(frozen=True)
class MaterializedTableSlice:
    """A table slice paired with source values in the requested order."""

    source: TableSlice
    rows: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if len(self.rows) != len(self.source.row_ids):
            raise ValueError("materialized rows do not match table slice row IDs")
        expected = set(self.source.columns)
        if any(set(row) != expected for row in self.rows):
            raise ValueError("materialized rows do not match table slice columns")


@dataclass(frozen=True)
class SerializedTablePredictionRequest:
    """Inference-only request with zero or selected visible table labels."""

    request_id: str
    protocol: SerializedTableProtocol
    scope: SerializedTableScope
    dataset_split: str
    task_type: str
    task_family: TaskFamily
    target_column: str
    table: MaterializedTableSlice
    class_labels: tuple[Any, ...] = ()
    visible_labels: MaterializedTableSlice | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "class_labels",
            tuple(normalize_value(value) for value in self.class_labels),
        )
        _validate_prediction_label_space(self.task_family, self.class_labels)
        if self.protocol not in {
            "zero_label_serialized_table",
            "partially_labeled_serialized_table",
        }:
            raise ValueError(f"unknown serialized-table protocol: {self.protocol}")
        if self.scope not in {"full_table", "episode"}:
            raise ValueError(f"unknown serialized-table scope: {self.scope}")
        if self.target_column in self.table.source.columns:
            raise ValueError("target column leaked into serialized prediction table")
        if self.protocol == "zero_label_serialized_table":
            if self.visible_labels is not None:
                raise ValueError("zero-label tables cannot contain visible labels")
            return
        if self.visible_labels is None:
            raise ValueError("partially labelled tables require visible labels")
        if self.visible_labels.source.dataset_id != self.table.source.dataset_id:
            raise ValueError("visible labels and feature table must share a dataset")
        if self.visible_labels.source.columns != (self.target_column,):
            raise ValueError("visible labels must contain only the target column")
        table_rows = set(self.table.source.row_ids)
        label_rows = set(self.visible_labels.source.row_ids)
        if not label_rows < table_rows:
            raise ValueError(
                "visible-label rows must be a non-empty proper subset of table rows"
            )

    @property
    def dataset_id(self) -> str:
        """Return the source dataset for the serialized table."""

        return self.table.source.dataset_id

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the serialized table's ordered feature columns."""

        return self.table.source.columns

    @property
    def query_row_ids(self) -> tuple[str, ...]:
        """Return target-hidden source rows in serialized table order."""

        visible = (
            set(self.visible_labels.source.row_ids) if self.visible_labels else set()
        )
        return tuple(
            row_id for row_id in self.table.source.row_ids if row_id not in visible
        )


@dataclass(frozen=True)
class ICLPredictionRequest:
    """Inference-only row demonstrations followed by target-hidden queries."""

    request_id: str
    protocol: ICLProtocol
    dataset_split: str
    task_type: str
    task_family: TaskFamily
    target_column: str
    query: MaterializedTableSlice
    shots: int
    class_labels: tuple[Any, ...] = ()
    demonstrations: MaterializedTableSlice | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "class_labels",
            tuple(normalize_value(value) for value in self.class_labels),
        )
        _validate_prediction_label_space(self.task_family, self.class_labels)
        if self.protocol not in {"zero_shot_icl", "few_shot_icl"}:
            raise ValueError(f"unknown ICL protocol: {self.protocol}")
        if self.shots < 0:
            raise ValueError("shot count cannot be negative")
        if self.target_column in self.query.source.columns:
            raise ValueError("target column leaked into in-context query")
        if self.protocol == "zero_shot_icl":
            if self.shots != 0:
                raise ValueError("zero-shot ICL must use zero shots")
            if self.demonstrations is not None:
                raise ValueError("zero-shot requests cannot contain demonstrations")
            return
        if self.shots == 0:
            raise ValueError("few-shot ICL requires a positive shot count")
        if self.demonstrations is None:
            raise ValueError("few-shot requests require demonstrations")
        if len(self.demonstrations.rows) != self.shots:
            raise ValueError("demonstration count does not match the shot count")
        if self.demonstrations.source.dataset_id != self.query.source.dataset_id:
            raise ValueError("demonstrations and query must come from the same dataset")
        expected = (*self.query.source.columns, self.target_column)
        if self.demonstrations.source.columns != expected:
            raise ValueError(
                "demonstration columns must be query features followed by the target"
            )

    @property
    def dataset_id(self) -> str:
        """Return the source dataset shared by demonstrations and query."""

        return self.query.source.dataset_id

    @property
    def feature_columns(self) -> tuple[str, ...]:
        """Return the ordered query feature columns."""

        return self.query.source.columns


@dataclass(frozen=True)
class PredictionGold:
    """Local evaluation targets kept outside the rendered model request."""

    request_id: str
    query_targets: tuple[Any, ...]


@dataclass(frozen=True)
class SerializedTablePredictionExample:
    """A serialized-table request paired with separate evaluation targets."""

    request: SerializedTablePredictionRequest
    gold: PredictionGold
    support: SupportLevel | None = None

    def __post_init__(self) -> None:
        if self.gold.request_id != self.request.request_id:
            raise ValueError("request and gold IDs differ")
        if len(self.gold.query_targets) != len(self.request.query_row_ids):
            raise ValueError("gold count does not match target-hidden table rows")


@dataclass(frozen=True)
class ICLPredictionExample:
    """A zero/few-shot request paired with separate evaluation targets."""

    request: ICLPredictionRequest
    gold: PredictionGold
    episode_id: str
    shots: int
    support: SupportLevel | None = None

    def __post_init__(self) -> None:
        if self.shots != self.request.shots:
            raise ValueError("example and request shot counts differ")
        if self.gold.request_id != self.request.request_id:
            raise ValueError("request and gold IDs differ")
        if len(self.gold.query_targets) != len(self.request.query.rows):
            raise ValueError("gold count does not match ICL query rows")


@dataclass(frozen=True)
class RenderedPrediction:
    """A deterministic text view of an input-only prediction request."""

    request_id: str
    view: PredictionView
    serialization_version: str
    input_text: str
    query_aliases: dict[str, str]


def _validate_prediction_label_space(
    task_family: TaskFamily,
    class_labels: tuple[Any, ...],
) -> None:
    if task_family == "regression":
        if class_labels:
            raise ValueError("regression requests cannot define class labels")
        return
    if not class_labels:
        raise ValueError("classification requests require allowed class labels")
    keys = [canonical_json(value) for value in class_labels]
    if "null" in keys:
        raise ValueError("classification labels cannot contain missing values")
    if len(keys) != len(set(keys)):
        raise ValueError("classification labels must be unique")
