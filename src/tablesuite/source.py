"""Selective access to locally downloaded OpenML Parquet tables."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tablesuite._util import normalize_value
from tablesuite.types import DatasetSpec, MaterializedTableSlice, TableSlice


class ParquetSource:
    """Resolve selected OpenML tables from ``<data_id>.parquet`` files."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._cached_dataset_id: str | None = None
        self._cached_rows: list[dict[str, Any]] | None = None

    def rows(
        self,
        dataset: DatasetSpec,
        row_ids: tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Return all or selected normalized rows from one source dataset."""

        if self._cached_dataset_id == dataset.dataset_id and self._cached_rows is not None:
            rows = self._cached_rows
        else:
            rows = self._load(dataset)
            self._cached_dataset_id = dataset.dataset_id
            self._cached_rows = rows
        if row_ids is None:
            return rows
        indices = [int(value) for value in row_ids]
        if any(index < 0 or index >= len(rows) for index in indices):
            raise IndexError(f"{dataset.dataset_id}: row reference is outside the table")
        return [rows[index] for index in indices]

    def materialize(
        self,
        dataset: DatasetSpec,
        source_slice: TableSlice,
    ) -> MaterializedTableSlice:
        """Resolve one row or subtable slice without adding implicit columns."""

        if source_slice.dataset_id != dataset.dataset_id:
            raise ValueError(
                f"slice dataset {source_slice.dataset_id!r} does not match "
                f"{dataset.dataset_id!r}"
            )
        available = {*dataset.feature_columns, dataset.target_column}
        if unknown := set(source_slice.columns) - available:
            raise ValueError(
                f"{dataset.dataset_id}: unknown slice columns: {sorted(unknown)}"
            )
        rows = self.rows(dataset, source_slice.row_ids)
        projected = tuple(
            {column: row[column] for column in source_slice.columns} for row in rows
        )
        return MaterializedTableSlice(source=source_slice, rows=projected)

    def clear(self) -> None:
        """Release cached source rows."""

        self._cached_dataset_id = None
        self._cached_rows = None

    def _load(self, dataset: DatasetSpec) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as error:
            raise RuntimeError("install tablesuite[local] to read source Parquet") from error
        path = self.root / f"{dataset.source_id}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        required = {*dataset.feature_columns, dataset.target_column}
        if missing := required - available:
            raise ValueError(f"{dataset.dataset_id}: source columns missing: {sorted(missing)}")
        if parquet.metadata.num_rows != dataset.n_rows:
            raise ValueError(
                f"{dataset.dataset_id}: expected {dataset.n_rows} rows, "
                f"found {parquet.metadata.num_rows}"
            )
        columns = [*dataset.feature_columns, dataset.target_column]
        rows = pq.read_table(path, columns=columns).to_pylist()
        for row in rows:
            for column in columns:
                row[column] = normalize_value(row[column])
            row[dataset.target_column] = _transform_target(
                row[dataset.target_column], dataset.target_transform
            )
        return rows


def _transform_target(value: Any, transform: str) -> Any:
    if transform in {"", "none"}:
        return value
    if transform == "numeric_thousands_separator":
        if value is None:
            return None
        try:
            return float(str(value).replace(",", ""))
        except ValueError as error:
            raise ValueError(f"cannot parse numeric target value {value!r}") from error
    raise ValueError(f"unsupported target transform: {transform}")


def materialize_openml_sources(
    datasets: Iterable[DatasetSpec],
    output_root: str | Path,
    *,
    accept_source_terms: bool,
    overwrite: bool = False,
    fetcher: Callable[[DatasetSpec], Any] | None = None,
) -> dict[str, Any]:
    """Download selected OpenML tables into the local Parquet source layout.

    The benchmark repository never redistributes these tables. Calling this
    function is an explicit user-directed download from OpenML, subject to each
    source dataset's terms. ``fetcher`` is injectable for deterministic tests.
    """

    if not accept_source_terms:
        raise ValueError(
            "source terms must be accepted before downloading; review each dataset's "
            "OpenML page and https://docs.openml.org/intro/terms/"
        )
    selected = sorted(datasets, key=lambda item: item.dataset_id)
    ids = [dataset.dataset_id for dataset in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("source selection contains duplicate dataset IDs")
    destination_root = Path(output_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    load = fetcher or _fetch_openml_table
    downloaded: list[str] = []
    reused: list[str] = []
    for dataset in selected:
        destination = destination_root / f"{dataset.source_id}.parquet"
        if destination.is_file() and not overwrite:
            _validate_materialized_source(destination, dataset)
            reused.append(dataset.dataset_id)
            continue
        table = load(dataset)
        _validate_source_table(table, dataset)
        _write_source_table(table, destination)
        downloaded.append(dataset.dataset_id)
    notices_path = destination_root / "SOURCE_NOTICES.json"
    _write_source_notices(notices_path, selected)
    return {
        "output_root": str(destination_root),
        "requested_datasets": len(selected),
        "downloaded_datasets": downloaded,
        "reused_datasets": reused,
        "source_notices": str(notices_path),
    }


def _fetch_openml_table(dataset: DatasetSpec) -> Any:
    try:
        import openml
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError(
            "install tablesuite[openml] to download source tables"
        ) from error
    try:
        source_id = int(dataset.source_id)
    except ValueError as error:
        raise ValueError(f"{dataset.dataset_id}: invalid OpenML data ID") from error
    source = openml.datasets.get_dataset(source_id, download_data=True)
    features, target, _, _ = source.get_data(
        dataset_format="dataframe",
        target=dataset.target_column,
    )
    if target is None:
        raise ValueError(f"{dataset.dataset_id}: OpenML returned no registered target")
    frame = features.copy()
    frame[dataset.target_column] = target
    return pa.Table.from_pandas(frame, preserve_index=False)


def _validate_source_table(table: Any, dataset: DatasetSpec) -> None:
    names = set(table.column_names)
    required = {*dataset.feature_columns, dataset.target_column}
    if missing := required - names:
        raise ValueError(
            f"{dataset.dataset_id}: downloaded source columns missing: {sorted(missing)}"
        )
    if table.num_rows != dataset.n_rows:
        raise ValueError(
            f"{dataset.dataset_id}: expected {dataset.n_rows} rows, found {table.num_rows}"
        )


def _validate_materialized_source(path: Path, dataset: DatasetSpec) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("install tablesuite[local] to inspect source Parquet") from error
    parquet = pq.ParquetFile(path)
    required = {*dataset.feature_columns, dataset.target_column}
    if missing := required - set(parquet.schema_arrow.names):
        raise ValueError(
            f"{dataset.dataset_id}: existing source columns missing: {sorted(missing)}"
        )
    if parquet.metadata.num_rows != dataset.n_rows:
        raise ValueError(
            f"{dataset.dataset_id}: expected {dataset.n_rows} rows, "
            f"found {parquet.metadata.num_rows}"
        )


def _write_source_table(table: Any, destination: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("install tablesuite[local] to write source Parquet") from error
    temporary = destination.with_suffix(".parquet.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_source_notices(path: Path, datasets: list[DatasetSpec]) -> None:
    existing: dict[str, dict[str, Any]] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing = {
            str(item["dataset_id"]): item for item in payload.get("datasets", [])
        }
    for dataset in datasets:
        existing[dataset.dataset_id] = {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.dataset_name,
            "openml_data_id": dataset.source_id,
            "openml_url": dataset.source_url,
            "openml_license_claim": dataset.license_claim,
            "openml_license_url": dataset.source_license_url,
        }
    payload = {
        "schema_version": "1.0",
        "notice": (
            "Source tables were downloaded directly from OpenML and retain their "
            "upstream licenses. TableSuite-1K does not relicense them."
        ),
        "datasets": [existing[dataset_id] for dataset_id in sorted(existing)],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
