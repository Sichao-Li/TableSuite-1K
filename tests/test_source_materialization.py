from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tablesuite import Catalog, materialize_openml_sources


def test_openml_materialization_is_bounded_and_records_source_notices(
    benchmark_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    reference, _ = benchmark_fixture
    dataset = Catalog.from_path(reference).datasets[0]
    destination = tmp_path / "openml"
    calls: list[str] = []

    def fetcher(spec):
        calls.append(spec.dataset_id)
        return pa.table(
            {
                "Age": [20, 21, 22, 23, 24, 25, 26, 27],
                "Income": [100, 200, 300, 400, 500, 600, 700, 800],
                "Default": [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

    summary = materialize_openml_sources(
        [dataset],
        destination,
        accept_source_terms=True,
        fetcher=fetcher,
    )

    assert calls == ["openml_1"]
    assert summary["downloaded_datasets"] == ["openml_1"]
    assert pq.read_table(destination / "1.parquet").num_rows == 8
    notices = json.loads((destination / "SOURCE_NOTICES.json").read_text())
    assert notices["datasets"][0]["openml_data_id"] == "1"

    reused = materialize_openml_sources(
        [dataset],
        destination,
        accept_source_terms=True,
        fetcher=fetcher,
    )
    assert calls == ["openml_1"]
    assert reused["reused_datasets"] == ["openml_1"]


def test_openml_materialization_requires_source_terms(
    benchmark_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    reference, _ = benchmark_fixture
    dataset = Catalog.from_path(reference).datasets[0]

    with pytest.raises(ValueError, match="source terms must be accepted"):
        materialize_openml_sources(
            [dataset],
            tmp_path / "openml",
            accept_source_terms=False,
        )
