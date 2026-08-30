from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tablesuite._cli import main


def test_cli_module_entrypoint(benchmark_fixture: tuple[Path, Path]) -> None:
    reference, _ = benchmark_fixture
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tablesuite._cli",
            "info",
            "--reference",
            str(reference),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout)["datasets"] == 2


def test_cli_info_and_prediction_preview(
    benchmark_fixture: tuple[Path, Path],
    capsys,
) -> None:
    reference, source = benchmark_fixture
    main(["info", "--reference", str(reference)])
    assert json.loads(capsys.readouterr().out)["datasets"] == 2

    main(
        [
            "prediction",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--protocol",
            "icl",
            "--support",
            "0.5",
            "--dataset-id",
            "openml_1",
            "--max-episodes-per-dataset",
            "1",
            "--limit",
            "1",
        ]
    )
    output = capsys.readouterr().out
    assert "Row A: Age=" in output
    assert "Query q0:" in output

    main(
        [
            "prediction",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--protocol",
            "serialized_table",
            "--support",
            "0.5",
            "--dataset-id",
            "openml_1",
            "--max-episodes-per-dataset",
            "1",
            "--limit",
            "1",
        ]
    )
    output = capsys.readouterr().out
    assert 'Predict "Default" for rows where the target is masked.' in output
    assert "| Default |" in output
    assert "| ? |" in output


def test_cli_bounds_openml_download(
    benchmark_fixture: tuple[Path, Path], tmp_path: Path
) -> None:
    reference, _ = benchmark_fixture

    with pytest.raises(SystemExit, match="bound the download"):
        main(
            [
                "fetch-openml",
                "--reference",
                str(reference),
                "--output",
                str(tmp_path / "source"),
                "--accept-source-terms",
            ]
        )
