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


def test_cli_info_select_and_preview(
    benchmark_fixture: tuple[Path, Path],
    tmp_path: Path,
    capsys,
) -> None:
    reference, source = benchmark_fixture
    main(["info", "--reference", str(reference)])
    assert json.loads(capsys.readouterr().out)["datasets"] == 2

    manifest = tmp_path / "selection.json"
    main(
        [
            "select",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--task",
            "few_shot_icl",
            "--output",
            str(manifest),
        ]
    )
    assert manifest.is_file()
    assert json.loads(capsys.readouterr().out)["eligible_episodes"] == 1

    main(
        [
            "preview",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--task",
            "zero_label_serialized_table",
            "--dataset-id",
            "openml_1",
            "--view",
            "key_value",
            "--limit",
            "1",
        ]
    )
    output = capsys.readouterr().out
    assert 'Predict "Default" for every row.' in output
    assert "Age =" in output

    main(
        [
            "preview",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--task",
            "partially_labeled_serialized_table",
            "--dataset-id",
            "openml_1",
            "--shots",
            "4",
            "--limit",
            "1",
        ]
    )
    output = capsys.readouterr().out
    assert 'Predict "Default" for rows where the target is masked.' in output
    assert "| r0 | 20 | 100 | 0 |" in output
    assert "| r4 | 24 | 500 | ? |" in output

    main(
        [
            "preview",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--task",
            "grounding",
            "--dataset-id",
            "openml_1",
            "--view",
            "key_value",
            "--limit",
            "1",
        ]
    )
    output = capsys.readouterr().out
    assert " = " in output
    assert '"text_views"' not in output

    main(
        [
            "table",
            "--reference",
            str(reference),
            "--source",
            str(source),
            "--dataset-id",
            "openml_1",
            "--row-id",
            "1",
            "--row-id",
            "3",
            "--column",
            "Age",
            "--column",
            "Income",
        ]
    )
    table_output = capsys.readouterr().out
    assert "| Age | Income |" in table_output
    assert "| 21 | 200 |" in table_output


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
