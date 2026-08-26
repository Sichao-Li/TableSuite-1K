"""Small public CLI for inspecting and consuming TableSuite-1K."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tablesuite.benchmark import Benchmark, BenchmarkSubset
from tablesuite.catalog import Catalog
from tablesuite.generation import generate_task
from tablesuite.release import (
    TaskGenerationConfig,
    build_huggingface_release,
    validate_huggingface_release,
)
from tablesuite.rendering import (
    render_icl_prediction,
    render_serialized_table_prediction,
    render_table,
)
from tablesuite.source import ParquetSource, materialize_openml_sources
from tablesuite.tasks import load_task
from tablesuite.types import Selection, TableSlice


def build_parser() -> argparse.ArgumentParser:
    """Create the public command-line parser."""

    parser = argparse.ArgumentParser(prog="tablesuite")
    commands = parser.add_subparsers(dest="command", required=True)

    release = commands.add_parser(
        "build-release",
        help="Author and audit the six-config Hugging Face release",
    )
    release.add_argument("--reference", type=Path, required=True)
    release.add_argument("--source", type=Path, required=True)
    release.add_argument("--dataset-card", type=Path, required=True)
    release.add_argument("--output", type=Path, required=True)
    release.add_argument("--seed", type=int, default=0)
    release.add_argument("--grounding-items-per-dataset", type=int, default=30)
    release.add_argument("--grounding-transfer-items-per-dataset", type=int, default=15)
    release.add_argument("--qa-items-per-dataset", type=int, default=12)
    release.add_argument("--qa-transfer-items-per-dataset", type=int, default=6)
    release.add_argument("--min-grounding-context-columns", type=int, default=4)
    release.add_argument("--max-grounding-context-columns", type=int, default=8)
    release.add_argument("--min-qa-context-columns", type=int, default=3)
    release.add_argument("--max-qa-context-columns", type=int, default=8)
    release.add_argument("--grounding-row-size", type=int, action="append", default=[])
    release.add_argument("--qa-row-size", type=int, action="append", default=[])
    release.add_argument("--shard-size", type=int, default=50_000)

    validate = commands.add_parser(
        "validate-release",
        help="Audit a built Hugging Face release",
    )
    validate.add_argument("--release", type=Path, required=True)
    validate.add_argument("--source", type=Path)

    generate = commands.add_parser(
        "generate",
        help="Generate a deterministic local task bundle",
    )
    _add_reference_arguments(generate)
    generate.add_argument("--source", type=Path, required=True)
    generate.add_argument(
        "--name",
        choices=["table_grounding", "table_question_answering"],
        required=True,
    )
    generate.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "episode_test",
            "dataset_test",
            "template_test",
            "composition_test",
        ],
        default="train",
    )
    generate.add_argument("--dataset-id", action="append", default=[])
    generate.add_argument(
        "--task-family",
        choices=["classification", "regression"],
        action="append",
        default=[],
    )
    generate.add_argument("--max-datasets", type=int)
    generate.add_argument("--items-per-dataset", type=int)
    generate.add_argument("--max-items", type=int)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--output", type=Path, required=True)

    info = commands.add_parser("info", help="Summarize a reference catalog")
    _add_reference_arguments(info)

    fetch = commands.add_parser(
        "fetch-openml",
        help="Download selected source tables directly from OpenML",
    )
    _add_reference_arguments(fetch)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--all-datasets", action="store_true")
    fetch.add_argument("--overwrite", action="store_true")
    fetch.add_argument(
        "--accept-source-terms",
        action="store_true",
        required=True,
        help="Confirm that upstream OpenML and per-dataset terms were reviewed",
    )
    _add_dataset_filter_arguments(fetch)

    select = commands.add_parser("select", help="Resolve and save a benchmark subset")
    _add_reference_arguments(select)
    select.add_argument("--source", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    _add_selection_arguments(select)

    preview = commands.add_parser("preview", help="Print selected examples")
    _add_reference_arguments(preview)
    preview.add_argument(
        "--source", type=Path, required=True
    )
    preview.add_argument(
        "--task",
        choices=[
            "zero_shot_icl",
            "few_shot_icl",
            "zero_label_serialized_table",
            "partially_labeled_serialized_table",
            "grounding",
        ],
        required=True,
    )
    preview.add_argument(
        "--view",
        choices=["json", "key_value", "markdown"],
        default="markdown",
    )
    preview.add_argument("--limit", type=int, default=3)
    preview.add_argument(
        "--show-targets",
        action="store_true",
        help="Print local evaluation targets for diagnostics",
    )
    preview.add_argument(
        "--show-gold",
        dest="show_targets",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    preview.add_argument("--rows-per-table", type=int)
    preview.add_argument(
        "--query-scope",
        choices=["full_table", "episode"],
        default="full_table",
    )
    preview.add_argument("--query-rows-per-table", type=int)
    _add_selection_arguments(preview, include_tasks=False)

    table = commands.add_parser("table", help="Render selected rows and columns")
    _add_reference_arguments(table)
    table.add_argument("--source", type=Path, required=True)
    table.add_argument("--dataset-id", required=True)
    table.add_argument("--row-id", action="append", required=True)
    table.add_argument("--column", action="append", required=True)
    table.add_argument(
        "--view",
        choices=["json", "key_value", "markdown"],
        default="markdown",
    )

    task = commands.add_parser(
        "task",
        help="Preview or score one official Hugging Face task example",
    )
    _add_reference_arguments(task)
    task.add_argument("--source", type=Path, required=True)
    task.add_argument(
        "--name",
        choices=[
            "table_grounding",
            "table_question_answering",
        ],
        required=True,
    )
    task.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "episode_test",
            "dataset_test",
            "template_test",
            "composition_test",
        ],
        required=True,
    )
    task.add_argument("--item-id")
    task.add_argument("--dataset-id", action="append", default=[])
    task.add_argument("--index", type=int, default=0)
    task.add_argument("--response")
    task.add_argument("--summary", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one public benchmark command."""

    args = build_parser().parse_args(argv)
    if args.command == "build-release":
        _run_build_release(args)
        return
    if args.command == "validate-release":
        print(
            json.dumps(
                validate_huggingface_release(args.release, source_root=args.source),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "generate":
        _run_generate(args)
        return
    if args.command == "task":
        _run_task(args)
        return
    catalog = _load_catalog(args.reference, args.revision)
    if args.command == "info":
        print(json.dumps(catalog.summary(), indent=2, sort_keys=True))
        return

    if args.command == "fetch-openml":
        if args.all_datasets and (args.dataset_id or args.max_datasets is not None):
            raise SystemExit("--all-datasets cannot be combined with dataset limits")
        if not args.all_datasets and not args.dataset_id and args.max_datasets is None:
            raise SystemExit(
                "bound the download with --dataset-id/--max-datasets, or pass --all-datasets"
            )
        selected = catalog.select(
            _selection_from_args(args, tasks=("zero_label_serialized_table",))
        )
        summary = materialize_openml_sources(
            selected.datasets,
            args.output,
            accept_source_terms=args.accept_source_terms,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    if args.command == "table":
        subset = Benchmark(catalog, ParquetSource(args.source)).select(
            Selection(tasks=(), dataset_ids=(args.dataset_id,))
        )
        table = subset.materialize(
            TableSlice(
                dataset_id=args.dataset_id,
                row_ids=tuple(args.row_id),
                columns=tuple(args.column),
            )
        )
        print(render_table(table, view=args.view))
        return

    tasks = (args.task,) if args.command == "preview" else tuple(args.tasks)
    selection = _selection_from_args(args, tasks=tasks)
    subset = Benchmark(catalog, ParquetSource(args.source)).select(selection)
    if args.command == "select":
        subset.manifest.save(args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "datasets": len(subset.manifest.dataset_ids),
                    "eligible_episodes": len(subset.manifest.episode_ids),
                },
                indent=2,
            )
        )
        return
    _preview(
        subset,
        args.task,
        args.view,
        args.limit,
        args.show_targets,
        args.rows_per_table,
        args.query_scope,
        args.query_rows_per_table,
    )


def _run_task(args: argparse.Namespace) -> None:
    task = load_task(
        args.reference,
        args.name,
        split=args.split,
        source=args.source,
        revision=args.revision,
        dataset_ids=args.dataset_id,
    )
    if args.summary:
        print(json.dumps(task.summary(), indent=2, sort_keys=True))
        return
    item = task[args.item_id] if args.item_id is not None else task[args.index]
    print(item.prompt)
    if args.response is not None:
        print("\nScore:")
        print(
            json.dumps(
                asdict(task.score(item.id, args.response)),
                ensure_ascii=False,
                sort_keys=True,
            )
        )


def _run_build_release(args: argparse.Namespace) -> None:
    grounding_row_sizes = (
        tuple(args.grounding_row_size) if args.grounding_row_size else (4, 8, 16)
    )
    row_sizes = tuple(args.qa_row_size) if args.qa_row_size else (4, 8, 16)
    summary = build_huggingface_release(
        reference_root=args.reference,
        source_root=args.source,
        dataset_card=args.dataset_card,
        output_dir=args.output,
        config=TaskGenerationConfig(
            seed=args.seed,
            grounding_items_per_dataset=args.grounding_items_per_dataset,
            grounding_transfer_items_per_dataset=args.grounding_transfer_items_per_dataset,
            qa_items_per_dataset=args.qa_items_per_dataset,
            qa_transfer_items_per_dataset=args.qa_transfer_items_per_dataset,
            min_grounding_context_columns=args.min_grounding_context_columns,
            max_grounding_context_columns=args.max_grounding_context_columns,
            min_qa_context_columns=args.min_qa_context_columns,
            max_qa_context_columns=args.max_qa_context_columns,
            grounding_row_sizes=grounding_row_sizes,
            qa_row_sizes=row_sizes,
            shard_size=args.shard_size,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def _run_generate(args: argparse.Namespace) -> None:
    generated = generate_task(
        args.reference,
        args.name,
        source=args.source,
        split=args.split,
        revision=args.revision,
        dataset_ids=args.dataset_id,
        task_families=args.task_family,
        max_datasets=args.max_datasets,
        items_per_dataset=args.items_per_dataset,
        max_items=args.max_items,
        seed=args.seed,
    )
    generated.save(args.output)
    summary = generated.manifest.to_dict()
    summary["output"] = str(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _preview(
    subset: BenchmarkSubset,
    task: str,
    view: str,
    limit: int,
    show_targets: bool,
    rows_per_table: int | None,
    query_scope: str,
    query_rows_per_table: int | None,
) -> None:
    if limit <= 0:
        raise SystemExit("--limit must be positive")
    if task == "grounding":
        for index, fact in enumerate(subset.grounding()):
            if index >= limit:
                break
            print(fact.text_views[view])
            print()
        return
    if rows_per_table is not None and task != "zero_label_serialized_table":
        raise SystemExit("--rows-per-table applies only to zero-label tables")
    if query_rows_per_table is not None and task != "partially_labeled_serialized_table":
        raise SystemExit(
            "--query-rows-per-table applies only to partially labelled tables"
        )
    serialized = {
        "zero_label_serialized_table",
        "partially_labeled_serialized_table",
    }
    if query_scope != "full_table" and task not in serialized:
        raise SystemExit("--query-scope applies only to serialized-table protocols")
    if task == "zero_label_serialized_table":
        examples = subset.zero_label_serialized_table(
            scope=query_scope,
            rows_per_table=rows_per_table,
        )
    elif task == "partially_labeled_serialized_table":
        examples = subset.partially_labeled_serialized_table(
            query_scope=query_scope,
            query_rows_per_table=query_rows_per_table,
        )
    elif task == "zero_shot_icl":
        examples = subset.zero_shot_icl()
    else:
        examples = subset.few_shot_icl()
    for index, example in enumerate(examples):
        if index >= limit:
            break
        rendered = (
            render_serialized_table_prediction(example.request, view=view)
            if task
            in {"zero_label_serialized_table", "partially_labeled_serialized_table"}
            else render_icl_prediction(example.request)
        )
        print(rendered.input_text)
        if show_targets:
            answers = {
                alias: target
                for alias, target in zip(
                    rendered.query_aliases,
                    example.gold.query_targets,
                    strict=True,
                )
            }
            print("\nEvaluation targets:")
            print(json.dumps(answers, ensure_ascii=False, sort_keys=True))
        print("\n---\n")


def _add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference",
        required=True,
        help="Downloaded reference directory or Hugging Face repository ID",
    )
    parser.add_argument("--revision")


def _add_selection_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_tasks: bool = True,
) -> None:
    if include_tasks:
        parser.add_argument(
            "--task",
            dest="tasks",
            choices=[
                "zero_shot_icl",
                "few_shot_icl",
                "zero_label_serialized_table",
                "partially_labeled_serialized_table",
                "grounding",
            ],
            action="append",
            required=True,
        )
    _add_dataset_filter_arguments(parser)
    parser.add_argument(
        "--shots",
        type=int,
        choices=[4, 16, 32],
        action="append",
        default=[],
    )
    parser.add_argument("--max-episodes-per-dataset-per-shot", type=int)
    parser.add_argument("--max-grounding-facts-per-dataset", type=int)


def _add_dataset_filter_arguments(parser: argparse.ArgumentParser) -> None:
    """Add filters shared by selection and source materialization commands."""

    parser.add_argument("--dataset-id", action="append", default=[])
    parser.add_argument(
        "--dataset-split",
        choices=["train", "validation", "test"],
        action="append",
        default=[],
    )
    parser.add_argument(
        "--task-family",
        choices=["classification", "regression"],
        action="append",
        default=[],
    )
    parser.add_argument("--max-datasets", type=int)
    parser.add_argument("--seed", type=int, default=0)


def _load_catalog(reference: str, revision: str | None) -> Catalog:
    path = Path(reference)
    if path.is_dir():
        return Catalog.from_path(path)
    return Catalog.from_huggingface(reference, revision=revision)


def _selection_from_args(args: argparse.Namespace, *, tasks: tuple[str, ...]) -> Selection:
    values: dict[str, Any] = {
        "tasks": tasks,
        "dataset_ids": tuple(args.dataset_id),
        "dataset_splits": tuple(args.dataset_split),
        "task_families": tuple(args.task_family),
        "shots": tuple(getattr(args, "shots", ())),
        "max_datasets": args.max_datasets,
        "max_episodes_per_dataset_per_shot": getattr(
            args, "max_episodes_per_dataset_per_shot", None
        ),
        "max_grounding_facts_per_dataset": getattr(
            args, "max_grounding_facts_per_dataset", None
        ),
        "seed": args.seed,
    }
    return Selection(**values)


if __name__ == "__main__":
    main()
