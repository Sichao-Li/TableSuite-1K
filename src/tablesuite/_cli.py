"""Public command-line interface for TableSuite-1K."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

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
)
from tablesuite.source import materialize_openml_sources
from tablesuite.suite import TableSuite
from tablesuite.tasks import load_task
from tablesuite.types import Selection

_SPLITS = (
    "train",
    "validation",
    "episode_test",
    "dataset_test",
    "template_test",
    "composition_test",
)


def build_parser() -> argparse.ArgumentParser:
    """Create the public command-line parser."""

    parser = argparse.ArgumentParser(
        prog="tablesuite",
        description="Inspect, materialize, and evaluate TableSuite-1K tasks.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    info = commands.add_parser("info", help="Summarize a benchmark reference")
    _add_reference_arguments(info)

    fetch = commands.add_parser(
        "fetch-openml",
        help="Materialize selected source tables from OpenML",
    )
    _add_reference_arguments(fetch)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--all-datasets", action="store_true")
    fetch.add_argument("--overwrite", action="store_true")
    fetch.add_argument(
        "--accept-source-terms",
        action="store_true",
        required=True,
        help="Confirm that OpenML and per-dataset terms were reviewed",
    )
    _add_dataset_filter_arguments(fetch)

    prediction = commands.add_parser(
        "prediction",
        help="Preview percentage-controlled table-prediction requests",
    )
    _add_reference_arguments(prediction)
    prediction.add_argument("--source", type=Path, required=True)
    prediction.add_argument(
        "--protocol",
        choices=["icl", "serialized_table"],
        required=True,
    )
    prediction.add_argument(
        "--support",
        type=float,
        action="append",
        required=True,
        help="Labelled support fraction in [0,1]; repeat for a support curve",
    )
    prediction.add_argument(
        "--view",
        choices=["json", "key_value", "markdown"],
        default="markdown",
    )
    prediction.add_argument("--max-episodes-per-dataset", type=int)
    prediction.add_argument("--limit", type=int, default=3)
    prediction.add_argument(
        "--show-targets",
        action="store_true",
        help="Print private local targets for evaluator diagnostics",
    )
    _add_dataset_filter_arguments(prediction)

    task = commands.add_parser(
        "task",
        help="Preview or score an official grounding or table-QA item",
    )
    _add_reference_arguments(task)
    task.add_argument("--source", type=Path, required=True)
    task.add_argument(
        "--name",
        choices=["table_grounding", "table_question_answering"],
        required=True,
    )
    task.add_argument("--split", choices=_SPLITS, required=True)
    task.add_argument("--item-id")
    task.add_argument("--dataset-id", action="append", default=[])
    task.add_argument("--index", type=int, default=0)
    task.add_argument("--response")
    task.add_argument("--summary", action="store_true")

    generate = commands.add_parser(
        "generate",
        help="Generate a deterministic local grounding or table-QA bundle",
    )
    _add_reference_arguments(generate)
    generate.add_argument("--source", type=Path, required=True)
    generate.add_argument(
        "--name",
        choices=["table_grounding", "table_question_answering"],
        required=True,
    )
    generate.add_argument("--split", choices=_SPLITS, default="train")
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

    release = commands.add_parser(
        "build-release",
        help="Author and audit the five-config Hugging Face release",
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run one TableSuite command."""

    args = build_parser().parse_args(argv)
    if args.command == "build-release":
        _run_build_release(args)
    elif args.command == "validate-release":
        _print_json(
            validate_huggingface_release(args.release, source_root=args.source)
        )
    elif args.command == "generate":
        _run_generate(args)
    elif args.command == "task":
        _run_task(args)
    elif args.command == "prediction":
        _run_prediction(args)
    elif args.command == "fetch-openml":
        _run_fetch(args)
    else:
        _print_json(_load_catalog(args.reference, args.revision).summary())


def _run_fetch(args: argparse.Namespace) -> None:
    if args.all_datasets and (args.dataset_id or args.max_datasets is not None):
        raise SystemExit("--all-datasets cannot be combined with dataset limits")
    if not args.all_datasets and not args.dataset_id and args.max_datasets is None:
        raise SystemExit(
            "bound the download with --dataset-id/--max-datasets, "
            "or pass --all-datasets"
        )
    catalog = _load_catalog(args.reference, args.revision)
    selected = catalog.select(
        Selection(
            tasks=(),
            dataset_ids=tuple(args.dataset_id),
            dataset_splits=tuple(args.dataset_split),
            task_families=tuple(args.task_family),
            max_datasets=args.max_datasets,
            seed=args.seed,
        )
    )
    _print_json(
        materialize_openml_sources(
            selected.datasets,
            args.output,
            accept_source_terms=args.accept_source_terms,
            overwrite=args.overwrite,
        )
    )


def _run_prediction(args: argparse.Namespace) -> None:
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    suite = TableSuite.open(
        args.reference,
        source=args.source,
        revision=args.revision,
    )
    examples = suite.prediction(
        args.protocol,
        support=tuple(args.support),
        dataset_ids=args.dataset_id,
        dataset_splits=args.dataset_split,
        task_families=args.task_family,
        max_datasets=args.max_datasets,
        max_episodes_per_dataset=args.max_episodes_per_dataset,
        seed=args.seed,
    )
    for index, example in enumerate(examples):
        if index >= args.limit:
            break
        rendered = (
            render_icl_prediction(example.request)
            if args.protocol == "icl"
            else render_serialized_table_prediction(example.request, view=args.view)
        )
        print(rendered.input_text)
        if args.show_targets:
            targets = dict(
                zip(
                    rendered.query_aliases,
                    example.gold.query_targets,
                    strict=True,
                )
            )
            print("\nEvaluation targets:")
            print(json.dumps(targets, ensure_ascii=False, sort_keys=True))
        print("\n---\n")


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
        _print_json(task.summary())
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
    _print_json(summary)


def _run_build_release(args: argparse.Namespace) -> None:
    grounding_row_sizes = (
        tuple(args.grounding_row_size) if args.grounding_row_size else (4, 8, 16)
    )
    qa_row_sizes = tuple(args.qa_row_size) if args.qa_row_size else (4, 8, 16)
    _print_json(
        build_huggingface_release(
            reference_root=args.reference,
            source_root=args.source,
            dataset_card=args.dataset_card,
            output_dir=args.output,
            config=TaskGenerationConfig(
                seed=args.seed,
                grounding_items_per_dataset=args.grounding_items_per_dataset,
                grounding_transfer_items_per_dataset=(
                    args.grounding_transfer_items_per_dataset
                ),
                qa_items_per_dataset=args.qa_items_per_dataset,
                qa_transfer_items_per_dataset=args.qa_transfer_items_per_dataset,
                min_grounding_context_columns=args.min_grounding_context_columns,
                max_grounding_context_columns=args.max_grounding_context_columns,
                min_qa_context_columns=args.min_qa_context_columns,
                max_qa_context_columns=args.max_qa_context_columns,
                grounding_row_sizes=grounding_row_sizes,
                qa_row_sizes=qa_row_sizes,
                shard_size=args.shard_size,
            ),
        )
    )


def _add_reference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reference",
        required=True,
        help="Downloaded release directory or Hugging Face repository ID",
    )
    parser.add_argument("--revision")


def _add_dataset_filter_arguments(parser: argparse.ArgumentParser) -> None:
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


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
