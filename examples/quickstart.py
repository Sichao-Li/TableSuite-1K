"""Load and inspect one official Hugging Face task example."""

from __future__ import annotations

import argparse

from tablesuite import load_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("source")
    parser.add_argument(
        "--name",
        default="table_question_answering",
        choices=[
            "table_grounding",
            "table_question_answering",
        ],
    )
    parser.add_argument("--split", default="dataset_test")
    parser.add_argument(
        "--dataset-id",
        action="append",
        default=[],
        help="Limit loading to one or more already materialized datasets",
    )
    args = parser.parse_args()

    task = load_task(
        args.reference,
        args.name,
        split=args.split,
        source=args.source,
        dataset_ids=args.dataset_id,
    )
    print(task.summary())
    print()
    print(task[0].prompt)


if __name__ == "__main__":
    main()
