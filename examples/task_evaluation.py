"""Load one official task split and evaluate model responses."""

from __future__ import annotations

import argparse
import json
from itertools import islice

from tablesuite import load_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="Hugging Face dataset ID or local snapshot")
    parser.add_argument("source", help="Directory containing OpenML Parquet sources")
    parser.add_argument(
        "name",
        choices=[
            "table_grounding",
            "table_question_answering",
        ],
    )
    parser.add_argument("split")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--dataset-id", action="append", default=[])
    args = parser.parse_args()

    task = load_task(
        args.reference,
        args.name,
        split=args.split,
        source=args.source,
        dataset_ids=args.dataset_id,
    )
    predictions: dict[str, str] = {}
    for example in islice(task, args.limit):
        print(example.prompt)
        predictions[example.id] = input("\nAnswer: ")
        print("\n---\n")

    report = task.evaluate(predictions, allow_partial=True)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
