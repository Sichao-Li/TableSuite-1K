"""Generate, save, and reload a small deterministic task bundle."""

from __future__ import annotations

import argparse
import json

from tablesuite import generate_task, load_generated_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", help="Hugging Face dataset ID or local snapshot")
    parser.add_argument("source", help="Directory containing OpenML Parquet sources")
    parser.add_argument("output", help="New directory for the value-free task bundle")
    parser.add_argument(
        "--name",
        choices=["cell_grounding", "table_question_answering"],
        default="table_question_answering",
    )
    parser.add_argument("--dataset-id", action="append", default=[])
    parser.add_argument("--items-per-dataset", type=int, default=12)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    task = generate_task(
        args.reference,
        args.name,
        source=args.source,
        dataset_ids=args.dataset_id,
        items_per_dataset=args.items_per_dataset,
        max_items=args.max_items,
        seed=args.seed,
    )
    task.save(args.output)
    restored = load_generated_task(args.output, source=args.source)
    print(json.dumps(restored.manifest.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
