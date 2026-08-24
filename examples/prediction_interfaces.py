"""Preview the four inference-only prediction protocols."""

from __future__ import annotations

import argparse

from tablesuite import (
    Benchmark,
    Selection,
    render_icl_prediction,
    render_serialized_table_prediction,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("source")
    parser.add_argument("dataset_id")
    parser.add_argument("--rows-per-table", type=int, default=16)
    args = parser.parse_args()

    benchmark = Benchmark.from_path(args.reference, args.source)
    zero_label = benchmark.select(
        Selection(
            tasks=("zero_label_serialized_table",),
            dataset_ids=(args.dataset_id,),
        )
    )
    zero_label_example = next(
        zero_label.zero_label_serialized_table(
            rows_per_table=args.rows_per_table
        )
    )
    print(render_serialized_table_prediction(zero_label_example.request).input_text)

    episode_protocols = benchmark.select(
        Selection(
            tasks=(
                "zero_shot_icl",
                "few_shot_icl",
                "partially_labeled_serialized_table",
            ),
            dataset_ids=(args.dataset_id,),
            shots=(4,),
        )
    )
    for example in (
        next(episode_protocols.zero_shot_icl()),
        next(episode_protocols.few_shot_icl()),
    ):
        print("\n---\n")
        print(render_icl_prediction(example.request).input_text)
    print("\n---\n")
    partially_labeled = next(
        episode_protocols.partially_labeled_serialized_table()
    )
    print(render_serialized_table_prediction(partially_labeled.request).input_text)


if __name__ == "__main__":
    main()
