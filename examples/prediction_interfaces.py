"""Preview percentage-controlled ICL and serialized-table prediction."""

from __future__ import annotations

import argparse

from tablesuite import (
    TableSuite,
    render_icl_prediction,
    render_serialized_table_prediction,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("source")
    parser.add_argument("dataset_id")
    parser.add_argument("--support", type=float, action="append")
    args = parser.parse_args()

    suite = TableSuite.open(args.reference, source=args.source)
    common = {
        "support": tuple(args.support or (0.1,)),
        "dataset_ids": (args.dataset_id,),
        "max_episodes_per_dataset": 1,
    }
    for protocol in ("icl", "serialized_table"):
        print(f"\n## {protocol}\n")
        for example in suite.prediction(protocol, **common):
            renderer = (
                render_icl_prediction
                if protocol == "icl"
                else render_serialized_table_prediction
            )
            print(example.support)
            print(renderer(example.request).input_text)
            print("\n---\n")


if __name__ == "__main__":
    main()
