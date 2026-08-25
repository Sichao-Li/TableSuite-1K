# Source Data Policy

## Distribution Boundary

The public Hugging Face repository is a benchmark-definition package. It
contains OpenML identities, schemas, targets, fixed partitions, episode row
references, and grounding sampler contracts. It does not contain source table
values, labels, rendered facts, predictions, embeddings, or model weights.

OpenML remains the source-data distributor. A user who materializes a table is
responsible for its upstream terms. TableSuite-1K's code or benchmark-metadata
license does not relicense any referenced table. An OpenML metadata value such as
`Public`, `CC BY 4.0`, or `ODbL` is recorded as `openml_license_claim` and is
not itself a legal conclusion by this project.

TableSuite-1K is an independent research benchmark and is not affiliated with
or endorsed by OpenML.

This follows the task-suite pattern used by OpenML benchmark suites and
TabArena: standard tasks remain tied to versioned OpenML sources, while the
benchmark publishes its protocol and reproducibility artifacts separately.

- OpenML benchmark suites: <https://docs.openml.org/benchmark/>
- OpenML terms: <https://docs.openml.org/intro/terms/>
- TabArena: <https://github.com/autogluon/tabarena>

## Reference Suite

The public suite contains 1,000 value-free OpenML references. Membership means
that a source identity, schema, target, task protocol, and split are recorded
reproducibly. It does not assert redistribution clearance or transfer the
upstream table's license to benchmark users.

## User-Directed Materialization

The package downloads only datasets explicitly bounded by `--dataset-id`,
`--max-datasets`, or `--all-datasets`. Users must acknowledge source terms. The
download is written to a local directory as `<openml_data_id>.parquet`, together
with `SOURCE_NOTICES.json`. These files are local working data and must not be
uploaded as part of the benchmark reference package without a separate rights
review.

## Publication Checklist

Before publishing a reference revision:

1. load all six Hugging Face configurations in a fresh cache;
2. verify that no source-value or model-artifact fields are present;
3. report counts by task family and dataset split;
4. confirm that the release contains exactly the intended OpenML IDs;
5. pin the Hugging Face revision used by every reported experiment;
6. keep code licensing and upstream data licensing visibly separate.
