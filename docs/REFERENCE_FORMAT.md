# Hugging Face Reference Format

The release contains six value-free configurations. Four form a compact
source/task catalog:

| Configuration | Records | Purpose |
| --- | ---: | --- |
| `datasets` | 1,000 | OpenML identity, schema, target, source adaptation, split, and license claim |
| `table_prediction_tasks` | 998 | Zero-label prediction protocol and metrics |
| `prediction_episodes` | 35,472 | Frozen support/query references and shot count |
| `grounding_tasks` | 989 | Eligible columns and deterministic sampler contract |

The catalog intentionally excludes feature values, targets, support labels,
query labels, rendered cell values, model outputs, and checkpoints.

`datasets` is the canonical metadata configuration. The other three catalog
configurations store only task-specific fields and join to it by `dataset_id`.
This avoids repeating source URL, schema, target, and license metadata in every
episode.

Published `table_prediction_tasks` records declare `protocol =
zero_label_serialized_table`, `input_interface = serialized_table`, and
`parameter_updates = false`. Published `prediction_episodes` records freeze
support/query references and `shots` for `few_shot_icl` and
`partially_labeled_serialized_table`. A `zero_shot_icl` request is derived by
removing support labels while retaining the frozen query references.

Two additional configurations contain official evaluation items:

| Configuration | Records | Purpose |
| --- | ---: | --- |
| `cell_grounding` | 50,012 | exact fact grounding |
| `table_question_answering` | 22,448 | programmatic table QA |

Each task configuration is divided into applicable evaluation splits such as
`train`, `validation`, `episode_test`, `dataset_test`, `template_test`, and
`composition_test`. A row contains an `item_id`, source references, a typed
operation, deterministic rendering policy, and scoring policy. It never stores
the rendered question or answer.

Prediction packets and integrated model outputs are not part of the first
public Hugging Face release. The `table_prediction_tasks` and
`prediction_episodes` configurations define inference-only prediction
interfaces without bundling any predictor's outputs or authorizing parameter
updates.

Each dataset record also carries its OpenML URL and upstream license claim.
`license_claim` is provenance metadata, not a license granted by TableSuite-1K.

## Source Resolution

Local source files use:

```text
<source-root>/<openml_data_id>.parquet
```

The resolver checks the expected row count, feature columns, and target column.
Declared source adaptations are applied deterministically. File checksums are
not part of the source contract.

`tablesuite fetch-openml` can create this layout for an explicit bounded
selection. It contacts OpenML directly and writes `SOURCE_NOTICES.json` locally;
downloaded source rows are never added to the reference repository.

## Selection

`Catalog.select` first resolves dataset and candidate-episode filters.
`Benchmark.select` then checks episode eligibility against source targets and
writes only executable episode IDs to `SelectionManifest`.

A selection manifest records:

```text
reference identity
selection parameters and seed
explicit dataset IDs
explicit eligible episode IDs
```

This keeps custom subsets flexible while making every reported subset exactly
reproducible.

## Rows and Subtables

`TableSlice` is the source-addressing primitive used by the companion package:

```text
dataset_id
row_ids
columns
```

A one-row slice represents a row; a multi-row, multi-column slice represents a
subtable. Every slice belongs to one dataset. An ordered collection of slices
may span multiple selected datasets without merging incompatible schemas.

Slices are source references rather than copied tables. Official task
configurations use the same primitive, while the companion package can also
create ad hoc slices from any selected datasets.
