# Hugging Face Reference Format

TableSuite-1K v1.3.0 contains six value-free Parquet configurations. The
`datasets` configuration is the canonical metadata table; every other
configuration joins to it by `dataset_id`.

No configuration contains feature values, targets, support labels, rendered
questions, gold answers, model outputs, embeddings, or checkpoints.

## Catalog Configurations

### `datasets`

One row per referenced OpenML dataset:

```text
dataset_id
dataset_split
openml_data_id
openml_url
dataset_name
task_type
target_column
feature_columns
target_transform
excluded_feature_columns
source_adaptation_rationale
n_rows
n_features
n_classes
dedup_cluster_id
openml_license_claim
```

`dataset_split` and `dedup_cluster_id` define the duplicate-aware transfer
partition. `openml_license_claim` records upstream metadata and is not a
license granted by TableSuite-1K.

### `table_prediction_tasks`

One row per prediction-eligible dataset:

```text
dataset_id
primary_metrics
```

The configuration name defines the zero-label serialized-table protocol. The
target, task type, features, and source identity come from `datasets`.

### `prediction_episodes`

One frozen support/query episode per row:

```text
episode_id
dataset_id
episode_split
support_row_ids
query_row_ids
shots
```

These records support few-shot ICL and partially labelled serialized-table
prediction. Zero-shot ICL reuses the query references without exposing support
labels.

### `grounding_tasks`

One row per grounding-eligible dataset:

```text
dataset_id
eligible_columns
excluded_identifier_columns
max_cells
```

The package derives deterministic source-cell facts from these fields. The
registered target is never an eligible grounding column.

## Official Evaluation Configurations

The official tasks freeze semantic operations and source references. Wording
and gold are generated from the referenced source table only when an item is
loaded.

### `cell_grounding`

```text
item_id
dataset_id
evaluation_split
render_seed
source_row_id
context_columns
answer_column
answer_type
absolute_tolerance
relative_tolerance
template_split
```

Each item shows one contextual row and asks for the exact value at
`source_row_id` and `answer_column`.

### `table_question_answering`

```text
item_id
dataset_id
evaluation_split
render_seed
source_row_ids
source_columns
operation
operation_arguments
answer_type
absolute_tolerance
relative_tolerance
template_split
```

`operation` is one of `aggregate`, `argmax_lookup`, or
`filtered_argmax_lookup`. `operation_arguments` is a named nullable struct:

```text
aggregation
column
filter_column
filter_value_row_id
maximize_column
return_column
```

Only fields required by the selected operation are populated. For a filtered
operation, `filter_value_row_id` references a source row; the filter value
itself is not copied into the public plan.

## Why The Rows Are Compact

Earlier releases embedded internal plan objects in every task row. v1.3.0
keeps only item-specific public fields. Dataset partition, deduplication
cluster, OpenML identity, and schema are joined from `datasets`; operation and
renderer versions are fixed by the tagged companion package. `render_seed` is
retained because it is an item-authoring choice and affects deterministic
wording.

`tablesuite.load_task` accepts both the compact v1.3.0 rows and legacy v1.2.1
nested rows. Users do not need to reconstruct internal plans themselves.

## Locally Generated Bundles

`tablesuite generate` writes a deliberately separate two-file bundle:

```text
generated-task/
  generation.json
  plans.jsonl
```

`generation.json` records the benchmark reference, selected datasets, task,
split, seed, scale, versions, eligibility shortfalls, and deterministic
fingerprints. `plans.jsonl` stores value-free semantic plans. It never stores
rendered questions, source values, gold answers, model responses, or metrics.

Generated bundles are loaded with `load_generated_task` and must not be mixed
with the frozen Hugging Face evaluation configurations.

## Source Resolution

Local source files use:

```text
<source-root>/<openml_data_id>.parquet
```

The resolver checks row count, feature columns, and target column, then applies
declared source adaptations deterministically. `tablesuite fetch-openml`
materializes only an explicit bounded selection and writes local
`SOURCE_NOTICES.json`; source rows are never added to the reference release.

## Rows And Subtables

`TableSlice(dataset_id, row_ids, columns)` is the package's source-addressing
primitive. One row ID represents a row; multiple row IDs and columns represent
a subtable. A collection of slices may span datasets, but one slice never
merges incompatible schemas.

## Reproducible Selection

`Catalog.select` resolves dataset and candidate-episode filters.
`Benchmark.select` then checks source-target eligibility and writes a
`SelectionManifest` containing the reference identity, selection parameters,
dataset IDs, and executable episode IDs.
