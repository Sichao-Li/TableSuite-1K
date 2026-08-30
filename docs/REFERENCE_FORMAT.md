# Hugging Face Reference Format

TableSuite-1K v2.1 publishes five value-free Parquet configurations. `datasets`
is the canonical metadata table; every other configuration joins to it by
`dataset_id`.

No configuration contains feature values, targets, support labels, rendered
questions, gold answers, model outputs, embeddings, or checkpoints.

## Catalog Configurations

### `datasets`

One row per OpenML reference:

```text
dataset_id, dataset_split, openml_data_id, openml_url, dataset_name
task_type, target_column, feature_columns, target_transform
excluded_feature_columns, semantic_columns, source_adaptation_rationale
n_rows, n_features, n_classes, dedup_cluster_id, openml_license_claim
```

`dataset_split` and `dedup_cluster_id` define duplicate-aware transfer.
`semantic_columns` is the ordered subset eligible for grounding and QA after
target and identifier exclusion. Keeping it beside the canonical schema
preserves the audited semantic contract without a separate sampler config.

### `table_prediction_tasks`

```text
dataset_id, primary_metrics
```

Task type, target, and features are joined from `datasets`.

### `prediction_episodes`

```text
episode_id, dataset_id, episode_split
support_row_ids, query_row_ids, shots
```

The rows freeze query anchors and retain v2 fixed-shot reproduction. The v2.1
percentage interface derives one deterministic non-query support ordering from
the local source table; labels and values remain local.

## Official Evaluation Configurations

`table_grounding` and `table_question_answering` share this compact schema:

```text
item_id, dataset_id, evaluation_split, render_seed, schema_language
source_row_ids, source_columns, operation, operation_arguments
answer_type, absolute_tolerance, relative_tolerance, template_split
```

### `table_grounding`

`operation` is `cell_lookup`, `row_lookup`, `column_values`, `distinct_values`,
or `value_counts`. All rows and columns belong to the displayed source slice.

### `table_question_answering`

`operation` is `aggregate`, `argmax_lookup`, or `filtered_argmax_lookup`.
Named nullable arguments specify aggregation, lookup, filter, maximize, and
return columns. A filtered operation stores a source row reference for the
filter value rather than copying that value into the plan.

## Runtime Materialization

Local source files use:

```text
<source-root>/<openml_data_id>.parquet
```

The resolver verifies row count, feature columns, and target column and applies
declared source adaptations. `TableSlice(dataset_id, row_ids, columns)` is the
internal source-addressing primitive. Task constructors enforce target
visibility before rendering.

## Generated Bundles

`tablesuite generate` writes:

```text
generated-task/
  generation.json
  plans.jsonl
```

The manifest records task, split, datasets, seed, scale, versions, eligibility
shortfalls, and deterministic fingerprints. Generated bundles are value-free
and must not be mixed with frozen official evaluation results.

## Versioning

The Hugging Face tag and Python package tag are both `v2.1.0`. Semantic task
records remain schema `2.0`; prediction percentage scheduling and context
budget reports are runtime protocol additions in package v2.1.
