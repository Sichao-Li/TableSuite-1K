# Benchmark Protocols

TableSuite-1K separates three capabilities and evaluates each against an
explicit source-grounded contract. Model training, prompting, and decoding are
left to the submitter unless a protocol states otherwise.

## Table Prediction

Prediction is inference-only: an official request permits no fitting,
fine-tuning, or per-dataset parameter update. Evaluation targets live in
`PredictionGold` and cannot be rendered from the request object.

| Interface | Input | Visible row labels | Query rows |
| --- | --- | --- | --- |
| `icl` | labelled row demonstrations | selected support prefix | frozen |
| `serialized_table` | one table with labelled support and masked queries | selected support prefix | frozen |

Every request states the task family. Classification requests also state the
complete output vocabulary; this is target-schema metadata, not a row-level
label assignment.

### Percentage Support

The official capacity schedule is `0/10/30/50/70/90/100%`. For `N` eligible
non-query rows, zero exposes no labels and positive fraction `p` exposes:

```text
min(N, max(1, ceil(p * N)))
```

Each query has one deterministic support ordering. Levels are nested prefixes,
and both interfaces use the same rows. Classification ordering is
label-stratified; regression ordering is quantile-stratified. Query rows are
never support rows.

The legacy v2.0 fixed-4/16/32-shot protocols remain available only through
`TableSuite.fixed_prediction()` for exact reproduction.

### Context-Budgeted Evaluation

Percentage support measures a model's use of available supervision. Context
limits are a separate systems constraint. To evaluate a model fairly:

1. create a prediction dataset with one maximum support fraction;
2. call `fit_context` with the model's actual prompt token counter;
3. include system text and chat-template tokens in that counter;
4. report the resulting `ContextBudgetReport` with task metrics.

The fitter finds the largest nested support prefix under the budget. It records
requested/realized support, row count, prompt tokens, and exclusions. It never
silently truncates. A request whose zero-support prompt is too long is excluded
with reason `zero_support_exceeds_prompt_budget`.

### Prediction Metrics

Structured predictions are keyed by request ID and follow query-row order (or
are mappings keyed by query row ID). The dependency-free evaluator reports:

- classification: dataset-macro accuracy, balanced accuracy, macro-F1, and
  dedup-cluster-macro balanced accuracy;
- regression: dataset-macro MAE, RMSE, normalized MAE/RMSE, R-squared, and
  dedup-cluster-macro normalized MAE;
- both: requested/scored records, datasets, and response coverage.

Results are reported separately for every support level and interface.

## Provided-Table Grounding

Grounding asks for facts or simple comprehension over only the displayed table.
The official operations are:

```text
cell_lookup
row_lookup
column_values
distinct_values
value_counts
```

Static and runtime audits reject operations or evidence outside the displayed
slice. Literal source headers are used in v2.1. Paraphrased or mapped schema
language requires a curated mapping and is not claimed by this release.

## Table Question Answering

QA applies typed operations to controlled 4/8/16-row subtables:

```text
count, sum, mean, min, max
argmax lookup
filter then argmax lookup
```

The operation plan and source references are frozen. Wording and gold are
materialized deterministically at access time. Gold is never LLM-authored.

## Transfer Splits

`dataset_split` is assigned by `dedup_cluster_id` to prevent duplicate-related
tables from crossing dataset partitions. Official semantic splits additionally
separate source rows, templates, and operation composition:

| Split | Held-out factor |
| --- | --- |
| `validation` | validation clusters |
| `episode_test` | rows within training clusters |
| `dataset_test` | dataset clusters |
| `template_test` | rows and wording |
| `composition_test` | composed QA operation |

## Required Result Metadata

Published results should include the package and dataset revisions, model and
tokenizer revisions, task/interface, split, seed, serialization, requested and
realized support, prompt-token budget, context coverage, response coverage,
metric aggregation, and failure counts. Scores without coverage are incomplete.
