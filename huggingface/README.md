---
pretty_name: "TableSuite-1K: A Comprehensive Benchmark for Tabular Prediction, Grounding, and Question Answering"
license: other
tags:
  - openml
  - tabular
  - in-context-learning
  - semantic-grounding
  - table-question-answering
configs:
  - config_name: datasets
    data_files:
      - split: train
        path: datasets/train/*.parquet
      - split: validation
        path: datasets/validation/*.parquet
      - split: test
        path: datasets/test/*.parquet
  - config_name: table_prediction_tasks
    data_files:
      - split: train
        path: table_prediction_tasks/train/*.parquet
      - split: validation
        path: table_prediction_tasks/validation/*.parquet
      - split: test
        path: table_prediction_tasks/test/*.parquet
  - config_name: prediction_episodes
    data_files:
      - split: train
        path: prediction_episodes/train/*.parquet
      - split: validation
        path: prediction_episodes/validation/*.parquet
      - split: test
        path: prediction_episodes/test/*.parquet
  - config_name: grounding_tasks
    data_files:
      - split: train
        path: grounding_tasks/train/*.parquet
      - split: validation
        path: grounding_tasks/validation/*.parquet
      - split: test
        path: grounding_tasks/test/*.parquet
  - config_name: cell_grounding
    data_files:
      - split: train
        path: tasks/cell_grounding/train/*.parquet
      - split: validation
        path: tasks/cell_grounding/validation/*.parquet
      - split: episode_test
        path: tasks/cell_grounding/episode_test/*.parquet
      - split: dataset_test
        path: tasks/cell_grounding/dataset_test/*.parquet
      - split: template_test
        path: tasks/cell_grounding/template_test/*.parquet
  - config_name: table_question_answering
    data_files:
      - split: train
        path: tasks/table_question_answering/train/*.parquet
      - split: validation
        path: tasks/table_question_answering/validation/*.parquet
      - split: episode_test
        path: tasks/table_question_answering/episode_test/*.parquet
      - split: dataset_test
        path: tasks/table_question_answering/dataset_test/*.parquet
      - split: template_test
        path: tasks/table_question_answering/template_test/*.parquet
      - split: composition_test
        path: tasks/table_question_answering/composition_test/*.parquet
---

# TableSuite-1K

**A comprehensive benchmark for tabular prediction, in-context learning, cell
grounding, and table question answering over 1,000 OpenML-referenced tables.**

This repository contains a value-free benchmark catalog over 1,000
OpenML-referenced datasets. It defines inference-only zero/few-shot ICL and
zero/partial-label serialized-table interfaces plus official cell-grounding
and programmatic table-QA tasks. It does **not** redistribute the referenced
source tables.

## Configurations

| Configuration | Unit | Contains source values? |
| --- | --- | --- |
| `datasets` | dataset identity and source/task contract | No |
| `table_prediction_tasks` | zero-label table target and metric contract | No |
| `prediction_episodes` | fixed support/query references and shot count | No |
| `grounding_tasks` | eligible columns and sampler contract | No |
| `cell_grounding` | contextual source-cell evaluation plans | No |
| `table_question_answering` | typed table operations and source slices | No |

The two executable task configurations store source references and operations,
not copied values, rendered questions, or gold answers. Runtime rendering and
programmatic scoring are deterministic. No LLM authors the gold data.

All prediction evaluation is parameter-update-free. The companion package
exposes four explicit protocols: `zero_shot_icl`, `few_shot_icl`,
`zero_label_serialized_table`, and `partially_labeled_serialized_table`.
Zero-label table prediction renders features only:

```text
Predict "loan_status" for every row.

| row_id | age | income |
| --- | --- | --- |
| r0 | 35 | 60000 |
| r1 | 52 | 30000 |
```

In-context episodes use a different row-example interface:

```text
Target: loan_status

Row A: age=35, income=60000 -> approved
Query q0: age=44, income=50000 -> ?
```

Positive shot counts expose frozen demonstrations. Zero-shot removes the
demonstrations while retaining a source-validated frozen query set. The same
frozen support/query references can instead be rendered as one partially
labelled table, with support targets visible and query targets replaced by
`?`. That protocol predicts all remaining eligible rows by default; an episode
scope is available for an exact serialization comparison with few-shot ICL.
None of these protocols updates model parameters.

```text
Predict "loan_status" for rows where the target is masked.

| row_id | age | income | loan_status |
| --- | --- | --- | --- |
| r0 | 35 | 60000 | approved |
| r1 | 52 | 30000 | denied |
| r2 | 44 | 50000 | ? |
```

## Evaluation Splits

`episode_test` uses disjoint rows from training dataset clusters;
`dataset_test` uses held-out dataset clusters; `template_test` uses disjoint
rows and held-out wording; and QA `composition_test` uses disjoint rows with a
filter-then-argmax operation. These axes are reported separately.

## Source Data and Licensing

Source tables remain hosted by OpenML and retain their individual upstream
licenses. Fields such as `license_claim` reproduce upstream provenance
metadata; they do not grant rights from this repository. The repository-level
`other` designation reflects those heterogeneous terms. Benchmark code is
licensed separately in its GitHub repository.

TableSuite-1K is an independent research benchmark and is not affiliated with
or endorsed by OpenML.

## Usage

```python
from tablesuite import load_task

task = load_task(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    split="dataset_test",
    source="openml-parquet",
)
example = task[0]
score = task.score(example.id, model(example.prompt))
```

Inspect the value-free specifications directly:

```python
from datasets import load_dataset

items = load_dataset(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    split="dataset_test",
)
```

Download only the selected OpenML sources:

```bash
tablesuite fetch-openml \
  --reference 'Lester1996/TableSuite-1K' \
  --output openml-parquet \
  --dataset-split validation \
  --max-datasets 10 \
  --accept-source-terms
```

See the GitHub repository for protocol definitions, deterministic rendering,
evaluation rules, and the source-data policy.

The companion package represents arbitrary rows and subtables with
`TableSlice(dataset_id, row_ids, columns)`. Collections of slices can span
multiple selected datasets while preserving each source schema.
