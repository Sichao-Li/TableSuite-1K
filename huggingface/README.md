---
pretty_name: "TableSuite-1K: Predictive and Language-Grounded Tabular Intelligence"
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
  - config_name: table_grounding
    data_files:
      - split: train
        path: tasks/table_grounding/train/*.parquet
      - split: validation
        path: tasks/table_grounding/validation/*.parquet
      - split: episode_test
        path: tasks/table_grounding/episode_test/*.parquet
      - split: dataset_test
        path: tasks/table_grounding/dataset_test/*.parquet
      - split: template_test
        path: tasks/table_grounding/template_test/*.parquet
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

TableSuite-1K is a source-grounded benchmark for predictive and
language-grounded tabular intelligence over 1,000 OpenML-referenced datasets.
It provides source metadata and value-free task plans, not copied OpenML
tables, model outputs, or checkpoints.

## Capabilities

| Configuration | Purpose |
| --- | --- |
| `datasets` | source, schema, target, split, and license metadata |
| `table_prediction_tasks` | prediction eligibility and metrics |
| `prediction_episodes` | frozen 4/16/32-shot support/query references |
| `grounding_tasks` | eligible non-target columns |
| `table_grounding` | provided-table cell/row/column operations |
| `table_question_answering` | programmatic aggregate and lookup operations |

Questions and gold answers are generated deterministically from the displayed
source slice at access time. No LLM authors benchmark gold.

## Quickstart

```bash
pip install \
  'tablesuite[local,hf,openml] @ git+https://github.com/Sichao-Li/TableSuite-1K.git@v2.0.0'

tablesuite fetch-openml \
  --reference Lester1996/TableSuite-1K \
  --revision v2.0.0 \
  --output openml-parquet \
  --dataset-id openml_45069 \
  --accept-source-terms
```

```python
from tablesuite import load_task

task = load_task(
    "Lester1996/TableSuite-1K",
    "table_grounding",
    split="dataset_test",
    source="openml-parquet",
    revision="v2.0.0",
    dataset_ids=("openml_45069",),
)

example = task[0]
result = task.score(example.id, model(example.prompt))
```

## Protocol Summary

Prediction supports zero-shot and 4/16/32-shot row ICL, plus zero-label and
partially labelled serialized-table inference. It is inference-only.

Table grounding operates only on the table shown in the request. Official v2
operations are `cell_lookup`, `row_lookup`, `column_values`,
`distinct_values`, and `value_counts`.

Table QA uses independently balanced 4/8/16-row subtables and typed count,
sum, mean, min, max, argmax, and filter-then-argmax operations.

The v2 semantic tasks use literal source-schema wording. This release does not
claim curated cross-dataset ontology equivalence.

## Source Terms

OpenML remains the source-table distributor. Each dataset retains its upstream
terms. `openml_license_claim` is provenance metadata, not a license granted by
TableSuite-1K. The repository-level `other` designation reflects heterogeneous
source terms.

Code and complete protocol documentation:
<https://github.com/Sichao-Li/TableSuite-1K>
