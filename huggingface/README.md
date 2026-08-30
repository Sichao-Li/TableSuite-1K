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

TableSuite-1K benchmarks predictive and language-grounded tabular intelligence
over 1,000 OpenML-referenced datasets.

| Task | Input | Evaluation |
| --- | --- | --- |
| Prediction | ICL rows or a partially labelled serialized table | classification and regression |
| Table grounding | a provided table plus a lookup/comprehension question | exact displayed-table facts |
| Table QA | a provided subtable plus a typed question | programmatic operations |

This repository contains metadata and value-free task plans. It contains no
OpenML source values, labels, rendered questions, gold answers, model outputs,
embeddings, or checkpoints. OpenML remains the source-table distributor.

## Configurations

| Configuration | Purpose |
| --- | --- |
| `datasets` | source identity, schema, target, split, and provenance |
| `table_prediction_tasks` | prediction eligibility and primary metrics |
| `prediction_episodes` | frozen prediction query anchors |
| `table_grounding` | official provided-table grounding plans |
| `table_question_answering` | official programmatic QA plans |

## Quickstart

Install the matching package and materialize only the OpenML sources you need:

```bash
pip install \
  'tablesuite[local,hf,openml] @ git+https://github.com/Sichao-Li/TableSuite-1K.git@v2.1.0'

tablesuite fetch-openml \
  --reference Lester1996/TableSuite-1K \
  --revision v2.1.0 \
  --output openml-parquet \
  --dataset-id openml_45069 \
  --accept-source-terms
```

```python
from tablesuite import TableSuite

suite = TableSuite.open(
    "Lester1996/TableSuite-1K",
    source="openml-parquet",
    revision="v2.1.0",
)

task = suite.official(
    "table_grounding",
    split="dataset_test",
    dataset_ids=("openml_45069",),
)
example = task[0]
score = task.score(example.id, model(example.prompt))
```

Prediction uses the same frozen queries across interfaces and a deterministic
nested support schedule:

```python
prediction = suite.prediction(
    "icl",
    support=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
    dataset_ids=("openml_45069",),
)
```

For model context limits, `PredictionDataset.fit_context` selects the largest
support prefix that fits the actual tokenized prompt and emits an auditable
coverage report. It never silently truncates.

## Evaluation Contracts

- Prediction is inference-only; no per-dataset parameter updates are allowed.
- Query targets are always private; only selected support labels are visible.
- Grounding and QA operate only on the displayed table slice.
- Wording and gold are generated deterministically from local source data.
- Dataset transfer uses duplicate-aware `dedup_cluster_id` partitions.
- Results must report response coverage and, for prediction, requested/realized
  support plus context coverage.

The v2.1 semantic tasks use literal source headers. This release does not claim
curated cross-dataset ontology equivalence.

## Source Terms

Each referenced OpenML dataset retains its upstream terms.
`openml_license_claim` is provenance metadata, not a license granted by
TableSuite-1K. The repository-level `other` designation reflects heterogeneous
source terms.

Code and full protocol documentation:
<https://github.com/Sichao-Li/TableSuite-1K>
