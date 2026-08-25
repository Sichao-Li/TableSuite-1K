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

TableSuite-1K is a source-grounded benchmark for tabular prediction, cell
grounding, and table question answering over 1,000 OpenML-referenced datasets.
This repository stores compact task specifications, not copied source tables,
rendered answers, model outputs, or checkpoints.

## Release Contents

| Configuration | Records | Unit |
| --- | ---: | --- |
| `datasets` | 1,000 | source, schema, target, split, and license metadata |
| `table_prediction_tasks` | 998 | executable zero-label prediction contracts |
| `prediction_episodes` | 35,472 | frozen 4/16/32-shot support/query references |
| `grounding_tasks` | 989 | eligible columns for custom grounding |
| `cell_grounding` | 50,012 | official exact-cell evaluation plans |
| `table_question_answering` | 22,448 | official programmatic table-QA plans |

The `datasets` configuration is the canonical metadata table. Other catalog
configurations contain task-specific fields and join to it by `dataset_id`.
The official grounding and QA plans cover 933 eligible datasets.

## Quickstart

Install the matching companion package:

```bash
pip install \
  'tablesuite[local,hf,openml] @ git+https://github.com/Sichao-Li/TableSuite-1K.git@v1.2.1'
```

Materialize one explicitly selected source table from OpenML:

```bash
tablesuite fetch-openml \
  --reference 'Lester1996/TableSuite-1K' \
  --revision v1.2.1 \
  --output openml-parquet \
  --dataset-id openml_45069 \
  --accept-source-terms
```

Load and score official QA examples for that dataset:

```python
from tablesuite import load_task

task = load_task(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    split="dataset_test",
    source="openml-parquet",
    revision="v1.2.1",
    dataset_ids=("openml_45069",),
)

example = task[0]
response = input(f"{example.prompt}\n\nAnswer: ")
print(task.score(example.id, response))
```

Questions and answers are produced deterministically from the source slice and
typed operation when an example is loaded. No LLM authors benchmark gold.

## Task Interfaces

**Cell grounding** presents one contextual source row with 4-8 eligible
feature columns and asks for one exact cell value.

**Table question answering** presents a 4/8/16-row subtable with 3-8 columns
and asks a programmatic aggregate, comparison, filter, or lookup question.

**Prediction** is inference-only and supports four renderings:

| Protocol | Input | Visible labels |
| --- | --- | --- |
| `zero_shot_icl` | target-hidden row queries | none |
| `few_shot_icl` | demonstrations plus row queries | 4/16/32 |
| `zero_label_serialized_table` | feature-only table | none |
| `partially_labeled_serialized_table` | support-labelled table with masked queries | 4/16/32 |

Prediction targets are excluded from rendered model input and returned
separately for local evaluation. They are open benchmark targets recoverable
from the referenced OpenML source, not secret server-side labels.

## Direct Specification Access

All six configurations are ordinary Parquet datasets:

```python
from datasets import load_dataset

plans = load_dataset(
    "Lester1996/TableSuite-1K",
    "cell_grounding",
    split="dataset_test",
    revision="v1.2.1",
)
print(plans[0]["item_id"])
```

These rows are value-free plans. Use the companion package when you need
source resolution, prompt rendering, and programmatic scoring.

## Evaluation Splits

- `episode_test`: held-out rows from training dataset clusters.
- `dataset_test`: held-out dataset clusters.
- `template_test`: held-out rows and question wording.
- `composition_test`: composed filter-then-argmax QA.

Report these transfer axes separately.

## Source and Licensing Boundary

OpenML remains the source-table distributor. Every source retains its upstream
terms; `license_claim` is provenance metadata, not a license granted by
TableSuite-1K. The repository-level `other` designation reflects heterogeneous
source terms. TableSuite-1K is independent from and not endorsed by OpenML.

Code, full protocol documentation, and release validation instructions are at
<https://github.com/Sichao-Li/TableSuite-1K>.
