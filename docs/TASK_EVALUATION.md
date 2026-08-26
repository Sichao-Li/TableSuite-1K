# Official Task Evaluation

Status: **public task-record schema 2.0, released with TableSuite-1K v2.0.0**.

Package v2.0.1 is the minimum supported runtime for prediction requests; it
adds explicit task-family and classification-label-space rendering.

Users load ordinary Hugging Face configurations. The companion package joins
each compact task row to `datasets`, resolves the referenced OpenML slice,
renders the model input, and computes gold programmatically.

## Tasks

| Configuration | Model input | Gold source |
| --- | --- | --- |
| `table_grounding` | provided row/subtable plus typed question | displayed source slice |
| `table_question_answering` | 4/8/16-row subtable plus typed question | deterministic table operation |

Table grounding uses 4-8 eligible non-target columns and cell, row, or column
operations. Table QA uses 3-8 columns and supports aggregate, argmax lookup,
and filtered argmax lookup.
Release authoring rejects missing operands, non-finite numeric values, empty
filters, and tied maxima.

## Inspect Specifications

```python
from datasets import load_dataset

specifications = load_dataset(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    split="dataset_test",
    revision="v2.0.0",
)
print(specifications.column_names)
print(specifications[0]["operation"])
```

The rows contain source references and scoring fields, but no copied values,
rendered questions, answers, responses, or experiment results. See
[REFERENCE_FORMAT.md](REFERENCE_FORMAT.md) for their exact columns.

## Run And Score

```python
from tablesuite import load_task

task = load_task(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    split="dataset_test",
    source="openml-parquet",
    revision="v2.0.0",
    dataset_ids=("openml_45069",),
)

example = task[0]
prediction = model(example.prompt)
score = task.score(example.id, prediction)
```

`TaskExample` is input-only. It exposes the prompt, question, rendered table,
metadata, and exact source slice, but not the answer.

Evaluate a complete split with responses keyed by stable item ID:

```python
predictions = {example.id: model(example.prompt) for example in task}
report = task.evaluate(predictions)
print(report.to_dict())
```

Reports include micro accuracy, dataset-macro accuracy,
dedup-cluster-macro accuracy, parse-failure rate, and accuracy by operation.
Missing responses fail by default; `allow_partial=True` is diagnostic only.

## Evaluation Splits

| Split | Held-out factor |
| --- | --- |
| `validation` | validation dataset clusters |
| `episode_test` | source rows within training dataset clusters |
| `dataset_test` | dataset clusters |
| `template_test` | source rows and question templates |
| `composition_test` | filter-then-argmax operation composition |

Dataset and template transfer must be reported separately. Exact source cells
do not cross evaluation splits, and only `template_test` uses held-out wording.

## Reproducibility Boundary

Per-item public rows freeze the source slice, typed operation, scoring
tolerances, template split, and render seed. The tagged package fixes the
operation executor, renderer, English language, Markdown view, missing-value
policy, and tie policy. Release validation reconstructs every internal plan,
checks catalog bindings and leakage, and executes every operation against the
source before upload.

OpenML remains the source-data distributor. Prediction packets and integrated
reasoning outputs are not included in this release.

## Deterministic Expansion

Use official plans for comparable evaluation. For training or controlled
stress tests, `generate_task` creates additional value-free plans with the same
operations, eligibility checks, split policy, renderer, and scorer:

```python
from tablesuite import generate_task

task = generate_task(
    "Lester1996/TableSuite-1K",
    "table_grounding",
    source="openml-parquet",
    revision="v2.0.0",
    split="train",
    items_per_dataset=250,
    max_items=10_000,
    seed=7,
)
```

`items_per_dataset` controls balanced source coverage. `max_items` is an
optional deterministic global cap, useful for smoke tests. A saved generated
bundle contains only `generation.json` and `plans.jsonl`; wording and gold are
still produced from the source at access time.

Each generation manifest records the reference, revision, selected datasets,
seed, scale, generator version, configuration fingerprint, and plan
fingerprint. Generated bundles must be reported separately from official
evaluation plans.
