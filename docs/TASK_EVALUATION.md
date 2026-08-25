# Official Task Evaluation

Status: **task-plan schema 1.1, frozen for TableSuite-1K v1.2.1**.

Users interact with ordinary Hugging Face configurations and splits. They do
not construct evaluation runtimes or manipulate semantic plans.

## Hugging Face Configurations

| Configuration | Capability | Answer source |
| --- | --- | --- |
| `cell_grounding` | retrieve a cell from a contextual source row | source table |
| `table_question_answering` | aggregate, compare, filter, and look up | deterministic table execution |

Each configuration can expose `train`, `validation`, `episode_test`,
`dataset_test`, `template_test`, and `composition_test` as applicable. The Hub
rows contain stable item IDs, source references, typed operations, rendering
policy, and scoring policy. They contain no copied source values, rendered
questions, gold answers, model responses, or experiment results.

Cell grounding uses a one-row slice with 4–8 eligible non-target columns and
an explicit lookup column. Table QA uses 4/8/16-row subtables with 3–8 columns.
Its typed operations are aggregate, argmax lookup, and filtered argmax lookup.
Missing operands, non-finite numeric values, empty filters, and tied maxima are
rejected during release authoring.

Inspect one split using the standard Datasets API:

```python
from datasets import load_dataset

specifications = load_dataset(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    split="dataset_test",
)
print(specifications[0]["item_id"])
```

## Run A Task

The companion package resolves the external OpenML rows, deterministically
renders the question, and keeps the computed answer away from the model input.

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
prediction = model(example.prompt)
score = task.score(example.id, prediction)
```

`TaskExample` is input-only. It exposes the prompt, question, rendered table,
task/split metadata, and exact source slice, but not the answer.

For a complete run, submit answers keyed by stable item ID:

```python
predictions = {example.id: model(example.prompt) for example in task}
report = task.evaluate(predictions)
print(report.to_dict())
```

The report includes micro accuracy, dataset-macro accuracy, dedup-cluster-macro
accuracy, parse failures, and accuracy by semantic operation. Missing examples
are rejected by default. `allow_partial=True` is reserved for diagnostic smoke
tests.

The CLI follows the same vocabulary:

```bash
tablesuite task \
  --reference 'Lester1996/TableSuite-1K' \
  --source openml-parquet \
  --name table_question_answering \
  --split dataset_test \
  --dataset-id openml_45069 \
  --index 0
```

## Source Boundary

OpenML remains the source-data distributor. The HF task specifications are
value-free, so users materialize only the source tables needed by their chosen
configuration and split.

The first release does not publish model prediction packets or an integrated
reasoning task. Inference-only prediction interfaces remain represented by
`table_prediction_tasks` and `prediction_episodes`, without treating one external
predictor's outputs as
benchmark ground truth.

## Reproducibility Contract

Internally, every HF task row freezes the source slice, typed operation,
template family, held-out-template policy, rendering seed, executor version,
and scorer. Loading validates source identity, dataset partition, and
deduplication cluster. The release-authoring audit additionally rejects
cross-split source and deduplication leakage before upload.

The split contract isolates dataset, row/episode, wording, and operation
composition transfer. Exact source cells cannot occur across evaluation
splits, and only `template_test` uses held-out question templates.

Benchmark maintainers can import the lower-level authoring contracts from
`tablesuite.evaluation`. Those classes are intentionally absent from the
top-level user API; official users should call `load_task`.
