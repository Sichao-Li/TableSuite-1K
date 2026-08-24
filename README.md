# TableSuite-1K

**A comprehensive benchmark for tabular prediction, in-context learning, cell
grounding, and table question answering over 1,000 OpenML-referenced tables.**

This repository provides the companion Python package and release-authoring
tools for a value-free Hugging Face benchmark over 1,000 OpenML-referenced
tables. The first release has two independently loadable official tasks:

| Hugging Face configuration | Model input | Evaluated output |
| --- | --- | --- |
| `cell_grounding` | contextual source row and question | exact cell fact |
| `table_question_answering` | source subtable and question | programmatic table answer |

Questions are rendered deterministically when an item is accessed. Gold is
computed from frozen source references and typed operations, never authored by
an LLM. The model receives a prompt without the answer.

Prediction evaluation is also inference-only. It exposes four explicit
protocols over two intentionally different interfaces:

| Protocol | Runtime input | Visible labels |
| --- | --- | --- |
| `zero_shot_icl` | target-hidden query rows | none |
| `few_shot_icl` | labelled row examples followed by target-hidden queries | 4/16/32 demonstrations |
| `zero_label_serialized_table` | one feature-only table | none |
| `partially_labeled_serialized_table` | one table with labelled support rows and masked query rows | 4/16/32 support labels |

All four protocols forbid fitting, fine-tuning, and per-dataset parameter
updates. OpenML targets are private evaluation truth unless they are explicitly
exposed as support labels by a label-visible protocol.

## Install

```bash
pip install 'tablesuite[local,hf] @ git+https://github.com/Sichao-Li/TableSuite-1K.git@v1.2.0'
```

Add `openml` when downloading source tables directly:

```bash
pip install 'tablesuite[local,hf,openml] @ git+https://github.com/Sichao-Li/TableSuite-1K.git@v1.2.0'
```

## Load One Task

```python
from tablesuite import load_task

qa = load_task(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    split="dataset_test",
    source="openml-parquet",
    revision="v1.2.0",
)

example = qa[0]
print(example.prompt)

response = model(example.prompt)
print(qa.score(example.id, response))
```

`TaskDataset` supports integer indexing, stable-ID indexing, iteration, and a
compact metadata summary:

```python
same_example = qa[example.id]
print(qa.summary())
```

Evaluate a complete submission with one mapping:

```python
predictions = {example.id: model(example.prompt) for example in qa}
report = qa.evaluate(predictions)
print(report.to_dict())
```

The report contains micro accuracy, dataset-macro accuracy,
dedup-cluster-macro accuracy, parse-failure rate, and accuracy by operation.
Missing examples fail loudly unless `allow_partial=True` is explicitly used
for a smoke test.

The first release does not bundle prediction packets or an integrated
reasoning task. Prediction interfaces are represented by the
`table_prediction_tasks` and `prediction_episodes` reference configurations.
No predictor or LLM is invoked implicitly.

## Prediction Interfaces

Zero-label serialized-table prediction evaluates every eligible row in one
feature-only table by default. Models with bounded input capacity may request
deterministic chunks; chunking preserves source order and does not sample or
drop rows.

```python
from tablesuite import (
    Benchmark,
    Selection,
    render_serialized_table_prediction,
)

benchmark = Benchmark.from_path("reference-package", "openml-parquet")
subset = benchmark.select(
    Selection(
        tasks=("zero_label_serialized_table",),
        dataset_ids=("openml_1",),
    )
)
example = next(subset.zero_label_serialized_table(rows_per_table=128))
print(render_serialized_table_prediction(example.request).input_text)
```

Few-shot ICL uses row demonstrations rather than table serialization. Zero-shot
ICL removes those demonstrations while preserving a source-validated frozen
query set. The same support/query references can also be rendered as one
partially labelled table. By default that table retains the frozen support rows
and predicts every other eligible source row. Pass `query_scope="episode"` to
compare serialization against few-shot ICL on exactly the same query rows.

```python
from tablesuite import (
    render_icl_prediction,
    render_serialized_table_prediction,
)

subset = benchmark.select(
    Selection(
        tasks=(
            "zero_shot_icl",
            "few_shot_icl",
            "partially_labeled_serialized_table",
        ),
        dataset_ids=("openml_1",),
        shots=(4,),
    )
)
print(render_icl_prediction(next(subset.zero_shot_icl()).request).input_text)
print(render_icl_prediction(next(subset.few_shot_icl()).request).input_text)
print(
    render_serialized_table_prediction(
        next(
            subset.partially_labeled_serialized_table(
                query_scope="episode"
            )
        ).request
    ).input_text
)
```

## Hugging Face-Native Access

The task specifications are ordinary Hugging Face dataset configurations and
splits. They can be inspected without the companion package:

```python
from datasets import load_dataset

items = load_dataset(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    split="dataset_test",
)
print(items[0]["item_id"])
```

These rows are deliberately value-free. Each freezes an item ID, source rows
and columns, semantic operation, rendering seed, and scoring policy. The
companion package turns that specification into the model-ready prompt and
scores the response against programmatically computed gold.

## Evaluation Axes

Official task plans isolate distinct transfer questions:

| Split | What changes? |
| --- | --- |
| `validation` | validation dataset clusters |
| `episode_test` | disjoint rows from training dataset clusters |
| `dataset_test` | held-out dataset clusters |
| `template_test` | disjoint rows and held-out question wording |
| `composition_test` | disjoint rows and composed filter-then-argmax QA |

Cell grounding presents one row with 4–8 eligible non-target columns. Table QA
presents compact 4/8/16-row subtables with 3–8 columns. Release authoring
rejects missing operands, non-finite values, empty filters, and tied maxima.

The repository also exposes four reference configurations used to select or
materialize source data:

| Configuration | Purpose |
| --- | --- |
| `datasets` | OpenML identity, schema, split, and licensing metadata |
| `table_prediction_tasks` | zero-label serialized-table target and metric contracts |
| `prediction_episodes` | frozen support/query references used by ICL and partially labelled tables |
| `grounding_tasks` | eligible non-target columns and sampling contracts |

## Roadmap

Post-v1.2.0 work is intentionally separated from the frozen release contract.
The ordered plan covers ablation, attribution, composed reasoning, and
multi-view evaluation; see [docs/ROADMAP.md](docs/ROADMAP.md).

## Source Tables

The HF repository publishes value-free references rather than copying 1,000
heterogeneously licensed OpenML tables. OpenML remains the source distributor,
and each materialized table retains its upstream terms. See
[docs/SOURCE_DATA.md](docs/SOURCE_DATA.md).

TableSuite-1K is an independent research benchmark and is not affiliated with
or endorsed by OpenML.

Download a bounded subset:

```bash
tablesuite fetch-openml \
  --reference 'Lester1996/TableSuite-1K' \
  --output openml-parquet \
  --dataset-split test \
  --max-datasets 10 \
  --accept-source-terms
```

Source files are stored as `<openml_data_id>.parquet`. TableSuite-1K validates row
count, required features, and the registered target; byte-identical source
files and SHA-256 checks are not required.

## CLI

Preview and optionally score one official task item:

```bash
tablesuite task \
  --reference 'Lester1996/TableSuite-1K' \
  --source openml-parquet \
  --name cell_grounding \
  --split dataset_test \
  --index 0 \
  --response '42'
```

Inspect reference metadata:

```bash
tablesuite info \
  --reference 'Lester1996/TableSuite-1K'
```

The existing `select`, `preview`, and `table` commands remain available for
building custom zero/few-shot ICL, zero/partial-label serialized-table,
grounding, row, and subtable subsets.

## Build The Release

Maintainers create the final six-configuration directory atomically:

```bash
tablesuite build-release \
  --reference reference-package \
  --source openml-parquet \
  --dataset-card huggingface/README.md \
  --output huggingface-release

tablesuite validate-release \
  --release huggingface-release \
  --source openml-parquet
```

The builder audits dataset-cluster partitions, source-cell overlap across
splits, target-column exclusion, value-free plan records, and execution of
every generated item before making the output visible.

## Stable Python API

The primary evaluator surface is intentionally small:

```text
load_task
TaskDataset
TaskExample
TaskScore
TaskReport
```

`Benchmark`, `Selection`, `TableSlice`, and rendering helpers form the advanced
source-selection API. Frozen task-authoring contracts live under
`tablesuite.evaluation`; maintainer APIs live under
`tablesuite.authoring` and `tablesuite.release`.

## Development

```bash
python -m pytest -q
ruff check src tests examples
```

See [docs/TASK_EVALUATION.md](docs/TASK_EVALUATION.md) for task semantics,
[docs/PROTOCOLS.md](docs/PROTOCOLS.md) for prediction/grounding protocols, and
[docs/REFERENCE_FORMAT.md](docs/REFERENCE_FORMAT.md) for the HF reference
schema. Maintainers should follow [docs/RELEASING.md](docs/RELEASING.md).

This public package contains no model weights, embeddings, private experiment
reports, cluster launch scripts, leaderboard, or LLM reasoner.
