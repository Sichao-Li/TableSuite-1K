# TableSuite-1K

**A source-grounded benchmark for tabular prediction, cell grounding, and
table question answering over 1,000 OpenML-referenced datasets.**

TableSuite-1K publishes deterministic task definitions and a small Python
interface. Source values remain on OpenML: the Hugging Face release stores
dataset identities, row/column references, operations, and scoring policies,
but no copied table values or model outputs.

## What Is Released

The catalog contains 1,000 OpenML dataset references. Task eligibility is
reported separately:

| Surface | Public records | Purpose |
| --- | ---: | --- |
| `datasets` | 1,000 | source, schema, split, target, and license metadata |
| `table_prediction_tasks` | 998 | prediction-eligible datasets and metrics |
| `prediction_episodes` | 35,472 | frozen 4/16/32-shot support/query references |
| `grounding_tasks` | 989 | eligible non-target columns and sampling caps |
| `cell_grounding` | 50,012 | official exact-cell evaluation plans |
| `table_question_answering` | 22,448 | official programmatic table-QA plans |

The two official task configurations cover 933 eligible datasets in total.
The 1,000-dataset catalog remains available for dataset-level studies even
when a source is too narrow for a particular task.

## Install

```bash
pip install \
  'tablesuite[local,hf,openml] @ git+https://github.com/Sichao-Li/TableSuite-1K.git@v1.4.0'
```

## Runnable Quickstart

Download one explicitly selected source table directly from OpenML:

```bash
tablesuite fetch-openml \
  --reference 'Lester1996/TableSuite-1K' \
  --revision v1.3.0 \
  --output openml-parquet \
  --dataset-id openml_45069 \
  --accept-source-terms
```

Load the matching official QA examples and score a response:

```python
from tablesuite import load_task

task = load_task(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    split="dataset_test",
    source="openml-parquet",
    revision="v1.3.0",
    dataset_ids=("openml_45069",),
)

example = task[0]
print(example.prompt)

response = input(f"{example.prompt}\n\nAnswer: ")
print(task.score(example.id, response))
```

Evaluate a complete split or bounded dataset subset with one mapping:

```python
predictions = {example.id: model(example.prompt) for example in task}
report = task.evaluate(predictions)
print(report.to_dict())
```

Reports include micro accuracy, dataset-macro accuracy,
dedup-cluster-macro accuracy, parse-failure rate, and accuracy by operation.
Missing responses fail unless `allow_partial=True` is requested explicitly.

## Generate Additional Tasks

Official evaluation always uses the frozen Hugging Face plans above. Additional
training or stress-test plans can be generated deterministically from the same
task recipe:

```python
from tablesuite import generate_task

generated = generate_task(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    source="openml-parquet",
    revision="v1.3.0",
    split="train",
    task_families=("classification",),
    max_datasets=10,
    items_per_dataset=100,
    seed=42,
)

example = generated[0]
print(example.prompt)
print(generated.score(example.id, model(example.prompt)))
generated.save("generated-qa")
```

`generated-qa` contains a value-free plan file and a reproducibility manifest.
Reload it without regenerating source choices:

```python
from tablesuite import load_generated_task

generated = load_generated_task("generated-qa", source="openml-parquet")
```

The CLI provides the same operation:

```bash
tablesuite generate \
  --reference 'Lester1996/TableSuite-1K' \
  --revision v1.3.0 \
  --source openml-parquet \
  --name table_question_answering \
  --split train \
  --items-per-dataset 100 \
  --max-datasets 10 \
  --seed 42 \
  --output generated-qa
```

Generated plans are deterministic but are not part of the official evaluation
set. Scale is balanced per dataset so large source tables do not dominate.

## Official Tasks

### Cell Grounding

Input: one contextual source row containing 4-8 eligible feature columns and a
lookup question. Output: the exact requested cell value.

### Table Question Answering

Input: a 4/8/16-row source subtable with 3-8 columns and a deterministic
question. Output: a programmatically computed aggregate or lookup answer.

Questions are rendered at access time. Gold answers are computed from source
references and typed operations, never authored by an LLM.

Official task plans provide distinct evaluation axes:

| Split | Held-out factor |
| --- | --- |
| `validation` | validation dataset clusters |
| `episode_test` | rows from training dataset clusters |
| `dataset_test` | dataset clusters |
| `template_test` | rows and question wording |
| `composition_test` | composed filter-then-argmax QA |

## Prediction Interfaces

Prediction is inference-only: no fitting, fine-tuning, or per-dataset parameter
updates are part of these protocols.

| Protocol | Input | Visible labels |
| --- | --- | --- |
| `zero_shot_icl` | target-hidden query rows | none |
| `few_shot_icl` | row demonstrations plus queries | 4/16/32 |
| `zero_label_serialized_table` | one feature-only table | none |
| `partially_labeled_serialized_table` | support-labelled table with masked queries | 4/16/32 |

Evaluation targets are stored separately from rendered model requests. They
are open benchmark targets recoverable from the referenced OpenML source, not
secret server-side labels.

```python
from tablesuite import Benchmark, Selection, render_icl_prediction

benchmark = Benchmark.from_huggingface(
    "Lester1996/TableSuite-1K",
    "openml-parquet",
    revision="v1.3.0",
)
subset = benchmark.select(
    Selection(
        tasks=("few_shot_icl",),
        dataset_ids=("openml_10",),
        shots=(4,),
        max_episodes_per_dataset_per_shot=1,
    )
)
case = next(subset.few_shot_icl())
print(render_icl_prediction(case.request).input_text)
```

See [docs/PROTOCOLS.md](docs/PROTOCOLS.md) for exact prediction and grounding
contracts.

## Hugging Face-Native Access

Every configuration is ordinary Parquet and can be inspected without the
companion package:

```python
from datasets import load_dataset

plans = load_dataset(
    "Lester1996/TableSuite-1K",
    "cell_grounding",
    split="dataset_test",
    revision="v1.3.0",
)
print(plans[0]["item_id"])
```

Rows in official task configs are value-free plans. The package resolves the
referenced OpenML source, renders model input, and computes gold at runtime.

## Source and Licensing Boundary

OpenML remains the source-table distributor. Each downloaded table retains its
upstream terms; `openml_license_claim` is provenance metadata, not a license
granted by TableSuite-1K. The package downloads only an explicit bounded
selection and writes a local `SOURCE_NOTICES.json`.

TableSuite-1K is independent from and not endorsed by OpenML. See
[docs/SOURCE_DATA.md](docs/SOURCE_DATA.md).

## Stable User API

```text
load_task
generate_task
load_generated_task
TaskDataset
GeneratedTaskDataset
TaskExample
TaskScore
TaskReport
```

`Benchmark`, `Selection`, `TableSlice`, and rendering helpers are the advanced
prediction/source-selection API. Maintainer authoring utilities are documented
separately and are not part of the stable user contract.

## CLI

```bash
tablesuite info --reference 'Lester1996/TableSuite-1K' --revision v1.3.0

tablesuite task \
  --reference 'Lester1996/TableSuite-1K' \
  --revision v1.3.0 \
  --source openml-parquet \
  --name table_question_answering \
  --split dataset_test \
  --dataset-id openml_45069 \
  --index 0
```

## Development and Release Authoring

```bash
python -m pytest -q
ruff check src tests examples
```

The deterministic authoring implementation is retained for scientific
reproducibility, but build audits stay outside the public dataset payload.
Maintainers should follow [docs/RELEASING.md](docs/RELEASING.md).
The repository contains no model weights, embeddings, raw OpenML tables,
cluster scripts, experiment reports, leaderboard, or LLM reasoner.

## Citation

See [CITATION.cff](CITATION.cff). A paper citation will be added when its
bibliographic metadata is final.
