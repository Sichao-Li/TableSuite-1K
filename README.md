# TableSuite-1K

**A source-grounded benchmark for predictive and language-grounded tabular
intelligence over 1,000 OpenML-referenced datasets.**

TableSuite-1K evaluates three complementary capabilities:

1. **Prediction:** infer a registered target from zero/few-shot rows or a
   serialized table.
2. **Table grounding:** retrieve or summarize facts from the table shown in
   the request.
3. **Table question answering:** execute typed operations over a displayed
   subtable.

The Hugging Face dataset stores source metadata and value-free task plans.
OpenML remains the source-table distributor. Questions and gold answers are
created deterministically after a user materializes the referenced tables;
benchmark gold is never authored by an LLM.

## Install

```bash
pip install \
  'tablesuite[local,hf,openml] @ git+https://github.com/Sichao-Li/TableSuite-1K.git@v2.0.0'
```

## Quickstart

Materialize an explicitly selected source table:

```bash
tablesuite fetch-openml \
  --reference Lester1996/TableSuite-1K \
  --revision v2.0.0 \
  --output openml-parquet \
  --dataset-id openml_45069 \
  --accept-source-terms
```

Load one official task split:

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
print(example.prompt)
print(task.score(example.id, model(example.prompt)))
```

Evaluate a complete split or an explicit dataset subset:

```python
predictions = {example.id: model(example.prompt) for example in task}
report = task.evaluate(predictions)
print(report.to_dict())
```

Missing predictions fail official evaluation. `allow_partial=True` is
available only for diagnostic runs.

## Public Configurations

| Configuration | Purpose |
| --- | --- |
| `datasets` | OpenML source, schema, target, split, and license metadata |
| `table_prediction_tasks` | prediction-eligible datasets and metrics |
| `prediction_episodes` | frozen 4/16/32-shot support/query references |
| `grounding_tasks` | eligible non-target columns and sampling policy |
| `table_grounding` | official provided-table lookup/comprehension plans |
| `table_question_answering` | official programmatic QA plans |

The catalog contains 1,000 datasets. Task-specific eligibility is reported
separately; users may select one dataset, a subset, or the full catalog.

## Task Contracts

### Prediction

Prediction is inference-only. No fitting, fine-tuning, or per-dataset
parameter updates are part of an official run.

| Protocol | Input | Visible labels |
| --- | --- | --- |
| `zero_shot_icl` | target-hidden row queries | none |
| `few_shot_icl` | demonstrations plus queries | 4/16/32 |
| `zero_label_serialized_table` | feature-only serialized table | none |
| `partially_labeled_serialized_table` | one table with labelled support and masked queries | 4/16/32 |

### Table Grounding

Every answer comes only from the table displayed in the request. Official v2
operations are:

```text
cell_lookup
row_lookup
column_values
distinct_values
value_counts
```

Source rows receive deterministic local aliases (`r0`, `r1`, ...). Column
names are quoted in questions, including numeric or otherwise ambiguous
headers. The v2 release uses literal source-schema wording and records this as
`schema_language="literal"`; it does not claim curated cross-dataset ontology
mapping.

### Table Question Answering

QA uses 4/8/16-row provided subtables and typed operations:

```text
count, sum, mean, min, max
argmax lookup
filter then argmax lookup
```

Operation and table size are scheduled independently. Gold and evidence are
computed programmatically from the exact displayed slice.

## Evaluation Splits

| Split | Held-out factor |
| --- | --- |
| `validation` | validation dataset clusters |
| `episode_test` | rows from training dataset clusters |
| `dataset_test` | deduplicated dataset clusters |
| `template_test` | rows and question wording |
| `composition_test` | filter-then-argmax composition (QA only) |

Dataset partitions are assigned by `dedup_cluster_id`, not raw dataset ID.

## Generate Additional Plans

Official evaluation uses frozen Hugging Face plans. Training and stress-test
plans can be generated with the same deterministic authoring path:

```python
from tablesuite import generate_task

generated = generate_task(
    "Lester1996/TableSuite-1K",
    "table_question_answering",
    source="openml-parquet",
    revision="v2.0.0",
    split="train",
    max_datasets=10,
    items_per_dataset=100,
    seed=42,
)
generated.save("generated-qa")
```

Saved bundles contain only plans and a reproducibility manifest. Generated
plans never replace the official evaluation set.

## Source Boundary

TableSuite-1K does not redistribute source table values. Each OpenML dataset
retains its upstream terms; `openml_license_claim` is provenance metadata, not
a license granted by this repository. Explicit source selection and
`--accept-source-terms` are required before download.

The repository contains no model weights, embeddings, raw OpenML tables,
cluster launch scripts, model outputs, leaderboard, or LLM reasoner.

## Development

```bash
python -m pytest -q
ruff check src tests examples
```

See [docs/PROTOCOLS.md](docs/PROTOCOLS.md) for task semantics and
[docs/RELEASING.md](docs/RELEASING.md) for the audited release process.
