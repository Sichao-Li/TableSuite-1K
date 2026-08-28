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
  'tablesuite[local,hf,openml] @ git+https://github.com/Sichao-Li/TableSuite-1K.git@v2.1.0'
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

Open the suite and load one official task split:

```python
from tablesuite import TableSuite

suite = TableSuite.open(
    "Lester1996/TableSuite-1K",
    source="openml-parquet",
    revision="v2.0.0",
)
print([task.name for task in suite.tasks()])

task = suite.official(
    "table_grounding",
    split="dataset_test",
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

Prediction uses one deterministic support pool per frozen query episode. A
single fraction runs one condition; a tuple runs a nested support curve:

```python
from tablesuite import render_icl_prediction

prediction = suite.prediction(
    "icl",
    support=(0.1, 0.3),
    dataset_ids=("openml_45069",),
    max_episodes_per_dataset=1,
)
for case in prediction:
    print(case.support)
    print(render_icl_prediction(case.request).input_text)
```

`support=0.1` runs only 10%. Python's `(0.1)` is also a scalar; `(0.1,)`
is the equivalent one-element tuple. The official curve is exported as
`OFFICIAL_SUPPORT_LEVELS`.

## Public Configurations

| Configuration | Purpose |
| --- | --- |
| `datasets` | OpenML source, schema, target, split, and license metadata |
| `table_prediction_tasks` | prediction-eligible datasets and metrics |
| `prediction_episodes` | frozen query anchors and fixed-4/16/32 compatibility records |
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
| `icl` | labelled row demonstrations plus fixed queries | selected support fraction |
| `serialized_table` | one table with labelled support and fixed masked queries | selected support fraction |

For a support pool of size `N`, fraction zero exposes no labels and every
positive fraction exposes `min(N, max(1, ceil(fraction * N)))` labels. The same
nested source rows are used by both interfaces. Query labels always remain
private. At 100%, every eligible non-query row is visible as support.

Every request states its task family. Classification requests also state the
allowed target-label vocabulary; zero-label means that no source row carries a
visible label assignment, not that the output space is undefined.

The fixed v2.0 4/16/32-shot records remain reproducible through
`suite.fixed_prediction(...)` and the lower-level `Benchmark.select(...)` API.

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
generated = suite.generate(
    "table_question_answering",
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
