# TableSuite-1K

**Benchmarking predictive and language-grounded tabular intelligence across
1,000 OpenML-referenced datasets.**

TableSuite-1K evaluates three capabilities through one source-grounded API:

| Task | Model input | What is scored |
| --- | --- | --- |
| Table prediction | ICL rows or a partially labelled serialized table | classification and regression |
| Table grounding | a provided table plus a lookup/comprehension question | exact facts from that table |
| Table question answering | a provided subtable plus a typed operation question | programmatic aggregate or lookup answers |

The Hugging Face release contains compact metadata and value-free task plans.
It does **not** redistribute OpenML tables. Questions and gold answers are
materialized deterministically from the user's local source tables; no LLM
authors benchmark gold.

## Install

```bash
pip install \
  'tablesuite[local,hf,openml] @ git+https://github.com/Sichao-Li/TableSuite-1K.git@v2.1.0'
```

## 1. Materialize Source Tables

Download only the datasets you intend to use:

```bash
tablesuite fetch-openml \
  --reference Lester1996/TableSuite-1K \
  --revision v2.1.0 \
  --output openml-parquet \
  --dataset-id openml_45069 \
  --accept-source-terms
```

OpenML remains the source-table distributor. The command records source notices
locally and never writes source values into the benchmark reference.

## 2. Open The Suite

```python
from tablesuite import TableSuite

suite = TableSuite.open(
    "Lester1996/TableSuite-1K",
    source="openml-parquet",
    revision="v2.1.0",
)
print(suite.catalog_summary())
print(suite.tasks())
```

## 3. Run A Task

### Table Prediction

The same frozen queries can be presented as row demonstrations or as one
serialized table. `support` is the fraction of eligible non-query rows whose
labels are visible; no model parameter updates are part of the protocol.

```python
from tablesuite import render_icl_prediction

task = suite.prediction(
    "icl",
    support=(0.0, 0.1, 0.3),
    dataset_ids=("openml_45069",),
    max_episodes_per_dataset=1,
)

predictions = {}
for example in task:
    prompt = render_icl_prediction(example.request).input_text
    predictions[example.request.request_id] = model(prompt)

report = task.evaluate(predictions)
print(report.to_dict())
```

Use `"serialized_table"` with `render_serialized_table_prediction` for the
matched table interface. The official support-capacity schedule is
`0/10/30/50/70/90/100%` and is exported as `OFFICIAL_SUPPORT_LEVELS`.

For a model-specific context limit, request one maximum support cap and fit the
largest nested prefix using the model's **actual** tokenizer and chat template:

```python
candidate = suite.prediction(
    "serialized_table",
    support=1.0,
    dataset_ids=("openml_45069",),
)
fitted = candidate.fit_context(
    max_prompt_tokens=32_768,
    count_tokens=count_final_model_prompt,
    tokenizer_id="model-name@revision",
)

print(fitted.report.to_dict())  # includes support, tokens, and exclusions
```

Requests that do not fit at zero support are explicitly excluded; TableSuite
never silently truncates a prediction request.

### Table Grounding And QA

```python
task = suite.official(
    "table_grounding",
    split="dataset_test",
    dataset_ids=("openml_45069",),
)

example = task[0]
print(example.prompt)
print(task.score(example.id, model(example.prompt)))

responses = {item.id: model(item.prompt) for item in task}
print(task.evaluate(responses).to_dict())
```

Replace `table_grounding` with `table_question_answering` for aggregate and
multi-step table operations. Missing responses fail official evaluation;
`allow_partial=True` is diagnostic only.

## Task Contracts

### Prediction

Each frozen query has one deterministic support ordering. Every percentage is
a prefix of that ordering, so support conditions are nested and the ICL and
serialized-table interfaces receive the same source rows. For a pool of size
`N`, positive fraction `p` exposes `min(N, max(1, ceil(p*N)))` labels.

Classification reports dataset-macro accuracy, balanced accuracy, and macro-F1.
Regression reports dataset-macro MAE, RMSE, normalized errors, and R-squared.
Every result should include response coverage, requested and realized support,
prompt-token counts, and the model context limit.

### Table Grounding

Answers are restricted to the table shown in the request. Operations are
`cell_lookup`, `row_lookup`, `column_values`, `distinct_values`, and
`value_counts`. The release uses literal source headers and makes no claim of a
curated cross-dataset ontology.

### Table Question Answering

QA uses controlled 4/8/16-row subtables and programmatic `count`, `sum`, `mean`,
`min`, `max`, argmax lookup, and filter-then-argmax operations. Gold and evidence
are computed from the exact displayed slice.

## Evaluation Splits

| Split | Held-out factor |
| --- | --- |
| `validation` | validation dataset clusters |
| `episode_test` | rows from training dataset clusters |
| `dataset_test` | deduplicated dataset clusters |
| `template_test` | rows and question wording |
| `composition_test` | filter-then-argmax composition (QA only) |

Dataset partitions are assigned by `dedup_cluster_id`, not raw dataset ID.

## Hugging Face Configurations

| Configuration | Purpose |
| --- | --- |
| `datasets` | OpenML identity, schema, target, split, and provenance |
| `table_prediction_tasks` | prediction eligibility and primary metrics |
| `prediction_episodes` | frozen query anchors and v2 fixed-shot compatibility records |
| `table_grounding` | official provided-table grounding plans |
| `table_question_answering` | official programmatic QA plans |

No configuration contains feature values, targets, rendered questions, gold
answers, responses, embeddings, checkpoints, or experiment logs.

## Generate Additional Training Plans

Official evaluation uses frozen Hugging Face plans. Additional deterministic
training or stress-test plans use the same operation engine:

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

Generated plans must be reported separately from official evaluation.

## Stable Public API

Start with `TableSuite`. The stable task surfaces are
`TableSuite.prediction`, `TableSuite.official`, and `TableSuite.generate`.
Renderer, report, and source-materialization helpers are exported from the
package root. Lower-level catalog and authoring modules are implementation
details and may change between minor releases.

## Development

```bash
python -m pytest -q
ruff check src tests examples
```

See [protocols](docs/PROTOCOLS.md), the [reference format](docs/REFERENCE_FORMAT.md),
and the [source-data policy](docs/SOURCE_DATA.md) for the full contracts.
