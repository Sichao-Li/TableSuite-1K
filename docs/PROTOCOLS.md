# Benchmark Protocols

## Prediction

Every prediction example consists of an input-only request and separate
`PredictionGold`. Evaluation targets cannot be rendered because they are
absent from the request object; only explicitly selected support labels may be
visible. All official prediction protocols are inference-only: no fine-tuning,
fitting, or per-dataset parameter updates are permitted.

The task family and complete classification output vocabulary are target-schema
metadata and are shown in every request. They reveal no row-to-label assignment;
"visible labels" below refers only to assignments attached to source rows.

### Protocol Matrix

| Protocol | Interface | Visible labels | Query scope |
| --- | --- | --- | --- |
| `icl` | row demonstrations | selected support fraction | frozen episode queries |
| `serialized_table` | one serialized table | selected support fraction | the same frozen queries |

The public `support` argument accepts one fraction or an ordered sequence. The
official schedule is `0/10/30/50/70/90/100%`. Each prediction plan has one
deterministic support ordering; every level is a prefix of that ordering. The
sets are therefore nested, and both interfaces receive the same labelled
source rows.

For `N` eligible non-query rows, zero maps to zero support rows. Every positive
fraction maps to:

```text
min(N, max(1, ceil(fraction * N)))
```

Classification support ordering is label-stratified. Regression ordering is
quantile-stratified. Query rows are always excluded from support.

`ICLPredictionRequest` renders selected rows as labelled demonstrations,
followed by target-hidden queries. `SerializedTablePredictionRequest` stores a
feature-only table containing the same support and query rows; visible support
targets live in a separate `visible_labels` slice and are merged only while
rendering. At 100%, the serialized request contains every eligible row for that
query plan. Query targets remain only in `PredictionGold`.

The frozen v2.0 4/16/32-shot protocols remain available through
`TableSuite.fixed_prediction()` solely for exact backward reproduction.

## Provided-Table Grounding

Table grounding samples non-target, non-identifier feature slices with a
deterministic operation-balanced policy. Cell lookup, row lookup, ordered
column extraction, distinct-value extraction, and value counting are executed
only over the table shown in the request. Static and runtime audits reject any
operation or evidence reference outside that slice.

V2 uses literal source headers and records `schema_language="literal"`.
Paraphrased or mapped schema language requires a separately curated mapping
and is not claimed by this release.

## Source Slices

`TableSlice` provides uniform access to a source row or subtable. It is a data
primitive, not an additional scored task. Serialized prediction uses a
multi-row feature slice; ICL uses demonstration and query slices; table
grounding uses one- or multi-row source slices. Task constructors control
target visibility so evaluation targets cannot enter model inputs.

## Partitions

`dataset_split` is the duplicate-aware train/validation/test partition for
cross-dataset studies. Prediction episodes retain fixed query identities.
Dataset transfer, requested and realized support, table size, and context
coverage must be reported separately.

## Metrics

Classification reports accuracy, balanced accuracy, and macro-F1 per dataset,
then macro-averages across datasets. Regression reports per-dataset MAE, RMSE,
R-squared, and scale-normalized errors before dataset-macro aggregation.
Prediction results are reported separately at every support fraction.

Grounding reports exact typed-answer accuracy, parse-failure rate,
dataset-macro accuracy, cluster-macro accuracy, and accuracy by operation.

Every published result must include its manifest, reference revision,
serialization version, requested and realized support, prompt tokens, model
context limit, context coverage, and metric aggregation version.

## Official Hugging Face Tasks

TableSuite-1K publishes independently loadable table-grounding and table-QA
configurations. Each item freezes its semantic operation and source references
while the package renders wording and computes gold only when accessed. Task
specifications remain value-free and are audited for deduplication-cluster
partitioning and exact source-cell overlap. See
[TASK_EVALUATION.md](TASK_EVALUATION.md) for loading, scoring, and transfer
contracts.
