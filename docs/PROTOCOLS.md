# Benchmark Protocols

## Prediction

Every prediction example consists of an input-only request and separate
`PredictionGold`. Evaluation targets cannot be rendered because they are
absent from the request object; only explicitly selected support labels may be
visible. All official prediction protocols are inference-only: no fine-tuning,
fitting, or per-dataset parameter updates are permitted.

### Protocol Matrix

| Protocol | Interface | Visible labels | Query scope |
| --- | --- | --- | --- |
| `zero_shot_icl` | row-query format | none | frozen episode query rows |
| `few_shot_icl` | row examples | 4/16/32 demonstrations | frozen episode query rows |
| `zero_label_serialized_table` | serialized table | none | every eligible row, or matched frozen episode queries |
| `partially_labeled_serialized_table` | serialized table | 4/16/32 support rows | all remaining rows by default |

`SerializedTablePredictionRequest` always stores a feature-only table. For the
partially labelled protocol, visible support targets live in a separate
`visible_labels` slice and are merged into the rendered table. Query targets
remain only in `PredictionGold`. Zero-label requests cover all eligible source
rows by default; deterministic source-order chunks are allowed for bounded
consumers and may not sample or drop rows. Its optional episode scope reuses
the frozen zero-shot query rows for a matched interface comparison. Partially
labelled requests retain a
frozen episode's support rows and predict every other eligible row by default.
The optional episode scope retains only that episode's frozen queries, enabling
a controlled comparison against few-shot ICL with identical source rows.

`ICLPredictionRequest` is deliberately rendered as row examples followed by
target-hidden queries, never as a Markdown/JSON table. Zero-shot requests omit
demonstrations. Few-shot requests expose exactly 4, 16, or 32 frozen support
rows.

The zero-label serialized protocol still asks for the registered OpenML target
for each row and scores against separate evaluation targets. It is label-free
inference, not a clustering task; arbitrary cluster identifiers require a
separate permutation-invariant evaluation contract.

Few-shot classification episodes are executable only when every query class
occurs in the visible support labels; regression episodes require finite
support and query targets. Every protocol is inference-only. OpenML targets
provide local evaluation targets, not permission to update model parameters.

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
grounding uses one- or multi-row source slices. Task constructors control target
visibility so evaluation targets cannot enter model inputs.

## Partitions

`dataset_split` is the duplicate-aware train/validation/test partition for
cross-dataset studies. ICL episodes retain their fixed demonstration/query row
identities. Serialized-table chunking is an input-capacity choice, not a train
fold. Dataset transfer, shot count, and chunk size must be reported separately.

## Metrics

Classification reports accuracy, balanced accuracy, and macro-F1 per dataset,
then macro-averages across datasets. Regression reports per-dataset MAE, RMSE,
R-squared, and scale-normalized errors before dataset-macro aggregation. ICL
results are reported separately by shot count.

Grounding reports exact typed-answer accuracy, parse-failure rate,
dataset-macro accuracy, cluster-macro accuracy, and accuracy by operation.

Every published result must include its saved `SelectionManifest`, reference
revision, serialization version, shot count or table scope, and metric
aggregation version.

## Official Hugging Face Tasks

TableSuite-1K publishes independently loadable table-grounding and table-QA
configurations. Each item freezes its semantic operation
and source references while the package renders wording and computes gold only
when accessed. Task specifications remain value-free and are audited for
deduplication-cluster partitioning and exact source-cell overlap. See
[TASK_EVALUATION.md](TASK_EVALUATION.md) for loading, scoring, and transfer
contracts.
