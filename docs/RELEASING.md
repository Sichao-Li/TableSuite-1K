# Publishing A Benchmark Release

This is the maintainer checklist for the value-free benchmark definition. It
never uploads OpenML source tables or experiment artifacts.

## Release Identity

- GitHub: `Sichao-Li/TableSuite-1K`
- Hugging Face: `Lester1996/TableSuite-1K`
- Python package and CLI: `tablesuite`
- release tag: `v2.1.0`

## Clean Build

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,hf,openml]'

python -m pytest -q
ruff check src tests examples

tablesuite build-release \
  --reference /path/to/prior-reference \
  --source /path/to/openml-parquet \
  --dataset-card huggingface/README.md \
  --output /path/to/tablesuite-v2.1 \
  > /path/to/release-audit.json

tablesuite validate-release \
  --release /path/to/tablesuite-v2.1 \
  --source /path/to/openml-parquet
```

`build-release` reconstructs and executes every official semantic task. Keep
the audit JSON outside the upload directory.

## Required Payload

The upload contains only:

```text
README.md
reference_summary.json
datasets/
table_prediction_tasks/
prediction_episodes/
tasks/table_grounding/
tasks/table_question_answering/
```

Reject the build if it contains source values, labels, rendered questions,
answers, model outputs, caches, logs, checkpoints, or experiment reports.

## Private Staging

Upload first to a private staging dataset repository. Use a write-scoped token
from the Hugging Face CLI; never place it in a script or commit.

```bash
hf auth login
hf upload Lester1996/TableSuite-1K-v2.1-staging \
  /path/to/tablesuite-v2.1 . \
  --repo-type dataset \
  --commit-message 'TableSuite-1K v2.1 release candidate'
```

From a fresh cache, require exactly these five configs:

```python
from datasets import get_dataset_config_names, load_dataset

repository = "Lester1996/TableSuite-1K-v2.1-staging"
expected = {
    "datasets",
    "table_prediction_tasks",
    "prediction_episodes",
    "table_grounding",
    "table_question_answering",
}
assert set(get_dataset_config_names(repository)) == expected
for name in sorted(expected):
    assert load_dataset(repository, name)
```

Also materialize and score at least one grounding item, one QA item, one ICL
request, and one serialized-table request against local source data.

## Publication

After staging passes:

1. upload the exact staged tree to the public dataset repository;
2. delete stale remote files, especially retired configurations;
3. tag the validated Hub commit `v2.1.0`;
4. tag the matching GitHub commit `v2.1.0`;
5. repeat the five-config and task smoke tests anonymously with no token;
6. record both immutable revisions with every reported experiment.

Relevant documentation:

- <https://huggingface.co/docs/huggingface_hub/en/guides/upload>
- <https://huggingface.co/docs/datasets/create_dataset>
