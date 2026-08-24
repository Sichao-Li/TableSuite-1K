# Publishing A Benchmark Release

This checklist publishes the value-free benchmark definition, not the OpenML
source tables or research artifacts.

## Release Identity

- GitHub: `Sichao-Li/TableSuite-1K`
- Hugging Face: `Lester1996/TableSuite-1K`
- Python distribution and CLI: `tablesuite`
- release tag: `v1.2.0`

The first Hub upload remains private until all six configurations pass the
fresh-cache validation below. Add the associated paper citation when its
bibliographic metadata is final.

## Clean-Room Build

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,hf,openml]'

python -m pytest -q
ruff check src tests examples

tablesuite build-release \
  --reference /path/to/reference-package \
  --source /path/to/openml-parquet \
  --dataset-card huggingface/README.md \
  --output /path/to/huggingface-release

tablesuite validate-release \
  --release /path/to/huggingface-release \
  --source /path/to/openml-parquet
```

Review `release_summary.json` and `release_summary.md`. The release directory
must contain exactly six configured datasets plus audit metadata. It must not
contain OpenML source Parquet files, model outputs, prediction packets,
embeddings, checkpoints, logs, or experiment reports.

## Hub Upload

Use a write-scoped token through the current Hugging Face CLI. Do not put the
token in a script or commit.

```bash
python -m pip install --upgrade huggingface_hub
hf auth login

REPO='Lester1996/TableSuite-1K'
hf repos create "$REPO" --repo-type dataset --private
hf upload "$REPO" /path/to/huggingface-release . \
  --repo-type dataset \
  --commit-message 'TableSuite-1K v1.2.0 release candidate'
```

`hf upload` is resumable and is the current replacement for the deprecated
`hf upload-large-folder` command. Keep the repository private until all six
configurations load in a fresh cache.

## Hub Validation

```python
from datasets import get_dataset_config_names, load_dataset

repository = "Lester1996/TableSuite-1K"
expected = {
    "datasets",
    "table_prediction_tasks",
    "prediction_episodes",
    "grounding_tasks",
    "cell_grounding",
    "table_question_answering",
}
assert set(get_dataset_config_names(repository)) == expected

for name in sorted(expected):
    dataset = load_dataset(repository, name)
    assert dataset
```

Also run one `load_task` example for each executable configuration against a
small locally materialized OpenML subset. Then record the immutable Hub commit
revision used by the paper and make the dataset repository public.

Current Hugging Face references:

- <https://huggingface.co/docs/huggingface_hub/en/guides/cli>
- <https://huggingface.co/docs/huggingface_hub/en/guides/upload>
- <https://huggingface.co/docs/datasets/create_dataset>
