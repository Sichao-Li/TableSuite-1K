"""Public API for TableSuite-1K's Hugging Face-hosted table tasks."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version

from tablesuite.benchmark import Benchmark, BenchmarkSubset
from tablesuite.catalog import Catalog
from tablesuite.rendering import (
    render_icl_prediction,
    render_serialized_table_prediction,
    render_table,
)
from tablesuite.source import ParquetSource, materialize_openml_sources
from tablesuite.tasks import (
    TaskDataset,
    TaskExample,
    TaskReport,
    TaskScore,
    load_task,
)
from tablesuite.types import (
    CellFact,
    DatasetSpec,
    ICLPredictionExample,
    ICLPredictionRequest,
    ICLProtocol,
    MaterializedTableSlice,
    PredictionGold,
    RenderedPrediction,
    Selection,
    SelectionManifest,
    SerializedTablePredictionExample,
    SerializedTablePredictionRequest,
    SerializedTableProtocol,
    SerializedTableScope,
    TableSlice,
)

try:
    __version__ = _distribution_version("tablesuite")
except _PackageNotFoundError:
    __version__ = "1.3.0"

__all__ = [
    "Benchmark",
    "BenchmarkSubset",
    "Catalog",
    "CellFact",
    "DatasetSpec",
    "ICLPredictionExample",
    "ICLPredictionRequest",
    "ICLProtocol",
    "MaterializedTableSlice",
    "ParquetSource",
    "PredictionGold",
    "RenderedPrediction",
    "Selection",
    "SelectionManifest",
    "SerializedTablePredictionExample",
    "SerializedTablePredictionRequest",
    "SerializedTableProtocol",
    "SerializedTableScope",
    "TableSlice",
    "TaskDataset",
    "TaskExample",
    "TaskReport",
    "TaskScore",
    "load_task",
    "materialize_openml_sources",
    "render_icl_prediction",
    "render_serialized_table_prediction",
    "render_table",
    "__version__",
]
