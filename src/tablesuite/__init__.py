"""Public API for TableSuite-1K's Hugging Face-hosted table tasks."""

from tablesuite.benchmark import Benchmark, BenchmarkSubset
from tablesuite.catalog import Catalog
from tablesuite.generation import (
    GeneratedTaskDataset,
    GenerationManifest,
    generate_task,
    load_generated_task,
)
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

__version__ = "2.0.0"

__all__ = [
    "Benchmark",
    "BenchmarkSubset",
    "Catalog",
    "CellFact",
    "DatasetSpec",
    "GeneratedTaskDataset",
    "GenerationManifest",
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
    "generate_task",
    "load_task",
    "load_generated_task",
    "materialize_openml_sources",
    "render_icl_prediction",
    "render_serialized_table_prediction",
    "render_table",
    "__version__",
]
