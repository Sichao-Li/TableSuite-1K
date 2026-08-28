"""Public API for TableSuite-1K's Hugging Face-hosted table tasks."""

from tablesuite.benchmark import Benchmark, BenchmarkSubset
from tablesuite.catalog import Catalog
from tablesuite.generation import (
    GeneratedTaskDataset,
    GenerationManifest,
    generate_task,
    load_generated_task,
)
from tablesuite.prediction import (
    OFFICIAL_SUPPORT_LEVELS,
    PredictionDataset,
    PredictionInterface,
    PredictionManifest,
    normalize_support_fractions,
    resolve_support_level,
)
from tablesuite.registry import (
    FixedPredictionProtocol,
    PredictionProtocol,
    PublicTaskName,
    TaskDescriptor,
    describe_task,
    list_tasks,
)
from tablesuite.rendering import (
    render_icl_prediction,
    render_serialized_table_prediction,
    render_table,
)
from tablesuite.source import ParquetSource, materialize_openml_sources
from tablesuite.suite import TableSuite
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
    SupportLevel,
    TableSlice,
)

__version__ = "2.1.0"

__all__ = [
    "Benchmark",
    "BenchmarkSubset",
    "Catalog",
    "CellFact",
    "DatasetSpec",
    "FixedPredictionProtocol",
    "GeneratedTaskDataset",
    "GenerationManifest",
    "ICLPredictionExample",
    "ICLPredictionRequest",
    "ICLProtocol",
    "MaterializedTableSlice",
    "OFFICIAL_SUPPORT_LEVELS",
    "ParquetSource",
    "PredictionDataset",
    "PredictionGold",
    "PredictionInterface",
    "PredictionManifest",
    "PredictionProtocol",
    "PublicTaskName",
    "RenderedPrediction",
    "Selection",
    "SelectionManifest",
    "SerializedTablePredictionExample",
    "SerializedTablePredictionRequest",
    "SerializedTableProtocol",
    "SerializedTableScope",
    "SupportLevel",
    "TableSlice",
    "TableSuite",
    "TaskDescriptor",
    "TaskDataset",
    "TaskExample",
    "TaskReport",
    "TaskScore",
    "describe_task",
    "generate_task",
    "list_tasks",
    "load_task",
    "load_generated_task",
    "materialize_openml_sources",
    "normalize_support_fractions",
    "render_icl_prediction",
    "render_serialized_table_prediction",
    "render_table",
    "resolve_support_level",
    "__version__",
]
