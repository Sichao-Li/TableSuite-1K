"""Stable public API for TableSuite-1K."""

from tablesuite.generation import (
    GeneratedTaskDataset,
    GenerationManifest,
    generate_task,
    load_generated_task,
)
from tablesuite.prediction import (
    OFFICIAL_SUPPORT_LEVELS,
    BudgetedPredictionDataset,
    ContextBudgetReport,
    ContextFit,
    PredictionDataset,
    PredictionExample,
    PredictionInterface,
    PredictionManifest,
)
from tablesuite.prediction_evaluation import (
    PredictionFamilyReport,
    PredictionReport,
    evaluate_predictions,
)
from tablesuite.registry import (
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
from tablesuite.source import materialize_openml_sources
from tablesuite.suite import TableSuite
from tablesuite.tasks import (
    TaskDataset,
    TaskExample,
    TaskReport,
    TaskScore,
    load_task,
)
from tablesuite.types import (
    DatasetSpec,
    ICLPredictionExample,
    ICLPredictionRequest,
    MaterializedTableSlice,
    PredictionGold,
    RenderedPrediction,
    SerializedTablePredictionExample,
    SerializedTablePredictionRequest,
    SupportLevel,
    TableSlice,
)

__version__ = "2.1.0"

__all__ = [
    "BudgetedPredictionDataset",
    "ContextBudgetReport",
    "ContextFit",
    "DatasetSpec",
    "GeneratedTaskDataset",
    "GenerationManifest",
    "ICLPredictionExample",
    "ICLPredictionRequest",
    "MaterializedTableSlice",
    "OFFICIAL_SUPPORT_LEVELS",
    "PredictionDataset",
    "PredictionExample",
    "PredictionFamilyReport",
    "PredictionGold",
    "PredictionInterface",
    "PredictionManifest",
    "PredictionReport",
    "PublicTaskName",
    "RenderedPrediction",
    "SerializedTablePredictionExample",
    "SerializedTablePredictionRequest",
    "SupportLevel",
    "TableSlice",
    "TableSuite",
    "TaskDescriptor",
    "TaskDataset",
    "TaskExample",
    "TaskReport",
    "TaskScore",
    "describe_task",
    "evaluate_predictions",
    "generate_task",
    "list_tasks",
    "load_task",
    "load_generated_task",
    "materialize_openml_sources",
    "render_icl_prediction",
    "render_serialized_table_prediction",
    "render_table",
    "__version__",
]
