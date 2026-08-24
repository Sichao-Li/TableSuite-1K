"""Authoring contracts for frozen, render-late task specifications."""

from tablesuite.evaluation.contracts import (
    BENCHMARK_VERSION,
    PLAN_SCHEMA_VERSION,
    CellReference,
    EvaluationGold,
    EvaluationItem,
    EvaluationPlan,
    EvaluationRequest,
    GenerationReceipt,
    OperationSpec,
    PlanRegistry,
    RenderingSpec,
    ScoreResult,
    ScoringSpec,
)
from tablesuite.evaluation.executor import PlanExecutor
from tablesuite.evaluation.operations import (
    MappingPredictionResolver,
    PredictionResolver,
)
from tablesuite.evaluation.scoring import score_response
from tablesuite.evaluation.validation import PlanAudit, audit_plans

__all__ = [
    "BENCHMARK_VERSION",
    "CellReference",
    "EvaluationGold",
    "EvaluationItem",
    "EvaluationPlan",
    "EvaluationRequest",
    "GenerationReceipt",
    "MappingPredictionResolver",
    "OperationSpec",
    "PLAN_SCHEMA_VERSION",
    "PlanAudit",
    "PlanRegistry",
    "PlanExecutor",
    "PredictionResolver",
    "RenderingSpec",
    "ScoreResult",
    "ScoringSpec",
    "audit_plans",
    "score_response",
]
