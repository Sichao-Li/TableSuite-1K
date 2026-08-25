"""Internal execution of frozen task specifications."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tablesuite.benchmark import Benchmark
from tablesuite.evaluation.contracts import (
    EvaluationGold,
    EvaluationItem,
    EvaluationPlan,
    GenerationReceipt,
    PlanRegistry,
    ScoreResult,
)
from tablesuite.evaluation.operations import (
    EXECUTOR_VERSION,
    PredictionResolver,
    execute_operation,
)
from tablesuite.evaluation.rendering import (
    GENERATOR_VERSION,
    render_evaluation_request,
)
from tablesuite.evaluation.scoring import score_response
from tablesuite.evaluation.validation import PlanAudit, audit_plans


class PlanExecutor:
    """Bind official task specifications to deterministic source execution."""

    def __init__(
        self,
        benchmark: Benchmark,
        plans: PlanRegistry,
        *,
        prediction_resolver: PredictionResolver | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.plans = plans
        self.prediction_resolver = prediction_resolver
        self._datasets = {
            dataset.dataset_id: dataset for dataset in benchmark.catalog.datasets
        }
        self.audit = audit_plans(plans.plans)
        self.audit.require_passed()
        if plans.plans and plans.plans[0].reference_id != benchmark.catalog.reference_id:
            raise ValueError(
                "plan reference does not match the loaded benchmark catalog: "
                f"{plans.plans[0].reference_id!r} != {benchmark.catalog.reference_id!r}"
            )

    @classmethod
    def from_jsonl(
        cls,
        benchmark: Benchmark,
        plans_path: str | Path,
        *,
        prediction_resolver: PredictionResolver | None = None,
    ) -> PlanExecutor:
        """Load a nested internal plan manifest and bind it to source data."""

        return cls(
            benchmark,
            PlanRegistry.load(plans_path),
            prediction_resolver=prediction_resolver,
        )

    def get_plan(self, item_id: str) -> EvaluationPlan:
        """Return one frozen plan by ID."""

        return self.plans.get_plan(item_id)

    def materialize(self, plan: str | EvaluationPlan) -> EvaluationItem:
        """Resolve source values, execute gold, and render deterministic wording."""

        resolved = self.get_plan(plan) if isinstance(plan, str) else plan
        if not isinstance(plan, str) and self.get_plan(plan.item_id) != plan:
            raise ValueError("materialized plan differs from its frozen registry record")
        self._validate_versions(resolved)
        try:
            dataset = self._datasets[resolved.source.dataset_id]
        except KeyError as error:
            raise KeyError(
                f"plan dataset {resolved.source.dataset_id!r} is not in the catalog"
            ) from error
        if resolved.source_id != dataset.source_id:
            raise ValueError(
                f"item {resolved.item_id!r} expects source {resolved.source_id!r}, "
                f"catalog has {dataset.source_id!r}"
            )
        if resolved.dataset_split != dataset.dataset_split:
            raise ValueError(
                f"item {resolved.item_id!r} has the wrong dataset partition"
            )
        if resolved.dedup_cluster_id != dataset.dedup_cluster_id:
            raise ValueError(
                f"item {resolved.item_id!r} has the wrong deduplication cluster"
            )
        table = self.benchmark.source.materialize(dataset, resolved.source)
        result = execute_operation(
            resolved,
            table,
            prediction_resolver=self.prediction_resolver,
        )
        request = render_evaluation_request(resolved, table, result)
        gold = EvaluationGold(
            item_id=resolved.item_id,
            answer=result.answer,
            evidence=result.evidence,
        )
        return EvaluationItem(
            request=request,
            gold=gold,
            receipt=GenerationReceipt(
                item_id=resolved.item_id,
                reference_id=resolved.reference_id,
                source_id=resolved.source_id,
                generator_version=resolved.generator_version,
                executor_version=resolved.executor_version,
                template_id=request.template_id,
                source=resolved.source,
            ),
        )

    def score(self, plan: str | EvaluationPlan, response: Any) -> ScoreResult:
        """Materialize one specification and score a response against computed gold."""

        resolved = self.get_plan(plan) if isinstance(plan, str) else plan
        item = self.materialize(resolved)
        return score_response(response, item.gold, resolved.scoring)

    def validate(self) -> PlanAudit:
        """Return the static plan-manifest audit performed at construction."""

        return self.audit

    @staticmethod
    def _validate_versions(plan: EvaluationPlan) -> None:
        if plan.generator_version != GENERATOR_VERSION:
            raise ValueError(
                f"unsupported generator version: {plan.generator_version!r}"
            )
        if plan.executor_version != EXECUTOR_VERSION:
            raise ValueError(
                f"unsupported executor version: {plan.executor_version!r}"
            )
