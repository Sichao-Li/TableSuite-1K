"""Static audits for official evaluation-plan manifests."""

from __future__ import annotations

from dataclasses import dataclass

from tablesuite.evaluation.contracts import EvaluationPlan
from tablesuite.evaluation.operations import validate_operation_spec


@dataclass(frozen=True)
class PlanAudit:
    """Result of checking identities, partitions, and source overlap."""

    plans: int
    datasets: int
    dedup_clusters: int
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether the manifest satisfies the official split contract."""

        return not self.errors

    def require_passed(self) -> None:
        """Raise one readable error when the manifest is invalid."""

        if self.errors:
            raise ValueError("invalid evaluation plan manifest:\n- " + "\n- ".join(self.errors))


def audit_plans(plans: tuple[EvaluationPlan, ...]) -> PlanAudit:
    """Check that frozen plans do not leak clusters or exact source cells."""

    errors: list[str] = []
    dataset_bindings: dict[str, tuple[str, str, str]] = {}
    cluster_partitions: dict[str, str] = {}
    cell_splits: dict[tuple[str, str, str], str] = {}
    expected_partitions = {
        "train": "train",
        "validation": "validation",
        "episode_test": "train",
        "dataset_test": "test",
        "template_test": "train",
        "composition_test": "train",
    }

    for plan in plans:
        try:
            validate_operation_spec(plan)
        except (TypeError, ValueError) as error:
            errors.append(f"item {plan.item_id!r}: {error}")

        binding = (plan.source_id, plan.dataset_split, plan.dedup_cluster_id)
        previous_binding = dataset_bindings.setdefault(plan.source.dataset_id, binding)
        if previous_binding != binding:
            errors.append(
                f"dataset {plan.source.dataset_id!r} has inconsistent source or split metadata"
            )

        previous_partition = cluster_partitions.setdefault(
            plan.dedup_cluster_id, plan.dataset_split
        )
        if previous_partition != plan.dataset_split:
            errors.append(
                f"dedup cluster {plan.dedup_cluster_id!r} spans dataset partitions "
                f"{previous_partition!r} and {plan.dataset_split!r}"
            )

        expected_partition = expected_partitions[plan.evaluation_split]
        if plan.dataset_split != expected_partition:
            errors.append(
                f"item {plan.item_id!r} in {plan.evaluation_split!r} must use "
                f"the {expected_partition!r} dataset partition"
            )
        expected_template_split = (
            "test" if plan.evaluation_split == "template_test" else "train"
        )
        if plan.rendering.template_split != expected_template_split:
            errors.append(
                f"item {plan.item_id!r} in {plan.evaluation_split!r} must use "
                f"{expected_template_split!r} templates"
            )

        for row_id in plan.source.row_ids:
            for column in plan.source.columns:
                cell = (plan.source.dataset_id, row_id, column)
                previous_split = cell_splits.setdefault(cell, plan.evaluation_split)
                if previous_split != plan.evaluation_split:
                    errors.append(
                        f"source cell {cell!r} appears in evaluation splits "
                        f"{previous_split!r} and {plan.evaluation_split!r}"
                    )

    return PlanAudit(
        plans=len(plans),
        datasets=len(dataset_bindings),
        dedup_clusters=len(cluster_partitions),
        errors=tuple(dict.fromkeys(errors)),
    )
