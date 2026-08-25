"""Stable contracts for plan-frozen, render-late evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from tablesuite.types import TableSlice, TextView

EvaluationTask = Literal["grounding", "qa", "prediction", "integrated_reasoning"]
EvaluationSplit = Literal[
    "train",
    "validation",
    "episode_test",
    "dataset_test",
    "template_test",
    "composition_test",
]
DatasetPartition = Literal["train", "validation", "test"]
TemplateSplit = Literal["train", "test"]
AnswerType = Literal["string", "integer", "float", "boolean", "json"]
OperationName = Literal[
    "cell_lookup",
    "aggregate",
    "argmax_lookup",
    "filtered_argmax_lookup",
    "prediction_lookup",
    "prediction_with_cell",
]

PLAN_SCHEMA_VERSION = "1.1"
BENCHMARK_VERSION = "1.1"

_TASK_OPERATIONS: dict[str, frozenset[str]] = {
    "grounding": frozenset({"cell_lookup"}),
    "qa": frozenset({"aggregate", "argmax_lookup", "filtered_argmax_lookup"}),
    "prediction": frozenset({"prediction_lookup"}),
    "integrated_reasoning": frozenset({"prediction_with_cell"}),
}


@dataclass(frozen=True)
class CellReference:
    """One exact source cell used as evaluation evidence."""

    dataset_id: str
    row_id: str
    column: str


@dataclass(frozen=True)
class OperationSpec:
    """A typed operation containing source references and closed-vocabulary choices."""

    name: OperationName
    arguments: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.name not in {
            "cell_lookup",
            "aggregate",
            "argmax_lookup",
            "filtered_argmax_lookup",
            "prediction_lookup",
            "prediction_with_cell",
        }:
            raise ValueError(f"unsupported operation: {self.name!r}")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.arguments.items()
        ):
            raise TypeError("operation arguments must be string-to-string source references")
        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(sorted(self.arguments.items()))),
        )


@dataclass(frozen=True)
class RenderingSpec:
    """Deterministic runtime wording and table-serialization policy."""

    template_family: str
    template_split: TemplateSplit
    render_seed: int
    view: TextView = "markdown"
    language: str = "en"

    def __post_init__(self) -> None:
        if not self.template_family:
            raise ValueError("template family cannot be empty")
        if self.template_split not in {"train", "test"}:
            raise ValueError(f"unsupported template split: {self.template_split!r}")
        if self.view not in {"json", "key_value", "markdown"}:
            raise ValueError(f"unsupported table view: {self.view!r}")
        if self.language != "en":
            raise ValueError("the initial public renderer supports English only")


@dataclass(frozen=True)
class ScoringSpec:
    """Structured answer parsing and comparison policy."""

    answer_type: AnswerType
    case_sensitive: bool = False
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 0.0
    tie_policy: Literal["reject"] = "reject"
    missing_policy: Literal["reject"] = "reject"

    def __post_init__(self) -> None:
        if self.answer_type not in {"string", "integer", "float", "boolean", "json"}:
            raise ValueError(f"unsupported answer type: {self.answer_type!r}")
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("numeric tolerances cannot be negative")
        if self.tie_policy != "reject" or self.missing_policy != "reject":
            raise ValueError("the initial scorer only supports reject policies")


@dataclass(frozen=True)
class EvaluationPlan:
    """One immutable semantic evaluation item without wording, values, or gold."""

    schema_version: str
    benchmark_version: str
    reference_id: str
    item_id: str
    task: EvaluationTask
    evaluation_split: EvaluationSplit
    dataset_split: DatasetPartition
    dedup_cluster_id: str
    source_id: str
    source: TableSlice
    operation: OperationSpec
    rendering: RenderingSpec
    scoring: ScoringSpec
    generator_version: str
    executor_version: str
    prediction_packet_id: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_SCHEMA_VERSION:
            raise ValueError(f"unsupported plan schema version: {self.schema_version!r}")
        if self.benchmark_version != BENCHMARK_VERSION:
            raise ValueError(f"unsupported benchmark version: {self.benchmark_version!r}")
        if self.task not in _TASK_OPERATIONS:
            raise ValueError(f"unsupported evaluation task: {self.task!r}")
        if self.evaluation_split not in {
            "train",
            "validation",
            "episode_test",
            "dataset_test",
            "template_test",
            "composition_test",
        }:
            raise ValueError(f"unsupported evaluation split: {self.evaluation_split!r}")
        if self.dataset_split not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported dataset partition: {self.dataset_split!r}")
        required = {
            "schema_version": self.schema_version,
            "benchmark_version": self.benchmark_version,
            "reference_id": self.reference_id,
            "item_id": self.item_id,
            "dedup_cluster_id": self.dedup_cluster_id,
            "source_id": self.source_id,
            "generator_version": self.generator_version,
            "executor_version": self.executor_version,
        }
        if missing := sorted(name for name, value in required.items() if not value):
            raise ValueError(f"plan fields cannot be empty: {missing}")
        if self.operation.name not in _TASK_OPERATIONS[self.task]:
            raise ValueError(
                f"operation {self.operation.name!r} is not valid for task {self.task!r}"
            )
        if self.rendering.template_family != self.operation.name:
            raise ValueError("template family must match the operation name")
        needs_prediction = self.task in {"prediction", "integrated_reasoning"}
        if needs_prediction != bool(self.prediction_packet_id):
            raise ValueError(
                "prediction and integrated-reasoning plans require exactly one "
                "prediction packet reference"
            )

    def to_record(self) -> dict[str, Any]:
        """Return the deterministic JSON-compatible plan representation."""

        record = {
            "schema_version": self.schema_version,
            "benchmark_version": self.benchmark_version,
            "reference_id": self.reference_id,
            "item_id": self.item_id,
            "task": self.task,
            "evaluation_split": self.evaluation_split,
            "dataset_split": self.dataset_split,
            "dedup_cluster_id": self.dedup_cluster_id,
            "source_id": self.source_id,
            "source": asdict(self.source),
            "operation": {
                "name": self.operation.name,
                "arguments": [
                    {"name": name, "value": value}
                    for name, value in self.operation.arguments.items()
                ],
            },
            "rendering": asdict(self.rendering),
            "scoring": asdict(self.scoring),
            "generator_version": self.generator_version,
            "executor_version": self.executor_version,
        }
        if self.prediction_packet_id is not None:
            record["prediction_packet_id"] = self.prediction_packet_id
        return record

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> EvaluationPlan:
        """Load a plan from a JSON-compatible record."""

        values = dict(record)
        values["source"] = TableSlice(**values["source"])
        operation = dict(values["operation"])
        raw_arguments = operation.get("arguments") or {}
        if isinstance(raw_arguments, Mapping):
            argument_items = raw_arguments.items()
        elif raw_arguments and isinstance(raw_arguments[0], Mapping):
            argument_items = (
                (item["name"], item["value"])
                for item in raw_arguments
            )
        else:
            argument_items = dict(raw_arguments).items()
        operation["arguments"] = {
            str(key): str(value)
            for key, value in argument_items
            if value is not None
        }
        values["operation"] = OperationSpec(**operation)
        values["rendering"] = RenderingSpec(**values["rendering"])
        values["scoring"] = ScoringSpec(**values["scoring"])
        return cls(**values)


class PlanRegistry:
    """An immutable, indexed collection of official evaluation plans."""

    def __init__(self, plans: tuple[EvaluationPlan, ...]) -> None:
        ordered = tuple(sorted(plans, key=lambda plan: plan.item_id))
        ids = [plan.item_id for plan in ordered]
        if len(ids) != len(set(ids)):
            raise ValueError("plan IDs must be unique")
        if len({plan.schema_version for plan in ordered}) > 1:
            raise ValueError("one registry cannot mix plan schema versions")
        if len({plan.benchmark_version for plan in ordered}) > 1:
            raise ValueError("one registry cannot mix benchmark versions")
        if len({plan.reference_id for plan in ordered}) > 1:
            raise ValueError("one registry cannot mix reference catalogs")
        self._plans = ordered
        self._by_id = {plan.item_id: plan for plan in ordered}

    @property
    def plans(self) -> tuple[EvaluationPlan, ...]:
        """Return plans in stable plan-ID order."""

        return self._plans

    def get_plan(self, item_id: str) -> EvaluationPlan:
        """Return one frozen plan by its stable identifier."""

        try:
            return self._by_id[item_id]
        except KeyError as error:
            raise KeyError(f"unknown evaluation item: {item_id}") from error

    def select(
        self,
        *,
        task: EvaluationTask | None = None,
        split: EvaluationSplit | None = None,
    ) -> PlanRegistry:
        """Return plans matching one task and/or evaluation split."""

        return PlanRegistry(
            tuple(
                plan
                for plan in self._plans
                if (task is None or plan.task == task)
                and (split is None or plan.evaluation_split == split)
            )
        )

    def save(self, path: str | Path) -> None:
        """Write plans as deterministic JSON Lines without materialized values."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(plan.to_record(), ensure_ascii=False, sort_keys=True)
            for plan in self._plans
        ]
        destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> PlanRegistry:
        """Load an official local JSONL or Parquet plan manifest."""

        source = Path(path)
        if source.is_dir() or source.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as error:
                raise RuntimeError(
                    "install tablesuite[local] to load Parquet task files"
                ) from error
            shards = sorted(source.rglob("*.parquet")) if source.is_dir() else [source]
            if not shards:
                raise FileNotFoundError(f"no Parquet task files under {source}")
            records = [row for shard in shards for row in pq.read_table(shard).to_pylist()]
        else:
            records = [
                json.loads(line)
                for line in source.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return cls(tuple(EvaluationPlan.from_record(record) for record in records))

    @classmethod
    def from_huggingface(
        cls,
        repository: str,
        *,
        name: str,
        split: str,
        revision: str | None = None,
    ) -> PlanRegistry:
        """Load one official task configuration and split from the HF Hub."""

        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "install tablesuite[hf] to load a task from Hugging Face"
            ) from error
        rows = load_dataset(
            repository,
            name,
            split=split,
            revision=revision,
        )
        return cls(tuple(EvaluationPlan.from_record(dict(row)) for row in rows))


@dataclass(frozen=True)
class EvaluationRequest:
    """Input-only text request produced from one frozen plan."""

    item_id: str
    task: EvaluationTask
    input_text: str
    question: str
    table_text: str
    template_id: str


@dataclass(frozen=True)
class EvaluationGold:
    """Runtime-computed gold kept separate from the model request."""

    item_id: str
    answer: Any
    evidence: tuple[CellReference, ...]


@dataclass(frozen=True)
class GenerationReceipt:
    """Auditable record of how one runtime item was materialized."""

    item_id: str
    reference_id: str
    source_id: str
    generator_version: str
    executor_version: str
    template_id: str
    source: TableSlice


@dataclass(frozen=True)
class EvaluationItem:
    """One runtime request paired with separate gold and an audit receipt."""

    request: EvaluationRequest
    gold: EvaluationGold
    receipt: GenerationReceipt


@dataclass(frozen=True)
class ScoreResult:
    """Structured score for one model response."""

    item_id: str
    correct: bool
    exact_match: bool
    numeric_within_tolerance: bool | None
    parsed_answer: Any = None
    parse_error: str | None = None
