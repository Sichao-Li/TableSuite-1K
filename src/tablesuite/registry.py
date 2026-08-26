"""User-facing descriptions of the TableSuite task families."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

PublicTaskName = Literal[
    "table_prediction",
    "table_grounding",
    "table_question_answering",
]
PredictionProtocol = Literal[
    "zero_shot_icl",
    "few_shot_icl",
    "zero_label_serialized_table",
    "partially_labeled_serialized_table",
]


@dataclass(frozen=True)
class TaskDescriptor:
    """Concise discovery metadata for one public task family."""

    name: PublicTaskName
    title: str
    purpose: str
    input: str
    output: str
    official: bool
    generatable: bool
    protocols: tuple[PredictionProtocol, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible task description."""

        return asdict(self)


_TASKS = (
    TaskDescriptor(
        name="table_prediction",
        title="Table Prediction",
        purpose="Predict the registered OpenML target without parameter updates.",
        input="Feature rows with zero or selected visible support labels.",
        output="One classification label or regression value per query row.",
        official=True,
        generatable=False,
        protocols=(
            "zero_shot_icl",
            "few_shot_icl",
            "zero_label_serialized_table",
            "partially_labeled_serialized_table",
        ),
    ),
    TaskDescriptor(
        name="table_grounding",
        title="Table Grounding",
        purpose="Recover or summarize facts from only the displayed table slice.",
        input="A provided row or subtable and a closed-world lookup question.",
        output="An exact source-grounded cell, row, value list, or value count.",
        official=True,
        generatable=True,
    ),
    TaskDescriptor(
        name="table_question_answering",
        title="Table Question Answering",
        purpose="Execute a typed operation over a displayed source subtable.",
        input="A source-grounded subtable and a deterministic question.",
        output="A programmatically computed table answer.",
        official=True,
        generatable=True,
    ),
)
_TASKS_BY_NAME: dict[str, TaskDescriptor] = {task.name: task for task in _TASKS}


def list_tasks() -> tuple[TaskDescriptor, ...]:
    """Return the public task families in display order."""

    return _TASKS


def describe_task(name: str) -> TaskDescriptor:
    """Return one task description or raise a clear discovery error."""

    task = _TASKS_BY_NAME.get(name)
    if task is not None:
        return task
    choices = ", ".join(_TASKS_BY_NAME)
    raise KeyError(f"unknown TableSuite task {name!r}; choose one of: {choices}")
