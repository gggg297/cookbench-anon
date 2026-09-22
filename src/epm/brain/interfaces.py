from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

from epm.brain.plan_schema import PlanResponse, PlanStep
from epm.core.epm_types import Observation


@dataclass(frozen=True)
class Percept:
    """
    Output of a perception module, intended to be injected into planner prompt.
    """

    text: str
    slots: Dict[str, Any]


@dataclass(frozen=True)
class PlannerContext:
    high_level_id: str
    high_level_goal: str
    observation: Observation
    recipe_text: str
    feedback: str
    memory_bundle: Dict[str, str]
    percept: Optional[Percept] = None


class PerceptionModule(Protocol):
    def run(self, *, observation: Observation) -> Percept: ...


class PlannerModule(Protocol):
    def plan(self, *, context: PlannerContext) -> PlanResponse: ...


class PromptPolicy(Protocol):
    """
    Decide which prompt sections should be included for this round.
    """

    def include_memory(self, *, context: PlannerContext) -> bool: ...
    def include_tool_schemas(self, *, context: PlannerContext) -> bool: ...
    def include_percept(self, *, context: PlannerContext) -> bool: ...
    def include_task_progress(self, *, context: PlannerContext) -> bool: ...


class BrainPipeline(Protocol):
    """
    A pipeline produces the next executable step (action or skill) for the executor.
    """

    def next_step(self, *, context: PlannerContext) -> PlanStep: ...
    def on_step_result(self, *, step: PlanStep, success: bool, error: str) -> None: ...
