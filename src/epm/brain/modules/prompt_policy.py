from __future__ import annotations

from dataclasses import dataclass

from epm.brain.interfaces import PlannerContext, PromptPolicy


@dataclass(frozen=True)
class DefaultPromptPolicy(PromptPolicy):
    """
    Minimal policy:
    - memory/tool-schema injection toggled by config flags injected into memory_bundle
    - percept included if present
    """

    inject_memory_default: bool = True
    inject_tool_schemas_default: bool = True
    include_percept_default: bool = True
    include_task_progress_default: bool = True

    def include_memory(self, *, context: PlannerContext) -> bool:
        return bool(context.memory_bundle.get("_inject_memory", self.inject_memory_default))

    def include_tool_schemas(self, *, context: PlannerContext) -> bool:
        return bool(context.memory_bundle.get("_inject_tool_schemas", self.inject_tool_schemas_default))

    def include_percept(self, *, context: PlannerContext) -> bool:
        if not bool(context.memory_bundle.get("_include_percept", self.include_percept_default)):
            return False
        return context.percept is not None and bool(context.percept.text.strip())

    def include_task_progress(self, *, context: PlannerContext) -> bool:
        if not bool(context.memory_bundle.get("_include_task_progress", self.include_task_progress_default)):
            return False
        return self.include_memory(context=context)
