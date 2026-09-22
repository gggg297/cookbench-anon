from __future__ import annotations

from dataclasses import dataclass
from loguru import logger as logging

from epm.brain.interfaces import BrainPipeline, PlannerContext, PlannerModule, PromptPolicy
from epm.brain.plan_schema import PlanResponse, PlanStep
from epm.brain.prompt_builder import PromptBuilder


@dataclass
class ReActState:
    current_plan: PlanResponse | None = None
    current_index: int = 0
    new_plan: PlanResponse | None = None
    atomic_step_counter: int = 0


class ReActPipeline(BrainPipeline):
    """
    ReAct-style pipeline (observe-plan-act interleaving):

    Observe -> Think/Plan(a short chunk) -> Act -> Observe ...

    Notes:
    - The planner is still a structured planner that outputs a JSON plan schema,
      but prompt/count constraints come from config.
    - We keep executing the current planned suffix until it succeeds or fails.
    """

    def __init__(self, *, planner: PlannerModule, prompter: PromptBuilder, policy: PromptPolicy) -> None:
        self.planner = planner
        self.prompter = prompter
        self.policy = policy
        self.state = ReActState()

    def _need_plan(self) -> bool:
        return self.state.current_plan is None or self.state.current_index >= len(self.state.current_plan.action_list)

    def _renumber_plan_steps(self, plan: PlanResponse, *, high_level_id: str) -> PlanResponse:
        start = int(self.state.atomic_step_counter) + 1
        hid = str(high_level_id or plan.high_level_id or "H1")
        new_steps: list[PlanStep] = []
        for offset, s in enumerate(plan.action_list):
            idx = start + offset
            new_steps.append(
                PlanStep(
                    step_id=f"{hid}.A{idx}",
                    type=s.type,
                    name=s.name,
                    args=dict(s.args),
                    expectation=s.expectation,
                )
            )
        self.state.atomic_step_counter += len(new_steps)
        return PlanResponse(
            high_level_id=plan.high_level_id,
            goal=plan.goal,
            explanation=plan.explanation,
            thoughts=plan.thoughts,
            action_list=new_steps,
        )

    def next_step(self, *, context: PlannerContext) -> PlanStep:
        if self._need_plan():
            inject_memory = self.policy.include_memory(context=context)
            inject_tool_schemas = self.policy.include_tool_schemas(context=context)
            percept_text = context.percept.text if (self.policy.include_percept(context=context) and context.percept) else ""
            include_task_progress = self.policy.include_task_progress(context=context)

            prompt = self.prompter.build_react_prompt(
                observation=context.observation,
                high_level_id=context.high_level_id,
                high_level_goal=context.high_level_goal,
                memory_bundle=context.memory_bundle,
                feedback=context.feedback,
                percept_text=percept_text,
                inject_memory=inject_memory,
                inject_tool_schemas=inject_tool_schemas,
                include_task_progress=include_task_progress,
            )
            if "tools_manifest_openai_truncated=true" in prompt:
                logging.warning("[EPM] tools_manifest_openai prompt text truncated (see prompt_last.txt).")
            plan_raw = self.planner.plan(context=context, prompt=prompt)
            plan = self._renumber_plan_steps(plan_raw, high_level_id=context.high_level_id)
            self.state.current_plan = plan
            self.state.current_index = 0
            self.state.new_plan = plan

        assert self.state.current_plan is not None
        return self.state.current_plan.action_list[self.state.current_index]

    def consume_new_plan(self) -> PlanResponse | None:
        p = self.state.new_plan
        self.state.new_plan = None
        return p

    def on_step_result(self, *, step: PlanStep, success: bool, error: str) -> None:
        if success:
            self.state.current_index += 1
        else:
            self.state.current_plan = None
            self.state.current_index = 0
