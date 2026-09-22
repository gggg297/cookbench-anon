from __future__ import annotations

import json
import re
from dataclasses import dataclass
from loguru import logger as logging

from epm.brain.interfaces import BrainPipeline, PlannerContext, PlannerModule, PromptPolicy
from epm.brain.plan_schema import PlanResponse, PlanStep
from epm.brain.prompt_builder import PromptBuilder
from epm.core.prompt_ablation import is_prompt_section_enabled


@dataclass
class PlannerExecutorState:
    current_plan: PlanResponse | None = None
    current_index: int = 0
    new_plan: PlanResponse | None = None
    atomic_step_counter: int = 0


class PlannerExecutorPipeline(BrainPipeline):
    """
    Rolling-horizon pipeline:
    - For the current high-level goal, request a multi-step plan (PlanResponse)
    - Execute steps sequentially; on failure stop and replan next round using feedback
    """

    def __init__(self, *, planner: PlannerModule, prompter: PromptBuilder, policy: PromptPolicy) -> None:
        self.planner = planner
        self.prompter = prompter
        self.policy = policy
        self.state = PlannerExecutorState()

    def _need_plan(self) -> bool:
        return self.state.current_plan is None or self.state.current_index >= len(self.state.current_plan.action_list)

    def set_atomic_counter(self, value: int) -> None:
        try:
            self.state.atomic_step_counter = max(0, int(value))
        except Exception:
            self.state.atomic_step_counter = 0

    @staticmethod
    def _episode_step_from_context(context: PlannerContext) -> int:
        try:
            bundle = context.memory_bundle or {}
            return max(0, int(bundle.get("_episode_step", 0) or 0))
        except Exception:
            return 0

    def _renumber_plan_steps(self, plan: PlanResponse, *, high_level_id: str, start_index: int | None = None) -> PlanResponse:
        # Keep plan step ids aligned with the current execution step so
        # `step=<episode_step>` and `step_id=H*.A<episode_step>` stay consistent after replans.
        if start_index is None:
            start = int(self.state.atomic_step_counter) + 1
        else:
            start = max(1, int(start_index))
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

    @staticmethod
    def _is_epm_context(context: PlannerContext) -> bool:
        name = str((context.memory_bundle or {}).get("_pipeline_name", "") or "").strip().lower()
        return name in ("epm", "epm_agent")

    @staticmethod
    def _planner_context_for_model(context: PlannerContext) -> PlannerContext:
        return context

    @staticmethod
    def _extract_task_progress_constraints(context: PlannerContext) -> tuple[str, list[str]]:
        bundle = context.memory_bundle or {}
        if not is_prompt_section_enabled(
            profile=bundle.get("_prompt_ablation_profile", "full"),
            section="task_progress",
            disabled_groups=bundle.get("_prompt_ablation_disabled_groups"),
        ):
            return "", []
        raw = str(bundle.get("task_progress", "") or "").strip()
        if not raw:
            return "", []
        try:
            obj = json.loads(raw)
        except Exception:
            return "", []
        if not isinstance(obj, dict):
            return "", []
        goal_state = obj.get("goal_state") if isinstance(obj.get("goal_state"), dict) else {}
        current_subtask = str(goal_state.get("current_subgoal") or "").strip()
        blockers_raw = obj.get("blocking_conditions") if isinstance(obj.get("blocking_conditions"), list) else []
        blockers: list[str] = []
        for item in blockers_raw:
            if isinstance(item, dict):
                detail = str(item.get("detail") or "").strip()
                if detail:
                    blockers.append(detail)
        return current_subtask, blockers

    @staticmethod
    def _extract_focus_terms(text: str) -> list[str]:
        raw = str(text or "").strip().lower()
        if not raw:
            return []
        generic_phrases = {
            "add to a",
            "combine",
            "current recipe step",
            "current subtask",
            "follow recipe",
        }
        stopwords = {
            "the", "and", "for", "with", "into", "onto", "from", "then", "that", "this",
            "step", "subtask", "recipe", "current", "must", "need", "required", "before",
            "after", "while", "until", "where", "which", "have", "been", "are", "all",
            "use", "using", "keep", "held", "item", "items", "tool", "tools", "container",
            "containers", "future", "later", "stage", "staging", "gather",
        }
        chunks = re.split(r"[,;:\n]+", raw)
        phrases: list[str] = []
        tokens: list[str] = []
        for chunk in chunks:
            cleaned = re.sub(r"\([^)]*\)", " ", chunk)
            cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:ml|g|kg|s|sec|secs|second|seconds|piece|pieces)\b", " ", cleaned)
            cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)
            cleaned = " ".join(cleaned.split())
            if not cleaned or cleaned in generic_phrases:
                continue
            if len(cleaned) >= 4:
                phrases.append(cleaned)
            for tok in cleaned.split():
                if len(tok) >= 4 and tok not in stopwords:
                    tokens.append(tok)
        seen: set[str] = set()
        out: list[str] = []
        for term in phrases + tokens:
            if term not in seen:
                seen.add(term)
                out.append(term)
        return out

    @staticmethod
    def _render_plan_text(plan: PlanResponse) -> str:
        parts: list[str] = [str(plan.goal or ""), str(plan.thoughts or "")]
        for step in list(plan.action_list or [])[:8]:
            parts.append(str(step.name or ""))
            if step.expectation:
                parts.append(str(step.expectation))
            for k, v in dict(step.args or {}).items():
                parts.append(f"{k}={v}")
        return " ".join(parts).lower()

    def _validate_epm_hard_subtask_alignment(self, *, context: PlannerContext, plan: PlanResponse) -> tuple[bool, str]:
        if not self._is_epm_context(context):
            return True, ""
        current_subtask, blockers = self._extract_task_progress_constraints(context)
        if not current_subtask:
            return True, ""
        plan_text = self._render_plan_text(plan)
        subtask_terms = self._extract_focus_terms(current_subtask)
        blocker_terms = self._extract_focus_terms(" ".join(blockers[:4]))
        matched_subtask = [t for t in subtask_terms if t and t in plan_text]
        matched_blockers = [t for t in blocker_terms if t and t in plan_text]
        if matched_subtask or matched_blockers:
            return True, ""

        staging_keywords = [
            "side table",
            "sink place point",
            "right sink place point",
            "left sink place point",
            "stage",
            "staging",
            "gather cookware",
            "gather utensils",
            "nearby side table",
            "placement point",
            "free hands",
        ]
        staging_hits = [kw for kw in staging_keywords if kw in plan_text]
        if staging_hits:
            return (
                False,
                f"hard_current_subtask={current_subtask!r}; "
                f"rejected_staging_keywords={staging_hits[:4]!r}"
            )
        return True, ""

    def next_step(self, *, context: PlannerContext) -> PlanStep:
        if self._need_plan():
            inject_memory = self.policy.include_memory(context=context)
            inject_tool_schemas = self.policy.include_tool_schemas(context=context)
            percept_text = context.percept.text if (self.policy.include_percept(context=context) and context.percept) else ""
            include_task_progress = self.policy.include_task_progress(context=context)
            planner_feedback = (context.feedback or "").strip()
            plan_raw: PlanResponse | None = None
            last_alignment_reason = ""
            for attempt in range(2):
                prompt = self.prompter.build_planner_prompt(
                    observation=context.observation,
                    high_level_id=context.high_level_id,
                    high_level_goal=context.high_level_goal,
                    memory_bundle=context.memory_bundle,
                    feedback=planner_feedback,
                    percept_text=percept_text,
                    inject_memory=inject_memory,
                    inject_tool_schemas=inject_tool_schemas,
                    include_task_progress=include_task_progress,
                )
                if "tools_manifest_openai_truncated=true" in prompt:
                    logging.warning("[EPM] tools_manifest_openai prompt text truncated (see prompt_last.txt). Consider increasing limits.tools_manifest_max_chars or reducing injected tool text.")
                candidate_plan = self.planner.plan(
                    context=self._planner_context_for_model(context),
                    prompt=prompt,
                )
                valid, reason = self._validate_epm_hard_subtask_alignment(context=context, plan=candidate_plan)
                if valid:
                    plan_raw = candidate_plan
                    break
                last_alignment_reason = reason
                logging.warning(f"[EPM] rejected_plan_due_to_subtask_drift attempt={attempt + 1} reason={reason}")
                planner_feedback = (
                    f"{planner_feedback}\n\n"
                    f"EPM hard-subtask alignment failure:\n{reason}\n"
                    "Rewrite the plan so it directly advances the current subtask. "
                    "Do not broad-stage future-step tools unless the current blocker explicitly requires them."
                ).strip()
            if plan_raw is None:
                raise RuntimeError(f"epm_subtask_alignment_rejected:{last_alignment_reason}")
            start_index = self._episode_step_from_context(context)
            plan = self._renumber_plan_steps(
                plan_raw,
                high_level_id=context.high_level_id,
                start_index=start_index if start_index > 0 else None,
            )
            self.state.current_plan = plan
            self.state.current_index = 0
            self.state.new_plan = plan

        assert self.state.current_plan is not None
        return self.state.current_plan.action_list[self.state.current_index]

    def consume_new_plan(self) -> PlanResponse | None:
        """
        Return the most recently generated plan (if any) and clear the flag.

        This allows the executor/agent to update task_progress with the new atomic plan.
        """
        p = self.state.new_plan
        self.state.new_plan = None
        return p

    def current_remaining_plan(self) -> list[PlanStep]:
        if self.state.current_plan is None:
            return []
        idx = max(0, int(self.state.current_index))
        return list(self.state.current_plan.action_list[idx:])

    def replace_remaining_plan(self, *, context: PlannerContext, plan: PlanResponse) -> PlanResponse:
        """
        Replace the current pending suffix with a repaired remaining plan.
        """
        start_index = self._episode_step_from_context(context)
        repaired = self._renumber_plan_steps(
            plan,
            high_level_id=context.high_level_id,
            start_index=start_index if start_index > 0 else None,
        )
        prefix: list[PlanStep] = []
        if self.state.current_plan is not None:
            idx = max(0, int(self.state.current_index))
            prefix = list(self.state.current_plan.action_list[:idx])
        merged = prefix + list(repaired.action_list)
        self.state.current_plan = PlanResponse(
            high_level_id=context.high_level_id,
            goal=repaired.goal,
            explanation=repaired.explanation,
            thoughts=repaired.thoughts,
            action_list=merged,
        )
        self.state.new_plan = repaired
        self.state.current_index = len(prefix)
        return repaired

    def on_step_result(self, *, step: PlanStep, success: bool, error: str) -> None:
        if success:
            self.state.current_index += 1
        else:
            # Force replan on the next call; do not keep executing a failed (or invalid) plan.
            self.state.current_plan = None
            self.state.current_index = 0
