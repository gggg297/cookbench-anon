from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from epm.brain.interfaces import BrainPipeline, PlannerContext
from epm.brain.plan_schema import PlanStep
from epm.brain.modules.epm import EPMConfig, GoalDecomposer, SubgoalDoneJudge, load_goal_tree, save_goal_tree


@dataclass
class EPMState:
    subgoals: list[str] = None  # filled lazily
    current_index: int = 0
    steps_in_subgoal: int = 0
    last_judge_reason: str = ""


class EPMPipeline(BrainPipeline):
    """
    EPM-style hierarchical baseline:
    - decompose high-level goal into an ordered list of subgoals
    - execute current subgoal using a base executor pipeline (typically ReAct)
    - periodically judge if the subgoal is done, then advance
    """

    def __init__(
        self,
        *,
        base_executor: BrainPipeline,
        memory_dir: Path,
        chat_cfg: Any,
        epm_cfg: Optional[EPMConfig] = None,
        context_isolation: bool = True,
        persist_goal_tree: bool = True,
    ) -> None:
        self.base = base_executor
        self.cfg = epm_cfg or EPMConfig()
        self.memory_dir = Path(memory_dir)
        self.context_isolation = bool(context_isolation)
        self.persist_goal_tree = bool(persist_goal_tree)
        self.decomposer = GoalDecomposer(chat_cfg=chat_cfg, cfg=self.cfg, memory_dir=self.memory_dir)
        self.judge = SubgoalDoneJudge(chat_cfg=chat_cfg, cfg=self.cfg, memory_dir=self.memory_dir)
        self.state = EPMState(subgoals=[])

        # Attempt to resume a previously saved goal tree in this run folder.
        if self.persist_goal_tree:
            saved = load_goal_tree(path=self.memory_dir / "goal_tree.json")
            if saved and isinstance(saved.get("subgoals"), list):
                self.state.subgoals = [x for x in saved["subgoals"] if isinstance(x, str) and x.strip()]
                self.state.current_index = int(saved.get("current_index", 0) or 0)

    def _current_subgoal(self) -> str:
        if 0 <= int(self.state.current_index) < len(self.state.subgoals):
            return self.state.subgoals[int(self.state.current_index)]
        return ""

    def _compose_epm_goal_text(self, *, subgoal: str) -> str:
        total = max(1, len(self.state.subgoals))
        idx = min(max(0, int(self.state.current_index)), total - 1)
        lines = [
            f"[EPM subgoal {idx + 1}/{total}] {subgoal}\n"
            "EPM hard constraints:\n"
            "- Refine this current subgoal into executable steps; do not replace it with a broader convenience goal.\n"
            "- Do not gather/stage future-step tools or containers unless the current blocker explicitly requires them."
        ]
        if self.cfg.section_enabled("task_progress") or self.cfg.section_enabled("precondition_feedback"):
            lines.append("- Resolve supplied subgoal constraints before convenience actions.")
        return "\n".join(lines)

    def _ensure_subgoals(self, *, context: PlannerContext) -> None:
        if self.state.subgoals:
            return
        subs = self.decomposer.decompose(context=context)
        self.state.subgoals = subs
        self.state.current_index = 0
        self.state.steps_in_subgoal = 0
        if self.persist_goal_tree:
            save_goal_tree(
                path=self.memory_dir / "goal_tree.json",
                high_level_goal=context.high_level_goal,
                subgoals=self.state.subgoals,
                current_index=self.state.current_index,
            )

    def _maybe_advance_subgoal(self, *, context: PlannerContext) -> None:
        if not self.state.subgoals:
            return
        if self.state.current_index >= len(self.state.subgoals):
            return
        if not bool(self.cfg.subgoal_done_judge_enabled):
            return
        every = int(self.cfg.check_done_every_n_steps or 1)
        if every <= 0:
            every = 1
        if self.state.steps_in_subgoal <= 0:
            return
        if (self.state.steps_in_subgoal % every) != 0:
            return
        sg = self._current_subgoal()
        if not sg:
            return
        done, reason = self.judge.is_done(context=context, subgoal=sg)
        self.state.last_judge_reason = reason
        if done:
            self.state.current_index += 1
            self.state.steps_in_subgoal = 0
            if self.persist_goal_tree:
                save_goal_tree(
                    path=self.memory_dir / "goal_tree.json",
                    high_level_goal=context.high_level_goal,
                    subgoals=self.state.subgoals,
                    current_index=self.state.current_index,
                )

    def _base_needs_plan(self) -> bool:
        getter = getattr(self.base, "current_remaining_plan", None)
        if callable(getter):
            try:
                return len(list(getter() or [])) <= 0
            except Exception:
                return True
        return True

    def needs_planner_round(self) -> bool:
        if not self.state.subgoals:
            return True
        return self._base_needs_plan()

    def next_step(self, *, context: PlannerContext) -> PlanStep:
        self._ensure_subgoals(context=context)
        if self._base_needs_plan():
            self._maybe_advance_subgoal(context=context)

        sg = self._current_subgoal()
        if sg:
            mb = dict(context.memory_bundle or {})
            if self.context_isolation:
                # Baseline: context isolation (avoid parent-task context inflation).
                mb["_inject_memory"] = False
                mb["_include_task_progress"] = False
            sub_ctx = PlannerContext(
                high_level_id=context.high_level_id,
                high_level_goal=self._compose_epm_goal_text(subgoal=sg),
                observation=context.observation,
                recipe_text=context.recipe_text,
                feedback=context.feedback,
                memory_bundle=mb,
                percept=context.percept,
            )
            return self.base.next_step(context=sub_ctx)

        # Fallback: no decomposition produced; behave like base executor on the original goal.
        return self.base.next_step(context=context)

    def current_remaining_plan(self) -> list[PlanStep]:
        getter = getattr(self.base, "current_remaining_plan", None)
        if callable(getter):
            return getter()
        return []

    def consume_new_plan(self):
        consume = getattr(self.base, "consume_new_plan", None)
        if callable(consume):
            return consume()
        return None

    def set_atomic_counter(self, value: int) -> None:
        setter = getattr(self.base, "set_atomic_counter", None)
        if callable(setter):
            setter(value)

    def replace_remaining_plan(self, *, context: PlannerContext, plan) -> object:
        replacer = getattr(self.base, "replace_remaining_plan", None)
        if callable(replacer):
            return replacer(context=context, plan=plan)
        return plan

    def on_step_result(self, *, step: PlanStep, success: bool, error: str) -> None:
        self.base.on_step_result(step=step, success=success, error=error)
        if success and self._current_subgoal():
            self.state.steps_in_subgoal += 1
