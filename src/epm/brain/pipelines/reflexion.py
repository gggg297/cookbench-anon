from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import traceback
from typing import Any, Optional

from epm.brain.interfaces import BrainPipeline, PlannerContext
from epm.brain.plan_schema import PlanResponse, PlanStep
from epm.brain.modules.reflexion import ReflexionConfig, ReflexionReflector
from epm.core.prompt_ablation import is_prompt_group_enabled


@dataclass
class ReflexionState:
    pending_step: Optional[PlanStep] = None
    pending_error: str = ""
    latest_step: Optional[PlanStep] = None
    latest_error: str = ""
    latest_success: bool = True
    last_reflection: str = ""


class ReflexionPipeline(BrainPipeline):
    """
    Reflexion wrapper:
    - run a base pipeline with queued execution
    - only reflect after a failed step, using the latest post-step context before replanning
    """

    def __init__(
        self,
        *,
        base: BrainPipeline,
        memory_dir: Path,
        chat_cfg: Any,
        reflexion_cfg: Optional[ReflexionConfig] = None,
    ) -> None:
        self.base = base
        self.state = ReflexionState()
        self.memory_dir = Path(memory_dir)
        self.cfg = reflexion_cfg or ReflexionConfig()
        self.reflector = ReflexionReflector(cfg=self.cfg, chat_cfg=chat_cfg, memory_dir=Path(memory_dir))
        self._reflection_enabled = (
            is_prompt_group_enabled(
                profile=self.cfg.prompt_ablation_profile,
                group="history",
                disabled_groups=self.cfg.prompt_disabled_groups,
            )
            and is_prompt_group_enabled(
                profile=self.cfg.prompt_ablation_profile,
                group="feedback",
                disabled_groups=self.cfg.prompt_disabled_groups,
            )
        )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _record_reflection_exception(self, *, stage: str) -> None:
        err = traceback.format_exc().strip()
        if not err:
            return
        line = f"[{self._utc_now_iso()}] stage={stage} exception={err}\n"
        try:
            path = self.memory_dir / "reflexion_errors.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
        try:
            print(f"[Reflexion][ERROR] {line.strip()}")
        except Exception:
            pass

    def _inject_latest_reflection(self, *, context: PlannerContext) -> PlannerContext:
        latest = str(self.state.last_reflection or "").strip()
        if not latest:
            return context
        bundle = dict(context.memory_bundle or {})
        existing = str(bundle.get("reflexion_memory", "") or "").strip()
        latest_block = (
            "# Latest reflection from the recent chunk.\n"
            f"- latest={latest}"
        )
        bundle["reflexion_memory"] = f"{existing}\n\n{latest_block}".strip() if existing else latest_block
        return PlannerContext(
            high_level_id=context.high_level_id,
            high_level_goal=context.high_level_goal,
            observation=context.observation,
            recipe_text=context.recipe_text,
            feedback=context.feedback,
            memory_bundle=bundle,
            percept=context.percept,
        )

    def next_step(self, *, context: PlannerContext) -> PlanStep:
        if not self._reflection_enabled:
            return self.base.next_step(context=context)
        if self.state.pending_step is not None:
            # Reflexion happens with the latest post-step context.
            try:
                patch = self.reflector.maybe_reflect(context=context, failed_step=self.state.pending_step, error=self.state.pending_error)
                if patch and isinstance(patch.get("reflection"), str):
                    self.state.last_reflection = patch["reflection"]
            except Exception:
                self._record_reflection_exception(stage="maybe_reflect")
            self.state.pending_step = None
            self.state.pending_error = ""
        else:
            try:
                patch = self.reflector.maybe_loop_reflect(
                    context=context,
                    latest_step=self.state.latest_step,
                    latest_error=self.state.latest_error,
                )
                if patch is None:
                    patch = self.reflector.maybe_periodic_reflect(
                        context=context,
                        latest_step=self.state.latest_step,
                        latest_error=self.state.latest_error,
                        latest_success=bool(self.state.latest_success),
                    )
                if patch and isinstance(patch.get("reflection"), str):
                    self.state.last_reflection = patch["reflection"]
            except Exception:
                self._record_reflection_exception(stage="maybe_loop_or_periodic_reflect")
        return self.base.next_step(context=self._inject_latest_reflection(context=context))

    def on_step_result(self, *, step: PlanStep, success: bool, error: str) -> None:
        self.base.on_step_result(step=step, success=success, error=error)
        if not self._reflection_enabled:
            return
        self.reflector.note_result(step=step, success=success, error=error)
        self.state.latest_step = step
        self.state.latest_error = error or ""
        self.state.latest_success = bool(success)
        if not success:
            self.state.pending_step = step
            self.state.pending_error = error or ""

    def consume_new_plan(self) -> Optional[PlanResponse]:
        consume = getattr(self.base, "consume_new_plan", None)
        if callable(consume):
            return consume()
        return None

    def current_remaining_plan(self) -> list[PlanStep]:
        getter = getattr(self.base, "current_remaining_plan", None)
        if callable(getter):
            return list(getter() or [])
        return []

    def needs_planner_round(self) -> bool:
        checker = getattr(self.base, "needs_planner_round", None)
        if callable(checker):
            return bool(checker())
        return len(self.current_remaining_plan()) <= 0
