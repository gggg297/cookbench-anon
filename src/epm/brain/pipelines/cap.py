from __future__ import annotations

import json
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from epm.brain.interfaces import BrainPipeline, PlannerContext
from epm.brain.plan_schema import PlanResponse, PlanStep
from epm.brain.modules.cap import CaPCodegen, CaPConfig, CapProgram


@dataclass
class CaPState:
    program: Optional[CapProgram] = None
    step_counter: int = 0
    last_error: str = ""
    pending_regen_reason: str = "initial"
    consecutive_failure_streak: int = 0
    last_failed_step_signature: str = ""
    empty_query_base: str = ""
    empty_query_streak: int = 0
    positive_query_base: str = ""
    positive_query_streak: int = 0
    last_high_level_id: str = ""
    last_high_level_goal: str = ""
    non_physical_streak: int = 0
    last_step_signature: str = ""
    repeated_step_streak: int = 0
    current_plan: Optional[PlanResponse] = None
    current_index: int = 0
    new_plan: Optional[PlanResponse] = None


_NON_PHYSICAL_STEPS = {
    ("skill", "auto_perception"),
    ("skill", "query_scene_objects"),
    ("skill", "list_supported_items"),
    ("action", "countdown_time"),
}
_STRICT_REPEAT_THRESHOLDS = {
    ("skill", "gui_order_dish_via_computer"): 2,
    ("action", "countdown_time"): 3,
}
_DEFAULT_REPEAT_THRESHOLD = 4
_LOCAL_REPAIR_ERROR_PATTERNS = (
    "target_not_found",
    "enter_pouring_mode_failed",
    "not in pouring mode",
    "still_holding_item",
    "interaction_mode_still_active",
    "navigation_failed_target_not_found",
    "invalid_instance",
    "invalid_target_instance",
)
_DEFAULT_FAILURE_REGEN_THRESHOLD = 2
_LOCAL_REPAIR_FAILURE_REGEN_THRESHOLD = 3


def _merge_preserving_non_null(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = dict(base or {})
    for key, value in dict(incoming or {}).items():
        if value is None and key in out and out.get(key) is not None:
            continue
        out[key] = value
    return out


class CaPPipeline(BrainPipeline):
    """
    CaP planning-text pipeline:
    - use the CaP prompt style to generate Python-like planning code
    - statically extract sequential yield steps from that code
    - execute the extracted action/skill queue
    - repair by regenerating only the remaining suffix
    """

    def __init__(
        self,
        *,
        memory_dir: Path,
        chat_cfg: Any,
        cap_cfg: Optional[CaPConfig] = None,
        plan_min_steps: int = 1,
        plan_max_steps: int = 10,
    ) -> None:
        self.memory_dir = Path(memory_dir)
        self.cfg = cap_cfg or CaPConfig()
        self.codegen = CaPCodegen(chat_cfg=chat_cfg, cfg=self.cfg, memory_dir=self.memory_dir)
        self.state = CaPState()
        self.plan_min_steps = max(1, int(plan_min_steps))
        self.plan_max_steps = max(self.plan_min_steps, int(plan_max_steps))

    @staticmethod
    def _load_json_dict(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}
        return {}

    @classmethod
    def _build_resume_state(
        cls,
        *,
        context: PlannerContext,
        runtime_payload: dict[str, Any],
        durable_runtime: dict[str, Any],
    ) -> dict[str, Any]:
        resume_state: dict[str, Any] = {"continue_from_existing_progress": True}
        if durable_runtime:
            if durable_runtime.get("stage") is not None:
                resume_state["stage"] = durable_runtime.get("stage")
            if durable_runtime.get("_last_error"):
                resume_state["last_error"] = durable_runtime.get("_last_error")
            if isinstance(durable_runtime.get("_last_yield"), dict):
                resume_state["last_yield"] = durable_runtime.get("_last_yield")
        if isinstance(runtime_payload.get("last_step_result"), dict):
            resume_state["last_step_result"] = dict(runtime_payload.get("last_step_result") or {})
        if isinstance(runtime_payload.get("postconditions"), dict):
            resume_state["postconditions"] = dict(runtime_payload.get("postconditions") or {})
        recent_steps = runtime_payload.get("recent_steps")
        if isinstance(recent_steps, list) and recent_steps:
            resume_state["recent_steps"] = [it for it in recent_steps[-6:] if isinstance(it, dict)]
        agent_state = cls._load_json_dict(context.memory_bundle.get("agent_state", ""))
        ordered_names = agent_state.get("ordered_dish_names")
        if isinstance(ordered_names, list) and ordered_names:
            resume_state["dish_already_ordered"] = True
            resume_state["ordered_dish_names"] = [str(it) for it in ordered_names[:8]]
        last_ordered_name = str(agent_state.get("last_ordered_dish_name") or "").strip()
        if last_ordered_name:
            resume_state["last_ordered_dish_name"] = last_ordered_name
        return resume_state

    @staticmethod
    def _runtime_state_path(*, memory_dir: Path) -> Path:
        return Path(memory_dir) / "cap_runtime_state.json"

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _load_runtime_state(self) -> dict[str, Any]:
        path = self._runtime_state_path(memory_dir=self.memory_dir)
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_runtime_state(self, payload: dict[str, Any]) -> None:
        path = self._runtime_state_path(memory_dir=self.memory_dir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def _snapshot_runtime_state(
        self,
        *,
        yielded: Optional[dict[str, Any]] = None,
        step: Optional[PlanStep] = None,
        success: Optional[bool] = None,
        error: str = "",
    ) -> None:
        current = self._load_runtime_state()
        payload: dict[str, Any] = dict(current)
        payload["version"] = 1
        payload["updated_at"] = self._utc_now_iso()
        payload["high_level_id"] = self.state.last_high_level_id
        payload["high_level_goal"] = self.state.last_high_level_goal

        runtime_state = {}
        if self.state.program is not None and isinstance(self.state.program.ctx.get("runtime_state"), dict):
            runtime_state = dict(self.state.program.ctx.get("runtime_state") or {})
        previous_runtime_state = dict(current.get("runtime_state") or {}) if isinstance(current.get("runtime_state"), dict) else {}
        payload["runtime_state"] = _merge_preserving_non_null(previous_runtime_state, runtime_state)

        if isinstance(yielded, dict):
            payload["last_yield"] = dict(yielded)
        if step is not None:
            payload["last_step_result"] = {
                "step_id": str(step.step_id or ""),
                "type": str(step.type or ""),
                "name": str(step.name or ""),
                "success": bool(success),
                "error": str(error or ""),
            }
            recent_steps = list(current.get("recent_steps") or []) if isinstance(current.get("recent_steps"), list) else []
            recent_steps.append(
                {
                    "step_id": str(step.step_id or ""),
                    "type": str(step.type or ""),
                    "name": str(step.name or ""),
                    "args": dict(step.args or {}),
                    "success": bool(success),
                    "error": str(error or ""),
                }
            )
            payload["recent_steps"] = [it for it in recent_steps[-8:] if isinstance(it, dict)]
            postconditions = dict(current.get("postconditions") or {}) if isinstance(current.get("postconditions"), dict) else {}
            if success is True:
                if step.name == "gui_order_dish_via_computer":
                    postconditions["dish_ordered"] = True
                    dish_name = str((step.args or {}).get("dish_name") or "").strip()
                    if dish_name:
                        postconditions["ordered_dish_name"] = dish_name
                elif step.name == "gui_submit_dish_via_checkout_stand":
                    postconditions["dish_submitted"] = True
            payload["postconditions"] = postconditions

        payload["loop_guard"] = {
            "non_physical_streak": int(self.state.non_physical_streak),
            "repeated_step_streak": int(self.state.repeated_step_streak),
            "last_step_signature": str(self.state.last_step_signature or ""),
            "consecutive_failure_streak": int(self.state.consecutive_failure_streak),
            "last_failed_step_signature": str(self.state.last_failed_step_signature or ""),
        }
        if error:
            payload["last_error"] = str(error or "")
        elif success is True:
            payload["last_error"] = ""
        self._write_runtime_state(payload)

    @staticmethod
    def _feedback_line_value(feedback: str, key: str) -> str:
        prefix = f"{key}="
        for raw in str(feedback or "").splitlines():
            line = raw.strip()
            if line.startswith(prefix):
                return line[len(prefix) :].strip()
        return ""

    @classmethod
    def _normalize_query_base(cls, query: str) -> str:
        text = str(query or "").strip()
        if not text:
            return ""
        text = re.sub(r"(?:_alt\d+)+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+(source|stack)\s*$", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip().lower()

    @staticmethod
    def _step_signature(*, step_type: str, step_name: str, args: dict[str, Any]) -> str:
        try:
            args_text = json.dumps(dict(args or {}), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            args_text = json.dumps({}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{str(step_type or '').strip().lower()}::{str(step_name or '').strip()}::{args_text}"

    @staticmethod
    def _repeat_threshold(*, step_type: str, step_name: str) -> int:
        return int(_STRICT_REPEAT_THRESHOLDS.get((str(step_type or ""), str(step_name or "")), _DEFAULT_REPEAT_THRESHOLD))

    @staticmethod
    def _is_local_repair_error(error: str) -> bool:
        text = str(error or "").strip().lower()
        if not text:
            return False
        return any(pat in text for pat in _LOCAL_REPAIR_ERROR_PATTERNS)

    @classmethod
    def _failure_regen_threshold(cls, *, error: str) -> int:
        if cls._is_local_repair_error(error):
            return int(_LOCAL_REPAIR_FAILURE_REGEN_THRESHOLD)
        return int(_DEFAULT_FAILURE_REGEN_THRESHOLD)

    def _record_local_repair_hint(self, *, step: PlanStep, error: str) -> None:
        if self.state.program is None:
            return
        runtime_state = self.state.program.ctx.get("runtime_state")
        if not isinstance(runtime_state, dict):
            runtime_state = {}
            self.state.program.ctx["runtime_state"] = runtime_state
        runtime_state["_last_error"] = str(error or "")
        runtime_state["_local_repair"] = {
            "step_type": str(step.type or ""),
            "step_name": str(step.name or ""),
            "error": str(error or ""),
            "failure_streak": int(self.state.consecutive_failure_streak),
            "repair_mode": "repair_remaining_suffix",
        }

    def set_atomic_counter(self, value: int) -> None:
        try:
            self.state.step_counter = max(0, int(value))
        except Exception:
            self.state.step_counter = 0

    def needs_planner_round(self) -> bool:
        return self.state.current_plan is None or self.state.current_index >= len(self.state.current_plan.action_list)

    def consume_new_plan(self) -> PlanResponse | None:
        plan = self.state.new_plan
        self.state.new_plan = None
        return plan

    def current_remaining_plan(self) -> list[PlanStep]:
        if self.state.current_plan is None:
            return []
        idx = max(0, int(self.state.current_index))
        return list(self.state.current_plan.action_list[idx:])

    def _clear_current_plan(self) -> None:
        self.state.current_plan = None
        self.state.current_index = 0
        self.state.new_plan = None

    def _next_atomic_step_id(self, *, high_level_id: str) -> str:
        self.state.step_counter += 1
        return f"{high_level_id}.A{self.state.step_counter}"

    @staticmethod
    def _build_plan_chunk(*, high_level_id: str, high_level_goal: str, steps: list[PlanStep]) -> PlanResponse:
        return PlanResponse(
            high_level_id=str(high_level_id or "H1"),
            goal=str(high_level_goal or ""),
            explanation=None,
            thoughts="",
            action_list=list(steps),
        )

    def _populate_program_ctx(self, *, program_ctx: dict[str, Any], context: PlannerContext) -> None:
        program_ctx["goal"] = context.high_level_goal
        program_ctx["feedback"] = context.feedback
        program_ctx["on_screen_names"] = (context.observation.state.get("_on_screen_names", []) or [])[:50]
        program_ctx["on_screen_objects"] = (context.observation.state.get("_on_screen_objects", []) or [])[:50]
        program_ctx["agent_state"] = context.memory_bundle.get("agent_state", "")
        program_ctx["query_memory"] = context.memory_bundle.get("query_memory", "")
        program_ctx["instance_query_snapshot"] = context.memory_bundle.get("instance_query_snapshot", "")

        runtime_payload = self._load_json_dict(context.memory_bundle.get("cap_runtime_state", ""))
        durable_runtime = {}
        if isinstance(runtime_payload.get("runtime_state"), dict):
            durable_runtime = _merge_preserving_non_null(durable_runtime, runtime_payload.get("runtime_state") or {})
        if isinstance(program_ctx.get("runtime_state"), dict):
            durable_runtime = _merge_preserving_non_null(durable_runtime, program_ctx.get("runtime_state") or {})
        if runtime_payload.get("last_error"):
            durable_runtime["_last_error"] = runtime_payload.get("last_error")
        if isinstance(runtime_payload.get("last_yield"), dict):
            durable_runtime["_last_yield"] = runtime_payload.get("last_yield")
        program_ctx["runtime_state"] = durable_runtime
        program_ctx["resume_state"] = self._build_resume_state(
            context=context,
            runtime_payload=runtime_payload,
            durable_runtime=durable_runtime,
        )

    @staticmethod
    def _candidate_name(item: dict[str, Any]) -> str:
        for key in ("name_en", "name", "name_cn"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    def _maybe_force_regen_for_empty_query_loop(self, *, context: PlannerContext) -> None:
        feedback = str(context.feedback or "")
        if "type=skill name=query_scene_objects" not in feedback:
            self.state.empty_query_base = ""
            self.state.empty_query_streak = 0
            self.state.positive_query_base = ""
            self.state.positive_query_streak = 0
            return

        results_len_raw = self._feedback_line_value(feedback, "results_len")
        try:
            results_len = int(results_len_raw)
        except Exception:
            results_len = -1

        args_raw = self._feedback_line_value(feedback, "args")
        query_text = ""
        if args_raw:
            try:
                parsed = json.loads(args_raw)
                if isinstance(parsed, dict):
                    query_text = str(parsed.get("query", "") or "")
            except Exception:
                query_text = ""

        if results_len > 0:
            self.state.empty_query_base = ""
            self.state.empty_query_streak = 0
            query_base = self._normalize_query_base(query_text)
            if query_base and query_base == self.state.positive_query_base:
                self.state.positive_query_streak += 1
            else:
                self.state.positive_query_base = query_base
                self.state.positive_query_streak = 1
            if self.state.positive_query_streak >= 2:
                self._snapshot_runtime_state(error=f"positive_query_loop:base={query_base or 'unknown'}")
                self.state.program = None
                self._clear_current_plan()
                self.state.pending_regen_reason = (
                    f"positive_query_loop:base={query_base or 'unknown'}:streak={self.state.positive_query_streak}"
                )
                self.state.positive_query_base = ""
                self.state.positive_query_streak = 0
            return

        if results_len != 0:
            return

        self.state.positive_query_base = ""
        self.state.positive_query_streak = 0
        query_base = self._normalize_query_base(query_text)
        if query_base and query_base == self.state.empty_query_base:
            self.state.empty_query_streak += 1
        else:
            self.state.empty_query_base = query_base
            self.state.empty_query_streak = 1

        alt_count = len(re.findall(r"_alt\d+", str(query_text or ""), flags=re.IGNORECASE))
        if self.state.empty_query_streak >= 2 or alt_count >= 2 or len(str(query_text or "")) >= 120:
            self._snapshot_runtime_state(error=f"empty_query_loop:base={query_base or 'unknown'}")
            self.state.program = None
            self._clear_current_plan()
            self.state.pending_regen_reason = (
                f"empty_query_loop:base={query_base or 'unknown'}:streak={self.state.empty_query_streak}:alts={alt_count}"
            )
            self.state.empty_query_base = ""
            self.state.empty_query_streak = 0

    def _ensure_program(self, *, context: PlannerContext, reason: Optional[str] = None) -> None:
        if self.state.program is not None:
            return
        regen_reason = str(reason or self.state.pending_regen_reason or "initial")
        program = self.codegen.generate(context=context, reason=regen_reason, max_steps=self.plan_max_steps)
        self._populate_program_ctx(program_ctx=program.ctx, context=context)
        self.state.program = program
        self.state.pending_regen_reason = ""

    def _materialize_plan_chunk(self, *, context: PlannerContext) -> PlanResponse:
        self.state.last_high_level_id = str(context.high_level_id or "")
        self.state.last_high_level_goal = str(context.high_level_goal or "")
        self._maybe_force_regen_for_empty_query_loop(context=context)
        self._ensure_program(context=context)
        assert self.state.program is not None

        steps: list[PlanStep] = []
        for raw in list(self.state.program.plan_steps or [])[: self.plan_max_steps]:
            typ = str(raw.get("type") or "").strip().lower()
            name = str(raw.get("name") or "").strip()
            args = raw.get("args") if isinstance(raw.get("args"), dict) else {}
            if typ not in {"action", "skill"} or not name:
                continue
            yielded_step = {"type": typ, "name": name, "args": dict(args)}
            if (typ, name) in _NON_PHYSICAL_STEPS:
                self.state.non_physical_streak += 1
            else:
                self.state.non_physical_streak = 0
            self._snapshot_runtime_state(yielded=yielded_step)
            steps.append(
                PlanStep(
                step_id=self._next_atomic_step_id(high_level_id=context.high_level_id),
                type=typ,
                name=name,
                args=dict(args),
                expectation="",
            )
            )
        if not steps:
            raise RuntimeError("cap_empty_plan_chunk")
        return self._build_plan_chunk(
            high_level_id=context.high_level_id,
            high_level_goal=context.high_level_goal,
            steps=steps,
        )

    def next_step(self, *, context: PlannerContext) -> PlanStep:
        self.state.last_high_level_id = str(context.high_level_id or "")
        self.state.last_high_level_goal = str(context.high_level_goal or "")
        self._maybe_force_regen_for_empty_query_loop(context=context)
        if self.needs_planner_round():
            plan = self._materialize_plan_chunk(context=context)
            self.state.current_plan = plan
            self.state.current_index = 0
            self.state.new_plan = plan
        assert self.state.current_plan is not None
        return self.state.current_plan.action_list[self.state.current_index]

    def on_step_result(self, *, step: PlanStep, success: bool, error: str) -> None:
        self._snapshot_runtime_state(step=step, success=success, error=error or "")
        step_signature = self._step_signature(step_type=step.type, step_name=step.name, args=dict(step.args or {}))

        if success:
            if (
                self.state.current_plan is not None
                and 0 <= int(self.state.current_index) < len(self.state.current_plan.action_list)
                and str(self.state.current_plan.action_list[self.state.current_index].step_id or "") == str(step.step_id or "")
            ):
                self.state.current_index += 1
                if self.state.current_plan is not None and self.state.current_index >= len(self.state.current_plan.action_list):
                    self.state.program = None
                    self.state.pending_regen_reason = "plan_exhausted"
            self.state.consecutive_failure_streak = 0
            self.state.last_failed_step_signature = ""
            if step_signature == self.state.last_step_signature:
                self.state.repeated_step_streak += 1
            else:
                self.state.last_step_signature = step_signature
                self.state.repeated_step_streak = 1
        else:
            self.state.last_error = error or ""
            self.state.consecutive_failure_streak += 1
            self.state.last_failed_step_signature = step_signature
            self.state.last_step_signature = ""
            self.state.repeated_step_streak = 0
            self._record_local_repair_hint(step=step, error=error or "")
            self.state.program = None
            self._clear_current_plan()
            threshold = self._failure_regen_threshold(error=str(error or "").strip().lower())
            self.state.pending_regen_reason = (
                "repair_suffix:"
                + (self.state.last_error or f"step_failed:{step.name}")
                + f":failure_streak={self.state.consecutive_failure_streak}"
                + f":regen_threshold={threshold}"
            )
            return

        repeat_threshold = self._repeat_threshold(step_type=step.type, step_name=step.name)
        if success and self.state.repeated_step_streak >= repeat_threshold:
            self._snapshot_runtime_state(step=step, success=success, error="repeated_same_step_detected")
            self.state.program = None
            self._clear_current_plan()
            self.state.pending_regen_reason = (
                f"repeated_same_step_detected:name={step.name}:streak={self.state.repeated_step_streak}"
            )
            self.state.last_step_signature = ""
            self.state.repeated_step_streak = 0
