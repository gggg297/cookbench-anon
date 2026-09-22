from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from epm.brain.chat_client import _normalize_timeout, chat_complete_text
from epm.brain.http_qwen3vl import Qwen3VlHttpClient
from epm.brain.model_output_trace import default_model_trace_dir, write_model_call_trace, write_trace_files_for_raw_dir
from epm.brain.modules.prompt_assets import load_asset
from epm.brain.plan_schema import extract_json_object
from epm.brain.provider_compat import ANTHROPIC_PROVIDER_TYPES, OPENAI_PROVIDER_TYPES
from epm.core.settings import VlmSettings


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ensure_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _ensure_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _truncate_text(value: Any, *, limit: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _slice_stm_window_text_by_steps(text: str, *, history_window_steps: int) -> str:
    raw = str(text or "")
    limit = max(0, int(history_window_steps or 0))
    if limit <= 0:
        return raw.strip()
    lines = raw.splitlines()
    step_start_indices: list[int] = []
    for idx, line in enumerate(lines):
        if line.startswith("  - step_id: "):
            step_start_indices.append(idx)
    if len(step_start_indices) <= limit:
        return raw.strip()
    keep_from = step_start_indices[-limit]
    header = lines[:keep_from]
    kept = lines[keep_from:]
    return "\n".join(header + kept).strip()


@dataclass(frozen=True)
class TaskProgressMaintenanceResult:
    current_high_level_id: str
    current_high_level_goal: str
    goal_state: Dict[str, Any]
    semantic_progress: Dict[str, Any]
    commitments: list[Any]
    resource_bindings: Dict[str, Any]
    blocking_conditions: list[Any]
    temporal_checkpoints: Dict[str, Any]
    strategy_notes: Dict[str, Any]
    evidence: Dict[str, Any]
    rationale: str = ""


def _parse_result(obj: Dict[str, Any]) -> TaskProgressMaintenanceResult:
    cur = str(obj.get("current_high_level_id") or "").strip() or "H1"
    cur_goal = str(obj.get("current_high_level_goal") or "").strip() or "Follow recipe"

    goal_state = _ensure_dict(obj.get("goal_state"))
    semantic_progress = _ensure_dict(obj.get("semantic_progress"))
    resource_bindings = _ensure_dict(obj.get("resource_bindings"))
    temporal_checkpoints = _ensure_dict(obj.get("temporal_checkpoints"))
    strategy_notes = _ensure_dict(obj.get("strategy_notes"))
    evidence = _ensure_dict(obj.get("evidence"))

    goal_state["current_high_level_id"] = cur
    goal_state["current_high_level_goal"] = cur_goal
    goal_state["current_subgoal"] = str(goal_state.get("current_subgoal") or cur_goal).strip()
    goal_state["next_subgoal"] = ""
    goal_state["next_subgoal_type"] = "unknown"
    goal_state["next_subgoal_reason"] = ""
    status = str(goal_state.get("status") or "doing").strip().lower()
    if status not in {"todo", "doing", "done", "blocked"}:
        status = "doing"
    goal_state["status"] = status
    goal_state["done_criteria"] = [str(x).strip() for x in _ensure_list(goal_state.get("done_criteria")) if str(x).strip()]
    goal_state["current_focus"] = ""
    goal_state["current_activity"] = _truncate_text(goal_state.get("current_activity"), limit=240)
    goal_state.pop("completed_milestones", None)
    goal_state.pop("summary", None)

    semantic_progress["completed_tasks"] = [
        str(x).strip() for x in _ensure_list(semantic_progress.get("completed_tasks")) if str(x).strip()
    ]
    semantic_progress["recent_high_level_actions"] = [
        str(x).strip() for x in _ensure_list(semantic_progress.get("recent_high_level_actions")) if str(x).strip()
    ]
    semantic_progress.pop("in_progress_tasks", None)

    commitments = []
    for item in _ensure_list(obj.get("commitments")):
        if isinstance(item, dict):
            entry = {
                "name": _truncate_text(item.get("name"), limit=120),
                "value": _truncate_text(item.get("value"), limit=160),
                "reason": _truncate_text(item.get("reason"), limit=240),
            }
            if entry["name"] or entry["value"] or entry["reason"]:
                commitments.append(entry)
        else:
            text = _truncate_text(item, limit=240)
            if text:
                commitments.append({"name": text, "value": "", "reason": ""})

    resource_bindings["held_item"] = _truncate_text(resource_bindings.get("held_item"), limit=120)
    resource_bindings["active_container"] = _truncate_text(resource_bindings.get("active_container"), limit=120)
    resource_bindings["active_workspace"] = _truncate_text(resource_bindings.get("active_workspace"), limit=120)
    resource_bindings["timers"] = _ensure_list(resource_bindings.get("timers"))

    blocking_conditions = []
    for item in _ensure_list(obj.get("blocking_conditions")):
        if isinstance(item, dict):
            entry = {
                "type": _truncate_text(item.get("type"), limit=80),
                "detail": _truncate_text(item.get("detail"), limit=240),
            }
            if entry["type"] or entry["detail"]:
                blocking_conditions.append(entry)
        else:
            text = _truncate_text(item, limit=240)
            if text:
                blocking_conditions.append({"type": "", "detail": text})

    remaining = temporal_checkpoints.get("force_submit_remaining_steps")
    if not isinstance(remaining, int):
        try:
            remaining = int(remaining)
        except Exception:
            remaining = None
    temporal_checkpoints["force_submit_remaining_steps"] = remaining
    temporal_checkpoints["timer_checkpoints"] = _ensure_list(temporal_checkpoints.get("timer_checkpoints"))

    strategy_notes["current_approach"] = ""
    strategy_notes["next_planning_focus"] = ""
    strategy_notes.pop("observed_problems", None)
    strategy_notes.pop("effective_repairs", None)

    evidence["recipe_constraints"] = [
        _truncate_text(x, limit=240) for x in _ensure_list(evidence.get("recipe_constraints")) if str(x).strip()
    ]
    evidence["last_feedback_summary"] = _truncate_text(evidence.get("last_feedback_summary"), limit=360)
    evidence["recent_steps_summary"] = _truncate_text(evidence.get("recent_steps_summary"), limit=500)
    evidence["agent_state_summary"] = _truncate_text(evidence.get("agent_state_summary"), limit=360)

    rationale = ""

    return TaskProgressMaintenanceResult(
        current_high_level_id=cur,
        current_high_level_goal=cur_goal,
        goal_state=goal_state,
        semantic_progress=semantic_progress,
        commitments=commitments,
        resource_bindings=resource_bindings,
        blocking_conditions=blocking_conditions,
        temporal_checkpoints=temporal_checkpoints,
        strategy_notes=strategy_notes,
        evidence=evidence,
        rationale=rationale,
    )


def apply_task_progress_maintenance(*, task_progress_path: Path, result: TaskProgressMaintenanceResult) -> None:
    """
    Overwrite the semantic EPM task_progress JSON while preserving static recipe info.
    """
    current: dict[str, Any] = {}
    if task_progress_path.exists():
        try:
            obj = json.loads(task_progress_path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, dict):
                current = obj
        except Exception:
            current = {}

    recipe = _ensure_dict(current.get("recipe"))
    payload: dict[str, Any] = {
        "version": 3,
        "update_policy": "semantic_overwrite",
        "last_updated": _utc_now_iso(),
        "recipe": recipe,
        "goal_state": dict(result.goal_state),
        "semantic_progress": dict(result.semantic_progress),
        "commitments": list(result.commitments),
        "resource_bindings": dict(result.resource_bindings),
        "blocking_conditions": list(result.blocking_conditions),
        "temporal_checkpoints": dict(result.temporal_checkpoints),
        "strategy_notes": dict(result.strategy_notes),
        "evidence": dict(result.evidence),
    }
    task_progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TaskProgressMaintainerConfig:
    vlm: VlmSettings
    enabled: bool = False
    on_new_plan_generated: bool = True
    on_need_replan: bool = True
    history_window_steps: int = 30
    min_step_gap_steps: int = 30
    save_prompt_dir: Optional[Path] = None
    save_raw_dir: Optional[Path] = None


class TaskProgressMaintainer:
    """
    Prompt-driven semantic maintenance for EPM task_progress.
    """

    def __init__(self, cfg: TaskProgressMaintainerConfig) -> None:
        self.cfg = cfg
        self._last_run_step_id: Optional[int] = None
        if self.cfg.vlm.provider != "qwen3vl_http" and self.cfg.vlm.provider not in OPENAI_PROVIDER_TYPES and self.cfg.vlm.provider not in ANTHROPIC_PROVIDER_TYPES:
            raise ValueError(f"unsupported_provider:{self.cfg.vlm.provider}")
        self._qwen: Optional[Qwen3VlHttpClient] = None
        if self.cfg.vlm.provider == "qwen3vl_http":
            self._qwen = Qwen3VlHttpClient(
                base_url=self.cfg.vlm.base_url,
                connect_timeout_s=5.0,
                read_timeout_s=_normalize_timeout(float(self.cfg.vlm.timeout_s)),
            )

    def should_run(self, *, step_id: int, new_plan_generated: bool, need_replan: bool) -> bool:
        if not bool(self.cfg.enabled):
            return False
        triggered = False
        if new_plan_generated and bool(self.cfg.on_new_plan_generated):
            triggered = True
        if need_replan and bool(self.cfg.on_need_replan):
            triggered = True
        if not triggered:
            return False

        if self._last_run_step_id is None:
            return True
        gap = max(0, int(self.cfg.min_step_gap_steps or 0))
        if gap <= 0:
            return True
        return (int(step_id) - int(self._last_run_step_id)) >= gap

    def run(
        self,
        *,
        step_id: int,
        task_progress_path: Path,
        stm_window_path: Path,
        recipe_text: str,
        last_feedback: str,
        current_high_level_id: str,
        current_high_level_goal: str,
        agent_state_text: str,
    ) -> TaskProgressMaintenanceResult:
        prompt = self._build_prompt(
            task_progress=task_progress_path.read_text(encoding="utf-8") if task_progress_path.exists() else "",
            stm_window=_slice_stm_window_text_by_steps(
                stm_window_path.read_text(encoding="utf-8") if stm_window_path.exists() else "",
                history_window_steps=int(self.cfg.history_window_steps),
            ),
            recipe_text=recipe_text,
            last_feedback=last_feedback,
            current_high_level_id=current_high_level_id,
            current_high_level_goal=current_high_level_goal,
            agent_state_text=agent_state_text,
        )

        step_tag = f"step_{int(step_id):06d}"
        self._last_run_step_id = int(step_id)
        if self.cfg.save_prompt_dir is not None:
            try:
                self.cfg.save_prompt_dir.mkdir(parents=True, exist_ok=True)
                (self.cfg.save_prompt_dir / f"{step_tag}.txt").write_text(prompt, encoding="utf-8")
            except Exception:
                pass

        text = self._chat(prompt=prompt)
        if self.cfg.save_raw_dir is not None:
            try:
                self.cfg.save_raw_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{step_tag}.txt"
                (self.cfg.save_raw_dir / filename).write_text(text, encoding="utf-8")
                write_trace_files_for_raw_dir(raw_dir=self.cfg.save_raw_dir, filename=filename, text=text)
            except Exception:
                pass

        last_err: Optional[Exception] = None
        last_text = text
        for attempt in range(int(self.cfg.vlm.max_retries) + 1):
            try:
                obj = extract_json_object(last_text, strip_think_tags=bool(self.cfg.vlm.strip_think_tags))
                return _parse_result(obj)
            except Exception as e:
                last_err = e
                time.sleep(0.3 * (attempt + 1))
                last_text = self._chat(prompt=self._repair_prompt(previous_output=last_text))

        raise RuntimeError(f"task_progress_maintenance_invalid_json:{last_err!r}")

    def _chat(self, *, prompt: str) -> str:
        if self.cfg.vlm.provider == "qwen3vl_http":
            assert self._qwen is not None
            try:
                text = self._qwen.chat(prompt=prompt, image_path=None, max_new_tokens=int(self.cfg.vlm.max_tokens))
                try:
                    write_model_call_trace(
                        trace_dir=default_model_trace_dir(request_metrics_path=self.cfg.vlm.request_metrics_path),
                        call_name="task_progress_maintainer_qwen3vl_http",
                        prompt_text=prompt,
                        response_text=text,
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        screenshot_paths=[],
                    )
                except Exception:
                    pass
                return text
            except Exception as e:
                try:
                    write_model_call_trace(
                        trace_dir=default_model_trace_dir(request_metrics_path=self.cfg.vlm.request_metrics_path),
                        call_name="task_progress_maintainer_qwen3vl_http",
                        prompt_text=prompt,
                        response_text="",
                        error_text=repr(e),
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        screenshot_paths=[],
                    )
                except Exception:
                    pass
                raise
        completion = chat_complete_text(cfg=self.cfg.vlm, prompt=prompt, screenshot_path=None)
        return completion.content

    @staticmethod
    def _build_prompt(
        *,
        task_progress: str,
        stm_window: str,
        recipe_text: str,
        last_feedback: str,
        current_high_level_id: str,
        current_high_level_goal: str,
        agent_state_text: str,
    ) -> str:
        role_text = load_asset(
            "task_progress_maintainer",
            "role.txt",
            (
                "You are a semantic state maintainer for an embodied cooking agent.\n"
                "Your ONLY job is to summarize long-horizon task state for the next planning round.\n"
                "Do not output action plans. Do not repeat low-level step sequences unless they support a high-level summary.\n"
                "Do not output markdown.\n"
                "Track progress in recipe order: prefer the earliest unfinished recipe step as the current subgoal.\n"
                "If the agent is spending time on later subtasks before earlier recipe-critical ingredients/processes are completed, "
                "state that ordering mismatch explicitly in current_activity or blocking_conditions.\n"
                "Be score-aware but objective: cookware setup/storage alone is setup progress, not direct scoring dish-content progress, "
                "unless it immediately enables the next recipe-critical step."
            ),
        )
        output_text = load_asset(
            "task_progress_maintainer",
            "output_format.txt",
            (
                "You must output a SINGLE JSON object and NOTHING else.\n"
                "\n"
                "Output schema:\n"
                "{\n"
                '  "current_high_level_id": "H1",\n'
                '  "current_high_level_goal": "current semantic goal",\n'
                '  "goal_state": {\n'
                '    "current_subgoal": "",\n'
                '    "status": "todo|doing|done|blocked",\n'
                '    "done_criteria": ["..."],\n'
                '    "current_activity": ""\n'
                "  },\n"
                '  "semantic_progress": {\n'
                '    "completed_tasks": ["..."],\n'
                '    "recent_high_level_actions": ["..."]\n'
                "  },\n"
                '  "commitments": [{"name":"","value":"","reason":""}],\n'
                '  "resource_bindings": {\n'
                '    "held_item": "",\n'
                '    "active_container": "",\n'
                '    "active_workspace": "",\n'
                '    "timers": []\n'
                "  },\n"
                '  "blocking_conditions": [{"type":"","detail":""}],\n'
                '  "temporal_checkpoints": {\n'
                '    "force_submit_remaining_steps": null,\n'
                '    "timer_checkpoints": []\n'
                "  },\n"
                '  "strategy_notes": {\n'
                "  },\n"
                '  "evidence": {\n'
                '    "recipe_constraints": ["..."],\n'
                '    "last_feedback_summary": "",\n'
                '    "recent_steps_summary": "",\n'
                '    "agent_state_summary": ""\n'
                "  }\n"
                "}"
            ),
        )
        rules_text = load_asset(
            "task_progress_maintainer",
            "rules.txt",
            (
                "Rules:\n"
                "- Treat recipe requirements as hard constraints; include the most relevant recipe constraints in evidence.recipe_constraints.\n"
                "- Summarize what has been completed, what is in progress, and what is blocked.\n"
                "- Record state objectively. Do NOT decide what the next planning round should do.\n"
                "- Do NOT propose plans, next-step hints, suggested fixes, or recommendations.\n"
                "- Prefer semantic state summaries over low-level action repetition.\n"
                "- You are given the FULL current task_progress file. Decide which parts should change, keep the rest semantically stable, and output the FULL updated JSON object.\n"
                "- Only record commitments that should remain stable across replans.\n"
                "- Resource bindings should describe semantic roles, not every visible object.\n"
                "- Blocking conditions should be grounded in evidence and describe the blocker itself, not how the planner should respond.\n"
                "- Temporal checkpoints should capture future constraints such as timers or submit deadlines.\n"
                "- If uncertain, keep fields empty instead of inventing facts."
            ),
        )
        return (
            "SYSTEM:\n"
            f"{role_text}\n"
            "\n"
            f"{output_text}\n"
            "\n"
            f"{rules_text}\n"
            "\n"
            "USER:\n"
            f"Current high-level id: {str(current_high_level_id or '').strip()}\n"
            f"Current high-level goal: {str(current_high_level_goal or '').strip()}\n"
            "\n"
            "Recipe (raw):\n"
            f"{(recipe_text or '').strip()}\n"
            "\n"
            "Current task_progress file:\n"
            f"{(task_progress or '').strip()}\n"
            "\n"
            "Current STM window:\n"
            f"{(stm_window or '').strip()}\n"
            "\n"
            "Last feedback:\n"
            f"{(last_feedback or '').strip()}\n"
            "\n"
            "Agent state (JSON):\n"
            f"{(agent_state_text or '').strip()}\n"
        )

    @staticmethod
    def _repair_prompt(*, previous_output: str) -> str:
        prev = (previous_output or "").strip()
        if len(prev) > 2500:
            prev = prev[:2500] + "..."
        return (
            "Rewrite your answer to be a SINGLE JSON object and NOTHING else.\n"
            "- No markdown.\n"
            "- No commentary.\n"
            "- No <think>...</think>.\n"
            "- Use valid JSON (double quotes, no trailing commas).\n"
            "- Keep the same semantic schema.\n"
            "\n"
            "Previous response (invalid):\n"
            f"{prev}\n"
        )
