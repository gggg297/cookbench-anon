from __future__ import annotations

from collections import deque
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from epm.brain.chat_client import chat_complete_text
from epm.brain.model_output_trace import write_trace_files_for_raw_dir
from epm.brain.modules.prompt_assets import load_asset
from epm.brain.plan_schema import extract_json_object
from epm.brain.plan_schema import PlanStep
from epm.brain.interfaces import PlannerContext


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge src into dst (in place) recursively for dict values; overwrite scalars/lists.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


@dataclass(frozen=True)
class ReflexionConfig:
    enabled: bool = True
    # Reflect based on recent failure density by default.
    reflect_on_failure: bool = True
    min_failures_to_reflect: int = 1
    failure_window_steps: int = 8
    min_reflection_gap_steps: int = 8
    reflect_every_n_steps: int = 0
    reflect_on_loop: bool = True
    loop_window_steps: int = 6
    loop_repeat_threshold: int = 3
    memory_window_size: int = 1
    max_reflections: int = 200
    history_window_steps: int = 12
    prompt_ablation_profile: str = "full"
    prompt_disabled_groups: frozenset[str] | None = None


def _slice_stm_window_text_by_steps(text: str, *, history_window_steps: int) -> str:
    raw = str(text or "").strip()
    limit = max(0, int(history_window_steps or 0))
    if limit <= 0 or not raw:
        return raw
    lines = raw.splitlines()
    step_start_indices: list[int] = []
    for idx, line in enumerate(lines):
        if line.startswith("  - step_id: "):
            step_start_indices.append(idx)
    if len(step_start_indices) <= limit:
        return raw
    keep_from = step_start_indices[-limit]
    header = lines[:keep_from]
    kept = lines[keep_from:]
    return "\n".join(header + kept).strip()


class ReflexionReflector:
    """
    Generate a short structured lesson and update:
    - reflexion_memory.txt (sliding window)
    """

    def __init__(self, *, cfg: ReflexionConfig, chat_cfg: Any, memory_dir: Path) -> None:
        self.cfg = cfg
        self.chat_cfg = chat_cfg
        self.memory_dir = Path(memory_dir)
        self._num_reflections = 0
        self._executed_steps = 0
        recent_cap = max(
            8,
            int(cfg.loop_window_steps or 6) + 4,
            int(cfg.failure_window_steps or 8) + 4,
            int(cfg.min_reflection_gap_steps or 0) + 4,
        )
        self._recent_steps: Deque[dict[str, Any]] = deque(maxlen=recent_cap)
        self._last_loop_reflection_key = ""
        self._last_reflection_step = 0

    @staticmethod
    def _step_signature(step: Optional[PlanStep]) -> str:
        if step is None:
            return ""
        return f"{str(step.type or '').strip().lower()}::{str(step.name or '').strip()}"

    def note_result(self, *, step: Optional[PlanStep], success: bool, error: str = "") -> None:
        self._executed_steps += 1
        self._recent_steps.append(
            {
                "step_index": int(self._executed_steps),
                "signature": self._step_signature(step),
                "type": str(getattr(step, "type", "") or ""),
                "name": str(getattr(step, "name", "") or ""),
                "args": dict(getattr(step, "args", {}) or {}),
                "success": bool(success),
                "error": str(error or ""),
            }
        )

    def _reflection_gap_active(self) -> bool:
        gap = max(0, int(self.cfg.min_reflection_gap_steps or 0))
        if gap <= 0 or self._last_reflection_step <= 0:
            return False
        return (self._executed_steps - self._last_reflection_step) < gap

    def _mark_reflection_emitted(self) -> None:
        self._num_reflections += 1
        self._last_reflection_step = int(self._executed_steps)

    def _recent_failure_count(self) -> tuple[int, int]:
        window_n = max(1, int(self.cfg.failure_window_steps or 0))
        recent = list(self._recent_steps)[-window_n:]
        failures = sum(1 for item in recent if not bool(item.get("success", False)))
        return failures, len(recent)

    def maybe_reflect(self, *, context: PlannerContext, failed_step: PlanStep, error: str) -> Optional[dict[str, Any]]:
        if not bool(self.cfg.enabled):
            return None
        if not bool(self.cfg.reflect_on_failure):
            return None
        if self._num_reflections >= int(self.cfg.max_reflections):
            return None
        if self._reflection_gap_active():
            return None
        failure_count, seen_steps = self._recent_failure_count()
        if seen_steps <= 0:
            return None
        if failure_count < int(self.cfg.min_failures_to_reflect):
            return None
        self._mark_reflection_emitted()
        return self._reflect(
            context=context,
            focus_step=failed_step,
            error=error,
            trigger=f"failure_window:{failure_count}/{max(1, int(self.cfg.failure_window_steps or 0))}",
        )

    def maybe_periodic_reflect(self, *, context: PlannerContext, latest_step: Optional[PlanStep], latest_error: str, latest_success: bool) -> Optional[dict[str, Any]]:
        if not bool(self.cfg.enabled):
            return None
        every_n = int(self.cfg.reflect_every_n_steps or 0)
        if every_n <= 0:
            return None
        if self._num_reflections >= int(self.cfg.max_reflections):
            return None
        if self._reflection_gap_active():
            return None
        if self._executed_steps <= 0 or (self._executed_steps % every_n) != 0:
            return None
        self._mark_reflection_emitted()
        return self._reflect(
            context=context,
            focus_step=latest_step,
            error=(latest_error or ""),
            trigger=("periodic_after_failure" if not latest_success else "periodic"),
        )

    def maybe_loop_reflect(self, *, context: PlannerContext, latest_step: Optional[PlanStep], latest_error: str) -> Optional[dict[str, Any]]:
        if not bool(self.cfg.enabled):
            return None
        if not bool(self.cfg.reflect_on_loop):
            return None
        if self._num_reflections >= int(self.cfg.max_reflections):
            return None
        if self._reflection_gap_active():
            return None
        if latest_step is None:
            return None
        window_n = max(1, int(self.cfg.loop_window_steps or 0))
        repeat_threshold = max(2, int(self.cfg.loop_repeat_threshold or 0))
        recent = list(self._recent_steps)[-window_n:]
        if len(recent) < repeat_threshold:
            return None
        latest_signature = self._step_signature(latest_step)
        if not latest_signature:
            return None
        count = sum(1 for item in recent if str(item.get("signature") or "") == latest_signature)
        if count < repeat_threshold:
            return None
        loop_key = f"{latest_signature}::count={count}"
        if loop_key == self._last_loop_reflection_key:
            return None
        self._last_loop_reflection_key = loop_key
        self._mark_reflection_emitted()
        return self._reflect(
            context=context,
            focus_step=latest_step,
            error=(latest_error or ""),
            trigger=f"loop:{loop_key}",
        )

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _write_text(self, path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except Exception:
            return

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_json(self, path: Path, obj: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            return

    @staticmethod
    def _clean_text(value: Any, *, max_len: int = 220) -> str:
        text = str(value or "").strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) > max_len:
            text = text[:max_len].rstrip() + "..."
        return text

    @classmethod
    def _normalize_error_key(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        if not text:
            return ""
        tags = [
            ("pick_up_post_check_failed:not_holding_after_click", "pick_up_not_holding"),
            ("enter_pouring_mode_failed:not_in_pouring_mode", "pour_mode_not_entered"),
            ("auto_pour_requires_pouring_mode", "auto_pour_requires_mode"),
            ("enter_spices_sprinkle_mode_failed:not_in_sprinkle_mode", "sprinkle_mode_not_entered"),
            ("auto_sprinkle_requires_sprinkle_mode", "auto_sprinkle_requires_mode"),
            ("navigation_failed_astar_path_not_found", "nav_astar_path_not_found"),
        ]
        for needle, label in tags:
            if needle in text:
                return label
        text = re.sub(r"post_state:.*$", "", text).strip()
        text = re.sub(r"[:;].*$", "", text).strip()
        return text[:80]

    @staticmethod
    def _step_target_text(item: dict[str, Any]) -> str:
        if not isinstance(item, dict):
            return ""
        args = item.get("args")
        if not isinstance(args, dict):
            return ""
        for key in ("target", "container_name", "item_name", "object_name", "name"):
            value = str(args.get(key) or "").strip()
            if value:
                return value
        return ""

    def _recent_successful_navigation_target(self, *, recent: list[dict[str, Any]]) -> str:
        for item in reversed(recent):
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or "").strip() != "auto_navigation":
                continue
            if not bool(item.get("success", False)):
                continue
            target = self._step_target_text(item)
            if target:
                return target
        return ""

    def _summarize_recent_progress(self, *, recent: list[dict[str, Any]]) -> str:
        progress: list[str] = []
        for item in recent:
            if not isinstance(item, dict) or not bool(item.get("success", False)):
                continue
            name = str(item.get("name") or "").strip()
            if name == "auto_navigation":
                target = self._step_target_text(item)
                if target:
                    progress.append(f"reached {target}")
            elif name == "pick_up":
                progress.append("acquired an item")
            elif name == "put_down":
                progress.append("put an item down")
            elif name == "auto_pour":
                progress.append("poured into the target container")
        out: list[str] = []
        for text in progress:
            if text not in out:
                out.append(text)
        return "; ".join(out[:2])

    def _heuristic_lesson_for_recent_loop(
        self,
        *,
        focus_step: Optional[PlanStep],
        error: str,
    ) -> dict[str, str] | None:
        recent = list(self._recent_steps)[-8:]
        if not recent:
            return None

        def _matching(name: str, err_key: str) -> list[dict[str, Any]]:
            out: list[dict[str, Any]] = []
            for item in recent:
                if not isinstance(item, dict):
                    continue
                if str(item.get("name") or "").strip() != name:
                    continue
                if bool(item.get("success", False)):
                    continue
                if self._normalize_error_key(item.get("error")) == err_key:
                    out.append(item)
            return out

        pick_failures = _matching("pick_up", "pick_up_not_holding")
        if len(pick_failures) >= 2:
            target = self._recent_successful_navigation_target(recent=recent)
            target_text = f" around {target}" if target else ""
            progress = self._summarize_recent_progress(recent=recent)
            return {
                "completed": progress or f"re-approached the pickup target{target_text}",
                "issues": f"repeated pick_up retries{target_text} still ended with empty hands; this is a pickup setup loop, not real progress",
                "next_focus": "stop retrying pick_up on the same setup; re-perceive/query or switch angle/instance before one more pickup attempt",
            }

        pour_failures = _matching("enter_pouring_mode", "pour_mode_not_entered")
        auto_pour_failures = _matching("auto_pour", "auto_pour_requires_mode")
        if len(pour_failures) + len(auto_pour_failures) >= 2:
            target = self._recent_successful_navigation_target(recent=recent)
            target_text = f" at {target}" if target else ""
            progress = self._summarize_recent_progress(recent=recent)
            return {
                "completed": progress or f"reached the pour target{target_text}",
                "issues": f"pouring kept retrying{target_text} without ever entering pouring mode; the missing precondition was not fixed",
                "next_focus": "do not call auto_pour until pouring mode is active; re-aim/re-approach the target container, then retry mode entry once only after setup changed",
            }

        sprinkle_failures = _matching("enter_spices_sprinkle_mode", "sprinkle_mode_not_entered")
        auto_sprinkle_failures = _matching("auto_sprinkle", "auto_sprinkle_requires_mode")
        if len(sprinkle_failures) + len(auto_sprinkle_failures) >= 2:
            target = self._recent_successful_navigation_target(recent=recent)
            target_text = f" near {target}" if target else ""
            progress = self._summarize_recent_progress(recent=recent)
            return {
                "completed": progress or f"reached the seasoning target{target_text}",
                "issues": f"seasoning retries{target_text} never established sprinkle mode, so the plan was looping on a missing precondition",
                "next_focus": "stop repeating sprinkle-mode entry after the same error; verify spice-in-hand and target alignment first, then season once",
            }

        focus_name = str(getattr(focus_step, "name", "") or "").strip()
        error_key = self._normalize_error_key(error)
        if focus_name == "pick_up" and error_key == "pick_up_not_holding":
            return {
                "completed": self._summarize_recent_progress(recent=recent),
                "issues": "the latest pick_up still ended with empty hands, so the pickup preconditions are not actually satisfied",
                "next_focus": "change the setup before another pick_up attempt; do not spam the same retry",
            }
        return None

    @classmethod
    def _normalize_lesson(cls, obj: dict[str, Any], *, fallback_error: str) -> dict[str, str]:
        if not isinstance(obj, dict):
            obj = {}
        reflection = cls._clean_text(obj.get("reflection", ""))
        lesson = {
            "completed": cls._clean_text(obj.get("completed", "")),
            "issues": cls._clean_text(obj.get("issues", obj.get("failure_pattern", obj.get("cause", "")))),
            "unfinished": cls._clean_text(
                obj.get("unfinished", obj.get("remaining_work", obj.get("remaining", "")))
            ),
            "visual_attribution": cls._clean_text(
                obj.get(
                    "visual_attribution",
                    obj.get("visual", obj.get("visual_evidence", obj.get("visual_failure_attribution", ""))),
                )
            ),
            "next_focus": cls._clean_text(
                obj.get("next_focus", obj.get("next", obj.get("fix", "")))
            ),
        }
        if not any(lesson.values()) and reflection:
            lesson["issues"] = reflection
        if not lesson["issues"]:
            lesson["issues"] = cls._clean_text(fallback_error or "execution issue")
        if not lesson["visual_attribution"]:
            lesson["visual_attribution"] = "visual evidence inconclusive"
        return lesson

    @classmethod
    def _parse_reflection_response(cls, text: str, *, fallback_error: str) -> dict[str, str]:
        raw = str(text or "").strip()
        obj = None
        try:
            obj = extract_json_object(raw, strip_think_tags=True)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            return cls._normalize_lesson(obj, fallback_error=fallback_error)

        lesson = {
            "completed": "",
            "issues": "",
            "unfinished": "",
            "visual_attribution": "",
            "next_focus": "",
        }
        label_map = {
            "completed": "completed",
            "progress": "completed",
            "done": "completed",
            "finished": "completed",
            "issues": "issues",
            "issue": "issues",
            "problem": "issues",
            "problems": "issues",
            "blocker": "issues",
            "blockers": "issues",
            "unfinished": "unfinished",
            "remaining_work": "unfinished",
            "remaining": "unfinished",
            "visual": "visual_attribution",
            "visual_evidence": "visual_attribution",
            "visual_attribution": "visual_attribution",
            "visual_failure_attribution": "visual_attribution",
            "next": "next_focus",
            "next_focus": "next_focus",
            "focus": "next_focus",
            "todo": "next_focus",
            "attention": "next_focus",
            "caution": "next_focus",
        }
        plain_parts: list[str] = []
        for raw_line in raw.splitlines():
            line = re.sub(r"^\s*[-*]\s*", "", str(raw_line or "").strip())
            if not line:
                continue
            m = re.match(r"(?i)^([a-z_ ]{2,24})\s*[:\-]\s*(.+)$", line)
            if m:
                label = re.sub(r"\s+", "_", m.group(1).strip().lower())
                value = cls._clean_text(m.group(2))
                key = label_map.get(label)
                if key and value:
                    if lesson.get(key):
                        lesson[key] = cls._clean_text(f"{lesson[key]}; {value}")
                    else:
                        lesson[key] = value
                    continue
            plain_parts.append(cls._clean_text(line))

        if plain_parts:
            if not lesson["issues"]:
                lesson["issues"] = cls._clean_text(" ".join(plain_parts), max_len=320)
            elif not lesson["next_focus"] and len(plain_parts) >= 2:
                lesson["next_focus"] = plain_parts[-1]
        if not lesson["issues"]:
            lesson["issues"] = cls._clean_text(fallback_error or "execution issue")
        if not lesson["visual_attribution"]:
            lesson["visual_attribution"] = "visual evidence inconclusive"
        return lesson

    @staticmethod
    def _parse_memory_entries(text: str) -> list[str]:
        raw = str(text or "").strip()
        if not raw:
            return []
        lines = raw.splitlines()
        entries: list[str] = []
        current: list[str] = []
        for line in lines:
            if line.startswith("- ["):
                if current:
                    entries.append("\n".join(current).rstrip())
                current = [line]
                continue
            if current:
                current.append(line)
        if current:
            entries.append("\n".join(current).rstrip())
        return entries

    @staticmethod
    def _render_memory_entry(*, ts: str, trigger: str, lesson: dict[str, str]) -> str:
        parts = [f"- [{ts}] ({trigger})"]
        completed = str(lesson.get("completed") or "").strip()
        issues = str(lesson.get("issues") or "").strip()
        unfinished = str(lesson.get("unfinished") or "").strip()
        visual = str(lesson.get("visual_attribution") or "").strip()
        next_focus = str(lesson.get("next_focus") or "").strip()
        if completed:
            parts.append(f"done={completed}")
        if issues:
            parts.append(f"issue={issues}")
        if unfinished:
            parts.append(f"unfinished={unfinished}")
        if visual:
            parts.append(f"visual={visual}")
        if next_focus:
            parts.append(f"next={next_focus}")
        return " | ".join(parts).rstrip()

    def _reflect(self, *, context: PlannerContext, focus_step: Optional[PlanStep], error: str, trigger: str) -> dict[str, Any]:
        ts = _utc_now_iso()
        reflexion_mem_path = self.memory_dir / "reflexion_memory.txt"
        raw_dir = self.memory_dir / "reflexion_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        current_reflexion_mem = self._read_text(reflexion_mem_path).strip()
        progress_mem = str((context.memory_bundle or {}).get("reflexion_progress_memory", "") or "").strip()
        stm_window = _slice_stm_window_text_by_steps(
            str((context.memory_bundle or {}).get("stm_window", "") or "").strip(),
            history_window_steps=int(self.cfg.history_window_steps),
        )
        role_text = load_asset(
            "reflexion/reflector",
            "role.txt",
            (
                "You are Reflexion, a short-horizon self-reflective agent for embodied cooking.\n"
                "Your job is to briefly reflect on the current planning chunk (about the recent 10 steps): what real progress was made toward the local subgoal, why that subgoal was not fully completed, what is still unfinished, what the next 10 steps should focus on, and what visual evidence in the screenshot supports the failure diagnosis."
            ),
        )
        rules_text = load_asset(
            "reflexion/reflector",
            "rules.txt",
            (
                "Rules:\n"
                "- Keep the reflection brief: prefer 3-5 short lines or one short paragraph.\n"
                "- Focus only on the current planning chunk (roughly the recent 10 steps), not on the whole episode.\n"
                "- Summarize five things in concise text: what progress was made toward the current subgoal, what problem blocked completion, what part is still unfinished, what the next 10 steps should focus on, and what visual evidence supports the failure attribution.\n"
                "- Avoid brittle pixel-level advice.\n"
                "- Do not mention file paths.\n"
                "- Do not write a detailed step-by-step script; summarize the next chunk's focus, caution, unfinished subgoal, and one key guardrail instead.\n"
                "- Do not rely on volatile current-state summaries such as 'currently holding X' unless the lesson is about why such state became stale.\n"
                "- Prefer planner-level lessons over low-level speculation: call out bad subgoal order, missing preconditions, stale state assumptions, repeated retries, or unresolved placeholders when those are the real issue.\n"
                "- Only state causes that are directly supported by the recent steps or error messages. Do not guess hidden causes such as bottle cap state, hand pose, affordance, or physics details unless the logs directly support them.\n"
                "- When a screenshot is available, inspect it for actionable evidence such as: crosshair not on target, wrong hovered object, PICK UP/PUT DOWN/THROW prompt mismatch, target on the floor, object dropped after a failed mode-entry, target off-center, or obvious occlusion.\n"
                "- If the screenshot does not provide useful evidence, explicitly say the visual evidence is inconclusive instead of guessing.\n"
                "- If there was no real progress, say so explicitly.\n"
                "- If repeated actions happened without progress, explain the missing completion criterion.\n"
                "- If actions happened but the local subgoal still was not achieved, say that there was activity without real progress.\n"
                "- If the trigger is periodic, summarize the most important current risk or recurring mistake, not a full episode postmortem.\n"
                "- If the main issue is a planning inconsistency rather than a single action failure, say so explicitly.\n"
                "- Plain text is preferred. Optional short prefixes such as 'Completed:', 'Issues:', 'Unfinished:', 'Visual:', 'Next:' are allowed but not required."
            ),
        )
        output_text = load_asset(
            "reflexion/reflector",
            "output_format.txt",
            (
            "Preferred response style:\n"
            "- Write brief plain text.\n"
            "- Include: what real progress was made in the current chunk, what blocked completion, what remains unfinished, what visual evidence supports the diagnosis, and what the next chunk should focus on.\n"
            "- Prefer one short paragraph or 3-5 short lines.\n"
            "- Optional prefixes are allowed: 'Completed:', 'Issues:', 'Unfinished:', 'Visual:', 'Next:'."
            ),
        )
        step_desc = "(none)"
        if focus_step is not None:
            step_desc = (
                f"type={focus_step.type} "
                f"name={focus_step.name} "
                f"args={json.dumps(focus_step.args or {}, ensure_ascii=False)}"
            )
        recent_steps_desc = json.dumps(list(self._recent_steps)[-10:], ensure_ascii=False, indent=2)
        heuristic_lesson = self._heuristic_lesson_for_recent_loop(focus_step=focus_step, error=error)
        heuristic_hint = ""
        if heuristic_lesson is not None:
            heuristic_hint = (
                "Local heuristic hint (use only if consistent with the screenshot and recent steps):\n"
                f"{json.dumps(heuristic_lesson, ensure_ascii=False)}\n\n"
            )

        prompt = (
            "SYSTEM:\n"
            f"{role_text}\n"
            "\n"
            f"{rules_text}\n"
            "\n"
            f"{output_text}\n"
            "\n"
            "USER:\n"
            f"Trigger: {trigger}\n"
            f"High-level goal: {context.high_level_goal}\n"
            f"Focus step: {step_desc}\n"
            f"Error: {error}\n"
            "\n"
            "Last feedback string:\n"
            f"{(context.feedback or '').strip()}\n"
            "\n"
            "Recent STM window:\n"
            f"{stm_window or '(none)'}\n"
            "\n"
            "Recent stable progress facts:\n"
            f"{progress_mem or '(none)'}\n"
            "\n"
            "Recent executed steps:\n"
            f"{recent_steps_desc}\n"
            "\n"
            f"{heuristic_hint}"
            "Existing reflexion memory:\n"
            f"{current_reflexion_mem or '(empty)'}\n"
        )

        raw_name = f"{ts.replace(':', '').replace('-', '')}_reflect.txt"
        reflection_source = "local_heuristic"
        screenshot_path = getattr(context.observation, "screenshot_path", None)
        use_visual_reflection = bool(getattr(self.chat_cfg, "use_vision", False)) and bool(screenshot_path)
        lesson = None
        reflection_text_raw = ""
        if use_visual_reflection:
            out = chat_complete_text(
                cfg=self.chat_cfg,
                prompt=prompt,
                screenshot_path=screenshot_path,
                force_use_vision=bool(getattr(self.chat_cfg, "use_vision", False)),
            )
            reflection_source = "llm"
            reflection_text_raw = str(out.content or "")
            lesson = self._parse_reflection_response(reflection_text_raw, fallback_error=error)
        elif heuristic_lesson is not None:
            lesson = heuristic_lesson
            reflection_text_raw = "\n".join(
                line
                for line in [
                    f"Completed: {lesson.get('completed', '').strip()}",
                    f"Issues: {lesson.get('issues', '').strip()}",
                    f"Unfinished: {lesson.get('unfinished', '').strip()}",
                    f"Visual: {lesson.get('visual_attribution', '').strip()}",
                    f"Next: {lesson.get('next_focus', '').strip()}",
                ]
                if line.split(": ", 1)[1].strip()
            ).strip()

        if lesson is None:
            lesson = heuristic_lesson or {
                "completed": "",
                "issues": self._clean_text(error or "execution issue"),
                "unfinished": "",
                "visual_attribution": "visual evidence unavailable",
                "next_focus": "",
            }
            if not reflection_text_raw:
                reflection_text_raw = "\n".join(
                    line
                    for line in [
                        f"Completed: {lesson.get('completed', '').strip()}",
                        f"Issues: {lesson.get('issues', '').strip()}",
                        f"Unfinished: {lesson.get('unfinished', '').strip()}",
                        f"Visual: {lesson.get('visual_attribution', '').strip()}",
                        f"Next: {lesson.get('next_focus', '').strip()}",
                    ]
                    if line.split(": ", 1)[1].strip()
                ).strip()

        (raw_dir / raw_name).write_text(reflection_text_raw, encoding="utf-8")
        write_trace_files_for_raw_dir(raw_dir=raw_dir, filename=raw_name, text=reflection_text_raw)

        # 1) Maintain a sliding window of structured reflexion entries.
        entries = self._parse_memory_entries(current_reflexion_mem)
        new_entry = self._render_memory_entry(ts=ts, trigger=trigger, lesson=lesson)
        duplicate_recent = bool(entries and new_entry == entries[-1])
        if not duplicate_recent and any(lesson.values()):
            entries.append(new_entry)
        keep_n = max(1, int(self.cfg.memory_window_size or 10))
        entries = entries[-keep_n:]
        memory_text = "# Reflexion Memory (Long-term)\n# Sliding window of recent reusable lessons.\n\n"
        if entries:
            memory_text += "\n".join(entries).rstrip() + "\n"
        self._write_text(reflexion_mem_path, memory_text)

        reflection_text = "; ".join(
            part
            for part in [
                lesson.get("completed", ""),
                lesson.get("issues", ""),
                lesson.get("unfinished", ""),
                lesson.get("visual_attribution", ""),
                lesson.get("next_focus", ""),
            ]
            if part
        ).strip("; ")

        return {
            "reflection": reflection_text,
            "lesson": lesson,
            "previous": {
                "reflexion_memory_len": len(current_reflexion_mem),
            },
            "trigger": str(trigger or ""),
            "source": reflection_source,
        }
