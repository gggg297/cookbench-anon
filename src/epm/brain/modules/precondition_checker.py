from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from epm.brain.chat_client import _normalize_timeout, chat_complete_text
from epm.brain.http_qwen3vl import Qwen3VlHttpClient
from epm.brain.model_output_trace import default_model_trace_dir, write_model_call_trace, write_trace_files_for_raw_dir
from epm.brain.plan_schema import PlanStep, extract_json_object
from epm.brain.provider_compat import ANTHROPIC_PROVIDER_TYPES, OPENAI_PROVIDER_TYPES
from epm.core.epm_types import Observation
from epm.core.settings import VlmSettings


@dataclass(frozen=True)
class PreconditionCheckResult:
    status: str
    reason_code: str
    reason: str
    feedback_to_planner: str
    relevant_state: list[str]
    checks: Dict[str, bool]
    visual_blockers: list[str]
    all_blocked_steps: list[Dict[str, Any]]
    raw: Optional[Dict[str, Any]] = None

    @property
    def passed(self) -> bool:
        return str(self.status or "").strip().lower() == "pass"


@dataclass(frozen=True)
class PreconditionCheckerConfig:
    enabled: bool = False
    mode: str = "key_rules"
    vlm: Optional[VlmSettings] = None
    max_retries: int = 1
    max_plan_repair_attempts: int = 2
    recheck_after_repair: bool = False
    strip_think_tags: bool = True
    save_prompt_dir: Optional[Path] = None
    save_raw_dir: Optional[Path] = None


class PreconditionChecker:
    """
    Prompt-driven step precondition checker.

    It focuses on whether the remaining sequence can be executed under the
    current symbolic/text state, without using screenshot/perception input.
    """

    def __init__(self, cfg: PreconditionCheckerConfig) -> None:
        self.cfg = cfg
        if not self.cfg.enabled:
            return
        if str(self.cfg.mode or "key_rules").strip().lower() == "key_rules":
            self._provider = "key_rules"
            self.client = None
            return
        if self.cfg.vlm is None:
            raise ValueError("PreconditionChecker enabled but vlm settings missing")
        provider = str(self.cfg.vlm.provider or "").strip().lower()
        if provider != "qwen3vl_http" and provider not in OPENAI_PROVIDER_TYPES and provider not in ANTHROPIC_PROVIDER_TYPES:
            raise ValueError(f"PreconditionChecker unsupported provider: {provider!r}")
        self._provider = provider
        self.client: Optional[Qwen3VlHttpClient] = None
        if provider == "qwen3vl_http":
            self.client = Qwen3VlHttpClient(
                base_url=self.cfg.vlm.base_url,
                connect_timeout_s=5.0,
                read_timeout_s=_normalize_timeout(float(self.cfg.vlm.timeout_s)),
            )

    @staticmethod
    def _step_tag(obs: Observation) -> str:
        fid = str(getattr(obs, "frame_id", "") or "").strip()
        if fid.isdigit():
            return f"step_{int(fid):06d}"
        return fid or "step_unknown"

    @staticmethod
    def _prompt_asset_root() -> Path:
        return Path(__file__).resolve().parents[4] / "memory" / "epm" / "precondition_checker"

    @classmethod
    def _load_prompt_asset(cls, name: str, fallback: str) -> str:
        path = cls._prompt_asset_root() / name
        try:
            if path.exists():
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    return text
        except Exception:
            pass
        return fallback.strip()

    def check(
        self,
        *,
        observation: Observation,
        step: PlanStep,
        remaining_plan_text: str,
        recipe_text: str,
        high_level_goal: str,
        task_progress_text: str,
        recent_history_text: str,
        latest_visual_feedback_text: str,
        agent_state_text: str,
        body_rules_text: str,
        strategy_notes_text: str,
        tool_manifest_text: str,
        last_feedback: str,
    ) -> PreconditionCheckResult:
        if not self.cfg.enabled:
            return PreconditionCheckResult(
                status="pass",
                reason_code="checker_disabled",
                reason="precondition checker disabled",
                feedback_to_planner="",
                relevant_state=[],
                checks={
                    "mode_ok": True,
                    "hand_state_ok": True,
                    "visibility_ok": True,
                    "interactability_ok": True,
                    "task_state_ok": True,
                },
                visual_blockers=[],
                all_blocked_steps=[],
                raw=None,
            )
        if str(self.cfg.mode or "key_rules").strip().lower() == "key_rules":
            return self._check_key_rules(
                step=step,
                remaining_plan_text=remaining_plan_text,
                task_progress_text=task_progress_text,
                recent_history_text=recent_history_text,
                latest_visual_feedback_text=latest_visual_feedback_text,
                last_feedback=last_feedback,
            )

        prompt = self._build_prompt(
            observation=observation,
            step=step,
            remaining_plan_text=remaining_plan_text,
            recipe_text=recipe_text,
            high_level_goal=high_level_goal,
            task_progress_text=task_progress_text,
            recent_history_text=recent_history_text,
            latest_visual_feedback_text=latest_visual_feedback_text,
            agent_state_text=agent_state_text,
            body_rules_text=body_rules_text,
            strategy_notes_text=strategy_notes_text,
            tool_manifest_text=tool_manifest_text,
            last_feedback=last_feedback,
        )
        tag = self._step_tag(observation)
        if self.cfg.save_prompt_dir is not None:
            try:
                self.cfg.save_prompt_dir.mkdir(parents=True, exist_ok=True)
                (self.cfg.save_prompt_dir / f"{tag}.txt").write_text(prompt, encoding="utf-8")
            except Exception:
                pass

        last_err: Optional[Exception] = None
        last_text = ""
        for attempt in range(int(self.cfg.max_retries) + 1):
            try:
                current_prompt = prompt if attempt == 0 else self._repair_prompt(previous_output=last_text)
                if self._provider == "qwen3vl_http":
                    assert self.client is not None
                    last_text = self.client.chat(
                        prompt=current_prompt,
                        image_path=(observation.screenshot_path if (self.cfg.vlm and bool(self.cfg.vlm.use_vision)) else None),
                        max_new_tokens=int(self.cfg.vlm.max_tokens) if self.cfg.vlm else None,
                    )
                    try:
                        write_model_call_trace(
                            trace_dir=default_model_trace_dir(request_metrics_path=(self.cfg.vlm.request_metrics_path if self.cfg.vlm else None)),
                            call_name="precondition_checker_qwen3vl_http",
                            prompt_text=current_prompt,
                            response_text=last_text,
                            provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                            model=str(getattr(self.cfg.vlm, "model", "") or ""),
                            attempt=int(attempt),
                            screenshot_paths=([observation.screenshot_path] if (observation.screenshot_path and self.cfg.vlm and bool(self.cfg.vlm.use_vision)) else []),
                        )
                    except Exception:
                        pass
                else:
                    completion = chat_complete_text(
                        cfg=self.cfg.vlm,
                        prompt=current_prompt,
                        screenshot_path=(observation.screenshot_path if (self.cfg.vlm and bool(self.cfg.vlm.use_vision)) else None),
                    )
                    last_text = completion.content
                if self.cfg.save_raw_dir is not None:
                    try:
                        self.cfg.save_raw_dir.mkdir(parents=True, exist_ok=True)
                        filename = f"{tag}.txt"
                        (self.cfg.save_raw_dir / filename).write_text(last_text, encoding="utf-8")
                        write_trace_files_for_raw_dir(raw_dir=self.cfg.save_raw_dir, filename=filename, text=last_text)
                    except Exception:
                        pass
                obj = extract_json_object(last_text, strip_think_tags=bool(self.cfg.strip_think_tags))
                if isinstance(obj, dict) and obj:
                    return self._parse(obj)
                return self._parse_text(last_text)
            except Exception as e:
                try:
                    write_model_call_trace(
                        trace_dir=default_model_trace_dir(request_metrics_path=(self.cfg.vlm.request_metrics_path if self.cfg.vlm else None)),
                        call_name="precondition_checker_qwen3vl_http" if self._provider == "qwen3vl_http" else "precondition_checker_chat_complete",
                        prompt_text=current_prompt if "current_prompt" in locals() else prompt,
                        response_text="",
                        error_text=repr(e),
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        attempt=int(attempt),
                        screenshot_paths=([observation.screenshot_path] if (observation.screenshot_path and self.cfg.vlm and bool(self.cfg.vlm.use_vision)) else []),
                    )
                except Exception:
                    pass
                last_err = e
                time.sleep(0.3 * (attempt + 1))

        return PreconditionCheckResult(
            status="pass",
            reason_code="checker_failed_open",
            reason=f"checker_failed:{last_err!r}",
            feedback_to_planner="",
            relevant_state=[],
            checks={
                "mode_ok": True,
                "hand_state_ok": True,
                "visibility_ok": True,
                "interactability_ok": True,
                "task_state_ok": True,
            },
            visual_blockers=[],
            all_blocked_steps=[],
            raw={"raw_text": last_text},
        )

    @staticmethod
    def _parse(obj: Dict[str, Any]) -> PreconditionCheckResult:
        status = str(obj.get("status") or "").strip().lower()
        if status not in {"pass", "fail"}:
            raise ValueError("missing_status_pass_fail")
        reason_code = str(obj.get("reason_code") or "").strip()
        reason = str(obj.get("reason") or "").strip()
        feedback_to_planner = str(obj.get("feedback_to_planner") or "").strip()
        checks_raw = obj.get("checks")
        checks = checks_raw if isinstance(checks_raw, dict) else {}
        parsed_checks = {
            "mode_ok": bool(checks.get("mode_ok", False)),
            "hand_state_ok": bool(checks.get("hand_state_ok", False)),
            "visibility_ok": bool(checks.get("visibility_ok", False)),
            "interactability_ok": bool(checks.get("interactability_ok", False)),
            "task_state_ok": bool(checks.get("task_state_ok", False)),
        }
        visual_blockers_raw = obj.get("visual_blockers")
        if isinstance(visual_blockers_raw, list):
            visual_blockers = [str(x).strip() for x in visual_blockers_raw if str(x).strip()]
        else:
            visual_blockers = []
        all_blocked_steps_raw = obj.get("all_blocked_steps")
        all_blocked_steps: list[Dict[str, Any]] = []
        if isinstance(all_blocked_steps_raw, list):
            for item in all_blocked_steps_raw:
                if not isinstance(item, dict):
                    continue
                idx = item.get("index")
                try:
                    idx = int(idx) if idx is not None else None
                except Exception:
                    idx = None
                entry = {
                    "index": idx,
                    "name": str(item.get("name") or "").strip(),
                    "reason_code": str(item.get("reason_code") or "").strip(),
                    "reason": str(item.get("reason") or "").strip(),
                }
                if entry["index"] is not None or entry["name"] or entry["reason_code"] or entry["reason"]:
                    all_blocked_steps.append(entry)
        relevant_state_raw = obj.get("relevant_state")
        if isinstance(relevant_state_raw, list):
            relevant_state = [str(x).strip() for x in relevant_state_raw if str(x).strip()]
        else:
            relevant_state = []
        return PreconditionCheckResult(
            status=status,
            reason_code=reason_code,
            reason=reason,
            feedback_to_planner=feedback_to_planner,
            relevant_state=relevant_state,
            checks=parsed_checks,
            visual_blockers=visual_blockers,
            all_blocked_steps=all_blocked_steps,
            raw=obj,
        )

    @staticmethod
    def _parse_text_fields(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        current_key = ""
        current_value: list[str] = []
        for raw in str(text or "").splitlines():
            line = raw.rstrip()
            if not line.strip():
                continue
            m = re.match(r"^([a-zA-Z_]+):\s*(.*)$", line.strip())
            if m:
                if current_key:
                    out[current_key] = "\n".join(current_value).strip()
                current_key = str(m.group(1) or "").strip().lower()
                current_value = [str(m.group(2) or "").strip()]
                continue
            if current_key:
                current_value.append(line.strip())
        if current_key:
            out[current_key] = "\n".join(current_value).strip()
        return out

    @classmethod
    def _parse_text(cls, text: str) -> PreconditionCheckResult:
        fields = cls._parse_text_fields(text)
        passed_raw = str(fields.get("passed") or "").strip().lower()
        if passed_raw in {"yes", "true", "pass"}:
            status = "pass"
        elif passed_raw in {"no", "false", "fail"}:
            status = "fail"
        else:
            raise ValueError("missing_passed_yes_no")
        reason_code = str(fields.get("reason_code") or "").strip()
        reason = str(fields.get("reason") or "").strip()
        missing = str(fields.get("missing_prerequisite") or "").strip()
        repair = str(fields.get("minimal_repair") or "").strip()
        feedback = reason
        if missing:
            feedback = f"missing_prerequisite={missing}"
        if repair:
            feedback = (feedback + "\nminimal_repair:\n" + repair).strip()
        inferred = str(fields.get("inferred_context") or "").strip()
        relevant_state = [line.strip("- ").strip() for line in inferred.splitlines() if line.strip()]
        return PreconditionCheckResult(
            status=status,
            reason_code=reason_code,
            reason=reason,
            feedback_to_planner=feedback,
            relevant_state=relevant_state[:5],
            checks={
                "mode_ok": status == "pass",
                "hand_state_ok": status == "pass",
                "visibility_ok": status == "pass",
                "interactability_ok": status == "pass",
                "task_state_ok": status == "pass",
            },
            visual_blockers=[],
            all_blocked_steps=[],
            raw={"raw_text": text},
        )

    @staticmethod
    def _parse_step_trace_compact(remaining_plan_text: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw in str(remaining_plan_text or "").splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^\d+\.\s+([^\s]+)\s+([a-zA-Z_]+):([a-zA-Z0-9_]+)\s+args=(\{.*\})$", line)
            if not m:
                continue
            step_id, step_type, name, args_text = m.groups()
            try:
                args = json.loads(args_text)
            except Exception:
                args = {}
            rows.append(
                {
                    "step_id": str(step_id or "").strip(),
                    "type": str(step_type or "").strip(),
                    "name": str(name or "").strip(),
                    "args": args if isinstance(args, dict) else {},
                }
            )
        return rows

    @staticmethod
    def _parse_stm_window_steps(text: str) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        cur: dict[str, Any] | None = None
        for raw in str(text or "").splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            if stripped.startswith("- step_id:"):
                if cur:
                    steps.append(cur)
                step_id = stripped.split(":", 1)[1].strip()
                try:
                    step_id_val: Any = int(step_id)
                except Exception:
                    step_id_val = step_id
                cur = {"step_id": step_id_val, "action_or_skill": "", "params": {}, "result_summary": "", "errors": ""}
                continue
            if cur is None:
                continue
            if stripped.startswith("action_or_skill:"):
                cur["action_or_skill"] = stripped.split(":", 1)[1].strip().strip('"')
                continue
            if stripped.startswith("params:"):
                payload = stripped.split(":", 1)[1].strip()
                try:
                    parsed = json.loads(payload)
                except Exception:
                    parsed = {}
                cur["params"] = parsed if isinstance(parsed, dict) else {}
                continue
            if stripped.startswith("result_summary:"):
                cur["result_summary"] = stripped.split(":", 1)[1].strip().strip('"')
                continue
            if stripped.startswith("errors:"):
                cur["errors"] = stripped.split(":", 1)[1].strip().strip('"')
                continue
        if cur:
            steps.append(cur)
        return steps

    @staticmethod
    def _extract_feedback_fields(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            key = str(k).strip()
            if key:
                out[key] = str(v).strip()
        return out

    @staticmethod
    def _parse_visual_tags(text: str) -> list[str]:
        fields = PreconditionChecker._extract_feedback_fields(text)
        raw = str(fields.get("tags") or "").strip()
        if not raw:
            return []
        try:
            obj = json.loads(raw)
        except Exception:
            obj = None
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        return []

    @staticmethod
    def _parse_task_progress_current_subtask(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except Exception:
            return ""
        if not isinstance(obj, dict):
            return ""
        goal_state = obj.get("goal_state") if isinstance(obj.get("goal_state"), dict) else {}
        return str(goal_state.get("current_subgoal") or "").strip()

    @staticmethod
    def _extract_held_item_from_error(text: str) -> str:
        raw = str(text or "")
        m = re.search(r"held_item='([^']+)'", raw)
        if m:
            return str(m.group(1) or "").strip()
        return ""

    @staticmethod
    def _is_container_like(name: str) -> bool:
        low = str(name or "").strip().lower()
        if not low:
            return False
        keywords = ("pot", "pan", "bowl", "plate", "tray", "casserole", "container")
        if any(k in low for k in keywords):
            return True
        return any(k in str(name or "") for k in ("锅", "盘", "碗", "盆", "托盘", "容器"))

    @staticmethod
    def _is_placement_like(name: str) -> bool:
        low = str(name or "").strip().lower()
        if not low:
            return False
        keywords = (
            "side table",
            "counter",
            "shelf",
            "stove place point",
            "table",
            "place point",
            "plate stack",
            "deep plate",
            "large table center",
            "top shelf",
            "surface",
        )
        return any(k in low for k in keywords)

    @staticmethod
    def _is_source_like(name: str) -> bool:
        low = str(name or "").strip().lower()
        if not low:
            return False
        keywords = ("source", "stack", "crate", "shelf")
        return any(k in low for k in keywords)

    @staticmethod
    def _is_liquid_like(name: str) -> bool:
        low = str(name or "").strip().lower()
        if not low:
            return False
        keywords = ("oil", "wine", "juice", "vinegar", "water", "milk", "broth", "sauce")
        return any(k in low for k in keywords)

    @staticmethod
    def _normalize_name(name: str) -> str:
        low = str(name or "").strip().lower()
        if not low:
            return ""
        low = re.sub(r"[<>{}\[\]\(\)_]+", " ", low)
        low = re.sub(r"[^a-z0-9\s]+", " ", low)
        return " ".join(low.split())

    @classmethod
    def _names_compatible(cls, left: str, right: str) -> bool:
        a = cls._normalize_name(left)
        b = cls._normalize_name(right)
        if not a or not b:
            return False
        return a == b or a in b or b in a

    @classmethod
    def _held_item_kind(cls, name: str) -> str:
        if cls._is_liquid_like(name):
            return "liquid"
        if cls._is_container_like(name):
            return "container"
        low = str(name or "").strip().lower()
        if "repair phone" in low or ("repair" in low and "phone" in low):
            return "repair_tool"
        if any(k in low for k in ("oregano", "thyme", "garlic dried", "pepper", "salt", "spice")):
            return "spice"
        if any(k in low for k in ("knife", "spatula", "ladle", "cutting board", "potato masher")):
            return "tool"
        return "unknown"

    @staticmethod
    def _step_target_name(step: dict[str, Any]) -> str:
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        for key in ("target", "container_name", "item", "object_name", "query"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @classmethod
    def _infer_symbolic_state_from_history(cls, recent_history_text: str) -> dict[str, Any]:
        steps = cls._parse_stm_window_steps(recent_history_text)
        held_item = ""
        mode = ""
        last_nav_target = ""
        nav_after_pick = ""
        picked_since_last_put_down = False
        for row in steps:
            name = str(row.get("action_or_skill") or "").strip()
            params = row.get("params") if isinstance(row.get("params"), dict) else {}
            result = str(row.get("result_summary") or "").strip().lower()
            errors = str(row.get("errors") or "").strip()
            target = str(params.get("target") or "").strip()

            hinted_held = cls._extract_held_item_from_error(errors)
            if hinted_held:
                held_item = hinted_held

            if result != "success":
                continue

            if name == "auto_navigation":
                last_nav_target = target
                if picked_since_last_put_down:
                    nav_after_pick = target
            elif name == "pick_up":
                picked_since_last_put_down = True
                nav_after_pick = ""
                if target:
                    held_item = target
                elif last_nav_target:
                    held_item = last_nav_target
            elif name == "gui_buy_new_item":
                item = str(params.get("item") or "").strip()
                if item:
                    held_item = item
                    picked_since_last_put_down = True
                    nav_after_pick = ""
            elif name == "put_down":
                held_item = ""
                picked_since_last_put_down = False
                nav_after_pick = ""
            elif name == "enter_pouring_mode":
                mode = "pour"
            elif name == "exit_pouring_mode":
                mode = ""
            elif name == "enter_spices_sprinkle_mode":
                mode = "sprinkle"
            elif name in {"exit_spices_sprinkle_mode", "exit_sprinkle_mode"}:
                mode = ""
        return {
            "held_item": held_item,
            "held_item_kind": cls._held_item_kind(held_item),
            "mode": mode,
            "last_nav_target": last_nav_target,
            "last_nav_target_kind": (
                "container" if cls._is_container_like(last_nav_target)
                else "placement_point" if cls._is_placement_like(last_nav_target)
                else "item_or_source" if last_nav_target else "unknown"
            ),
            "nav_after_pick": nav_after_pick,
            "nav_after_pick_kind": (
                "container" if cls._is_container_like(nav_after_pick)
                else "placement_point" if cls._is_placement_like(nav_after_pick)
                else "item_or_source" if nav_after_pick else "unknown"
            ),
        }

    @classmethod
    def _evaluate_key_rule(
        cls,
        *,
        step: dict[str, Any],
        state: dict[str, Any],
        current_subtask: str,
        latest_visual_feedback_text: str,
        last_feedback: str,
    ) -> dict[str, Any] | None:
        name = str(step.get("name") or "").strip()
        target = cls._step_target_name(step)
        held_item = str(state.get("held_item") or "").strip()
        held_item_kind = str(state.get("held_item_kind") or cls._held_item_kind(held_item)).strip()
        mode = str(state.get("mode") or "").strip()
        last_nav_target = str(state.get("last_nav_target") or "").strip()
        last_nav_target_kind = str(state.get("last_nav_target_kind") or "").strip()
        nav_after_pick = str(state.get("nav_after_pick") or "").strip()
        nav_after_pick_kind = str(state.get("nav_after_pick_kind") or "").strip()
        visual_tags = cls._parse_visual_tags(latest_visual_feedback_text)
        visual_text = str(latest_visual_feedback_text or "").lower()
        last_feedback_l = str(last_feedback or "").lower()
        current_subtask_l = str(current_subtask or "").lower()

        if mode and name == "auto_navigation":
            return {
                "reason_code": "auto_navigation_blocked_by_mode",
                "reason": f"history implies the agent is still in interaction mode {mode}.",
                "feedback_to_planner": f"before retrying {name}, first exit the current {mode} mode because navigation is blocked while that mode is active.",
                "checks": {"mode_ok": False},
                "relevant_state": [f"interaction_mode={mode}"],
                "visual_blockers": [],
            }

        if held_item and name in {"pick_up", "pick_up_into_the_container"}:
            return {
                "reason_code": "pick_up_while_already_holding",
                "reason": f"history implies the agent is already holding {held_item}.",
                "feedback_to_planner": f"before retrying {name}, first use or put down {held_item} because the history already shows it is being held.",
                "checks": {"hand_state_ok": False},
                "relevant_state": [f"held_item={held_item}"],
                "visual_blockers": [],
            }

        if name == "pick_up" and not last_nav_target:
            return {
                "reason_code": "pick_up_requires_recent_navigation",
                "reason": "history does not show a recent successful navigation to the target item or source.",
                "feedback_to_planner": "before retrying pick_up, first navigate to the target item or source because pick_up requires a recent approach and alignment.",
                "checks": {"interactability_ok": False},
                "relevant_state": ["last_nav_target=none"],
                "visual_blockers": [],
            }

        if name == "pick_up" and last_nav_target_kind in {"placement_point", "container"} and not cls._is_source_like(last_nav_target):
            return {
                "reason_code": "pick_up_requires_item_navigation",
                "reason": f"the most recent navigation was to {last_nav_target}, which does not look like the pickup target item or source.",
                "feedback_to_planner": f"before retrying pick_up, first navigate to the actual target item or source because the current approach target is {last_nav_target}.",
                "checks": {"interactability_ok": False},
                "relevant_state": [f"last_nav_target={last_nav_target}", f"last_nav_target_kind={last_nav_target_kind}"],
                "visual_blockers": [],
            }

        if name == "pick_up" and target and last_nav_target and not cls._names_compatible(target, last_nav_target):
            return {
                "reason_code": "pick_up_navigation_target_mismatch",
                "reason": f"the pickup target is {target}, but the most recent navigation target was {last_nav_target}.",
                "feedback_to_planner": f"before retrying pick_up, first navigate to {target} because the current approach target does not match the intended pickup target.",
                "checks": {"interactability_ok": False},
                "relevant_state": [f"pickup_target={target}", f"last_nav_target={last_nav_target}"],
                "visual_blockers": [],
            }

        if name == "pick_up_into_the_container" and (not held_item or held_item_kind != "container"):
            return {
                "reason_code": "pick_up_into_container_requires_holding_container",
                "reason": "history does not show that the agent is currently holding a container.",
                "feedback_to_planner": "before retrying pick_up_into_the_container, first hold the destination container because this action requires a container already in hand.",
                "checks": {"hand_state_ok": False},
                "relevant_state": [f"held_item={held_item or 'none'}", f"held_item_kind={held_item_kind or 'unknown'}"],
                "visual_blockers": [],
            }

        if name == "put_down" and not held_item:
            return {
                "reason_code": "put_down_requires_holding_item",
                "reason": "history implies the agent is not currently holding any item.",
                "feedback_to_planner": "before retrying put_down, first hold an item because put_down only makes sense while carrying something.",
                "checks": {"hand_state_ok": False},
                "relevant_state": ["held_item=none"],
                "visual_blockers": [],
            }

        if name == "put_down" and nav_after_pick_kind != "placement_point":
            return {
                "reason_code": "put_down_requires_placement_navigation",
                "reason": "history does not show a recent successful navigation to a placement point before put_down.",
                "feedback_to_planner": "before retrying put_down, first navigate to a valid placement point because put_down requires a placement approach immediately beforehand.",
                "checks": {"interactability_ok": False},
                "relevant_state": [f"held_item={held_item or 'unknown'}", f"nav_after_pick={nav_after_pick or 'none'}"],
                "visual_blockers": [],
            }

        if name == "put_down" and target and nav_after_pick and not cls._names_compatible(target, nav_after_pick):
            return {
                "reason_code": "put_down_navigation_target_mismatch",
                "reason": f"the put_down target is {target}, but the most recent post-pick navigation target was {nav_after_pick}.",
                "feedback_to_planner": f"before retrying put_down, first navigate to {target} because the current placement approach does not match the intended put-down target.",
                "checks": {"interactability_ok": False},
                "relevant_state": [f"put_down_target={target}", f"nav_after_pick={nav_after_pick}"],
                "visual_blockers": [],
            }

        if name == "repair" and held_item_kind != "repair_tool":
            return {
                "reason_code": "repair_requires_repair_phone",
                "reason": f"history implies the held item is {held_item or 'unknown'}, not a repair phone.",
                "feedback_to_planner": "before retrying repair, first hold the repair phone because repair can only be executed while holding it.",
                "checks": {"hand_state_ok": False},
                "relevant_state": [f"held_item={held_item or 'none'}", f"held_item_kind={held_item_kind or 'unknown'}"],
                "visual_blockers": [],
            }

        if name == "gui_buy_new_item" and held_item:
            return {
                "reason_code": "gui_buy_new_item_requires_empty_hands",
                "reason": f"history implies the agent is already holding {held_item}, but buying a new item requires empty hands.",
                "feedback_to_planner": f"before retrying gui_buy_new_item, first put down or use the currently held {held_item} because store purchase requires empty hands and will otherwise fail.",
                "checks": {"hand_state_ok": False},
                "relevant_state": [f"held_item={held_item}", f"held_item_kind={held_item_kind or 'unknown'}"],
                "visual_blockers": [],
            }

        if name == "enter_pouring_mode" and held_item_kind not in {"liquid", "container"}:
            return {
                "reason_code": "enter_pouring_mode_requires_liquid_or_container",
                "reason": f"history implies the held item is {held_item or 'unknown'}, not a liquid bottle or pourable container.",
                "feedback_to_planner": "before retrying enter_pouring_mode, first hold a liquid bottle or a pourable container because pouring mode requires a liquid source/container in hand.",
                "checks": {"hand_state_ok": False},
                "relevant_state": [f"held_item={held_item or 'none'}", f"held_item_kind={held_item_kind}"],
                "visual_blockers": [],
            }

        if held_item and held_item_kind in {"liquid", "container"} and name in {"enter_pouring_mode", "auto_pour"}:
            if not cls._is_container_like(nav_after_pick):
                return {
                    "reason_code": "pour_requires_container_navigation",
                    "reason": f"history shows {held_item} was picked up but the agent has not navigated to a container after that pickup.",
                    "feedback_to_planner": f"before retrying {name}, first navigate to the target container because {held_item} was picked up away from the container.",
                    "checks": {"interactability_ok": False},
                    "relevant_state": [f"held_item={held_item}", f"nav_after_pick={nav_after_pick or 'none'}"],
                    "visual_blockers": [],
                }

        if name == "auto_pour" and mode != "pour":
            return {
                "reason_code": "auto_pour_requires_pouring_mode",
                "reason": "history does not show that the agent is currently in pouring mode.",
                "feedback_to_planner": "before retrying auto_pour, first enter pouring mode because auto_pour is only valid after pouring mode is active.",
                "checks": {"mode_ok": False},
                "relevant_state": [f"interaction_mode={mode or 'idle'}", f"held_item={held_item or 'unknown'}"],
                "visual_blockers": [],
            }

        if name == "auto_pour" and held_item_kind not in {"liquid", "container"}:
            return {
                "reason_code": "auto_pour_requires_liquid_or_container_in_hand",
                "reason": "history does not support that a liquid bottle or pourable container is currently held.",
                "feedback_to_planner": "before retrying auto_pour, first ensure a liquid bottle or pourable container is in hand because auto_pour requires pouring from a liquid source/container.",
                "checks": {"hand_state_ok": False},
                "relevant_state": [f"held_item={held_item or 'none'}", f"held_item_kind={held_item_kind}"],
                "visual_blockers": [],
            }

        if name == "enter_spices_sprinkle_mode" and held_item_kind != "spice":
            return {
                "reason_code": "enter_sprinkle_mode_requires_spice",
                "reason": f"history implies the held item is {held_item or 'unknown'}, not a spice container.",
                "feedback_to_planner": "before retrying enter_spices_sprinkle_mode, first hold the spice because sprinkle mode requires a spice container in hand.",
                "checks": {"hand_state_ok": False},
                "relevant_state": [f"held_item={held_item or 'none'}", f"held_item_kind={held_item_kind}"],
                "visual_blockers": [],
            }

        if name == "enter_spices_sprinkle_mode" and nav_after_pick_kind != "container":
            return {
                "reason_code": "sprinkle_requires_container_navigation",
                "reason": "history does not show a recent successful navigation to the target container before entering sprinkle mode.",
                "feedback_to_planner": "before retrying enter_spices_sprinkle_mode, first navigate to the target container because sprinkle mode requires container alignment.",
                "checks": {"interactability_ok": False},
                "relevant_state": [f"nav_after_pick={nav_after_pick or 'none'}"],
                "visual_blockers": [],
            }

        if name == "auto_sprinkle" and mode != "sprinkle":
            return {
                "reason_code": "auto_sprinkle_requires_sprinkle_mode",
                "reason": "history does not show that the agent is currently in sprinkle mode.",
                "feedback_to_planner": "before retrying auto_sprinkle, first enter sprinkle mode because auto_sprinkle is only valid after sprinkle mode is active.",
                "checks": {"mode_ok": False},
                "relevant_state": [f"interaction_mode={mode or 'idle'}", f"held_item={held_item or 'unknown'}"],
                "visual_blockers": [],
            }

        if name == "auto_sprinkle" and held_item_kind != "spice":
            return {
                "reason_code": "auto_sprinkle_requires_spice_in_hand",
                "reason": "history does not support that a spice container is currently held.",
                "feedback_to_planner": "before retrying auto_sprinkle, first ensure a spice container is in hand because auto_sprinkle requires sprinkling from a spice source.",
                "checks": {"hand_state_ok": False},
                "relevant_state": [f"held_item={held_item or 'none'}", f"held_item_kind={held_item_kind}"],
                "visual_blockers": [],
            }

        if ("held_item_occlusion" in visual_tags or "held_item_occlusion" in visual_text) and name in {
            "query_scene_objects",
            "pick_up",
            "enter_pouring_mode",
            "enter_spices_sprinkle_mode",
        }:
            return {
                "reason_code": "held_item_occlusion",
                "reason": "latest visual feedback says the held item is occluding the workspace.",
                "feedback_to_planner": f"before retrying {name}, first clear the held-item occlusion because the latest visual feedback says the workspace is blocked.",
                "checks": {"visibility_ok": False},
                "relevant_state": [f"held_item={held_item or 'unknown'}"],
                "visual_blockers": ["held_item_occlusion"],
            }

        if (
            held_item
            and ("not_in_pouring_mode" in last_feedback_l or "pouring_mode_unavailable" in last_feedback_l)
            and name in {"query_scene_objects", "pick_up", "auto_navigation"}
            and target
            and target.strip().lower() == held_item.strip().lower()
        ):
            return {
                "reason_code": "retry_same_held_item_instead_of_fixing_pour_setup",
                "reason": f"the last failure was a pouring-setup failure, but this step goes back to {held_item} instead of fixing the pour position.",
                "feedback_to_planner": f"before retrying {name}, first re-approach the target container and fix pouring setup because the previous failure was not caused by losing {held_item}.",
                "checks": {"task_state_ok": False},
                "relevant_state": [f"held_item={held_item}", "last_failure=not_in_pouring_mode"],
                "visual_blockers": [],
            }

        if current_subtask_l and "add to a pot" in current_subtask_l and name == "gui_buy_new_item" and held_item:
            return {
                "reason_code": "buying_while_current_blocker_unresolved",
                "reason": f"the current recipe subtask is still active and {held_item} is still in hand.",
                "feedback_to_planner": f"before retrying {name}, first finish or clear the currently held {held_item} because the current pot subtask is still unresolved.",
                "checks": {"task_state_ok": False},
                "relevant_state": [f"held_item={held_item}", "subtask=pot_addition"],
                "visual_blockers": [],
            }

        return None

    @classmethod
    def _apply_virtual_step(cls, *, step: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        out = dict(state or {})
        name = str(step.get("name") or "").strip()
        target = cls._step_target_name(step)
        if name == "auto_navigation":
            out["last_nav_target"] = target
            out["last_nav_target_kind"] = (
                "container" if cls._is_container_like(target)
                else "placement_point" if cls._is_placement_like(target)
                else "item_or_source" if target else "unknown"
            )
            if str(out.get("held_item") or "").strip():
                out["nav_after_pick"] = target
                out["nav_after_pick_kind"] = out["last_nav_target_kind"]
        elif name in {"pick_up", "pick_up_into_the_container"}:
            if target:
                out["held_item"] = target
            elif str(out.get("last_nav_target") or "").strip():
                out["held_item"] = str(out.get("last_nav_target") or "").strip()
            out["held_item_kind"] = cls._held_item_kind(str(out.get("held_item") or "").strip())
            out["nav_after_pick"] = ""
            out["nav_after_pick_kind"] = "unknown"
        elif name == "gui_buy_new_item":
            item = str((step.get("args") or {}).get("item") or "").strip()
            if item:
                out["held_item"] = item
                out["held_item_kind"] = cls._held_item_kind(item)
                out["nav_after_pick"] = ""
                out["nav_after_pick_kind"] = "unknown"
        elif name == "put_down":
            out["held_item"] = ""
            out["held_item_kind"] = "unknown"
            out["nav_after_pick"] = ""
            out["nav_after_pick_kind"] = "unknown"
        elif name == "enter_pouring_mode":
            out["mode"] = "pour"
        elif name == "exit_pouring_mode":
            out["mode"] = ""
        elif name == "enter_spices_sprinkle_mode":
            out["mode"] = "sprinkle"
        elif name in {"exit_spices_sprinkle_mode", "exit_sprinkle_mode"}:
            out["mode"] = ""
        return out

    @classmethod
    def _check_key_rules(
        cls,
        *,
        step: PlanStep,
        remaining_plan_text: str,
        task_progress_text: str,
        recent_history_text: str,
        latest_visual_feedback_text: str,
        last_feedback: str,
    ) -> PreconditionCheckResult:
        remaining_steps = cls._parse_step_trace_compact(remaining_plan_text)
        if not remaining_steps:
            remaining_steps = [{"step_id": str(step.step_id or ""), "type": str(step.type or ""), "name": str(step.name or ""), "args": dict(step.args or {})}]
        current_subtask = cls._parse_task_progress_current_subtask(task_progress_text)
        state = cls._infer_symbolic_state_from_history(recent_history_text)
        blocked_steps: list[Dict[str, Any]] = []
        first_rule: dict[str, Any] | None = None
        failed_dims: dict[str, bool] = {
            "mode_ok": True,
            "hand_state_ok": True,
            "visibility_ok": True,
            "interactability_ok": True,
            "task_state_ok": True,
        }
        relevant_state: list[str] = []
        visual_blockers: list[str] = []

        for idx, cur in enumerate(remaining_steps, start=1):
            rule = cls._evaluate_key_rule(
                step=cur,
                state=state,
                current_subtask=current_subtask,
                latest_visual_feedback_text=latest_visual_feedback_text,
                last_feedback=last_feedback,
            )
            if rule is not None:
                if first_rule is None:
                    first_rule = rule
                    for key, value in dict(rule.get("checks") or {}).items():
                        if key in failed_dims and not bool(value):
                            failed_dims[key] = False
                    relevant_state = list(rule.get("relevant_state") or [])[:3]
                    visual_blockers = list(rule.get("visual_blockers") or [])[:3]
                blocked_steps.append(
                    {
                        "index": idx,
                        "name": str(cur.get("name") or "").strip(),
                        "reason_code": str(rule.get("reason_code") or "").strip(),
                        "reason": str(rule.get("reason") or "").strip(),
                    }
                )
                continue
            state = cls._apply_virtual_step(step=cur, state=state)

        if first_rule is None:
            return PreconditionCheckResult(
                status="pass",
                reason_code="pass",
                reason="key rules passed",
                feedback_to_planner="",
                relevant_state=[],
                checks=failed_dims,
                visual_blockers=[],
                all_blocked_steps=[],
                raw={"mode": "key_rules", "history_state": state},
            )

        return PreconditionCheckResult(
            status="fail",
            reason_code=str(first_rule.get("reason_code") or "").strip(),
            reason=str(first_rule.get("reason") or "").strip(),
            feedback_to_planner=str(first_rule.get("feedback_to_planner") or "").strip(),
            relevant_state=relevant_state,
            checks=failed_dims,
            visual_blockers=visual_blockers,
            all_blocked_steps=blocked_steps,
            raw={"mode": "key_rules", "history_state": state, "current_subtask": current_subtask},
        )

    def _build_prompt(
        self,
        *,
        observation: Observation,
        step: PlanStep,
        remaining_plan_text: str,
        recipe_text: str,
        high_level_goal: str,
        task_progress_text: str,
        recent_history_text: str,
        latest_visual_feedback_text: str,
        agent_state_text: str,
        body_rules_text: str,
        strategy_notes_text: str,
        tool_manifest_text: str,
        last_feedback: str,
    ) -> str:
        role_text = self._load_prompt_asset(
            "role.txt",
            (
                "You are an action feasibility checker for EPM.\n"
                "Your role:\n"
                "- Judge whether the candidate next step is immediately executable.\n"
                "- Use only short-horizon history, latest feedback, and concise visual feedback.\n"
                "- Perform lightweight heuristic inference from recent execution context."
            ),
        )
        rules_text = self._load_prompt_asset(
            "validation_rules.txt",
            (
                "Infer only this minimal execution context:\n"
                "- hands_state: empty / holding / unknown\n"
                "- held_item_kind: liquid / spice / container / tool / unknown\n"
                "- interaction_mode: idle / pouring / sprinkling / cutting / mixing / flipping / unknown\n"
                "- recent_navigation_target\n"
                "- recent_navigation_target_kind: item / container / placement_point / tool_point / unknown\n"
                "\n"
                "Validation rules:\n"
                "- pick_up requires empty hands and recent navigation to the target item or source.\n"
                "- put_down requires holding an item and recent navigation to a placement point.\n"
                "- gui_buy_new_item requires empty hands before opening the store flow.\n"
                "- enter_pouring_mode requires holding a liquid bottle or a pourable container and recent navigation to the target container.\n"
                "- auto_pour requires pouring mode and holding a liquid bottle or a pourable container.\n"
                "- enter_spices_sprinkle_mode requires holding a spice and recent navigation to the target container.\n"
                "- auto_sprinkle requires sprinkling mode and holding a spice.\n"
                "- auto_navigation should not be executed while blocked by an incompatible interaction mode.\n"
                "- Apply the same pattern to other interaction actions: first check the immediately required hand state, mode state, and recent navigation/alignment context, then judge whether the candidate action is executable.\n"
                "- When an action is not explicitly listed above, infer its immediate prerequisites from short-horizon interaction history and common embodied interaction constraints.\n"
                "- If an action is invalid, explain the reason and suggest the smallest useful modification rather than only rejecting it."
            ),
        )
        output_text = self._load_prompt_asset(
            "output_format.txt",
            (
                "Output plain text only.\n"
                "Use this exact format:\n"
                "passed: yes/no\n"
                "reason_code: ...\n"
                "reason: ...\n"
                "inferred_context:\n"
                "- hands_state=...\n"
                "- held_item_kind=...\n"
                "- interaction_mode=...\n"
                "- recent_navigation_target=...\n"
                "- recent_navigation_target_kind=...\n"
                "missing_prerequisite: ...\n"
                "minimal_repair:\n"
                "- ...\n"
                "- ..."
            ),
        )
        return (
            "SYSTEM:\n"
            f"{role_text}\n"
            "\n"
            f"{rules_text}\n"
            "\n"
            "Validation procedure:\n"
            "1. Replay recent history in order.\n"
            "2. Infer the minimal execution context.\n"
            "3. Check the candidate step against its immediate prerequisites.\n"
            "4. If invalid, reject it and identify the most important missing prerequisite.\n"
            "5. Suggest only the minimal repair action(s).\n"
            "\n"
            f"{output_text}\n"
            "\n"
            "USER:\n"
            f"High-level goal: {str(high_level_goal or '').strip()}\n"
            f"Current step: type={step.type} name={step.name} args={dict(step.args or {})}\n"
            "\n"
            "Remaining planned sequence:\n"
            f"{(remaining_plan_text or '').strip()}\n"
            "\n"
            "Recipe (raw):\n"
            f"{(recipe_text or '').strip()}\n"
            "\n"
            "Body rules:\n"
            f"{(body_rules_text or '').strip()}\n"
            "\n"
            "Strategy notes:\n"
            f"{(strategy_notes_text or '').strip()}\n"
            "\n"
            "Tool manifest:\n"
            f"{(tool_manifest_text or '').strip()}\n"
            "\n"
            "Current task_progress:\n"
            f"{(task_progress_text or '').strip()}\n"
            "\n"
            "Recent STM window:\n"
            f"{(recent_history_text or '').strip()}\n"
            "\n"
            "Latest visual feedback:\n"
            f"{(latest_visual_feedback_text or '').strip()}\n"
            "\n"
            "Recent feedback:\n"
            f"{(last_feedback or '').strip()}\n"
            "\n"
            "Agent state (JSON):\n"
            f"{(agent_state_text or '').strip()}\n"
        )

    @staticmethod
    def _repair_prompt(*, previous_output: str) -> str:
        prev = (previous_output or "").strip()
        if len(prev) > 2000:
            prev = prev[:2000] + "..."
        return (
            "Rewrite your answer to be a SINGLE JSON object and NOTHING else.\n"
            "- No markdown.\n"
            "- No commentary.\n"
            "- No <think>...</think>.\n"
            "- Use valid JSON (double quotes, no trailing commas).\n"
            "- status must be pass or fail.\n"
            "- feedback_to_planner must use: before retrying X, first do Y because Z.\n"
            "\n"
            "Schema:\n"
            "{\n"
            '  "status": "pass|fail",\n'
            '  "reason_code": "short_code",\n'
            '  "reason": "short explanation",\n'
            '  "all_blocked_steps": [{"index": 1, "name": "step_name", "reason_code": "short_code", "reason": "short explanation"}],\n'
            '  "checks": {\n'
            '    "mode_ok": true,\n'
            '    "hand_state_ok": true,\n'
            '    "visibility_ok": true,\n'
            '    "interactability_ok": true,\n'
            '    "task_state_ok": true\n'
            "  },\n"
            '  "visual_blockers": ["held_item_occludes_view"],\n'
            '  "feedback_to_planner": "concrete sequence-repair feedback",\n'
            '  "relevant_state": ["grounded observations"]\n'
            "}\n"
            "\n"
            "Previous response (invalid):\n"
            f"{prev}\n"
        )
