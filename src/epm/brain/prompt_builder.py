from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

from epm.core.epm_types import Observation
from epm.brain.modules.prompt_assets import load_asset
from epm.brain.skills_prompt import list_skill_card_names, render_skill_cards_for_prompt, select_skill_cards
from epm.core.prompt_ablation import assert_prompt_ablation, is_prompt_group_enabled, is_prompt_section_enabled


_SKILL_INTERFACE_NAME_HINTS = (
    "auto_cut",
    "auto_filp",
    "auto_flip",
    "auto_mix",
    "auto_navigation",
    "auto_perception",
    "auto_pour",
    "auto_sprinkle",
    "query_scene_objects",
    "query_product_catalog",
    "query_tool_manual",
    "gui_buy_new_item",
    "gui_order_dish_via_computer",
    "gui_submit_dish_via_checkout_stand",
)


@dataclass(frozen=True)
class PromptBuilderConfig:
    """
    Controls which memory snippets are injected into the prompt.
    """

    inject_memory: bool = True
    inject_tool_schemas: bool = True
    save_last_prompt_path: Optional[Path] = None
    save_per_step_dir: Optional[Path] = None
    plan_min_steps: int = 3
    plan_max_steps: int = 8
    force_submit_step_threshold: int = 0
    force_submit_within_steps: int = 50
    # If True, the planner is expected to emit steps via OpenAI-style tool calls.
    # In this mode, it's normal to emit 1 step per round (incremental planning).
    tool_calling: bool = False
    # Optional layout config loaded from JSON (e.g. epm/memory/prompt_layout.json).
    prompt_layout: Optional[Dict[str, Any]] = None
    prompt_ablation_profile: str = "full"
    prompt_disabled_groups: frozenset[str] | None = None
    vision_image_max_side: int = 768


class PromptBuilder:
    def __init__(self, cfg: PromptBuilderConfig) -> None:
        self.cfg = cfg

    def _section_enabled(self, section: str, *, memory_bundle: Dict[str, str]) -> bool:
        # The run-level config is authoritative. A memory bundle may carry the
        # value for traceability, but may not silently change the experiment arm.
        return is_prompt_section_enabled(
            profile=self.cfg.prompt_ablation_profile,
            section=section,
            disabled_groups=self.cfg.prompt_disabled_groups,
        )

    def _skill_interface_enabled(self) -> bool:
        return is_prompt_group_enabled(
            profile=self.cfg.prompt_ablation_profile,
            group="skill_interface",
            disabled_groups=self.cfg.prompt_disabled_groups,
        )

    def _strip_skill_interface_references(self, text: str) -> str:
        if self._skill_interface_enabled():
            return str(text or "")
        replacements = (
            ("`yield action(...)` / `yield skill(...)`", "`yield action(...)`"),
            ("yield action(...) / yield skill(...)", "yield action(...)"),
            ("action/skill", "action"),
            ("action / skill", "action"),
            ("action or skill", "action"),
            ("actions and skills", "actions"),
            ("actions/skills", "actions"),
            ("action/skills", "actions"),
        )
        kept: list[str] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line
            for source, replacement in replacements:
                line = line.replace(source, replacement)
            lowered = line.lower()
            if "skill" in lowered or any(name in lowered for name in _SKILL_INTERFACE_NAME_HINTS):
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    def _audit_final_prompt(self, *, source: str, prompt: str) -> None:
        assert_prompt_ablation(
            profile=self.cfg.prompt_ablation_profile,
            disabled_groups=self.cfg.prompt_disabled_groups,
            source=source,
            prompt_text=prompt,
        )

    @staticmethod
    def _normalize_pipeline_name(name: str) -> str:
        raw = str(name or "").strip().lower()
        if raw in ("planner-executor", "pe"):
            return "planner_executor"
        if raw in ("epm_agent",):
            return "epm"
        return raw

    @classmethod
    def _load_method_unique_asset(cls, pipeline_name: str, asset_name: str, fallback: str) -> str:
        normalized = cls._normalize_pipeline_name(pipeline_name)
        if normalized == "planner_executor":
            return load_asset("planner_executor/prompt_builder", asset_name, fallback)
        if normalized == "react":
            return load_asset("react/prompt_builder", asset_name, fallback)
        return str(fallback or "").strip()

    @staticmethod
    def _safe_template_format(template: str, values: Dict[str, Any]) -> str:
        text = str(template or "")
        if not text:
            return ""
        try:
            return text.format(**values).strip()
        except Exception:
            return text.strip()

    def _plan_step_window_text(self) -> tuple[int, int, str, str]:
        pmin = int(self.cfg.plan_min_steps)
        pmax = int(self.cfg.plan_max_steps)
        if pmax >= 0 and pmin == pmax:
            step_count_phrase = f"exactly {pmin} step{'s' if pmin != 1 else ''}"
            next_step_phrase = f"decide exactly {pmin} next step{'s' if pmin != 1 else ''}"
        elif pmax >= 0:
            step_count_phrase = f"between {pmin} and {pmax} steps"
            next_step_phrase = f"decide the next {pmin}-{pmax} step window"
        else:
            step_count_phrase = f"at least {pmin} steps"
            next_step_phrase = f"decide at least the next {pmin} steps"
        return pmin, pmax, step_count_phrase, next_step_phrase

    def _build_response_format_action_example_objects(self) -> list[str]:
        pmin = int(self.cfg.plan_min_steps)
        pmax = int(self.cfg.plan_max_steps)
        example_step_count = pmin if (pmax >= 0 and pmin == pmax) else max(2, pmin)
        example_step_count = max(3, min(example_step_count, 8))
        ordinal_words = [
            "first", "second", "third", "fourth",
            "fifth", "sixth", "seventh", "eighth",
        ]
        example_specs: list[tuple[str, str, str, str]] = [
            (
                "action",
                "EXACT_ACTION_NAME_FROM_SCHEMA",
                '"required_arg_name":"required_arg_value"',
                "the correct object, tool, or workspace is reached or made ready",
            ),
            (
                "action" if not self._skill_interface_enabled() else "skill",
                "EXACT_ACTION_NAME_FROM_SCHEMA" if not self._skill_interface_enabled() else "EXACT_SKILL_NAME_FROM_SCHEMA",
                '"required_arg_name":"required_arg_value","optional_arg_name":"optional_arg_value_if_needed"',
                "the next required state change is completed and sets up the following step",
            ),
            (
                "action",
                "ANOTHER_EXACT_ACTION_NAME_FROM_SCHEMA",
                '"required_arg_name":"required_arg_value","destination_arg_name":"destination_arg_value_if_needed"',
                "the object is moved, transformed, or aligned into the needed intermediate state",
            ),
        ]
        while len(example_specs) < example_step_count:
            idx = len(example_specs) + 1
            label = ordinal_words[idx - 1] if idx - 1 < len(ordinal_words) else f"step_{idx}"
            example_specs.append(
                (
                    "action" if (not self._skill_interface_enabled() or idx % 2 == 1) else "skill",
                    f"EXACT_{label.upper()}_{'ACTION' if (not self._skill_interface_enabled() or idx % 2 == 1) else 'SKILL'}_NAME_FROM_SCHEMA",
                    f'"required_arg_name":"{label}_required_arg_value","optional_arg_name":"optional_arg_value_if_needed"',
                    f"the {label} planned subgoal is completed with a concrete observable result",
                )
            )
        return [
            (
                f'    {{"step_id":"Hk.A{i}","type":"{step_type}","name":"{name}",'
                f'"args":{{{args}}},"expectation":"{expectation}"}}'
            )
            for i, (step_type, name, args, expectation) in enumerate(example_specs[:example_step_count], start=1)
        ]

    def _build_response_format_template_values(
        self,
        *,
        pmin: int,
        pmax: int,
        step_count_phrase: str,
        next_step_phrase: str,
    ) -> Dict[str, str]:
        example_objects = self._build_response_format_action_example_objects()
        extra_objects = example_objects[3:]
        return {
            "next_step_phrase": next_step_phrase,
            "plan_min_steps": str(pmin),
            "plan_max_steps": str(pmax),
            "step_count_phrase": step_count_phrase,
            "response_format_action_examples": ",\n".join(example_objects),
            "response_format_action_example_1": example_objects[0],
            "response_format_action_example_2": example_objects[1],
            "response_format_action_example_3": example_objects[2],
            "response_format_action_examples_extra": (",\n" + ",\n".join(extra_objects)) if extra_objects else "",
        }

    @staticmethod
    def _is_instance_disambiguation_feedback(feedback: str) -> bool:
        return "blocking=instance_disambiguation" in str(feedback or "").strip().lower()

    @staticmethod
    def _append_instance_id_selection_mode(
        *,
        system_parts: list[str],
    ) -> None:
        system_parts.append("Instance_id Selection Mode (blocking):")
        system_parts.append("- The previous step failed due to instance ambiguity and must be retried immediately.")
        system_parts.append("- Choose exactly one `instance_id` from feedback `candidates`.")
        system_parts.append("- Output EXACTLY one JSON object with ONLY two keys: `name` and `instance_id`.")
        system_parts.append("- Do NOT output action_list/tool_calls/thoughts/explanation/markdown.")
        system_parts.append("- The executor will fill this `name` + `instance_id` into the previous failed action/skill and retry it.")
        system_parts.append("- Prefer candidates with `is_on_screen=true`, `container=Scene`, and smaller `distance` when available.")
        system_parts.append("")

    def _layout_list(self, key: str, default: list[str]) -> list[str]:
        lay = self.cfg.prompt_layout or {}
        cur: Any = lay
        for part in key.split("."):
            if not isinstance(cur, dict):
                return list(default)
            cur = cur.get(part, None)
        v = cur
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            return [str(x) for x in v]
        return list(default)

    def _layout_str(self, key: str, default: str) -> str:
        lay = self.cfg.prompt_layout or {}
        cur: Any = lay
        for part in key.split("."):
            if not isinstance(cur, dict):
                return default
            cur = cur.get(part, None)
        v = cur
        if isinstance(v, str) and v.strip():
            return v.strip()
        return default

    def _layout_int(self, dotted_key: str, default: int) -> int:
        lay = self.cfg.prompt_layout or {}
        cur: Any = lay
        for part in dotted_key.split("."):
            if not isinstance(cur, dict):
                return int(default)
            cur = cur.get(part, None)
        try:
            return int(cur)
        except Exception:
            return int(default)

    @staticmethod
    def _filter_body_rules(text: str) -> str:
        """
        Remove legacy capability shorthands from body_rules.txt.

        The planner should rely on the tool/action schemas, not on deprecated pseudo-APIs like:
          - move: goto(target)
          - grasp: pick_up(object)
        """
        if not text:
            return ""
        lines = str(text).splitlines()
        out: list[str] = []
        skipping = False
        for raw in lines:
            s = raw.strip()
            if s.startswith("[Capabilities - cookbench_api list]"):
                skipping = True
                continue
            # Force-submit rule is rendered dynamically (only when active) with concrete numbers.
            if ("force_submit_active" in s) or ("强制提交覆盖" in s):
                continue
            if skipping:
                # Drop everything after the capabilities block header. In the current
                # repository template this block is at the end of the file.
                continue
            out.append(raw)
        return "\n".join(out).strip()

    @staticmethod
    def _filter_pipeline_specific_body_rules(text: str, *, pipeline_name: str) -> str:
        """
        Hide pipeline-specific body rules unless the current pipeline matches.

        Current convention:
        - Rule line containing "only for PE / planner_executor" is shown only for PE.
        """
        raw = str(text or "").strip()
        if not raw:
            return ""
        name = str(pipeline_name or "").strip().lower()
        is_pe = name in ("planner_executor", "planner-executor", "pe")

        lines = raw.splitlines()
        out: list[str] = []
        skipping = False
        for line in lines:
            s = line.strip()
            sl = s.lower()
            pe_only = ("only for pe / planner_executor" in sl) or ("pe / planner_executor only" in sl)
            if pe_only and not is_pe:
                skipping = True
                continue
            if skipping and re.match(r"^\d+\)\s+", s):
                skipping = False
            if not skipping:
                out.append(line)
        return "\n".join(out).strip()

    @staticmethod
    def _filter_strategy_notes(text: str) -> str:
        """
        Strategy notes are guidance-level prompt modules and do not need body-rule
        legacy filtering. Keep the content stable, only trimming whitespace.
        """
        return str(text or "").strip()

    @staticmethod
    def _render_force_submit_active_rule(agent_state_text: str) -> str:
        raw = str(agent_state_text or "").strip()
        if not raw:
            return ""
        try:
            st = json.loads(raw)
        except Exception:
            return ""
        if not isinstance(st, dict):
            return ""
        if not bool(st.get("force_submit_active", False)):
            return ""
        remaining = st.get("force_submit_remaining_steps")
        deadline = st.get("force_submit_deadline_step")
        threshold = st.get("force_submit_step_threshold")
        triggered = st.get("force_submit_triggered_step")
        return (
            "强制提交已激活："
            f"remaining_steps={remaining!r}, "
            f"deadline_step={deadline!r}, "
            f"triggered_step={triggered!r}, "
            f"threshold={threshold!r}."
        )

    @staticmethod
    def _sanitize_agent_state_snapshot(agent_state_text: str) -> str:
        """
        Normalize known game-feed inconsistencies for planner-facing prompt text.

        In some interaction modes (e.g. pour), the game feed may temporarily expose
        mode flags while `is_held` is reported as false and held item is missing.
        This is misleading for planning, so mark `is_held` as unknown in this case.
        """
        raw = str(agent_state_text or "").strip()
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except Exception:
            return raw
        if not isinstance(obj, dict):
            return raw

        mode = str(obj.get("mode") or "").strip().lower()
        held_item = obj.get("held_item")
        held_item_missing = (held_item is None) or (isinstance(held_item, str) and not held_item.strip())
        mode_like = ("pour", "cut", "mix", "sprinkle", "flip")
        if (
            any(mode.startswith(prefix) for prefix in mode_like)
            and (obj.get("is_held") is False)
            and held_item_missing
        ):
            obj["is_held"] = None
            obj["is_held_note"] = "unknown_in_mode_from_game_feed"

        # Hide force-submit runtime internals from planner prompt to avoid biasing
        # the model toward premature submit behavior.
        for k in (
            "force_submit_active",
            "force_submit_triggered_step",
            "force_submit_deadline_step",
            "force_submit_remaining_steps",
            "force_submit_overdue",
            "force_submit_stage",
            "force_submit_place_index",
            "force_submit_used_place_indices",
            "force_submit_step_threshold",
            "force_submit_within_steps",
        ):
            if k in obj:
                obj.pop(k, None)

        try:
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return raw

    @staticmethod
    def _sanitize_resume_state_snapshot(resume_state_text: str) -> str:
        raw = str(resume_state_text or "").strip()
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except Exception:
            return raw
        if not isinstance(obj, dict):
            return raw

        keep_order = (
            "continue_from_existing_progress",
            "status",
            "reason",
            "dish_id",
            "last_step_id",
            "last_completed_step_id",
            "last_physical_step_id",
            "last_network_resume_step_id",
            "last_http_403_resume_step_id",
            "updated_at",
            "last_time_iso",
        )
        out: dict[str, Any] = {}
        for key in keep_order:
            if key in obj:
                out[key] = obj.get(key)
        for key in ("last_network_waited_s", "last_http_403_waited_s"):
            if key in obj:
                out[key] = obj.get(key)
        try:
            return json.dumps(out if out else obj, ensure_ascii=False, indent=2)
        except Exception:
            return raw

    def _oracle_observation_text(
        self,
        *,
        observation: Observation,
        max_items: int,
        max_chars: int,
    ) -> str:
        objs = observation.objects if isinstance(observation.objects, list) else []
        visible: list[dict[str, Any]] = []
        for it in objs:
            if not isinstance(it, dict):
                continue
            on_screen = it.get("is_on_screen")
            if isinstance(on_screen, bool) and not on_screen:
                continue
            visible.append(it)
        if not visible:
            return ""

        def _dist(x: dict[str, Any]) -> float:
            v = x.get("distance", 9999.0)
            try:
                return float(v)
            except Exception:
                return 9999.0

        def _name(x: dict[str, Any]) -> str:
            for k in ("name_en", "name", "name_cn", "label"):
                v = x.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            return "unknown"

        def _num(v: Any, *, digits: int = 3) -> str:
            try:
                return f"{float(v):.{digits}f}"
            except Exception:
                return "null"

        def _uv(x: Any, y: Any, w: Any, h: Any) -> str:
            try:
                xf = float(x)
                yf = float(y)
                wf = float(w)
                hf = float(h)
                if wf <= 0.0 or hf <= 0.0:
                    return "(null, null)"
                return f"({xf / wf:.3f}, {yf / hf:.3f})"
            except Exception:
                return "(null, null)"

        vision_max_side = int(getattr(self.cfg, "vision_image_max_side", 768) or 768)

        def _scaled_dims(w: Any, h: Any, *, max_side: int = vision_max_side) -> tuple[Optional[int], Optional[int]]:
            try:
                wf = float(w)
                hf = float(h)
                if wf <= 0.0 or hf <= 0.0:
                    return None, None
                scale = min(1.0, float(max_side) / max(wf, hf))
                return max(1, int(round(wf * scale))), max(1, int(round(hf * scale)))
            except Exception:
                return None, None

        def _scaled_xy(x: Any, y: Any, w: Any, h: Any, *, max_side: int = vision_max_side) -> str:
            try:
                xf = float(x)
                yf = float(y)
                wf = float(w)
                hf = float(h)
                if wf <= 0.0 or hf <= 0.0:
                    return "(null, null)"
                scale = min(1.0, float(max_side) / max(wf, hf))
                return f"({xf * scale:.1f}, {yf * scale:.1f})"
            except Exception:
                return "(null, null)"

        state = observation.state if isinstance(observation.state, dict) else {}
        screen_w = state.get("_screen_width")
        screen_h = state.get("_screen_height")
        vlm_w, vlm_h = _scaled_dims(screen_w, screen_h)

        visible.sort(key=_dist)
        total = len(visible)

        lines: list[str] = []
        lines.append(f"Oracle visible objects (count={total}, sorted by distance):")
        if vlm_w is not None and vlm_h is not None:
            lines.append(
                "Oracle screen coordinates use the original realtime_products screen size; "
                f"the VLM image is compressed to approx ({vlm_w}, {vlm_h})."
            )
        for it in visible:
            pos = it.get("position") if isinstance(it.get("position"), dict) else {}
            line = (
                f"- name={_name(it)!r} instance_id={it.get('instance_id')} kind={it.get('kind')!r} "
                f"distance={_num(it.get('distance'), digits=2)} on_screen={it.get('is_on_screen')} "
                f"is_held={it.get('is_held')} screen=({_num(it.get('screen_x'), digits=1)}, {_num(it.get('screen_y'), digits=1)}) "
                f"vlm_screen={_scaled_xy(it.get('screen_x'), it.get('screen_y'), screen_w, screen_h)} "
                f"screen_uv={_uv(it.get('screen_x'), it.get('screen_y'), screen_w, screen_h)} "
                f"pos=({_num(pos.get('x'))}, {_num(pos.get('y'))}, {_num(pos.get('z'))})"
            )
            lines.append(line)

        text = "\n".join(lines).strip()
        return text

    @staticmethod
    def _render_tools_manifest_for_prompt(tools_json: str) -> str:
        # Keep full manifest text as-is (no filtering/truncation/transformation).
        return str(tools_json or "").strip()

    @staticmethod
    def _remove_keys_recursive(obj: Any, *, banned: set[str]) -> Any:
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for k, v in obj.items():
                ks = str(k)
                if ks in banned:
                    continue
                if ks.endswith("_path") or ks == "path":
                    continue
                out[ks] = PromptBuilder._remove_keys_recursive(v, banned=banned)
            return out
        if isinstance(obj, list):
            return [PromptBuilder._remove_keys_recursive(v, banned=banned) for v in obj]
        return obj

    @staticmethod
    def _sanitize_supported_items_snapshot(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""

        # Best effort: preserve title prefix then sanitize JSON payload.
        idx = raw.find("{")
        if idx >= 0:
            head = raw[:idx].strip()
            payload = raw[idx:].strip()
            try:
                obj = json.loads(payload)
                cleaned = PromptBuilder._remove_keys_recursive(obj, banned={"object_id"})
                body = json.dumps(cleaned, ensure_ascii=False, indent=2)
                return f"{head}\n{body}".strip() if head else body
            except Exception:
                pass

        # Fallback for non-JSON text.
        out_lines: list[str] = []
        for line in raw.splitlines():
            low = line.lower()
            if "object_id" in low:
                continue
            if "_path" in low or re.search(r"\bpath\s*[:=]", low):
                continue
            out_lines.append(line)
        return "\n".join(out_lines).strip()

    @staticmethod
    def _render_tool_en_catalog(text: str) -> str:
        """
        Render a compact tool catalog from epm/data/tool_en.json with only key fields.
        """
        raw = str(text or "").strip()
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except Exception:
            return ""
        if not isinstance(obj, list):
            return ""
        keep = (
            "tool_name",
            "tool_type",
            "container_type",
            "tool_function",
            "capacity",
            "supported_operation_states",
        )
        out: list[dict[str, Any]] = []
        for it in obj:
            if not isinstance(it, dict):
                continue
            row: dict[str, Any] = {}
            for k in keep:
                if k in it:
                    row[k] = it.get(k)
            if row:
                out.append(row)
        if not out:
            return ""
        try:
            return json.dumps(out, ensure_ascii=False, indent=2)
        except Exception:
            return ""

    @staticmethod
    def _strip_path_like_lines(text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        out_lines: list[str] = []
        for line in raw.splitlines():
            low = line.lower()
            if "_path" in low or re.search(r"\bpath\s*[:=]", low):
                continue
            out_lines.append(line)
        return "\n".join(out_lines).strip()

    @staticmethod
    def _parse_key_value_lines(text: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for raw in str(text or "").splitlines():
            line = raw.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            k = str(key).strip()
            if not k:
                continue
            out[k] = str(value).strip()
        return out

    @staticmethod
    def _parse_json_list(text: str) -> list[Any]:
        raw = str(text or "").strip()
        if not raw:
            return []
        try:
            obj = json.loads(raw)
        except Exception:
            return []
        return obj if isinstance(obj, list) else []

    @classmethod
    def _render_task_progress_prompt_text(cls, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except Exception:
            return cls._strip_path_like_lines(raw)
        if not isinstance(obj, dict):
            return cls._strip_path_like_lines(raw)

        goal_state = obj.get("goal_state") if isinstance(obj.get("goal_state"), dict) else {}
        blocking_conditions = obj.get("blocking_conditions") if isinstance(obj.get("blocking_conditions"), list) else []

        lines: list[str] = []
        current_subtask = ""
        for candidate in (
            goal_state.get("current_subgoal"),
            goal_state.get("current_activity"),
        ):
            txt = str(candidate or "").strip()
            if txt:
                current_subtask = txt
                break
        if current_subtask:
            lines.append(f"hard_current_subtask={current_subtask}")

        current_activity = str(goal_state.get("current_activity") or "").strip()
        if current_activity:
            lines.append(f"current_activity={current_activity}")

        problems: list[str] = []
        for item in blocking_conditions:
            if isinstance(item, dict):
                detail = str(item.get("detail") or "").strip()
                if detail:
                    problems.append(detail)
            else:
                txt = str(item).strip()
                if txt:
                    problems.append(txt)
        if problems:
            lines.append(f"hard_blockers={json.dumps(problems[:4], ensure_ascii=False)}")

        if current_subtask or problems:
            lines.append(
                "planner_rule=Treat hard_current_subtask and hard_blockers as higher-priority constraints than convenience staging."
            )

        return "\n".join(lines).strip()

    @classmethod
    def _render_precondition_feedback_prompt_text(cls, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        fields = cls._parse_key_value_lines(raw)
        if not fields:
            return cls._strip_path_like_lines(raw)
        lines: list[str] = []
        if fields.get("reason_code"):
            lines.append(f"reason_code={fields['reason_code']}")
        if fields.get("reason"):
            lines.append(f"reason={fields['reason']}")
        if fields.get("feedback_to_planner"):
            lines.append(f"feedback_to_planner={fields['feedback_to_planner']}")
        return "\n".join(lines).strip()

    @staticmethod
    def _strip_execution_outcomes_from_stm(text: str) -> str:
        """Retain action history while hiding outcome feedback for ``no_feedback``."""
        blocked_fields = {
            "result_summary",
            "errors",
            "error",
            "failure_reason",
            "last_error",
            "feedback",
        }
        kept: list[str] = []
        for line in str(text or "").splitlines():
            key_match = re.match(r"^\s*([a-z_]+)\s*:", line, flags=re.IGNORECASE)
            if key_match and key_match.group(1).lower() in blocked_fields:
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    @classmethod
    def _render_visual_anomaly_feedback_prompt_text(cls, text: str) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        fields = cls._parse_key_value_lines(raw)
        if not fields:
            return cls._strip_path_like_lines(raw)
        lines: list[str] = []
        for key in ("severity", "tags", "observed_issue", "feedback"):
            value = fields.get(key)
            if value:
                lines.append(f"{key}={value}")
        return "\n".join(lines).strip()

    @staticmethod
    def _render_percept_prompt_text(text: str, *, max_chars: int = 4000) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        compact = "\n".join(line.rstrip() for line in raw.splitlines() if line.strip()).strip()
        if max_chars > 0 and len(compact) > int(max_chars):
            return compact[: int(max_chars) - 3].rstrip() + "..."
        return compact

    @staticmethod
    def _render_task_execution_context_prompt_text(text: str, *, max_chars: int = 2000) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        compact = "\n".join(line.rstrip() for line in raw.splitlines() if line.strip()).strip()
        if max_chars > 0 and len(compact) > int(max_chars):
            return compact[: int(max_chars) - 3].rstrip() + "..."
        return compact

    @staticmethod
    def _feedback_already_contains_precondition_summary(feedback: str) -> bool:
        raw = str(feedback or "").strip().lower()
        return "precondition_check_failed=true" in raw or "deterministic_precondition_failure=true" in raw

    @staticmethod
    def _render_countdown_status(agent_state_text: str, *, current_step: str = "", current_step_status_text: str = "") -> str:
        def _parse_dict(raw_text: str) -> dict[str, Any]:
            text = str(raw_text or "").strip()
            if not text:
                return {}
            try:
                data = json.loads(text)
            except Exception:
                return {}
            return data if isinstance(data, dict) else {}

        def _maybe_int(value: Any) -> Optional[int]:
            try:
                if value is None or value == "":
                    return None
                return int(value)
            except Exception:
                return None

        st = _parse_dict(agent_state_text)
        current_status = _parse_dict(current_step_status_text)
        step_text = str(current_step or "").strip()

        episode_step = _maybe_int(current_status.get("episode_step"))
        logical_step = _maybe_int(current_status.get("logical_step"))
        threshold = _maybe_int(st.get("force_submit_step_threshold"))
        within = _maybe_int(st.get("force_submit_within_steps"))
        active = bool(st.get("force_submit_active", False))
        active_remaining = _maybe_int(st.get("force_submit_remaining_steps"))
        deadline_step = _maybe_int(st.get("force_submit_deadline_step"))

        lines: list[str] = []
        if episode_step is not None:
            lines.append(f"- episode_step={episode_step}")
        elif step_text:
            lines.append(f"- current_step={step_text}")

        if logical_step is not None and logical_step != episode_step:
            lines.append(f"- logical_committed_step={logical_step}")

        if threshold is not None and threshold > 0:
            lines.append(f"- force_submit_step_threshold={threshold}")
        if within is not None and within > 0:
            lines.append(f"- force_submit_within_steps={within}")

        if episode_step is not None and threshold is not None and threshold > 0 and not active:
            remaining_trigger = max(0, threshold - episode_step)
            lines.append(f"- remaining_until_force_submit_trigger={remaining_trigger}")

        if active:
            lines.append("- force_submit_mode=active")
            if active_remaining is not None:
                lines.append(f"- remaining_in_force_submit_window={max(0, active_remaining)}")
            if episode_step is not None and deadline_step is not None:
                lines.append(f"- remaining_until_force_submit_deadline={max(0, deadline_step - episode_step)}")
        elif threshold is not None and threshold > 0:
            lines.append("- force_submit_mode=inactive")

        phase = ""
        if active:
            phase = "final"
        elif episode_step is not None and threshold is not None and threshold > 0:
            remaining = max(0, threshold - episode_step)
            if remaining <= 15:
                phase = "final"
            elif remaining <= 40:
                phase = "late"
            elif remaining <= 80:
                phase = "mid"
            else:
                phase = "early"
        if phase:
            lines.append(f"- deadline_phase={phase}")

        timers = st.get("timers")
        if not isinstance(timers, list):
            timers = []
        timer_events = st.get("timer_events")
        if not isinstance(timer_events, list):
            timer_events = []
        running = [t for t in timers if isinstance(t, dict) and str(t.get("status", "running")) == "running"]
        if running:
            next_timer = None
            next_deadline = None
            for t in running:
                try:
                    d = float(t.get("deadline_ts"))
                except Exception:
                    continue
                if next_deadline is None or d < next_deadline:
                    next_deadline = d
                    next_timer = t
            if isinstance(next_timer, dict):
                lines.append(
                    "- timer_running: "
                    f"id={next_timer.get('id')!r} "
                    f"deadline_ts={next_timer.get('deadline_ts')!r} "
                    f"duration={next_timer.get('duration')!r}"
                )
        else:
            lines.append("- timer_running: none")

        if timer_events:
            last_ev = None
            for ev in reversed(timer_events):
                if isinstance(ev, dict):
                    last_ev = ev
                    break
            if isinstance(last_ev, dict):
                lines.append(
                    "- latest_timer_event: "
                    f"id={last_ev.get('id')!r} status={last_ev.get('status')!r} at_ts={last_ev.get('at_ts')!r}"
                )
        else:
            lines.append("- latest_timer_event: none")

        if not lines:
            return "- timers: unavailable"
        return "\n".join(lines).strip()

    def _append_system_injected_sections(
        self,
        *,
        system_parts: list[str],
        memory_bundle: Dict[str, str],
        inject_tool_schemas: bool,
        high_level_goal: str = "",
        recipe_text: str = "",
        feedback: str = "",
        instance_id_selection_mode: bool = False,
    ) -> None:
        """
        Append memory/tool sections into SYSTEM according to prompt_layout.json.
        """
        def _truthy(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            s = str(v or "").strip().lower()
            return s in {"1", "true", "yes", "y", "on"}

        force_submit_deadline = str(memory_bundle.get("force_submit_deadline", "") or "").strip()
        force_submit_deadline_emitted = False
        order = self._layout_list(
            "system_sections",
            [
                "body_rules",
                "strategy_notes",
                "supported_items",
                "tool_interaction_point_list",
                "tools_manifest_openai",
                "reflexion_memory",
                "put_place_occupancy",
            ],
        )
        if instance_id_selection_mode:
            order = self._layout_list(
                "instance_id_selection.system_sections",
                ["system_role", "body_rules", "strategy_notes", "tools_manifest_openai", "skill_specs"],
            )
        for sec in order:
            s = sec.strip().lower()
            if not self._section_enabled(s, memory_bundle=memory_bundle):
                continue
            if s == "system_role":
                role = memory_bundle.get("system_role", "").strip()
                if role and not self._section_enabled("feedback", memory_bundle=memory_bundle):
                    role = role.replace("execution feedback (feedback/agent_state/stm_window)", "current agent state")
                if role and not self._section_enabled("stm_window", memory_bundle=memory_bundle):
                    role = role.replace("/stm_window", "")
                if role:
                    system_parts.append("Role:")
                    system_parts.append(role)
                    system_parts.append("")
                continue
            if s == "body_rules":
                body_rules = self._filter_body_rules(memory_bundle.get("body_rules", "").strip())
                body_rules = self._filter_pipeline_specific_body_rules(
                    body_rules,
                    pipeline_name=str(memory_bundle.get("_pipeline_name", "") or ""),
                )
                if body_rules:
                    system_parts.append("Body rules (global constraints):")
                    system_parts.append(body_rules)
                    system_parts.append("")
                if force_submit_deadline:
                    system_parts.append("Step-budget and scoring rule:")
                    system_parts.append(force_submit_deadline)
                    system_parts.append("")
                    force_submit_deadline_emitted = True
                continue
            if s == "strategy_notes":
                strategy_notes = self._filter_strategy_notes(memory_bundle.get("strategy_notes", "").strip())
                if strategy_notes:
                    system_parts.append("Strategy notes (planning guidance and heuristics):")
                    system_parts.append(strategy_notes)
                    system_parts.append("")
                continue
            if s == "action_catalog":
                # Disabled: requested to remove Action catalog from prompt.
                continue
            if s == "force_submit_deadline":
                # Disabled by policy: do not inject force_submit_deadline prompt module.
                continue
            if s == "put_place_list":
                continue
            if s == "tool_interaction_point_list":
                tip = memory_bundle.get("tool_interaction_point_list", "").strip()
                if tip:
                    system_parts.append("Tool interaction point list:")
                    system_parts.append(tip)
                    system_parts.append("")
                continue
            if s == "tool_en_catalog":
                # Disabled: moved to explicit skill `query_tool_manual` to reduce prompt size/latency.
                continue
            if s == "put_place_occupancy":
                occupancy = memory_bundle.get("put_place_occupancy", "").strip()
                system_parts.append("Placement occupancy (oracle snapshot):")
                system_parts.append(occupancy if occupancy else "(no snapshot yet)")
                system_parts.append("")
                continue
            if s == "supported_items":
                supported = self._sanitize_supported_items_snapshot(memory_bundle.get("supported_items", ""))
                if supported:
                    system_parts.append("Supported items (static snapshot):")
                    system_parts.append(supported)
                    system_parts.append("")
                continue
            if s == "reflexion_memory":
                reflexion_memory = memory_bundle.get("reflexion_memory", "").strip()
                if reflexion_memory:
                    system_parts.append("Reflexion memory (lessons learned):")
                    system_parts.append(reflexion_memory)
                    system_parts.append("")
                continue
            if s == "reflexion_progress_memory":
                reflexion_progress = memory_bundle.get("reflexion_progress_memory", "").strip()
                if reflexion_progress:
                    system_parts.append("Reflexion progress memory (recent stable progress facts):")
                    system_parts.append(reflexion_progress)
                    system_parts.append("")
                continue
            if s == "tools_manifest_openai":
                if inject_tool_schemas and self.cfg.inject_tool_schemas:
                    tools_json = memory_bundle.get("tools_manifest_openai", "").strip()
                    tools_text = self._render_tools_manifest_for_prompt(tools_json)
                    if tools_text.strip():
                        label = (
                            "Allowed actions/skills (authoritative schemas):"
                            if self._skill_interface_enabled()
                            else "Allowed actions (authoritative schemas):"
                        )
                        system_parts.append(label)
                        system_parts.append(tools_text.strip())
                        system_parts.append("")
                continue
            if s == "skill_cards":
                if not self._skill_interface_enabled():
                    continue
                # Optional: inject a compact category->actions view (like "tools") to reduce schema search space.
                skills_catalog_json = memory_bundle.get("skills_catalog", "").strip()
                action_specs_text = memory_bundle.get("action_specs", "").strip()
                heuristics_json = memory_bundle.get("parameter_heuristics", "").strip()
                if skills_catalog_json and action_specs_text:
                    mode = self._layout_str("skill_cards_mode", "selected").strip().lower()
                    if mode == "all":
                        selected = list_skill_card_names(skills_catalog_json=skills_catalog_json)
                    else:
                        max_cards = self._layout_int("limits.skill_cards_max_cards", 0)
                        selected = select_skill_cards(
                            skills_catalog_json=skills_catalog_json,
                            high_level_goal=high_level_goal,
                            recipe_text=recipe_text,
                            feedback=feedback,
                            max_cards=int(max_cards),
                        )

                    if selected:
                        text = render_skill_cards_for_prompt(
                            skills_catalog_json=skills_catalog_json,
                            action_specs_text=action_specs_text,
                            parameter_heuristics_json=heuristics_json,
                            selected_skill_names=selected,
                        ).strip()
                        system_parts.append(text)
                        system_parts.append("")
                continue
            if s == "parameter_heuristics":
                heuristics_json = memory_bundle.get("parameter_heuristics", "").strip()
                if heuristics_json:
                    text = heuristics_json
                    system_parts.append("Parameter heuristics (action templates):")
                    system_parts.append(text.strip())
                    system_parts.append("")
                continue
            if s == "skill_specs":
                if not self._skill_interface_enabled():
                    continue
                builtin_skill_specs = memory_bundle.get("skill_specs", "").strip()
                if builtin_skill_specs:
                    system_parts.append(builtin_skill_specs)
                    system_parts.append("")
                continue

        if force_submit_deadline and not force_submit_deadline_emitted:
            system_parts.append("Step-budget and scoring rule:")
            system_parts.append(force_submit_deadline)
            system_parts.append("")

    def _append_user_sections(
        self,
        *,
        user_parts: list[str],
        observation: Observation,
        high_level_id: str,
        high_level_goal: str,
        memory_bundle: Dict[str, str],
        feedback: str,
        percept_text: str,
        inject_memory: bool,
        include_task_progress: bool,
        instance_id_selection_mode: bool = False,
    ) -> None:
        def _truthy(v: Any) -> bool:
            if isinstance(v, bool):
                return v
            s = str(v or "").strip().lower()
            return s in {"1", "true", "yes", "y", "on"}

        show_high_level_id = _truthy(memory_bundle.get("_show_high_level_id", True))
        if instance_id_selection_mode:
            sections = self._layout_list(
                "instance_id_selection.user_sections",
                ["goal", "feedback", "observation"],
            )
            for sec in sections:
                s = str(sec).strip().lower()
                if not self._section_enabled(s, memory_bundle=memory_bundle):
                    continue
                if s == "goal":
                    if show_high_level_id:
                        user_parts.append(f"Current high-level goal id: {high_level_id}")
                    user_parts.append(f"Goal: {high_level_goal}")
                    user_parts.append("")
                    continue
                if s == "feedback":
                    user_parts.append("Feedback from last execution:")
                    user_parts.append(self._strip_path_like_lines(feedback) or "(none)")
                    iq = str(memory_bundle.get("instance_query_snapshot", "") or "").strip()
                    if iq:
                        user_parts.append("")
                        user_parts.append("Auto query for instance disambiguation:")
                        user_parts.append(iq)
                    user_parts.append("")
                    continue
                if s == "countdown_status":
                    countdown_text = self._render_countdown_status(
                        memory_bundle.get("agent_state", ""),
                        current_step=str(getattr(observation, "frame_id", "") or ""),
                        current_step_status_text=memory_bundle.get("current_step_status", ""),
                    )
                    if countdown_text:
                        user_parts.append("Countdown status:")
                        user_parts.append(countdown_text)
                        user_parts.append("")
                    continue
                if s == "observation":
                    user_parts.append("Observation:")
                    if observation.screenshot_path:
                        user_parts.append("- A screenshot image is provided as the input.")
                    else:
                        user_parts.append("- No screenshot image is available.")
                    user_parts.append("")
                    continue
            return

        order = self._layout_list(
            "user_sections",
            [
                "goal",
                "recipe_text",
                "task_execution_context",
                "action_memory",
                "put_place_occupancy",
                "reflexion_progress_memory",
                "observation",
                "agent_state",
                "resume_state",
                "countdown_status",
                "task_progress",
                "precondition_feedback",
                "visual_anomaly_feedback",
                "feedback",
                "query_memory",
                "stm_window",
                "percept_text",
                "task_progress_feedback",
            ],
        )
        obs_mode = self._layout_str("observation_mode", "image_only").lower()
        include_oracle_observation = _truthy(memory_bundle.get("_include_oracle_observation", True))
        pipeline_name = str(memory_bundle.get("_pipeline_name", "") or "").strip().lower()

        for sec in order:
            s = sec.strip().lower()
            section_enabled = self._section_enabled(s, memory_bundle=memory_bundle)
            if not section_enabled and s != "query_memory":
                continue
            if s == "goal":
                if show_high_level_id:
                    user_parts.append(f"Current high-level goal id: {high_level_id}")
                user_parts.append(f"Goal: {high_level_goal}")
                user_parts.append("")
                continue
            if s == "recipe_text":
                user_parts.append("Full recipe:")
                user_parts.append(memory_bundle.get("recipe_text", "").strip())
                user_parts.append("")
                continue
            if s == "task_execution_context":
                task_execution_context = memory_bundle.get("task_execution_context", "").strip()
                rendered = self._render_task_execution_context_prompt_text(task_execution_context)
                if rendered:
                    user_parts.append("Task execution context:")
                    user_parts.append(rendered)
                    user_parts.append("")
                continue
            if s == "put_place_occupancy":
                occupancy = memory_bundle.get("put_place_occupancy", "").strip()
                user_parts.append("Placement occupancy (oracle snapshot):")
                user_parts.append(occupancy if occupancy else "(no snapshot yet)")
                user_parts.append("")
                continue
            if s == "reflexion_progress_memory":
                reflexion_progress = memory_bundle.get("reflexion_progress_memory", "").strip()
                if reflexion_progress:
                    user_parts.append("Reflexion progress memory (recent stable progress facts):")
                    user_parts.append(reflexion_progress)
                    user_parts.append("")
                continue
            if s == "observation":
                user_parts.append("Observation:")
                if obs_mode == "image_only":
                    if observation.screenshot_path:
                        user_parts.append("- A screenshot image is provided as the input.")
                    else:
                        user_parts.append("- No screenshot image is available.")
                else:
                    user_parts.append(f"time: {observation.time}")
                    user_parts.append(f"frame_id: {observation.frame_id}")
                    user_parts.append(f"state_keys: {list(observation.state.keys())}")
                    user_parts.append(f"num_objects: {len(observation.objects)}")
                use_percept_summary = bool(str(percept_text or "").strip())
                if (
                    include_oracle_observation
                    and self._section_enabled("oracle_observation", memory_bundle=memory_bundle)
                    and not (pipeline_name in ("epm", "epm_agent") and use_percept_summary)
                ):
                    oracle_text = self._oracle_observation_text(
                        observation=observation,
                        max_items=self._layout_int("limits.oracle_observation_max_items", 40),
                        max_chars=self._layout_int("limits.oracle_observation_max_chars", 12000),
                    )
                    if oracle_text:
                        user_parts.append(oracle_text)
                user_parts.append("")
                continue
            if s == "agent_state":
                agent_state = memory_bundle.get("agent_state", "").strip()
                if agent_state:
                    user_parts.append("Agent state snapshot:")
                    user_parts.append(self._sanitize_agent_state_snapshot(agent_state))
                    user_parts.append("")
                continue
            if s == "resume_state":
                resume_state = memory_bundle.get("resume_state", "").strip()
                if resume_state:
                    user_parts.append("Resume state snapshot:")
                    user_parts.append(self._sanitize_resume_state_snapshot(resume_state))
                    user_parts.append("")
                continue
            if s == "countdown_status":
                countdown_text = self._render_countdown_status(
                    memory_bundle.get("agent_state", ""),
                    current_step=str(getattr(observation, "frame_id", "") or ""),
                    current_step_status_text=memory_bundle.get("current_step_status", ""),
                )
                if countdown_text:
                    user_parts.append("Countdown status:")
                    user_parts.append(countdown_text)
                    user_parts.append("")
                continue
            if s == "feedback":
                user_parts.append("Feedback from last execution:")
                user_parts.append(self._render_precondition_feedback_prompt_text(feedback) or self._strip_path_like_lines(feedback) or "(none)")
                user_parts.append("")
                continue
            if s == "precondition_feedback":
                if pipeline_name not in ("epm", "epm_agent"):
                    continue
                if self._feedback_already_contains_precondition_summary(feedback):
                    continue
                precondition_feedback = memory_bundle.get("latest_precondition_feedback", "").strip()
                if precondition_feedback:
                    user_parts.append("Must-fix precondition blocker:")
                    user_parts.append(self._render_precondition_feedback_prompt_text(precondition_feedback))
                    user_parts.append("")
                continue
            if s == "visual_anomaly_feedback":
                visual_anomaly_feedback = memory_bundle.get("latest_visual_anomaly_feedback", "").strip()
                if visual_anomaly_feedback:
                    user_parts.append("Must-respect visual warning:")
                    user_parts.append(self._render_visual_anomaly_feedback_prompt_text(visual_anomaly_feedback))
                    user_parts.append("")
                continue
            if s == "query_memory":
                iq = str(memory_bundle.get("instance_query_snapshot", "") or "").strip()
                if iq:
                    user_parts.append("Auto query for disambiguation:")
                    user_parts.append(iq)
                    user_parts.append("")
                query_memory = memory_bundle.get("query_memory", "").strip()
                if section_enabled and query_memory:
                    user_parts.append("Cross-step query memory:")
                    user_parts.append(query_memory)
                    user_parts.append("")
                continue
            if s == "stm_window":
                if inject_memory and self.cfg.inject_memory:
                    stm = memory_bundle.get("stm_window", "").strip()
                    if stm:
                        if not self._section_enabled("feedback", memory_bundle=memory_bundle):
                            stm = self._strip_execution_outcomes_from_stm(stm)
                        user_parts.append("STM window snapshot:")
                        user_parts.append(self._strip_path_like_lines(stm))
                        user_parts.append("")
                continue
            if s == "percept_text":
                rendered = self._render_percept_prompt_text(percept_text)
                if rendered:
                    user_parts.append("Planner-facing visual summary:")
                    user_parts.append(rendered)
                    user_parts.append("")
                continue
            if s == "action_memory":
                continue
            if s == "task_progress":
                if inject_memory and self.cfg.inject_memory and include_task_progress:
                    task_progress = memory_bundle.get("task_progress", "").strip()
                    if task_progress:
                        user_parts.append("Task progress hard constraints:")
                        user_parts.append(self._render_task_progress_prompt_text(task_progress))
                        user_parts.append("")
                continue
            if s == "task_progress_feedback":
                continue

    @staticmethod
    def _step_tag(obs: Observation) -> str:
        fid = str(getattr(obs, "frame_id", "") or "").strip()
        if fid.isdigit():
            return f"step_{int(fid):06d}"
        return fid or "step_unknown"

    def build_planner_prompt(
        self,
        *,
        observation: Observation,
        high_level_id: str,
        high_level_goal: str,
        memory_bundle: Dict[str, str],
        feedback: str,
        percept_text: str = "",
        inject_memory: bool = True,
        inject_tool_schemas: bool = True,
        include_task_progress: bool = True,
    ) -> str:
        """
        EPM-style prompt for a planner LLM:
        - SYSTEM: interface + constraints + response schema + loop rules
        - USER: goal + state + feedback + memory modules
        """
        pmin, pmax, step_count_phrase, next_step_phrase = self._plan_step_window_text()
        response_format_template_values = self._build_response_format_template_values(
            pmin=pmin,
            pmax=pmax,
            step_count_phrase=step_count_phrase,
            next_step_phrase=next_step_phrase,
        )

        instance_id_selection_mode = (
            self._section_enabled("feedback", memory_bundle=memory_bundle)
            and self._is_instance_disambiguation_feedback(feedback)
        )
        has_instance_query_snapshot = (
            bool(str(memory_bundle.get("instance_query_snapshot", "") or "").strip())
        )

        system_parts: list[str] = []
        system_parts.append("SYSTEM:")
        pipeline_name = str(memory_bundle.get("_pipeline_name", "") or "").strip().lower()
        output_mode_phrase = (
            "a plan using ONLY the allowed actions/skills, expressed as tool calls (one call per step)"
            if bool(self.cfg.tool_calling)
            else "a JSON plan for the current high-level goal, using ONLY the allowed actions/skills"
        )
        self._append_system_injected_sections(
            system_parts=system_parts,
            memory_bundle=memory_bundle,
            inject_tool_schemas=inject_tool_schemas,
            high_level_goal=high_level_goal,
            recipe_text=memory_bundle.get("recipe_text", ""),
            feedback=feedback,
            instance_id_selection_mode=instance_id_selection_mode,
        )
        system_parts.append("")
        if instance_id_selection_mode:
            system_parts.append("You are in instance_id selection mode.")
            system_parts.append("You must select one candidate and return ONLY `name` + `instance_id` in JSON.")
        elif bool(self.cfg.tool_calling):
            if self._normalize_pipeline_name(pipeline_name) == "planner_executor":
                role_text = self._load_method_unique_asset(
                    pipeline_name,
                    "role.txt",
                    "You must output {output_mode_phrase}.",
                )
                system_parts.append(self._safe_template_format(role_text, {"output_mode_phrase": output_mode_phrase}))
            else:
                system_parts.append("You must output a plan using ONLY the allowed actions/skills, expressed as tool calls (one call per step).")
            # Apply explicit tool-call count constraints only for PE/planner_executor.
            if pipeline_name in ("planner_executor", "planner-executor", "pe"):
                pe_text = self._load_method_unique_asset(
                    pipeline_name,
                    "hard_constraints.txt",
                    (
                        "PE reminder: this is rolling-horizon planning; each round is partial. "
                        "Do NOT force task completion in one round.\n"
                        "Do NOT include `gui_submit_dish_via_checkout_stand` unless there is clear evidence the dish is ready to serve "
                        "(or force-submit mode is explicitly active).\n"
                        "Each step object MUST include `step_id`, `type`, `name`, `args`, and `expectation`.\n"
                        "- `type` must be exactly `action` or `skill`.\n"
                        "- `name` must be the exact action/skill name from the schemas above.\n"
                        "- `args` must be a JSON object using the actual parameter names for that action/skill; use `{}` when no args are needed.\n"
                        "{count_constraint}"
                    ),
                )
                count_constraint = (
                    f"Tool-call count constraint: output exactly {pmin} tool calls in this planning round."
                    if pmax >= 0 and pmin == pmax
                    else (
                        f"Tool-call count constraint: output between {pmin} and {pmax} tool calls in this planning round."
                        if pmax >= 0
                        else f"Tool-call count constraint: output at least {pmin} tool calls in this planning round."
                    )
                )
                formatted = self._safe_template_format(
                    pe_text,
                    {
                        "count_constraint": count_constraint,
                        "plan_min_steps": pmin,
                        "plan_max_steps": pmax,
                    },
                )
                system_parts.extend([line for line in formatted.splitlines() if line.strip()])
            if has_instance_query_snapshot:
                system_parts.append(
                    "Disambiguation reminder: if auto-query candidates are already provided, choose a concrete `instance_id` "
                    "from them and continue toward physical execution now. Do not schedule another `query_scene_objects` "
                    "unless the candidate list is empty or clearly stale."
                )
        else:
            if self._normalize_pipeline_name(pipeline_name) == "planner_executor":
                role_text = self._load_method_unique_asset(
                    pipeline_name,
                    "role.txt",
                    "You must output {output_mode_phrase}.",
                )
                system_parts.append(self._safe_template_format(role_text, {"output_mode_phrase": output_mode_phrase}))
            else:
                system_parts.append("You must output a JSON plan for the current high-level goal, using ONLY the allowed actions/skills.")
            if pipeline_name in ("planner_executor", "planner-executor", "pe"):
                pe_text = self._load_method_unique_asset(
                    pipeline_name,
                    "hard_constraints.txt",
                    (
                        "PE reminder: this is rolling-horizon planning; each round is partial. "
                        "Do NOT force task completion in one round.\n"
                        "Do NOT include `gui_submit_dish_via_checkout_stand` unless there is clear evidence the dish is ready to serve "
                        "(or force-submit mode is explicitly active).\n"
                        "Each step object MUST include `step_id`, `type`, `name`, `args`, and `expectation`.\n"
                        "- `type` must be exactly `action` or `skill`.\n"
                        "- `name` must be the exact action/skill name from the schemas above.\n"
                        "- `args` must be a JSON object using the actual parameter names for that action/skill; use `{}` when no args are needed.\n"
                        "{count_constraint}"
                    ),
                )
                count_constraint = (
                    f"Plan length constraint: output exactly {pmin} steps in this planning round."
                    if pmax >= 0 and pmin == pmax
                    else (
                        f"Plan length constraint: output between {pmin} and {pmax} steps in this planning round."
                        if pmax >= 0
                        else f"Plan length constraint: output at least {pmin} steps in this planning round."
                    )
                )
                formatted = self._safe_template_format(
                    pe_text,
                    {
                        "count_constraint": count_constraint,
                        "plan_min_steps": pmin,
                        "plan_max_steps": pmax,
                    },
                )
                system_parts.extend([line for line in formatted.splitlines() if line.strip()])
            if has_instance_query_snapshot:
                system_parts.append(
                    "Disambiguation reminder: if auto-query candidates are already provided, choose a concrete `instance_id` "
                    "from them and continue toward physical execution now. Do not schedule another `query_scene_objects` "
                    "unless the candidate list is empty or clearly stale."
                )
        system_parts.append("")
        if instance_id_selection_mode:
            system_parts.append("Instance-ID Selection Mode:")
            self._append_instance_id_selection_mode(system_parts=system_parts)
            system_parts.append("")
        if instance_id_selection_mode:
            system_parts.append("Response Format (JSON only):")
            system_parts.append('{"name":"candidate name","instance_id":123}')
        else:
            system_parts.append("Response Format (JSON only):")
            if self._normalize_pipeline_name(pipeline_name) == "planner_executor":
                response_format_text = self._load_method_unique_asset(
                    pipeline_name,
                    "response_format.txt",
                    "{\n"
                    '  "high_level_id": "Hk",\n'
                    '  "goal": "one sentence goal",\n'
                    '  "explanation": null | "why last attempt failed",\n'
                    '  "thoughts": "brief planning rationale",\n'
                    '  "action_list": [\n'
                    "{response_format_action_example_1},\n"
                    "{response_format_action_example_2},\n"
                    "{response_format_action_example_3}{response_format_action_examples_extra}\n"
                    "  ]\n"
                    "}",
                )
                response_format_text = self._safe_template_format(response_format_text, response_format_template_values)
                system_parts.append(response_format_text)
            else:
                system_parts.append(
                    "{\n"
                    '  "high_level_id": "Hk",\n'
                    '  "goal": "one sentence goal",\n'
                    '  "explanation": null | "why last attempt failed",\n'
                    '  "thoughts": "brief planning rationale",\n'
                    '  "action_list": [\n'
                    '    {"step_id":"Hk.A1","type":"action","name":"EXACT_ACTION_NAME_FROM_SCHEMA","args":{"required_arg_name":"required_arg_value"},"expectation":"a concrete observable outcome"}\n'
                    "  ]\n"
                    "}"
                )

        user_parts: list[str] = []
        user_parts.append("USER:")
        self._append_user_sections(
            user_parts=user_parts,
            observation=observation,
            high_level_id=high_level_id,
            high_level_goal=high_level_goal,
            memory_bundle=memory_bundle,
            feedback=feedback,
            percept_text=percept_text,
            inject_memory=inject_memory,
            include_task_progress=include_task_progress,
            instance_id_selection_mode=instance_id_selection_mode,
        )

        prompt = self._strip_skill_interface_references(
            "\n".join(system_parts).strip() + "\n\n" + "\n".join(user_parts).strip() + "\n"
        ) + "\n"

        if self.cfg.save_last_prompt_path is not None:
            self.cfg.save_last_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.save_last_prompt_path.write_text(prompt, encoding="utf-8")
        if self.cfg.save_per_step_dir is not None:
            try:
                self.cfg.save_per_step_dir.mkdir(parents=True, exist_ok=True)
                step_tag = self._step_tag(observation)
                out_path = self.cfg.save_per_step_dir / f"{step_tag}.txt"
                out_path.write_text(prompt, encoding="utf-8")
            except Exception:
                pass
        self._audit_final_prompt(source="prompt_builder.planner", prompt=prompt)
        return prompt

    def build_react_prompt(
        self,
        *,
        observation: Observation,
        high_level_id: str,
        high_level_goal: str,
        memory_bundle: Dict[str, str],
        feedback: str,
        percept_text: str = "",
        inject_memory: bool = True,
        inject_tool_schemas: bool = True,
        include_task_progress: bool = True,
    ) -> str:
        """
        ReAct-style prompt:
        - same structured JSON output schema
        - action_list step count follows configured planner bounds
        - always interleave observe -> plan(a short chunk) -> execute -> observe
        """
        pmin, pmax, step_count_phrase, next_step_phrase = self._plan_step_window_text()
        response_format_template_values = self._build_response_format_template_values(
            pmin=pmin,
            pmax=pmax,
            step_count_phrase=step_count_phrase,
            next_step_phrase=next_step_phrase,
        )

        instance_id_selection_mode = (
            self._section_enabled("feedback", memory_bundle=memory_bundle)
            and self._is_instance_disambiguation_feedback(feedback)
        )

        system_parts: list[str] = []
        system_parts.append("SYSTEM:")
        pipeline_name = str(memory_bundle.get("_pipeline_name", "") or "").strip().lower()
        self._append_system_injected_sections(
            system_parts=system_parts,
            memory_bundle=memory_bundle,
            inject_tool_schemas=inject_tool_schemas,
            high_level_goal=high_level_goal,
            recipe_text=memory_bundle.get("recipe_text", ""),
            feedback=feedback,
            instance_id_selection_mode=instance_id_selection_mode,
        )
        system_parts.append("")
        if instance_id_selection_mode:
            system_parts.append("You are in instance_id selection mode.")
            system_parts.append("You must select one candidate and return ONLY `name` + `instance_id` in JSON.")
        else:
            role_text = self._load_method_unique_asset(
                pipeline_name,
                "role.txt",
                f"You must {next_step_phrase} based on the latest observation and feedback.\nYou must output STRICT JSON only.",
            )
            role_text = self._safe_template_format(role_text, response_format_template_values)
            system_parts.extend([line for line in role_text.splitlines() if line.strip()])
        system_parts.append("Hard constraints:")
        if instance_id_selection_mode:
            system_parts.append("- Output ONLY two keys: `name` and `instance_id`.")
            system_parts.append("- Do NOT output action_list/tool_calls.")
            self._append_instance_id_selection_mode(system_parts=system_parts)
        else:
            hard_constraints = self._load_method_unique_asset(
                pipeline_name,
                "hard_constraints.txt",
                (
                    f"- Output {step_count_phrase}: action_list length MUST be {step_count_phrase}.\n"
                    "- Each step object MUST include `step_id`, `type`, `name`, `args`, and `expectation`.\n"
                    "- `type` must be exactly `action` or `skill`.\n"
                    "- Do NOT invent actions. Use exact action/skill names from the schemas above.\n"
                    "- `args` must be a JSON object using the actual parameter names from the action signature exactly (no extra keys); use `{}` when no args are needed.\n"
                    "- The expectation must name the intended target object/tool and a concrete outcome.\n"
                    "- If feedback includes `blocking=instance_disambiguation`, you MUST choose exactly one `instance_id` from "
                    "`candidates` and retry the same skill with the specified `instance_arg`."
                ),
            )
            hard_constraints = self._safe_template_format(hard_constraints, response_format_template_values)
            system_parts.extend([line for line in hard_constraints.splitlines() if line.strip()])
        system_parts.append("")
        if instance_id_selection_mode:
            system_parts.append("Response Format (JSON only):")
            system_parts.append('{"name":"candidate name","instance_id":123}')
        else:
            system_parts.append("Response Format (JSON only):")
            response_format_text = self._load_method_unique_asset(
                pipeline_name,
                "response_format.txt",
                "{\n"
                '  "high_level_id": "Hk",\n'
                '  "goal": "one sentence goal",\n'
                '  "explanation": null | "why last attempt failed",\n'
                '  "thoughts": "brief rationale",\n'
                '  "action_list": [\n'
                "{response_format_action_example_1},\n"
                "{response_format_action_example_2},\n"
                "{response_format_action_example_3}{response_format_action_examples_extra}\n"
                "  ]\n"
                "}",
            )
            response_format_text = self._safe_template_format(response_format_text, response_format_template_values)
            system_parts.append(response_format_text)

        user_parts: list[str] = []
        user_parts.append("USER:")
        self._append_user_sections(
            user_parts=user_parts,
            observation=observation,
            high_level_id=high_level_id,
            high_level_goal=high_level_goal,
            memory_bundle=memory_bundle,
            feedback=feedback,
            percept_text=percept_text,
            inject_memory=inject_memory,
            include_task_progress=include_task_progress,
            instance_id_selection_mode=instance_id_selection_mode,
        )

        prompt = self._strip_skill_interface_references(
            "\n".join(system_parts).strip() + "\n\n" + "\n".join(user_parts).strip() + "\n"
        ) + "\n"

        # Use the same prompt logging paths.
        if self.cfg.save_last_prompt_path is not None:
            self.cfg.save_last_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.save_last_prompt_path.write_text(prompt, encoding="utf-8")
        if self.cfg.save_per_step_dir is not None:
            try:
                self.cfg.save_per_step_dir.mkdir(parents=True, exist_ok=True)
                step_tag = self._step_tag(observation)
                out_path = self.cfg.save_per_step_dir / f"{step_tag}.txt"
                out_path.write_text(prompt, encoding="utf-8")
            except Exception:
                pass
        self._audit_final_prompt(source="prompt_builder.react", prompt=prompt)
        return prompt

    def build_open_loop_prompt(
        self,
        *,
        observation: Observation,
        high_level_id: str,
        high_level_goal: str,
        memory_bundle: Dict[str, str],
        feedback: str,
        percept_text: str = "",
        inject_memory: bool = True,
        inject_tool_schemas: bool = True,
        include_task_progress: bool = True,
    ) -> str:
        """
        Open-loop sequential baseline:
        - Plan once upfront, then execute sequentially without replanning.
        - Do not rely on future execution feedback.
        - Uses the same JSON plan schema.
        """
        pmin, pmax, step_count_phrase, next_step_phrase = self._plan_step_window_text()
        response_format_template_values = self._build_response_format_template_values(
            pmin=pmin,
            pmax=pmax,
            step_count_phrase=step_count_phrase,
            next_step_phrase=next_step_phrase,
        )
        instance_id_selection_mode = False

        system_parts: list[str] = []
        system_parts.append("SYSTEM:")
        self._append_system_injected_sections(
            system_parts=system_parts,
            memory_bundle=memory_bundle,
            inject_tool_schemas=inject_tool_schemas,
            high_level_goal=high_level_goal,
            recipe_text=memory_bundle.get("recipe_text", ""),
            feedback="",
            instance_id_selection_mode=instance_id_selection_mode,
        )
        system_parts.append("")
        system_parts.append(
            "You must output one complete JSON action plan for the current high-level goal, using ONLY the allowed actions/skills."
        )
        system_parts.append(
            "This is a strict open-loop setting: plan from the current observation only, and do not assume you will receive later execution feedback or replanning chances."
        )
        system_parts.append(
            f"Plan length constraint: output {step_count_phrase} in this planning round."
        )
        system_parts.append(
            "Each step object MUST include `step_id`, `type`, `name`, `args`, and `expectation`."
        )
        system_parts.append("- `type` must be exactly `action` or `skill`.")
        system_parts.append("- `name` must be the exact action/skill name from the schemas above.")
        system_parts.append("- `args` must be a JSON object using the actual parameter names for that action/skill; use `{}` when no args are needed.")
        system_parts.append("")
        system_parts.append("Response Format (JSON only):")
        response_format_text = (
            "{\n"
            '  "high_level_id": "Hk",\n'
            '  "goal": "one sentence goal",\n'
            '  "explanation": null,\n'
            '  "thoughts": "brief rationale",\n'
            '  "action_list": [\n'
            "{response_format_action_example_1},\n"
            "{response_format_action_example_2},\n"
            "{response_format_action_example_3}{response_format_action_examples_extra}\n"
            "  ]\n"
            "}"
        )
        system_parts.append(self._safe_template_format(response_format_text, response_format_template_values))

        user_parts: list[str] = []
        user_parts.append("USER:")
        self._append_user_sections(
            user_parts=user_parts,
            observation=observation,
            high_level_id=high_level_id,
            high_level_goal=high_level_goal,
            memory_bundle=memory_bundle,
            feedback="",
            percept_text=percept_text,
            inject_memory=inject_memory,
            include_task_progress=include_task_progress,
            instance_id_selection_mode=instance_id_selection_mode,
        )

        prompt = self._strip_skill_interface_references(
            "\n".join(system_parts).strip() + "\n\n" + "\n".join(user_parts).strip() + "\n"
        ) + "\n"
        if self.cfg.save_last_prompt_path is not None:
            self.cfg.save_last_prompt_path.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.save_last_prompt_path.write_text(prompt, encoding="utf-8")
        if self.cfg.save_per_step_dir is not None:
            try:
                self.cfg.save_per_step_dir.mkdir(parents=True, exist_ok=True)
                step_tag = self._step_tag(observation)
                out_path = self.cfg.save_per_step_dir / f"{step_tag}.txt"
                out_path.write_text(prompt, encoding="utf-8")
            except Exception:
                pass
        self._audit_final_prompt(source="prompt_builder.open_loop", prompt=prompt)
        return prompt

