from __future__ import annotations

import json
from typing import Any

from epm.brain.plan_schema import PlanStep
from epm.cerebellum.cookbench_api import ActionResult


def _compact_json(obj: Any, *, max_chars: int = -1) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        s = repr(obj)
    if int(max_chars) > 0 and len(s) > int(max_chars):
        return s[: max_chars - 3] + "..."
    return s


def _summarize_return_value(value: Any) -> str:
    if value is None:
        return ""

    # Common structured skill outputs (prefer stable keys).
    if isinstance(value, dict):
        # Mode enter/exit hints (best-effort, for prompt debugging).
        # Only surface this when we positively know entering failed, to avoid
        # hiding more informative summaries (e.g., poured_ml, completed/requested).
        if isinstance(value.get("mode"), str) and value.get("mode_entered") is False:
            return f"return: mode={value.get('mode')!r} mode_entered=False"

        if "blender_id" in value and ("container_name" in value or "container" in value):
            c = value.get("container_name", value.get("container"))
            return f"return: container={c!r} blender_id={value.get('blender_id')}"
        if "completed" in value and "requested" in value:
            completed = value.get("completed")
            requested = value.get("requested")
            target = value.get("target")
            return f"return: target={target!r} completed={completed} requested={requested}"

        if "cut_num" in value and "item_name" in value:
            w = value.get("weight", None)
            if w is not None:
                return f"return: item={value.get('item_name')!r} cut_num={value.get('cut_num')} weight={w}"
            return f"return: item={value.get('item_name')!r} cut_num={value.get('cut_num')}"

        if "target" in value and ("on_screen" in value or "distance" in value):
            return (
                f"return: target={value.get('target')!r} on_screen={value.get('on_screen')!r} "
                f"distance={value.get('distance')!r}"
            )

        if "target" in value and set(value.keys()) <= {"success", "target", "error"}:
            return f"return: target={value.get('target')!r}"

        if "poured_ml" in value and "target_ml" in value:
            poured = value.get("poured_ml")
            target = value.get("target_ml")
            tol = value.get("tolerance")
            container = value.get("container_name")
            parts: list[str] = []
            if container:
                parts.append(f"container={container!r}")
            parts.append(f"poured_ml={poured} target_ml={target}")
            if tol is not None:
                parts.append(f"tolerance={tol}")
            status = value.get("pour_status")
            if status:
                parts.append(f"pour_status={status}")
            return "return: " + " ".join(parts)

        # Generic dict: show a small subset of keys.
        keys = list(value.keys())
        preview_keys = keys[:8]
        preview = {k: value.get(k) for k in preview_keys}
        extra = ""
        if len(keys) > len(preview_keys):
            extra = f" (+{len(keys) - len(preview_keys)} keys)"
        return "return: " + _compact_json(preview) + extra

    if isinstance(value, list):
        return f"return: list(len={len(value)}) " + _compact_json(value)

    return "return: " + _compact_json(value)


def _snapshot_active_modes(snap: Any) -> list[str]:
    if not isinstance(snap, dict):
        return []
    out: list[str] = []
    if bool(snap.get("is_pouring_mode")):
        out.append("pour")
    if bool(snap.get("is_sprinkle_mode")):
        out.append("sprinkle")
    if bool(snap.get("is_cutting_mode")):
        out.append("cut")
    if bool(snap.get("is_mixing_mode")):
        out.append("mix")
    if bool(snap.get("is_flip_mode")):
        out.append("flip")
    return out


def render_action_feedback(
    *,
    step: PlanStep,
    action_result: ActionResult,
    final_success: bool,
    error: str,
) -> str:
    """
    Render a compact, prompt-friendly feedback block for the *next* planning round.

    Goals:
    - Deterministic + easy to parse by LLMs
    - Includes args and (when available) structured return values from skills/actions
    """

    lines: list[str] = []
    lines.append(f"last_step_id={step.step_id}")
    lines.append(f"type={step.type} name={step.name}")
    if step.args:
        lines.append("args=" + _compact_json(step.args))
    lines.append(f"result={'success' if final_success else 'failure'}")
    if error:
        lines.append("error=" + error.strip())
        if step.name == "put_down" and "post_check_failed:put_down_not_idle" in str(error):
            lines.append("blocking=put_down_failed_place_invalid")
            lines.append(
                "instruction=Put-down likely failed because the place point is occupied or current held item "
                "cannot be placed there. Navigate to another empty place point and retry put_down."
            )
            lines.append(
                "hint=Use put_place_occupancy snapshot to select an empty point; avoid previously failed point."
            )
        if step.name == "put_down" and (
            "put_down_post_check_failed:still_holding_item" in str(error)
            or "put_down_post_check_failed:interaction_mode_still_active" in str(error)
        ):
            lines.append("blocking=put_down_failed_place_invalid")
            lines.append(
                "instruction=Put-down did not release the held item. This place is likely occupied or invalid for current item. "
                "Navigate to another empty place point and retry put_down."
            )
            lines.append(
                "hint=Use put_place_occupancy snapshot and choose a different free point than the last attempt."
            )

    if isinstance(action_result.raw, dict) and action_result.raw:
        # For query skills, include structured results directly so the planner can
        # bind instance_id (and avoid re-query loops).
        if step.type == "skill" and step.name == "query_scene_objects" and isinstance(action_result.raw.get("results"), list):
            results = action_result.raw.get("results") or []
            compact: list[dict[str, Any]] = []
            for it in results:
                if not isinstance(it, dict):
                    continue
                # Keep only the most useful disambiguation/navigation fields.
                keep_keys = (
                    "name_en",
                    "name_cn",
                    "name",
                    "kind",
                    "instance_id",
                    "distance",
                    "is_on_screen",
                    "container",
                    "position",
                    "is_open",
                    "open_angle",
                )
                compact.append({k: it.get(k) for k in keep_keys if k in it})
            lines.append(f"results_len={len(results)}")
            if compact:
                lines.append("results=" + _compact_json(compact, max_chars=-1))
            grouped = action_result.raw.get("results_by_query")
            if isinstance(grouped, dict) and grouped:
                grouped_summary: dict[str, int] = {}
                for q, vals in grouped.items():
                    if isinstance(vals, list):
                        grouped_summary[str(q)] = len(vals)
                if grouped_summary:
                    lines.append("results_by_query=" + _compact_json(grouped_summary, max_chars=-1))
                lines.append(
                    "hint=Select one result; use its instance_id for actions/skills that accept *_instance_id "
                    "(e.g., PickupObject, PutObject, auto_navigation)."
                )

        # Navigation target resolution failures: surface suggestions so the planner can recover.
        if step.type == "skill" and step.name == "auto_navigation" and error.strip() == "invalid_target_name":
            sugg = action_result.raw.get("mapping_suggestions")
            if isinstance(sugg, list) and sugg:
                lines.append("mapping_suggestions=" + _compact_json(sugg[:20], max_chars=-1))
                lines.append("hint=Choose one mapping_suggestions value and retry auto_navigation(target=...).")
        if step.type == "skill" and step.name == "auto_navigation" and error.strip() == "navigation_blocked_by_mode":
            mode_raw = action_result.raw.get("blocked_by_mode")
            if isinstance(mode_raw, str) and mode_raw.strip():
                lines.append(f"blocked_by_mode={mode_raw.strip()!r}")
            mode_class = action_result.raw.get("blocked_mode_class")
            if isinstance(mode_class, str) and mode_class.strip():
                lines.append(f"blocked_mode_class={mode_class.strip()!r}")
            hint = action_result.raw.get("hint")
            if isinstance(hint, str) and hint.strip():
                lines.append("hint=" + hint.strip())
            mode_hint = action_result.raw.get("mode_hint")
            if isinstance(mode_hint, str) and mode_hint.strip():
                lines.append("mode_hint=" + mode_hint.strip())
        if "interaction_mode_still_active" in error or "cutting_mode_still_active" in error:
            for label, snap_key in (("pre_active_modes", "precheck"), ("post_active_modes", "postcheck")):
                modes = _snapshot_active_modes(action_result.raw.get(snap_key))
                if modes:
                    lines.append(f"{label}=" + _compact_json(modes))

        if step.name == "auto_pour":
            feedback_code = str(action_result.raw.get("feedback_code") or "").strip()
            feedback_to_planner = str(action_result.raw.get("feedback_to_planner") or "").strip()
            pour_status = str(action_result.raw.get("pour_status") or "").strip()
            if pour_status:
                lines.append(f"pour_status={pour_status}")
            if feedback_code:
                lines.append(f"feedback_code={feedback_code}")
            if feedback_to_planner:
                lines.append("instruction=" + feedback_to_planner)

        # list_supported_items: provide a compact summary of supported names/platforms/tool points.
        if step.type == "skill" and step.name == "list_supported_items":
            items = action_result.raw.get("items")
            platforms = action_result.raw.get("platforms")
            tool_points = action_result.raw.get("tool_points")
            if isinstance(items, list):
                names: list[str] = []
                for it in items:
                    if isinstance(it, str) and it.strip():
                        names.append(it.strip())
                    elif isinstance(it, dict):
                        # best-effort schema
                        for k in ("name_en", "name_cn", "name"):
                            v = it.get(k)
                            if isinstance(v, str) and v.strip():
                                names.append(v.strip())
                                break
                lines.append(f"supported_items_len={len(items)}")
                if names:
                    lines.append("supported_items_sample=" + _compact_json(names, max_chars=-1))
            if isinstance(platforms, list):
                lines.append(f"platforms_len={len(platforms)}")
                sample = []
                for it in platforms:
                    if isinstance(it, str) and it.strip():
                        sample.append(it.strip())
                    elif isinstance(it, dict):
                        v = it.get("name") or it.get("platform") or it.get("id")
                        if isinstance(v, str) and v.strip():
                            sample.append(v.strip())
                if sample:
                    lines.append("platforms_sample=" + _compact_json(sample, max_chars=-1))
            if isinstance(tool_points, list):
                lines.append(f"tool_points_len={len(tool_points)}")
                sample = []
                for it in tool_points:
                    if isinstance(it, str) and it.strip():
                        sample.append(it.strip())
                    elif isinstance(it, dict):
                        v = it.get("name") or it.get("tool") or it.get("id")
                        if isinstance(v, str) and v.strip():
                            sample.append(v.strip())
                if sample:
                    lines.append("tool_points_sample=" + _compact_json(sample, max_chars=-1))

        # Special-case instance disambiguation: this is meant to be injected into the next prompt
        # so the planner can immediately pick an instance_id and retry.
        if error in {"instance_id_required", "instance_id_not_found"} and isinstance(action_result.raw.get("candidates"), list):
            lines.append("blocking=instance_disambiguation")
            # Structured meta for instance_id selection mode (planner parses these fields).
            for k in ("action", "name_arg", "instance_arg", "name"):
                v = action_result.raw.get(k)
                if isinstance(v, str) and v.strip():
                    lines.append(f"instance_meta.{k}={v.strip()}")
            if isinstance(action_result.raw.get("selection_prompt"), str) and action_result.raw.get("selection_prompt"):
                lines.append("instruction=" + str(action_result.raw.get("selection_prompt")).strip())
            # Include full candidate dicts (schemas vary). Keep larger budget for this case.
            lines.append("candidates=" + _compact_json(action_result.raw.get("candidates"), max_chars=-1))
        else:
            ret = action_result.raw.get("return")
            ret_summary = _summarize_return_value(ret)
            if ret_summary:
                lines.append(ret_summary)

    return "\n".join(lines).strip()
