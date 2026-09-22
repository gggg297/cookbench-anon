from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
import io as _pyio
from pathlib import Path
from typing import Optional
import re
from contextlib import redirect_stdout

from epm.cerebellum.cookbench_api import ActionAPI, ActionResult
from epm.cerebellum.game_hotkeys import ensure_alt_j_interaction, ensure_f11_radar_scan, ensure_f12_products_scan
from epm.cerebellum.interaction_snapshot import read_interaction_snapshot, render_interaction_snapshot
from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.realtime_products import best_match_by_name, extract_items, item_display_name, read_realtime_products
from epm.cerebellum.skills._instance_resolution import resolve_instance_id
from epm.cerebellum.skills._shared_paths import realtime_radar_scan_path, userdata_root
from epm.cerebellum.skills._shared_paths import realtime_products_json
from epm.vision.screen_capture import activate_window

from .simple_astar_radar_navigator import SimpleAstarRadarNavigator


@dataclass(frozen=True)
class NavigateArgs:
    """
    Navigate to a target object using the A* navigator.

    - `map_file` optional: falls back to `epm/epm_config.json` -> `paths.auto_nav_map_path`.
    """

    target: str
    target_instance_id: int | None = None
    map_file: str | None = None
    auto_activate_window: bool = True
    impl: str = "simple_astar_radar"
    docking_distance_m: float = 1.5
    docking_distance_tolerance_m: float = 0.2
    internal_rect_strategy: str = "normalized_ratio"

_PLATFORM_SUFFIX_RE = re.compile(r".*-\d+$")
_INDEX_SUFFIX_RE = re.compile(r".*\d+$")
_PLANE_AXES = ("x", "z")  # ground-plane axes in realtime_products.json


def _norm(s: object) -> str:
    return (str(s or "")).strip().lower()


def _item_text(it: dict) -> str:
    return " ".join(
        str(it.get(k) or "").strip().lower()
        for k in ("name_en", "name_cn", "game_object")
    )


def _is_cutting_board_like(it: dict) -> bool:
    text = _item_text(it)
    return any(token in text for token in ("cutting board", "chopping board", "菜板", "砧板", "cutting_board"))


def _is_knife_like(it: dict) -> bool:
    return "knife" in _item_text(it)


def _detect_interaction_mode(items: list[dict]) -> str:
    """
    Best-effort mode detection directly from realtime_products items.
    Returns empty string when no interaction mode is active.
    """
    def _any_true(keys: tuple[str, ...]) -> bool:
        for it in items:
            if not isinstance(it, dict):
                continue
            for k in keys:
                if bool(it.get(k, False)):
                    return True
        return False

    for it in items:
        if not isinstance(it, dict):
            continue
        if _is_knife_like(it) and bool(it.get("is_cut_mode") or it.get("is_cutting") or it.get("is_cutting_mode")):
            return "cutting_mode"
    if _any_true(("is_cut_mode", "is_cutting", "is_cutting_mode")):
        return "cutting_mode"
    for it in items:
        if not isinstance(it, dict) or _is_cutting_board_like(it):
            continue
        if bool(it.get("is_pouring_mode") or it.get("is_pouring")):
            return "pouring_mode"
    if _any_true(("is_mixing_mode",)):
        return "mixing_mode"
    if _any_true(("is_filp_mode", "is_flip_mode", "is_flipping_mode")):
        return "flip_mode"
    if _any_true(("is_sprinkle_mode", "is_sprinkle")):
        return "sprinkle_mode"
    return ""


def _find_held_item(items: list[dict]) -> dict | None:
    for it in items:
        if not isinstance(it, dict):
            continue
        if bool(it.get("is_held", False)):
            return it
    return None


def _tail_lines(text: str, *, max_lines: int = 40, max_chars: int = 8000) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = raw.splitlines()
    tail = "\n".join(lines[-int(max_lines) :]).strip()
    if len(tail) <= int(max_chars):
        return tail
    return tail[-int(max_chars) :]


def _classify_navigation_failure(stdout_text: str) -> tuple[str, list[str]]:
    raw = str(stdout_text or "")
    text = raw.lower()
    signals: list[str] = []

    def _hit(snippet: str, signal: str) -> bool:
        if snippet.lower() in text:
            signals.append(signal)
            return True
        return False

    if _hit("未找到目标物品", "target_not_found"):
        return "navigation_failed_target_not_found", signals
    if _hit("无法加载相机信息", "camera_info_unavailable"):
        return "navigation_failed_camera_info_unavailable", signals
    if _hit("a*路径规划失败", "astar_path_not_found"):
        return "navigation_failed_astar_path_not_found", signals
    if _hit("路径点用完且无最终目标", "missing_final_target"):
        return "navigation_failed_missing_final_target", signals
    if _hit("所有方向都不安全", "all_directions_blocked"):
        return "navigation_failed_all_directions_blocked", signals
    if _hit("前方被堵死", "front_hemisphere_blocked"):
        return "navigation_failed_front_blocked", signals
    if _hit("移动受阻", "local_movement_blocked"):
        return "navigation_failed_local_movement_blocked", signals
    if _hit("超过最大尝试次数", "max_attempts_exceeded"):
        return "navigation_failed_max_attempts_exceeded", signals
    if _hit("未满足精确调整要求", "precise_adjustment_failed"):
        return "navigation_failed_precise_adjustment_failed", signals
    if _hit("未满足所有条件", "final_validation_failed"):
        return "navigation_failed_final_validation_failed", signals
    if _hit("目标不在视野内", "target_not_in_view"):
        return "navigation_failed_target_not_in_view", signals
    if _hit("无法获取目标物品信息", "target_info_unavailable"):
        return "navigation_failed_target_info_unavailable", signals
    if _hit("无法读取雷达数据", "radar_unavailable"):
        return "navigation_failed_radar_unavailable", signals
    return "navigation_failed_unknown", signals


def _classify_navigation_failure_from_raw(raw: dict, stdout_text: str) -> tuple[str, list[str]]:
    fallback_code, fallback_signals = _classify_navigation_failure(stdout_text)
    if fallback_code != "navigation_failed_unknown":
        return fallback_code, fallback_signals

    path_debug = raw.get("path_debug")
    if isinstance(path_debug, dict):
        reason = str(path_debug.get("reason") or "").strip().lower()
        start_original = path_debug.get("start_original")
        goal_original = path_debug.get("goal_original")
        signals: list[str] = []
        if reason:
            signals.append(f"path_reason:{reason}")
        if isinstance(start_original, dict) and bool(start_original.get("hard_obstacle")):
            signals.append("path_start_in_hard_obstacle")
        if isinstance(goal_original, dict) and bool(goal_original.get("hard_obstacle")):
            signals.append("path_goal_in_hard_obstacle")
        if reason == "path_not_found_after_search":
            return "navigation_failed_astar_path_not_found", signals or ["astar_path_not_found"]
        if reason:
            return f"navigation_failed_path_{reason}", signals

    docking_debug = raw.get("docking_debug")
    if isinstance(docking_debug, dict):
        chosen = docking_debug.get("chosen")
        if chosen is None:
            signals = ["no_valid_docking_pose"]
            candidates = docking_debug.get("candidate_directions")
            if isinstance(candidates, list) and candidates:
                blocked = 0
                for cand in candidates:
                    if not isinstance(cand, dict):
                        continue
                    if bool(cand.get("hard_obstacle")) or bool(cand.get("effective_obstacle")):
                        blocked += 1
                if blocked:
                    signals.append(f"docking_candidates_blocked:{blocked}")
            return "navigation_failed_no_valid_docking_pose", signals

    precise_adjust_debug = raw.get("precise_adjust_debug")
    if isinstance(precise_adjust_debug, dict):
        signals = []
        if precise_adjust_debug.get("mode") == "fallback_face_target":
            signals.append("fallback_face_target_only")
        if precise_adjust_debug.get("position_adjusted") is False:
            signals.append("precise_position_not_reached")
        if precise_adjust_debug.get("position_phase_within_tolerance") is False:
            signals.append("position_out_of_tolerance")
        if precise_adjust_debug.get("center_adjusted") is False:
            signals.append("target_not_centered")

        container_debug = raw.get("container_docking_debug")
        chosen_distance = raw.get("chosen_docking_distance_m")
        try:
            chosen_distance_f = float(chosen_distance) if chosen_distance is not None else None
        except Exception:
            chosen_distance_f = None
        if (
            isinstance(container_debug, dict)
            and chosen_distance_f is not None
            and chosen_distance_f > 0.30
        ):
            signals.append("container_target_far_from_walkable_space")

        if signals:
            if "container_target_far_from_walkable_space" in signals:
                return "navigation_failed_container_target_too_far", signals
            return "navigation_failed_precise_adjustment_failed", signals

    container_debug = raw.get("container_docking_debug")
    chosen_distance = raw.get("chosen_docking_distance_m")
    try:
        chosen_distance_f = float(chosen_distance) if chosen_distance is not None else None
    except Exception:
        chosen_distance_f = None
    if (
        isinstance(container_debug, dict)
        and chosen_distance_f is not None
        and chosen_distance_f > 0.30
    ):
        return "navigation_failed_container_target_too_far", ["container_target_far_from_walkable_space"]

    return fallback_code, fallback_signals


def _current_alt_j_snapshot_text() -> str:
    try:
        snap = read_interaction_snapshot(userdata_root=userdata_root(), max_age_s=2.5)
        if snap is None:
            return ""
        return render_interaction_snapshot(snap)
    except Exception:
        return ""


def _pos_of(item: dict) -> tuple[float, float, float] | None:
    pos = item.get("position")
    if not isinstance(pos, dict):
        return None
    try:
        x = float(pos.get("x"))
        y = float(pos.get("y"))
        z = float(pos.get("z"))
        return (x, y, z)
    except Exception:
        return None


def _pos_plane(item: dict) -> tuple[float, float] | None:
    pos = item.get("position")
    if not isinstance(pos, dict):
        return None
    try:
        a = float(pos.get(_PLANE_AXES[0]))
        b = float(pos.get(_PLANE_AXES[1]))
        return (a, b)
    except Exception:
        return None


def _distance_plane(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _platform_category_from_name(name: str) -> str:
    s = (name or "").strip()
    if _PLATFORM_SUFFIX_RE.match(s):
        return s[: s.rfind("-")].strip()
    return s


def _platform_index_from_name(name: str) -> int | None:
    s = (name or "").strip()
    if not _PLATFORM_SUFFIX_RE.match(s):
        return None
    try:
        return int(s[s.rfind("-") + 1 :])
    except Exception:
        return None


def _platform_spacing(items: list[dict], *, target_name: str, target_pos: dict | None) -> float | None:
    if not target_pos:
        return None
    cat = _platform_category_from_name(str(target_name)).lower()
    if not cat:
        return None
    center = _pos_plane({"position": target_pos})
    if center is None:
        return None
    min_dist: float | None = None
    for it in items:
        if not isinstance(it, dict):
            continue
        if _norm(it.get("kind")) != "platform_point":
            continue
        name = item_display_name(it)
        if _platform_category_from_name(name).lower() != cat:
            continue
        pos = _pos_plane(it)
        if pos is None:
            continue
        d = _distance_plane(center, pos)
        if d <= 0:
            continue
        if min_dist is None or d < min_dist:
            min_dist = d
    return min_dist


def _pick_resolved_item(resolved: object, *, target_name: str) -> dict | None:
    if not resolved or not hasattr(resolved, "candidates"):
        return None
    candidates = getattr(resolved, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return None
    iid = getattr(resolved, "instance_id", None)
    if iid is not None:
        for it in candidates:
            if not isinstance(it, dict):
                continue
            try:
                if int(it.get("instance_id")) == int(iid):
                    return it
            except Exception:
                continue
    return best_match_by_name(candidates, target_name) or next((it for it in candidates if isinstance(it, dict)), None)


def _bounds_of(item: dict) -> tuple[dict, dict] | None:
    bmin = item.get("bounds_min")
    bmax = item.get("bounds_max")
    if not isinstance(bmin, dict) or not isinstance(bmax, dict):
        return None
    try:
        _ = float(bmin.get("x"))
        _ = float(bmin.get("y"))
        _ = float(bmin.get("z"))
        _ = float(bmax.get("x"))
        _ = float(bmax.get("y"))
        _ = float(bmax.get("z"))
        return bmin, bmax
    except Exception:
        return None


def _platform_point_occupied_with_items(
    *, items: list[dict], target_name: str, target_instance_id: int | None, resolved_item: dict | None
) -> tuple[bool, dict]:
    """
    Best-effort occupancy check for a platform point.

    Heuristics (in order):
    1) container/parent field matches target_name or target_instance_id
    2) distance-to-platform-position < threshold
    """
    occupied_items: list[dict] = []
    seen: set[tuple[str, int | None]] = set()
    spacing = 0.18
    spacing_source = "fallback"
    min_above_dy = 0.01

    resolved_iid = None
    if isinstance(resolved_item, dict):
        try:
            resolved_iid = int(resolved_item.get("instance_id"))
        except Exception:
            resolved_iid = None

    def _add(item: dict, reason: str) -> None:
        name = item_display_name(item)
        iid = None
        try:
            iid = int(item.get("instance_id"))
        except Exception:
            iid = None
        key = (name, iid)
        if key in seen:
            return
        seen.add(key)
        occupied_items.append(
            {
                "name": name,
                "instance_id": iid,
                "reason": reason,
            }
        )

    # Distance-based match on ground-plane (if we know platform position)
    if not isinstance(resolved_item, dict):
        return False, {"occupied": False, "occupied_distance_threshold_m": spacing}
    pos = _pos_of(resolved_item)
    if pos is None:
        return False, {"occupied": False, "occupied_distance_threshold_m": spacing}

    platform_plane = {"x": pos[0], "y": pos[1], "z": pos[2]}
    spacing_nn = _platform_spacing(items, target_name=str(target_name), target_pos=platform_plane)
    if spacing_nn is not None:
        spacing = spacing_nn
        spacing_source = "nearest_neighbor"

    p0 = _pos_plane({"position": platform_plane})
    if p0 is None:
        return False, {"occupied": False, "occupied_distance_threshold_m": spacing}

    for it in items:
        if not isinstance(it, dict):
            continue
        if bool(it.get("is_held")):
            continue
        kind = _norm(it.get("kind"))
        if kind in {"platform_point", "tool_point"}:
            continue
        try:
            if resolved_iid is not None and int(it.get("instance_id")) == int(resolved_iid):
                continue
        except Exception:
            pass
        p3 = _pos_of(it)
        above = False
        if p3 is not None and (p3[1] - pos[1]) >= min_above_dy:
            above = True
        bounds = _bounds_of(it)
        if bounds is not None:
            bmin, bmax = bounds
            try:
                by = float(bmax.get("y"))
                if (by - pos[1]) >= min_above_dy:
                    above = True
            except Exception:
                pass
            try:
                x0, z0 = p0
                if float(bmin.get("x")) <= x0 <= float(bmax.get("x")) and float(bmin.get("z")) <= z0 <= float(bmax.get("z")):
                    if above:
                        _add(it, "bbox_cover")
                        if len(occupied_items) >= 8:
                            break
                    continue
            except Exception:
                pass
        if not above:
            continue
        p2 = _pos_plane(it)
        if p2 is None:
            continue
        if _distance_plane(p2, p0) <= spacing:
            _add(it, "plane_distance_match")
            if len(occupied_items) >= 8:
                break

    if occupied_items:
        return True, {
            "occupied": True,
            "occupied_items": occupied_items,
            "occupied_by": [x.get("name") for x in occupied_items[:8]],
            "occupied_distance_threshold_m": spacing,
            "occupied_spacing_source": spacing_source,
            "occupied_plane_axes": f"{_PLANE_AXES[0]}-{_PLANE_AXES[1]}",
            "occupied_above_min_dy": min_above_dy,
        }
    return False, {
        "occupied": False,
        "occupied_items": [],
        "occupied_distance_threshold_m": spacing,
        "occupied_spacing_source": spacing_source,
        "occupied_plane_axes": f"{_PLANE_AXES[0]}-{_PLANE_AXES[1]}",
        "occupied_above_min_dy": min_above_dy,
    }


def _load_put_place_list_full() -> dict:
    p = _put_place_list_path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _free_ranges_from_indices(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    indices = sorted(set(indices))
    ranges: list[list[int]] = []
    start = prev = indices[0]
    for x in indices[1:]:
        if x == prev + 1:
            prev = x
            continue
        ranges.append([start, prev])
        start = prev = x
    ranges.append([start, prev])
    return ranges


def _build_put_place_occupancy_snapshot(*, items: list[dict]) -> dict:
    platform_points = [it for it in items if isinstance(it, dict) and _norm(it.get("kind")) == "platform_point"]
    occupied_indices: dict[str, set[int]] = {}
    occupied_by_item: dict[str, dict] = {}
    occupied_places_by_item: dict[str, set[str]] = {}

    for pp in platform_points:
        name = item_display_name(pp)
        cat = _platform_category_from_name(name)
        idx = _platform_index_from_name(name)
        iid = None
        try:
            iid = int(pp.get("instance_id"))
        except Exception:
            iid = None
        occ, occ_raw = _platform_point_occupied_with_items(
            items=items,
            target_name=name,
            target_instance_id=iid,
            resolved_item=pp,
        )
        if occ and idx is not None:
            occupied_indices.setdefault(cat, set()).add(int(idx))
            for it in occ_raw.get("occupied_items", []) or []:
                if not isinstance(it, dict):
                    continue
                iname = str(it.get("name") or "unknown").strip()
                try:
                    inst = int(it.get("instance_id")) if it.get("instance_id") is not None else None
                except Exception:
                    inst = None
                key = f"{iname}#{inst}" if inst is not None else iname
                seen_places = occupied_places_by_item.setdefault(key, set())
                if name in seen_places:
                    continue
                entry = occupied_by_item.get(
                    key,
                    {"name": iname, "instance_id": inst, "places": []},
                )
                entry["places"].append(
                    {
                        "place_point": name,
                        "reason": str(it.get("reason") or ""),
                    }
                )
                occupied_by_item[key] = entry
                seen_places.add(name)

    put_place = _load_put_place_list_full()
    platforms = put_place.get("platforms") if isinstance(put_place.get("platforms"), list) else []
    free_platforms: list[dict] = []
    for p in platforms:
        if not isinstance(p, dict):
            continue
        cat = str(p.get("category") or "").strip()
        name_template = str(p.get("name_template") or "").strip()
        ranges = p.get("index_ranges")
        free_indices: list[int] = []
        if isinstance(ranges, list):
            for rr in ranges:
                if not isinstance(rr, list) or len(rr) != 2:
                    continue
                try:
                    lo, hi = int(rr[0]), int(rr[1])
                except Exception:
                    continue
                for i in range(lo, hi + 1):
                    if i not in occupied_indices.get(cat, set()):
                        free_indices.append(i)
        free_platforms.append(
            {
                "category": cat,
                "name_template": name_template,
                "index_ranges": _free_ranges_from_indices(free_indices),
            }
        )

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "kind": str(put_place.get("kind") or "platform_point"),
        "note": str(put_place.get("note") or ""),
        "free-place-point": free_platforms,
        "occupied_by_item": occupied_by_item,
    }


def _put_place_list_path() -> Path:
    # <repo>/epm/src/epm/cerebellum/skills/auto_navigation/skill.py -> parents[6] == <repo>
    return Path(__file__).resolve().parents[6] / "epm" / "data" / "put_place_list.json"


def _load_platforms() -> list[dict]:
    p = _put_place_list_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
        plats = raw.get("platforms")
        if isinstance(plats, list):
            return [x for x in plats if isinstance(x, dict)]
    except Exception:
        return []
    return []

def _clean_platform_category_name(name: str) -> str:
    # Tolerate accidental trailing '-' or whitespace, e.g. "Griddle Surface-".
    return (name or "").strip().rstrip().rstrip("-").rstrip()


def _platform_input_diag(*, target: str, target_instance_id: int | None, platforms: list[dict]) -> dict:
    clean = _clean_platform_category_name(str(target))
    cats: dict[str, dict] = {}
    for p in platforms:
        cat = str(p.get("category") or "").strip()
        if not cat:
            continue
        cats[cat.lower()] = p
    hit = cats.get(clean.lower())
    diag: dict = {
        "target_raw": str(target),
        "target_clean": clean,
        "category_match": bool(hit is not None),
        "index_provided": target_instance_id is not None,
        "index_value": (int(target_instance_id) if target_instance_id is not None else None),
    }
    if hit is not None:
        diag["category"] = str(hit.get("category") or clean)
        diag["index_ranges"] = hit.get("index_ranges")
        diag["example"] = {
            "as_platform_index": f"target={clean!r} target_instance_id=3",
            "as_full_name": f"target={str(hit.get('name_template') or (clean + '-{index}')).strip().replace('{index}', '3')!r}",
        }
        # If index is provided, validate it against ranges.
        idx = target_instance_id
        ok = None
        if idx is not None:
            ok = False
            ranges = hit.get("index_ranges")
            if isinstance(ranges, list):
                for rr in ranges:
                    if isinstance(rr, list) and len(rr) == 2:
                        try:
                            lo, hi = int(rr[0]), int(rr[1])
                        except Exception:
                            continue
                        if lo <= int(idx) <= hi:
                            ok = True
                            break
            diag["index_valid"] = bool(ok)
    return diag


def _try_expand_platform_point_name(*, name: str, index: int | None) -> tuple[str | None, str | None]:
    """
    Compatibility for navigation:
    - If `target` is a platform category (e.g. "Side Table") and `target_instance_id` is provided as an index (e.g. 3),
      expand to "Side Table-3" before resolve_instance_id/navigate.
    """
    n = _clean_platform_category_name(str(name))
    if not n or _PLATFORM_SUFFIX_RE.match(n):
        return None, None
    if index is None:
        return None, None

    platforms = _load_platforms()
    if not platforms:
        return None, "put_place_list_missing"

    want = n.lower()
    hit = None
    for p in platforms:
        cat = str(p.get("category") or "").strip()
        if cat.lower() == want:
            hit = p
            break
    if hit is None:
        return None, None

    tmpl = str(hit.get("name_template") or "").strip()
    if "{index}" not in tmpl:
        return None, "put_place_template_invalid"

    ranges = hit.get("index_ranges")
    ok = False
    if isinstance(ranges, list) and ranges:
        for rr in ranges:
            if isinstance(rr, list) and len(rr) == 2:
                try:
                    lo, hi = int(rr[0]), int(rr[1])
                except Exception:
                    continue
                if lo <= int(index) <= hi:
                    ok = True
                    break
        if not ok:
            return None, "platform_index_out_of_range"

    return tmpl.format(index=int(index)), None


def _tool_interaction_point_list_path() -> Path:
    # <repo>/epm/src/epm/cerebellum/skills/auto_navigation/skill.py -> parents[6] == <repo>
    return Path(__file__).resolve().parents[6] / "epm" / "data" / "tool_interaction_point_list.json"


def _load_tool_points() -> list[dict]:
    p = _tool_interaction_point_list_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
        pts = raw.get("tool_points")
        if isinstance(pts, list):
            return [x for x in pts if isinstance(x, dict)]
    except Exception:
        return []
    return []


def _clean_tool_point_category_name(name: str) -> str:
    return (name or "").strip()


def _tool_point_input_diag(*, target: str, target_instance_id: int | None, tool_points: list[dict]) -> dict:
    clean = _clean_tool_point_category_name(str(target))
    cats: dict[str, dict] = {}
    for p in tool_points:
        cat = str(p.get("category") or "").strip()
        if not cat:
            continue
        cats[cat.lower()] = p
    hit = cats.get(clean.lower())
    diag: dict = {
        "target_raw": str(target),
        "target_clean": clean,
        "category_match": bool(hit is not None),
        "index_provided": target_instance_id is not None,
        "index_value": (int(target_instance_id) if target_instance_id is not None else None),
    }
    if hit is not None:
        diag["category"] = str(hit.get("category") or clean)
        diag["index_ranges"] = hit.get("index_ranges")
        tmpl = str(hit.get("name_template") or hit.get("category") or clean).strip()
        diag["example"] = {
            "as_indexed_category": f"target={clean!r} target_instance_id=1",
            "as_full_name": f"target={tmpl.replace('{index}', '1')!r}",
        }
        idx = target_instance_id
        ranges = hit.get("index_ranges")
        if idx is not None:
            if ranges is None:
                diag["index_valid"] = False
                diag["index_reason"] = "index_not_applicable"
            else:
                ok = False
                if isinstance(ranges, list):
                    for rr in ranges:
                        if isinstance(rr, list) and len(rr) == 2:
                            try:
                                lo, hi = int(rr[0]), int(rr[1])
                            except Exception:
                                continue
                            if lo <= int(idx) <= hi:
                                ok = True
                                break
                diag["index_valid"] = bool(ok)
    return diag


def _try_expand_tool_point_name(*, name: str, index: int | None) -> tuple[str | None, str | None]:
    """
    Compatibility for navigation:
    - If `target` is a tool-point category (listed in epm/data/tool_interaction_point_list.json) and `target_instance_id`
      is provided as an index, expand it into a concrete tool-point name using `name_template`.
    - If `index_ranges` is null for that category, the input is treated as a single name (no index).
    """
    n = _clean_tool_point_category_name(str(name))
    if not n or _INDEX_SUFFIX_RE.match(n):
        return None, None
    if index is None:
        return None, None

    tool_points = _load_tool_points()
    if not tool_points:
        return None, "tool_interaction_point_list_missing"

    want = n.lower()
    hit = None
    for p in tool_points:
        cat = str(p.get("category") or "").strip()
        if cat.lower() == want:
            hit = p
            break
    if hit is None:
        return None, None

    ranges = hit.get("index_ranges")
    if ranges is None:
        return None, "tool_point_index_not_applicable"

    ok = False
    if isinstance(ranges, list) and ranges:
        for rr in ranges:
            if isinstance(rr, list) and len(rr) == 2:
                try:
                    lo, hi = int(rr[0]), int(rr[1])
                except Exception:
                    continue
                if lo <= int(index) <= hi:
                    ok = True
                    break
    if not ok:
        return None, "tool_point_index_out_of_range"

    tmpl = str(hit.get("name_template") or "").strip()
    if "{index}" not in tmpl:
        return None, "tool_point_template_invalid"
    return tmpl.replace("{index}", str(int(index))), None


def _repo_root() -> Path:
    """Locate the repository root, tolerating either checkout layout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").is_dir() and (parent / "src").is_dir():
            return parent
    # <repo>/src/epm/cerebellum/skills/auto_navigation/skill.py -> parents[5] == <repo>
    return here.parents[5]


def _resolve_default_map_file() -> Optional[str]:
    cfg_path = (_repo_root() / "epm_config.json").resolve()
    if not cfg_path.exists():
        return None
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        paths = raw.get("paths") or {}
        v = paths.get("auto_nav_map_path")
        if not isinstance(v, str) or not v.strip():
            return None
        p = Path(v.strip())
        if not p.is_absolute():
            p = (cfg_path.parent / p).resolve()
        return str(p)
    except Exception:
        return None


def run(api: ActionAPI, args: NavigateArgs) -> ActionResult:
    map_file = args.map_file or _resolve_default_map_file()
    if not map_file:
        return ActionResult(
            success=False,
            error="Missing map_file. Provide map_file or set epm/epm_config.json: paths.auto_nav_map_path",
            raw={},
        )

    target = (args.target or "").strip()
    if not target:
        return ActionResult(
            success=False,
            error="Missing target. Provide target=<object_name> (e.g., target=lemon).",
            raw={},
        )

    # Strict pre-check: do not navigate while still in interaction modes.
    items_mode: list[dict] = []
    try:
        data_mode = read_realtime_products(realtime_products_json())
        items_mode = extract_items(data_mode)
        mode_active = _detect_interaction_mode(items_mode)
        if mode_active:
            return ActionResult(
                success=False,
                error="navigation_blocked_by_mode",
                raw={
                    "target": str(args.target),
                    "blocked_by_mode": mode_active,
                    "hint": (
                        "Cannot navigate while currently in interaction mode. "
                        "Exit the current mode first (e.g., right click to exit pouring/cutting/mixing/flip/sprinkle), "
                        "then retry auto_navigation."
                    ),
                },
            )
    except Exception:
        # Keep compatibility: if realtime read fails here, continue to downstream preflight checks.
        pass

    # Compatibility: if target is a platform category without "-<index>", allow passing the platform index in
    # target_instance_id, expand to full name for navigation. For normal objects, target_instance_id remains an instance_id.
    platforms = _load_platforms()
    diag = _platform_input_diag(
        target=str(target),
        target_instance_id=(int(args.target_instance_id) if args.target_instance_id is not None else None),
        platforms=platforms,
    )

    tool_points = _load_tool_points()
    tool_diag = _tool_point_input_diag(
        target=str(target),
        target_instance_id=(int(args.target_instance_id) if args.target_instance_id is not None else None),
        tool_points=tool_points,
    )
    expanded, expand_err = _try_expand_platform_point_name(
        name=str(target),
        index=(int(args.target_instance_id) if args.target_instance_id is not None else None),
    )
    if diag.get("category_match") is True:
        # Platform category input: require a valid index to expand.
        if args.target_instance_id is None:
            return ActionResult(
                success=False,
                error="platform_point_index_required",
                raw={
                    "action": "auto_navigation",
                    "target": str(args.target),
                    "target_instance_id": args.target_instance_id,
                    "platform_input": diag,
                    "put_place_platforms": platforms,
                    "selection_prompt": (
                        "Target is a platform category (kind='platform_point'). Provide `target_instance_id` as the "
                        "platform index (e.g. target='Side Table', target_instance_id=3), or pass the full name "
                        "(e.g. target='Side Table-3')."
                    ),
                },
            )
        if expand_err in {"put_place_list_missing", "put_place_template_invalid", "platform_index_out_of_range"} or not expanded:
            return ActionResult(
                success=False,
                error="platform_point_index_invalid",
                raw={
                    "action": "auto_navigation",
                    "target": str(args.target),
                    "target_instance_id": args.target_instance_id,
                    "reason": expand_err or "platform_expand_failed",
                    "platform_input": diag,
                    "put_place_platforms": platforms,
                    "selection_prompt": (
                        "Invalid platform-point input. If target is a platform category, pass "
                        "`target_instance_id` as the platform index and ensure it is within the allowed ranges. "
                        "Example: target='Side Table', target_instance_id=3."
                    ),
                },
            )
        target = expanded
        target_instance_id: int | None = None  # index consumed into name; instance_id still resolved from realtime
        kind_required: str | None = "platform_point"
    elif tool_diag.get("category_match") is True:
        # Tool-point category input: either expand with an index (for indexed categories) or accept as a single name
        # (for index_ranges=null categories).
        if args.target_instance_id is None:
            if tool_diag.get("index_ranges") is None:
                target_instance_id = None
                kind_required = "tool_point"
            else:
                return ActionResult(
                    success=False,
                    error="tool_point_index_required",
                    raw={
                        "action": "auto_navigation",
                        "target": str(args.target),
                        "target_instance_id": args.target_instance_id,
                        "tool_point_input": tool_diag,
                        "tool_points": tool_points,
                        "selection_prompt": (
                            "Target is a tool interaction point category (kind='tool_point'). Provide "
                            "`target_instance_id` as the category index (e.g. target='Stove Switch', "
                            "target_instance_id=1), or pass the full name (e.g. target='Stove Switch 1')."
                        ),
                    },
                )
        else:
            tool_expanded, tool_expand_err = _try_expand_tool_point_name(
                name=str(target),
                index=int(args.target_instance_id),
            )
            if tool_expand_err == "tool_point_index_not_applicable":
                return ActionResult(
                    success=False,
                    error="tool_point_index_not_applicable",
                    raw={
                        "action": "auto_navigation",
                        "target": str(args.target),
                        "target_instance_id": args.target_instance_id,
                        "tool_point_input": tool_diag,
                        "tool_points": tool_points,
                        "selection_prompt": (
                            "This tool interaction point does not accept an index. Pass only `target` "
                            "(e.g. target='Top Oven Switch') and omit `target_instance_id`."
                        ),
                    },
                )
            if tool_expand_err in {"tool_interaction_point_list_missing", "tool_point_template_invalid", "tool_point_index_out_of_range"} or not tool_expanded:
                return ActionResult(
                    success=False,
                    error="tool_point_index_invalid",
                    raw={
                        "action": "auto_navigation",
                        "target": str(args.target),
                        "target_instance_id": args.target_instance_id,
                        "reason": tool_expand_err or "tool_point_expand_failed",
                        "tool_point_input": tool_diag,
                        "tool_points": tool_points,
                        "selection_prompt": (
                            "Invalid tool interaction point input. If target is a tool-point category, pass "
                            "`target_instance_id` as the category index and ensure it is within the allowed ranges. "
                            "Example: target='Stove Switch', target_instance_id=1."
                        ),
                    },
                )
            target = tool_expanded
            target_instance_id = None  # index consumed into name; instance_id still resolved from realtime
            kind_required = "tool_point"
    else:
        # Normal object input: treat target_instance_id as a realtime instance_id.
        target_instance_id = (int(args.target_instance_id) if args.target_instance_id is not None else None)
        kind_required = None

    impl = (args.impl or "").strip().lower()
    if impl in ("", "simple_astar_radar", "radar", "simple_astar_radar_navigator"):
        # Preflight: make sure the mod outputs required by navigation are updating.
        # User preference: try to enable and continue; only warn on stale/missing signals.
        io = RawInputController()
        products_path = realtime_products_json()
        radar_path = realtime_radar_scan_path()
        ud = radar_path.parent if str(radar_path).strip() else userdata_root()
        if bool(args.auto_activate_window):
            try:
                activate_window("CookingSimulator")
            except Exception:
                pass

        f12_ok = ensure_f12_products_scan(
            realtime_products_path=products_path,
            window_title="CookingSimulator",
            activate_window=activate_window,
            io_controller=io,
            verbose=True,
        )
        f11_ok = ensure_f11_radar_scan(
            userdata_root=ud,
            window_title="CookingSimulator",
            activate_window=activate_window,
            io_controller=io,
            verbose=True,
        )
        alt_j_ok = ensure_alt_j_interaction(
            userdata_root=ud,
            window_title="CookingSimulator",
            activate_window=activate_window,
            io_controller=io,
            verbose=True,
        )
        if not f12_ok:
            print("[auto_navigation] warning: F12 products scan not confirmed fresh; continuing anyway")
        if not f11_ok:
            print("[auto_navigation] warning: F11 radar scan not confirmed fresh; continuing anyway")
        if not alt_j_ok:
            print("[auto_navigation] warning: Alt+J interaction scan not confirmed fresh; continuing anyway")

        resolved, err = resolve_instance_id(
            realtime_products_path=realtime_products_json(),
            name=str(target),
            instance_id=target_instance_id,
            action="auto_navigation",
            name_arg="target",
            instance_arg="target_instance_id",
            kind_required=kind_required,
        )
        if err is not None:
            raw = dict(getattr(err, "raw", None) or {})
            # Provide extra feedback about platform-input validity so the model can decide how to interpret target_instance_id.
            raw.setdefault("platform_input", diag)
            if platforms:
                raw.setdefault("put_place_platforms", platforms)
            raw.setdefault("tool_point_input", tool_diag)
            if tool_points:
                raw.setdefault("tool_points", tool_points)
            sel = raw.get("selection_prompt")
            if isinstance(sel, str) and sel.strip():
                notes: list[str] = []
                if isinstance(diag, dict) and diag.get("category_match"):
                    notes.append(
                        "Platform-point note: when target is a platform category, use target_instance_id as the platform "
                        "index (e.g. target='Side Table', target_instance_id=3) to expand to 'Side Table-3'."
                    )
                if isinstance(tool_diag, dict) and tool_diag.get("category_match"):
                    notes.append(
                        "Tool-point note: when target is a tool interaction point category, use target_instance_id as the "
                        "category index (e.g. target='Stove Switch', target_instance_id=1) to expand to 'Stove Switch 1'. "
                        "If the category has index_ranges=null, omit target_instance_id and pass only target."
                    )
                if notes:
                    raw["selection_prompt"] = sel.rstrip() + "\n\n" + "\n".join(notes)
            return ActionResult(False, raw=raw, error=str(err.error or ""))

        # Minimal deadlock guard: if the resolved target object is already the held item,
        # navigation is semantically invalid and should fail fast before touching A*.
        if kind_required is None and items_mode:
            held = _find_held_item(items_mode)
            if isinstance(held, dict):
                held_iid = held.get("instance_id")
                resolved_iid = getattr(resolved, "instance_id", None)
                try:
                    same_held_target = (
                        held_iid is not None
                        and resolved_iid is not None
                        and int(held_iid) == int(resolved_iid)
                    )
                except Exception:
                    same_held_target = False
                if same_held_target:
                    return ActionResult(
                        success=False,
                        error="navigation_target_already_held",
                        raw={
                            "target": str(args.target),
                            "resolved_target": target,
                            "requested_target_instance_id": args.target_instance_id,
                            "resolved_target_instance_id": resolved_iid,
                            "held_item": item_display_name(held),
                            "held_item_instance_id": held_iid,
                            "hint": (
                                "The requested navigation target is already in hand. "
                                "Do not auto_navigation to the held object."
                            ),
                        },
                    )

        navigator = SimpleAstarRadarNavigator(
            map_file=str(map_file),
            radar_file_path=str(radar_path),
            docking_distance_m=float(args.docking_distance_m),
            docking_distance_tolerance_m=float(args.docking_distance_tolerance_m),
            internal_rect_strategy=str(args.internal_rect_strategy or "normalized_ratio"),
        )
        # Ensure radar scan is available; upstream expects F11 to be enabled.
        try:
            init_ok = bool(navigator.initialize())
        except Exception:
            init_ok = False
        if not init_ok:
            # Best-effort retry: toggle/refresh F11 and re-initialize (navigator checks file contents).
            try:
                ensure_f11_radar_scan(
                    userdata_root=ud,
                    window_title="CookingSimulator",
                    activate_window=activate_window,
                    io_controller=io,
                    wait_after_press_s=0.9,
                )
                init_ok = bool(navigator.initialize())
            except Exception:
                init_ok = False

        if not init_ok:
            return ActionResult(
                success=False,
                error=(
                    "Radar not ready. Make sure the C# mod is running and press F11 in-game to enable "
                    "`realtime_radar_scan.txt`."
                ),
                raw={"target": args.target, "map_file": map_file, "preflight": {"f12_ok": bool(f12_ok), "f11_ok": bool(f11_ok), "alt_j_ok": bool(alt_j_ok)}},
            )

        buf = _pyio.StringIO()
        with redirect_stdout(buf):
            ok = bool(
                navigator.navigate_to_object(
                    object_name=target,
                    object_instance_id=(resolved.instance_id if resolved is not None else None),
                    auto_activate_window=bool(args.auto_activate_window),
                )
            )
    else:
        return ActionResult(
            success=False,
            error=f"Unknown auto_navigation impl: {args.impl!r}",
            raw={"supported": ["simple_astar_radar"]},
        )
    raw = {
        "target": args.target,
        "resolved_target": target,
        "requested_target_instance_id": args.target_instance_id,
        "resolved_target_instance_id": (resolved.instance_id if resolved is not None else None),
        "map_file": map_file,
        "preflight": {"f12_ok": bool(f12_ok), "f11_ok": bool(f11_ok), "alt_j_ok": bool(alt_j_ok)},
        "docking_config": {
            "docking_distance_m": float(args.docking_distance_m),
            "docking_distance_tolerance_m": float(args.docking_distance_tolerance_m),
            "internal_rect_strategy": str(args.internal_rect_strategy or "normalized_ratio"),
        },
    }
    nav_stdout = buf.getvalue().strip()
    if nav_stdout:
        raw["navigator_stdout_tail"] = _tail_lines(nav_stdout)
    requested_docking_distance_m = getattr(navigator, "last_requested_docking_distance_m", None)
    if requested_docking_distance_m is not None:
        try:
            raw["requested_docking_distance_m"] = float(requested_docking_distance_m)
        except Exception:
            pass
    chosen_docking_distance_m = getattr(navigator, "last_chosen_docking_distance_m", None)
    if chosen_docking_distance_m is not None:
        try:
            raw["chosen_docking_distance_m"] = float(chosen_docking_distance_m)
        except Exception:
            pass
    try:
        docking_debug = getattr(getattr(navigator, "pathfinder", None), "last_docking_debug", None)
        if isinstance(docking_debug, dict) and docking_debug:
            raw["docking_debug"] = docking_debug
    except Exception:
        pass
    try:
        path_debug = getattr(getattr(navigator, "pathfinder", None), "last_path_debug", None)
        if isinstance(path_debug, dict) and path_debug:
            raw["path_debug"] = path_debug
    except Exception:
        pass
    try:
        plan_docking_debug = getattr(navigator, "last_plan_docking_debug", None)
        if isinstance(plan_docking_debug, dict) and plan_docking_debug:
            raw["plan_docking_debug"] = plan_docking_debug
    except Exception:
        pass
    try:
        container_docking_debug = getattr(navigator, "last_container_docking_debug", None)
        if isinstance(container_docking_debug, dict) and container_docking_debug:
            raw["container_docking_debug"] = container_docking_debug
    except Exception:
        pass
    try:
        precise_adjust_debug = getattr(navigator, "last_precise_adjust_debug", None)
        if isinstance(precise_adjust_debug, dict) and precise_adjust_debug:
            raw["precise_adjust_debug"] = precise_adjust_debug
    except Exception:
        pass
    raw["entered_precise_adjust"] = bool(raw.get("precise_adjust_debug"))
    # Always attach placement occupancy snapshot when realtime items are readable,
    # so prompt injection is continuously refreshed across steps.
    try:
        data_all = read_realtime_products(realtime_products_json())
        items_all = extract_items(data_all)
        if items_all:
            raw["put_place_occupancy_snapshot"] = _build_put_place_occupancy_snapshot(items=items_all)
    except Exception as e:
        raw["put_place_occupancy_error"] = str(e)

    if ok and kind_required == "platform_point":
        try:
            data = read_realtime_products(realtime_products_json())
            items = extract_items(data)
        except Exception as e:
            items = []
            raw["occupied_check_error"] = str(e)
        resolved_item = _pick_resolved_item(resolved, target_name=str(target))
        occupied, occ_raw = _platform_point_occupied_with_items(
            items=items,
            target_name=str(target),
            target_instance_id=(resolved.instance_id if resolved is not None else None),
            resolved_item=resolved_item,
        )
        raw.update(occ_raw)
        if items:
            raw["put_place_occupancy_snapshot"] = _build_put_place_occupancy_snapshot(items=items)
    error_code = ""
    if not ok:
        error_code, signals = _classify_navigation_failure_from_raw(raw, nav_stdout)
        raw["navigation_failure_signals"] = signals
        if signals:
            raw["navigation_failure_summary"] = "; ".join(signals)
        else:
            raw["navigation_failure_summary"] = error_code
        container_debug = raw.get("container_docking_debug")
        chosen_distance = raw.get("chosen_docking_distance_m")
        if isinstance(container_debug, dict):
            try:
                if chosen_distance is None:
                    chosen_distance = float(container_debug.get("actual_docking_distance_m"))
                else:
                    chosen_distance = float(chosen_distance)
            except Exception:
                chosen_distance = None
        if (
            bool(raw.get("entered_precise_adjust"))
            and isinstance(container_debug, dict)
            and chosen_distance is not None
            and float(chosen_distance) > 0.30
        ):
            alt_j_text = _current_alt_j_snapshot_text()
            if alt_j_text:
                raw["navigation_failure_alt_j_snapshot"] = alt_j_text
            raw["navigation_failure_feedback_cn"] = (
                (
                    f"当前交互信息：\n{alt_j_text}\n\n"
                    if alt_j_text
                    else ""
                )
                + "如果想拿起容器中的物品，但拿取失败，可能是因为容器放置位置有些远，"
                "容器本身挡住了容器内物体，导致导航虽然已经接近，但仍难以稳定对准并交互到容器中的目标物品。"
                "推荐把该容器重新放置到更靠近可通行区域、且容器开口更容易正对玩家的位置，再尝试拿取其中的物品。"
            )
            raw["navigation_failure_feedback_en"] = (
                (
                    f"Current interaction info:\n{alt_j_text}\n\n"
                    if alt_j_text
                    else ""
                )
                + "If picking up an item inside the container fails, the container may be placed a bit too far "
                "from reachable walkable space, and the container body may be occluding the item inside. "
                "Even if navigation gets close, alignment and interaction can still be unreliable. "
                "Move the container closer to reachable space and place it so the opening is easier to face, "
                "then try picking up the item again."
            )
    posture_info = getattr(navigator, "last_posture_decision", None)
    if isinstance(posture_info, dict) and posture_info:
        raw["posture_decision"] = posture_info
    return ActionResult(
        success=ok,
        error="" if ok else error_code,
        raw=raw,
    )
