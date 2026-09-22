# -*- coding: utf-8 -*-
"""
Simple A* + Radar Navigator v3.8.2 (vendored) — EPM-adapted.

Upstream file: `obtain_camera_info/auto_navigation/simple_astar_radar_navigator.py`

Key adaptations:
- Removed `sys.path` hacks and bare-module imports.
- Uses `epm.cerebellum.local_actions` for keyboard/mouse actions.
- Uses `epm.vision.screen_capture.activate_window` for focusing the game window.
"""

from __future__ import annotations

import os
import math
import time
import ctypes
import json
from pathlib import Path
from typing import List, Optional, Tuple

from epm.vision.screen_capture import activate_window as _activate_window
from epm.cerebellum.realtime_products import extract_items, item_display_name, read_realtime_products
from epm.cerebellum.skills._shared_paths import userdata_root

from .astar_pathfinder import AStarPathfinder
from .camera_projection import CameraProjection
from .data_loader import DataLoader
from .realtime_radar_reader import RealtimeRadarReader

try:
    from epm.cerebellum import local_actions

    LOCAL_ACTIONS_AVAILABLE = True
except Exception:  # pragma: no cover
    LOCAL_ACTIONS_AVAILABLE = False
    local_actions = None  # type: ignore

annotate_target_on_screenshot = None  # loaded lazily when needed to avoid import-time side effects
ENABLE_CAPTURE_AND_ANNOTATE = False  # disable screenshot/annotation to avoid import/runtime issues

DEFAULT_DOCKING_DISTANCE_M = 1.5
DEFAULT_DOCKING_DISTANCE_TOLERANCE_M = 0.2
CONTAINER_CLOSE_DOCKING_REQUEST_M = 0.30
CONTAINER_DOCKING_TOO_FAR_THRESHOLD_M = 0.30

# Try to import visualizer (optional; kept disabled by default).
try:  # pragma: no cover
    from .realtime_navigator_visualizer import init_visualizer, update_navigation_state  # type: ignore

    VISUALIZER_AVAILABLE = False  # disabled (threading issues upstream)
except Exception:  # pragma: no cover
    VISUALIZER_AVAILABLE = False


class SimpleAstarRadarNavigator:
    """
    Simple A* + Radar Navigator

    Features:
    1. A* pathfinding for global path planning
    2. Real-time radar for local obstacle avoidance
    3. 8-direction movement (WASD combinations)
    4. Simple decision tree for obstacle avoidance
    """

    _FRIDGE_STAND_TARGETS = {
        "lemon",
        "lime",
        "radish",
        "mozzarella",
        "egg",
        "strawberry",
        "asparagus",
        "brussels sprouts",
    }

    def __init__(
        self,
        map_file: str,
        radar_file_path: str | None = None,
        move_duration: float = 0.20,  # v3.7.3: 从0.08提高到0.12，步子更大
        turn_sensitivity: int = 30,
        arrival_threshold: float = 0.25,  # v3.7.3: 从5cm放宽到20cm
        waypoint_threshold: float = 1.5,
        max_attempts: int = 500,
        safe_distance: float = 1.0,
        docking_distance_m: float = DEFAULT_DOCKING_DISTANCE_M,
        docking_distance_tolerance_m: float = DEFAULT_DOCKING_DISTANCE_TOLERANCE_M,
        internal_rect_strategy: str = "normalized_ratio",
    ):
        """Initialize navigator"""
        self.move_duration = move_duration
        self.turn_sensitivity = turn_sensitivity
        self.arrival_threshold = arrival_threshold
        self.waypoint_threshold = waypoint_threshold
        self.max_attempts = max_attempts
        self.safe_distance = safe_distance
        self.map_file = map_file
        self.default_docking_distance_m = float(docking_distance_m)
        self.docking_distance_m = float(docking_distance_m)
        self.docking_distance_tolerance_m = float(docking_distance_tolerance_m)
        self.internal_rect_strategy = str(internal_rect_strategy or "normalized_ratio").strip() or "normalized_ratio"

        # Components
        self.data_loader = DataLoader()
        self.radar_reader = RealtimeRadarReader(radar_file_path=radar_file_path)
        self.pathfinder = AStarPathfinder()
        self.camera_projection = CameraProjection()  # 相机投影计算器

        # State
        self.current_path: List[Tuple[float, float]] = []
        self.current_waypoint_index: int = 0
        self.final_target: Optional[Tuple[float, float]] = None
        self.is_crouching: bool = False  # 蹲下状态标记
        self.last_posture_decision: dict = {}
        self.last_precise_adjust_debug: dict = {}
        self.last_container_docking_debug: dict = {}
        self.last_plan_docking_debug: dict = {}
        self.last_forward_settle_debug: dict = {}
        self.last_requested_docking_distance_m: float = float(docking_distance_m)
        self.last_chosen_docking_distance_m: float | None = None

        # Load map
        if not self.pathfinder.load_map(map_file):
            raise Exception(f"Failed to load map: {map_file}")

    def _require_valid_pathfinder(self, *, caller: str) -> AStarPathfinder:
        pathfinder = getattr(self, "pathfinder", None)
        if isinstance(pathfinder, AStarPathfinder):
            return pathfinder
        raise TypeError(
            "[NAV] invalid_pathfinder_reference:"
            f" caller={caller}"
            f" pathfinder_type={type(pathfinder).__name__}"
            f" pathfinder_repr={pathfinder!r}. "
            "Expected `self.pathfinder` to be an AStarPathfinder instance. "
            "Likely causes: assigning `AStarPathfinder` instead of `AStarPathfinder()`, "
            "overwriting `self.pathfinder`, or passing an unbound method somewhere upstream."
        )

    @staticmethod
    def _norm_text(value: object) -> str:
        return str(value or "").strip().lower()

    @staticmethod
    def _bounds_contains_point(item: dict, point: Tuple[float, float, float], *, margin: float = 0.0) -> bool:
        bmin = item.get("bounds_min")
        bmax = item.get("bounds_max")
        if not isinstance(bmin, dict) or not isinstance(bmax, dict):
            return False
        try:
            px, py, pz = float(point[0]), float(point[1]), float(point[2])
            return (
                float(bmin.get("x")) - margin <= px <= float(bmax.get("x")) + margin
                and float(bmin.get("y")) - margin <= py <= float(bmax.get("y")) + margin
                and float(bmin.get("z")) - margin <= pz <= float(bmax.get("z")) + margin
            )
        except Exception:
            return False

    @staticmethod
    def _bbox_volume(item: dict) -> float:
        bmin = item.get("bounds_min")
        bmax = item.get("bounds_max")
        if not isinstance(bmin, dict) or not isinstance(bmax, dict):
            return float("inf")
        try:
            dx = max(0.0, float(bmax.get("x")) - float(bmin.get("x")))
            dy = max(0.0, float(bmax.get("y")) - float(bmin.get("y")))
            dz = max(0.0, float(bmax.get("z")) - float(bmin.get("z")))
            return dx * dy * dz
        except Exception:
            return float("inf")

    def _load_realtime_items(self) -> list[dict]:
        try:
            data = read_realtime_products(Path(self.data_loader.realtime_products_file))
            items = extract_items(data)
            return items if isinstance(items, list) else []
        except Exception:
            return []

    def _is_open_container_like(self, item: dict) -> bool:
        kind = self._norm_text(item.get("kind"))
        text = " ".join(
            self._norm_text(item.get(k))
            for k in ("name_en", "name_cn", "game_object", "name")
        )
        if kind == "container":
            return True
        return any(
            token in text
            for token in (
                "pot",
                "pan",
                "bowl",
                "tray",
                "plate",
                "cup",
                "mug",
                "glass",
                "jar",
                "wok",
                "skillet",
                "saucepan",
                "container",
            )
        )

    def _should_use_container_contents_navigation(self, item: dict | None) -> bool:
        if not isinstance(item, dict):
            return False
        kind = self._norm_text(item.get("kind"))
        if kind in {"product", "products", "liquid", "liquids", "spice", "spices"}:
            return True
        text = " ".join(
            self._norm_text(item.get(k))
            for k in ("name_en", "name_cn", "game_object", "name")
        )
        return any(
            token in text
            for token in (
                "food processor container",
                "liquid bottle",
                "seasoning",
                "spice jar",
                "spice bottle",
                "sauce bottle",
                "oil bottle",
                "vinegar bottle",
            )
        )

    def _find_container_bbox_close_docking(
        self,
        *,
        target_obj,
        object_name: str,
        object_instance_id: int | None,
    ) -> dict | None:
        items = self._load_realtime_items()
        if not items:
            return None

        target_item = None
        target_instance_id = object_instance_id
        if target_instance_id is None:
            try:
                target_instance_id = int(getattr(target_obj, "instance_id", 0) or 0)
            except Exception:
                target_instance_id = None

        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                if target_instance_id is not None and int(item.get("instance_id")) == int(target_instance_id):
                    target_item = item
                    break
            except Exception:
                continue

        if target_item is None:
            want_names = {
                self._norm_text(object_name),
                self._norm_text(getattr(target_obj, "name_en", "")),
                self._norm_text(getattr(target_obj, "name_cn", "")),
                self._norm_text(self.data_loader.resolve_alias(object_name)),
            }
            best_plane_dist = float("inf")
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_names = {
                    self._norm_text(item.get("name_en")),
                    self._norm_text(item.get("name_cn")),
                    self._norm_text(item.get("game_object")),
                    self._norm_text(item.get("name")),
                }
                if not want_names.intersection({n for n in item_names if n}):
                    continue
                try:
                    pos = item.get("position") or {}
                    dx = float(pos.get("x")) - float(target_obj.position[0])
                    dz = float(pos.get("z")) - float(target_obj.position[2])
                    plane_dist = math.sqrt(dx * dx + dz * dz)
                except Exception:
                    plane_dist = float("inf")
                if plane_dist < best_plane_dist:
                    best_plane_dist = plane_dist
                    target_item = item

        if not isinstance(target_item, dict):
            return None

        if not self._should_use_container_contents_navigation(target_item):
            return None

        target_pos = (
            float(target_obj.position[0]),
            float(target_obj.position[1]),
            float(target_obj.position[2]),
        )
        target_container_name = self._norm_text(target_item.get("container"))
        container_candidates: list[tuple[int, float, dict]] = []

        for item in items:
            if not isinstance(item, dict) or item is target_item:
                continue
            if not self._is_open_container_like(item):
                continue
            if not self._bounds_contains_point(item, target_pos, margin=0.01):
                continue

            item_names = {
                self._norm_text(item.get("name_en")),
                self._norm_text(item.get("name_cn")),
                self._norm_text(item.get("game_object")),
                self._norm_text(item.get("name")),
            }
            matches_container_name = (
                target_container_name not in {"", "scene", "refrigerator"}
                and target_container_name in {n for n in item_names if n}
            )
            container_candidates.append((0 if matches_container_name else 1, self._bbox_volume(item), item))

        if not container_candidates:
            return None

        _, _, container_item = min(container_candidates, key=lambda x: (x[0], x[1]))
        try:
            container_instance_id = int(container_item.get("instance_id"))
        except Exception:
            container_instance_id = None

        return {
            "mode": "container_bbox_close_docking",
            "target_name": str(getattr(target_obj, "name_en", "") or object_name),
            "target_instance_id": target_instance_id,
            "container_name": item_display_name(container_item),
            "container_instance_id": container_instance_id,
            "container_bounds_min": container_item.get("bounds_min"),
            "container_bounds_max": container_item.get("bounds_max"),
            "requested_docking_distance_m": CONTAINER_CLOSE_DOCKING_REQUEST_M,
            "too_far_feedback_threshold_m": CONTAINER_DOCKING_TOO_FAR_THRESHOLD_M,
        }

    def _extract_last_chosen_docking_distance_m(self) -> float | None:
        try:
            docking_debug = getattr(self.pathfinder, "last_docking_debug", None)
            chosen = docking_debug.get("chosen") if isinstance(docking_debug, dict) else None
            if isinstance(chosen, dict) and chosen.get("target_distance_m") is not None:
                return float(chosen.get("target_distance_m"))
        except Exception:
            pass
        return None

    def _current_container_bounds_for_docking(self) -> tuple[dict | None, dict | None]:
        info = self.last_container_docking_debug if isinstance(self.last_container_docking_debug, dict) else {}
        bounds_min = info.get("container_bounds_min")
        bounds_max = info.get("container_bounds_max")
        if isinstance(bounds_min, dict) and isinstance(bounds_max, dict):
            return bounds_min, bounds_max
        return None, None

    def _interaction_target_name_candidates(self, target_obj=None) -> list[str]:
        names: set[str] = set()

        def _add(value: object) -> None:
            norm = self._norm_text(value)
            if norm:
                names.add(norm)

        target_name = getattr(self, "target_name", "")
        _add(target_name)
        try:
            aliases = getattr(self.data_loader, "aliases", {}) or {}
            if isinstance(aliases, dict):
                target_name_raw = str(target_name or "").strip()
                target_name_lower = target_name_raw.lower()
                _add(aliases.get(target_name_raw))
                _add(aliases.get(target_name_lower))
                _add(aliases.get(target_name_raw.replace("_", " ")))
                _add(aliases.get(target_name_lower.replace("_", " ")))
        except Exception:
            pass

        if target_obj is not None:
            _add(getattr(target_obj, "name_en", ""))
            _add(getattr(target_obj, "name_cn", ""))

        return sorted(names)

    def _read_target_interaction_state(self, *, target_obj=None, max_age_s: float = 2.5) -> dict:
        debug: dict = {
            "snapshot_available": False,
            "target_name_candidates": self._interaction_target_name_candidates(target_obj),
            "matched": False,
            "matched_any": False,
            "matched_fields": [],
            "matched_name": "",
            "match_type": "none",
        }

        try:
            from epm.cerebellum.skills.auto_navigation.interaction_detector import (
                get_udp_debug_snapshot,
                init_udp_mode,
                read_interaction_info,
            )
        except Exception as exc:
            debug["read_error"] = f"{exc.__class__.__name__}: {exc}"
            return debug

        latest_age_s = None
        udp_running = False
        try:
            init_udp_mode()
            udp_snapshot = get_udp_debug_snapshot()
            if isinstance(udp_snapshot, dict):
                udp_running = bool(udp_snapshot.get("running"))
                age = udp_snapshot.get("latest_age_s", None)
                latest_age_s = (float(age) if age is not None else None)
                debug["udp_running"] = udp_running
                debug["udp_latest_age_s"] = latest_age_s
        except Exception as exc:
            debug["udp_debug_error"] = f"{exc.__class__.__name__}: {exc}"

        info = None
        source = "unavailable"
        if udp_running and latest_age_s is not None and 0.0 <= latest_age_s <= float(max_age_s):
            try:
                info = read_interaction_info(use_udp=True)
                source = "udp"
            except Exception as exc:
                debug["udp_read_error"] = f"{exc.__class__.__name__}: {exc}"

        if info is None:
            try:
                from epm.cerebellum.interaction_snapshot import read_interaction_snapshot

                snap = read_interaction_snapshot(userdata_root=userdata_root(), max_age_s=max_age_s)
                if snap is not None:
                    info = snap
                    source = "file"
            except Exception as exc:
                debug["file_read_error"] = f"{exc.__class__.__name__}: {exc}"

        if info is None:
            return debug

        item_name = self._norm_text(getattr(info, "item_name", ""))
        container_name = self._norm_text(getattr(info, "container_name", ""))
        container_contents = self._norm_text(getattr(info, "container_contents", ""))
        action = str(getattr(info, "action", "") or "").strip()
        matched_fields: list[str] = []
        matched_name = ""
        match_type = "none"

        for candidate in debug["target_name_candidates"]:
            if not candidate:
                continue
            if item_name and candidate in item_name:
                matched_fields.append("item_name")
                matched_name = candidate
                match_type = "direct_item"
                break
            if container_contents and candidate in container_contents:
                matched_fields.append("container_contents")
                matched_name = candidate
                match_type = "container_contents_only"
                break

        debug.update(
            {
                "snapshot_available": True,
                "source": source,
                "has_target": bool(getattr(info, "has_target", False)),
                "item_name": str(getattr(info, "item_name", "") or "").strip(),
                "container_name": str(getattr(info, "container_name", "") or "").strip(),
                "container_contents": str(getattr(info, "container_contents", "") or "").strip(),
                "action": action,
                "timestamp": str(getattr(info, "timestamp", "") or "").strip(),
                "matched": bool(match_type == "direct_item"),
                "matched_any": bool(matched_fields),
                "matched_fields": matched_fields,
                "matched_name": matched_name,
                "match_type": match_type,
                "raw_item_name_norm": item_name,
                "raw_container_name_norm": container_name,
                "raw_container_contents_norm": container_contents,
            }
        )
        return debug

    def _ensure_alt_j_interaction_enabled(self, *, verbose: bool = False) -> bool:
        try:
            from epm.cerebellum.game_hotkeys import ensure_alt_j_interaction
            from epm.cerebellum.raw_input_controller import RawInputController
        except Exception as exc:
            if verbose:
                print(f"            [Alt+J] ensure unavailable: {exc.__class__.__name__}: {exc}")
            return False

        try:
            io = RawInputController()
            ok = ensure_alt_j_interaction(
                userdata_root=userdata_root(),
                window_title="CookingSimulator",
                activate_window=_activate_window,
                io_controller=io,
                verbose=verbose,
            )
            if verbose:
                print(f"            [Alt+J] ensure {'OK' if ok else 'FAILED'}")
            return bool(ok)
        except Exception as exc:
            if verbose:
                print(f"            [Alt+J] ensure failed: {exc.__class__.__name__}: {exc}")
            return False

    def _probe_target_interaction_near_center(
        self,
        *,
        target_obj=None,
        probe_pixels: int = 35,
        settle_s: float = 0.05,
        step_pixels: int = 1,
        hold_after_detect_s: float = 0.45,
        max_raw_pixels_per_direction: int = 180,
        max_age_s: float = 0.8,
    ) -> dict:
        debug: dict = {
            "attempted": True,
            "probe_pixels": int(probe_pixels),
            "probe_pixels_semantics": "screen_space_target_projection_delta",
            "settle_s": round(float(settle_s), 3),
            "step_pixels": int(step_pixels),
            "hold_after_detect_s": round(float(hold_after_detect_s), 3),
            "max_raw_pixels_per_direction": int(max_raw_pixels_per_direction),
            "steps": [],
            "success": False,
            "stop_reason": "not_detected_after_probe",
            "snapshot_available": False,
        }

        target_world_pos = None
        try:
            if target_obj is not None and getattr(target_obj, "position", None) is not None:
                target_world_pos = (
                    float(target_obj.position[0]),
                    float(target_obj.position[1]),
                    float(target_obj.position[2]),
                )
        except Exception:
            target_world_pos = None

        def _current_target_projection():
            if target_world_pos is None:
                return None
            camera_info = self._load_camera_info_with_retry()
            if not camera_info:
                return None
            return self._project_to_screen_v2(target_world_pos, camera_info)

        base_projection = _current_target_projection()
        if base_projection is not None:
            debug["base_projection"] = [int(base_projection[0]), int(base_projection[1])]

        initial_state = self._read_target_interaction_state(target_obj=target_obj, max_age_s=max_age_s)
        debug["initial_state"] = initial_state
        debug["snapshot_available"] = bool(initial_state.get("snapshot_available"))

        if not LOCAL_ACTIONS_AVAILABLE:
            debug["stop_reason"] = "local_actions_unavailable"
            return debug

        debug["alt_j_ensure_ok"] = bool(self._ensure_alt_j_interaction_enabled(verbose=True))
        initial_state_after_ensure = self._read_target_interaction_state(target_obj=target_obj, max_age_s=max_age_s)
        debug["initial_state_after_ensure"] = initial_state_after_ensure
        debug["snapshot_available"] = bool(
            debug["snapshot_available"] or initial_state_after_ensure.get("snapshot_available")
        )
        if initial_state.get("matched"):
            print("            [Alt+J] centered state already matches target; continue directional probe anyway")
        elif initial_state.get("matched_any"):
            print(
                "            [Alt+J] centered state only sees container/contents, "
                "not direct target item; continue probing"
            )
        elif initial_state_after_ensure.get("matched"):
            print("            [Alt+J] target visible after ensure; continue directional probe anyway")
        elif initial_state_after_ensure.get("matched_any"):
            print(
                "            [Alt+J] after ensure only container/contents mention target, "
                "not direct item; continue probing"
            )

        probes = [
            ("left", local_actions.look_left, local_actions.look_right),
            ("right", local_actions.look_right, local_actions.look_left),
            ("up", local_actions.look_up, local_actions.look_down),
            ("down", local_actions.look_down, local_actions.look_up),
        ]

        for direction, move_fn, undo_fn in probes:
            step_debug = {
                "direction": direction,
                "probe_pixels": int(probe_pixels),
                "step_pixels": int(step_pixels),
                "substeps": [],
            }
            print(
                "            [Alt+J] begin directional probe: "
                f"direction={direction}, target_screen_delta={probe_pixels}px"
            )
            moved_raw_total = 0
            while moved_raw_total < max(1, int(max_raw_pixels_per_direction)):
                current_step = min(
                    max(1, int(step_pixels)),
                    max(1, int(max_raw_pixels_per_direction)) - moved_raw_total,
                )
                move_fn(pixels=current_step)
                time.sleep(settle_s)
                state_after_probe = self._read_target_interaction_state(target_obj=target_obj, max_age_s=max_age_s)
                projection_after_probe = _current_target_projection()
                axis_index = 0 if direction in ("left", "right") else 1
                axis_delta = None
                if base_projection is not None and projection_after_probe is not None:
                    axis_delta = abs(int(projection_after_probe[axis_index]) - int(base_projection[axis_index]))
                substep_debug = {
                    "phase": "probe",
                    "pixels": int(current_step),
                    "state": state_after_probe,
                    "projection": (
                        None
                        if projection_after_probe is None
                        else [int(projection_after_probe[0]), int(projection_after_probe[1])]
                    ),
                    "screen_axis_delta_px": (None if axis_delta is None else int(axis_delta)),
                }
                step_debug["substeps"].append(substep_debug)
                debug["snapshot_available"] = bool(
                    debug["snapshot_available"] or state_after_probe.get("snapshot_available")
                )
                if state_after_probe.get("matched"):
                    debug["success"] = True
                    debug["stop_reason"] = "detected_after_probe"
                    debug["matched_direction"] = direction
                    debug["matched_state"] = state_after_probe
                    print(
                        "            [Alt+J] detected target during probe: "
                        f"direction={direction}, step_pixels={current_step}"
                    )
                    time.sleep(hold_after_detect_s)
                    debug["steps"].append(step_debug)
                    return debug
                moved_raw_total += current_step
                if axis_delta is not None and axis_delta >= int(probe_pixels):
                    print(
                        "            [Alt+J] reached probe screen delta: "
                        f"direction={direction}, screen_delta={axis_delta}px, raw_total={moved_raw_total}"
                    )
                    break

            step_debug["raw_probe_total"] = int(moved_raw_total)
            remaining_reset = int(moved_raw_total)
            while remaining_reset > 0:
                current_step = min(max(1, int(step_pixels)), remaining_reset)
                undo_fn(pixels=current_step)
                time.sleep(settle_s)
                state_after_reset = self._read_target_interaction_state(target_obj=target_obj, max_age_s=max_age_s)
                projection_after_reset = _current_target_projection()
                axis_index = 0 if direction in ("left", "right") else 1
                axis_delta = None
                if base_projection is not None and projection_after_reset is not None:
                    axis_delta = abs(int(projection_after_reset[axis_index]) - int(base_projection[axis_index]))
                substep_debug = {
                    "phase": "reset",
                    "pixels": int(current_step),
                    "state": state_after_reset,
                    "projection": (
                        None
                        if projection_after_reset is None
                        else [int(projection_after_reset[0]), int(projection_after_reset[1])]
                    ),
                    "screen_axis_delta_px": (None if axis_delta is None else int(axis_delta)),
                }
                step_debug["substeps"].append(substep_debug)
                debug["snapshot_available"] = bool(
                    debug["snapshot_available"] or state_after_reset.get("snapshot_available")
                )
                if state_after_reset.get("matched"):
                    debug["success"] = True
                    debug["stop_reason"] = "detected_during_reset"
                    debug["matched_direction"] = f"{direction}_reset"
                    debug["matched_state"] = state_after_reset
                    print(
                        "            [Alt+J] detected target during reset: "
                        f"direction={direction}, step_pixels={current_step}"
                    )
                    time.sleep(hold_after_detect_s)
                    debug["steps"].append(step_debug)
                    return debug
                remaining_reset -= current_step

            debug["steps"].append(step_debug)

        return debug

    def initialize(self) -> bool:
        """Initialize navigator"""
        print("\n" + "=" * 70)
        print("  简化版A*+雷达导航器 - 初始化")
        print("=" * 70)

        # Check realtime radar data
        print(f"\n[1/2] 检查实时雷达数据: {self.radar_reader.radar_file_path}")
        if not os.path.exists(self.radar_reader.radar_file_path):
            print("   ERROR 实时雷达数据文件不存在")
            return False
        print("   OK 实时雷达数据文件已存在")

        # Test radar reading
        print(f"\n[2/2] 测试读取雷达数据...")
        readings = self.radar_reader.get_all_readings()
        if readings is None or len(readings) == 0:
            print("   ERROR 无法读取雷达数据")
            return False
        print(f"   OK 雷达数据正常（{len(readings)}个方向）")

        # Initialize visualizer
        if VISUALIZER_AVAILABLE:
            print(f"\n[3/3] 启动实时可视化...")
            try:
                init_visualizer(self.map_file)
                print("   OK 可视化窗口已启动")
            except Exception as e:
                print(f"   WARNING 可视化启动失败: {e}")

        print("\n" + "=" * 70)
        print("  初始化完成！")
        print("=" * 70)
        return True

    def _read_posture_state(self) -> str:
        """Best-effort read of current posture based on Ctrl key state."""
        try:
            ctrl_down = bool(ctypes.windll.user32.GetAsyncKeyState(0x11) & 0x8000)
            return "crouching" if ctrl_down else "standing"
        except Exception:
            return "unknown"

    def _target_name_norm(self) -> str:
        return str(getattr(self, "target_name", "") or "").strip().lower()

    def _is_knife_target(self) -> bool:
        raw_name = str(getattr(self, "target_name", "") or "")
        norm_name = raw_name.strip().lower()
        return ("knife" in norm_name) or ("刀" in raw_name)

    def _center_tolerance_pixels_for_target(self, default_pixels: float = 10.0) -> float:
        return 2.0 if self._is_knife_target() else float(default_pixels)

    def _fixed_docking_pose_for_special_target(
        self,
        target_name_norm: str,
        target_pos: Tuple[float, float],
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        if target_name_norm in ("checkout stand", "checkout_stand", "checkout"):
            ideal_pos = (target_pos[0] - self.docking_distance_m, target_pos[1])
            yaw_dx = target_pos[0] - ideal_pos[0]
            yaw_dz = target_pos[1] - ideal_pos[1]
            ideal_yaw = math.degrees(math.atan2(yaw_dx, yaw_dz)) % 360
            print(f"   OK Checkout Stand fixed docking point: ({ideal_pos[0]:.2f}, {ideal_pos[1]:.2f})")
            print("      Fixed direction: (-1, 0)")
            return (ideal_pos, ideal_yaw)

        fridge_front_targets = {
            "refrigerator right door",
            "refrigerator left door",
            "refrigerator left drawer",
            "refrigerator right drawer",
        }
        target_in_refrigerator = False
        _tname = getattr(self, "target_name", "") or ""
        _tid = getattr(self, "target_instance_id", None)
        try:
            if _tid is not None:
                target_obj = self.data_loader.find_object_by_name_and_instance_id(str(_tname), int(_tid))
            else:
                target_obj = self.data_loader.find_object_by_name(str(_tname))
            target_in_refrigerator = bool(target_obj and getattr(target_obj, "layer", None) == "Refrigerator")
        except Exception:
            target_in_refrigerator = False

        if target_name_norm not in fridge_front_targets and not target_in_refrigerator:
            return None

        fridge_obj = self.data_loader.find_object_by_name("refrigerator")
        if fridge_obj:
            _, _, fridge_z = fridge_obj.position
            # For ingredients inside the refrigerator, keep approaching from the refrigerator-front side,
            # but measure docking distance relative to the concrete item rather than uniformly from the door/front plane.
            if target_in_refrigerator and target_name_norm not in fridge_front_targets:
                ideal_pos = (target_pos[0], target_pos[1] + self.docking_distance_m)
            else:
                ideal_pos = (target_pos[0], fridge_z + self.docking_distance_m)
            yaw_dx = target_pos[0] - ideal_pos[0]
            yaw_dz = target_pos[1] - ideal_pos[1]
            ideal_yaw = math.degrees(math.atan2(yaw_dx, yaw_dz)) % 360
            debug_label = (
                f"{target_name_norm} (in refrigerator)"
                if target_in_refrigerator and target_name_norm not in fridge_front_targets
                else target_name_norm
            )
            print(
                f"   OK Refrigerator front docking point: ({ideal_pos[0]:.2f}, {ideal_pos[1]:.2f}) "
                f"for {debug_label}"
            )
            print(f"      Refrigerator front z: {fridge_z:.2f}")
            if target_in_refrigerator and target_name_norm not in fridge_front_targets:
                print(
                    f"      Refrigerator item docking distance: {self.docking_distance_m:.2f}m "
                    f"(measured from item position along fridge-front direction)"
                )
            return (ideal_pos, ideal_yaw)

        ideal_pos = (target_pos[0], target_pos[1] + self.docking_distance_m)
        yaw_dx = target_pos[0] - ideal_pos[0]
        yaw_dz = target_pos[1] - ideal_pos[1]
        ideal_yaw = math.degrees(math.atan2(yaw_dx, yaw_dz)) % 360
        print(
            f"   WARNING Refrigerator object not found; using fallback front docking point: "
            f"({ideal_pos[0]:.2f}, {ideal_pos[1]:.2f})"
        )
        return (ideal_pos, ideal_yaw)

    def navigate_to_object(self, object_name: str, object_instance_id: int | None = None, auto_activate_window: bool = True) -> bool:
        """Navigate to target object"""
        # 保存目标名称用于截图
        self.target_name = object_name
        self.target_instance_id = int(object_instance_id) if object_instance_id is not None else None
        self.last_container_docking_debug = {}
        self.last_plan_docking_debug = {}
        self.last_precise_adjust_debug = {}
        self.last_forward_settle_debug = {}
        self.last_requested_docking_distance_m = float(self.default_docking_distance_m)
        self.last_chosen_docking_distance_m = None

        print("\n" + "=" * 70)
        print("  简化版A*+雷达导航器")
        print("=" * 70)

        # Activate game window
        print("\n[1/5] 激活游戏窗口...")
        if auto_activate_window and _activate_window:
            _activate_window(window_title="CookingSimulator")
        print("   OK")

        # Find target object
        print(f"\n[2/5] 查找目标物品: {object_name}")
        if object_instance_id is not None:
            target_obj = self.data_loader.find_object_by_name_and_instance_id(object_name, int(object_instance_id))
        else:
            target_obj = self.data_loader.find_object_by_name(object_name)
        if not target_obj:
            print(f"   ERROR 未找到目标物品: {object_name}")
            return False
        target_x, target_y, target_z = target_obj.position
        print(f"   OK 找到目标: {target_obj.name_cn or target_obj.name_en}")
        print(f"      位置: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")

        # 检查目标高度，低于50cm需要蹲下
        CROUCH_HEIGHT_THRESHOLD = 0.6  # 60cm
        FRIDGE_ITEM_DOCKING_DISTANCE_M = 1.3
        target_name_norm = (target_obj.name_en or object_name or "").strip().lower()
        fridge_front_targets = {
            "refrigerator right door",
            "refrigerator left door",
            "refrigerator left drawer",
            "refrigerator right drawer",
        }
        target_in_refrigerator = getattr(target_obj, "layer", None) == "Refrigerator"
        force_fridge_standing = (
            target_in_refrigerator
            and target_name_norm in self._FRIDGE_STAND_TARGETS
        )
        fridge_item_close_docking = (
            target_in_refrigerator
            and target_name_norm not in fridge_front_targets
            and not force_fridge_standing
        )
        if force_fridge_standing:
            self.docking_distance_m = 0.5
        elif fridge_item_close_docking:
            self.docking_distance_m = FRIDGE_ITEM_DOCKING_DISTANCE_M
        else:
            self.docking_distance_m = self.default_docking_distance_m
        container_docking_debug = self._find_container_bbox_close_docking(
            target_obj=target_obj,
            object_name=object_name,
            object_instance_id=object_instance_id,
        )
        if isinstance(container_docking_debug, dict):
            self.docking_distance_m = CONTAINER_CLOSE_DOCKING_REQUEST_M
            self.last_container_docking_debug = container_docking_debug
            print(
                "      Target is inside container bbox: "
                f"{container_docking_debug['container_name']}; "
                f"request close docking distance {self.docking_distance_m:.2f}m"
            )
        self.last_requested_docking_distance_m = float(self.docking_distance_m)
        posture = self._read_posture_state()
        if posture in ("crouching", "standing"):
            self.is_crouching = posture == "crouching"

        force_crouch = "oven bottom place point" in target_name_norm
        desired_posture = "standing" if force_fridge_standing else (
            "crouching" if (target_y < CROUCH_HEIGHT_THRESHOLD or force_crouch) else "standing"
        )
        posture_action = "none"
        if force_fridge_standing:
            print(
                f"      目标 {object_name} 在冰箱内，命中特殊站立规则，"
                f"使用 0.50m 理想停靠距离 (current_posture={posture})"
            )
            if LOCAL_ACTIONS_AVAILABLE and hasattr(local_actions, "stand_up") and self.is_crouching:
                local_actions.stand_up()
                self.is_crouching = False
                posture_action = "stand_up"
                print("      已站起（松开Ctrl）")
                time.sleep(0.3)
            else:
                print("      当前已是站立状态，无需执行 stand_up")
        elif target_y < CROUCH_HEIGHT_THRESHOLD or force_crouch:
            if force_crouch:
                print(f"      target {object_name} matched forced crouch rule for Oven Bottom place point (current_posture={posture})")
            else:
                print(f"      目标高度 {target_y*100:.0f}cm < {CROUCH_HEIGHT_THRESHOLD*100:.0f}cm，需要蹲下操作 (current_posture={posture})")
            if LOCAL_ACTIONS_AVAILABLE and hasattr(local_actions, "kneel_down") and not self.is_crouching:
                local_actions.kneel_down()
                self.is_crouching = True
                posture_action = "kneel_down"
                print("      已蹲下（按住Ctrl）")
                time.sleep(0.3)  # 等待蹲下动作完成
            else:
                print("      当前已是蹲下状态，无需重复执行 kneel_down")
        else:
            print(f"      目标高度 {target_y*100:.0f}cm >= {CROUCH_HEIGHT_THRESHOLD*100:.0f}cm，站立操作 (current_posture={posture})")
            if LOCAL_ACTIONS_AVAILABLE and hasattr(local_actions, "stand_up") and self.is_crouching:
                local_actions.stand_up()
                self.is_crouching = False
                posture_action = "stand_up"
                print("      已站起（松开Ctrl）")
                time.sleep(0.3)
            else:
                print("      当前已是站立状态，无需执行 stand_up")

        self.last_posture_decision = {
            "current_posture": posture,
            "desired_posture": desired_posture,
            "action": posture_action,
            "target_height_m": float(target_y),
            "threshold_m": float(CROUCH_HEIGHT_THRESHOLD),
        }

        self.final_target = (target_x, target_z)

        # Get current position
        print("\n[3/5] 获取当前位置...")
        camera_info = self.data_loader.load_camera_info()
        if not camera_info:
            print("   ERROR 无法加载相机信息")
            return False
        current_x, current_y, current_z = camera_info.position
        print(f"   OK 当前位置: ({current_x:.2f}, {current_z:.2f})")

        # Calculate ideal docking position (130cm from target, perpendicular to nearest obstacle)
        print("\n[3.5/5] 计算理想停靠位置...")

        target_name_norm = (object_name or "").strip().lower()
        pose_info = self._fixed_docking_pose_for_special_target(
            target_name_norm,
            (target_x, target_z),
        )

        prefer_closer_to = None
        if target_obj.layer == "Refrigerator":
            # 动态查询冰箱位置 (注意: realtime_products.json 中是小写 refrigerator)
            fridge_obj = self.data_loader.find_object_by_name("refrigerator")
            if fridge_obj:
                fridge_x, _, fridge_z = fridge_obj.position
                prefer_closer_to = (fridge_x, fridge_z)
                print(f"      目标在冰箱内，优先选择靠近冰箱的位置 ({fridge_x:.2f}, {fridge_z:.2f})")

        if pose_info is None:
            pathfinder = self._require_valid_pathfinder(caller="plan_navigation.calculate_ideal_docking_pose_v2")
            container_bounds_min, container_bounds_max = self._current_container_bounds_for_docking()
            pose_info = pathfinder.calculate_ideal_docking_pose_v2(
                (target_x, target_z),
                safe_distance=self.docking_distance_m,
                search_radius=0.1,
                prefer_closer_to=prefer_closer_to,
                internal_rect_strategy=self.internal_rect_strategy,
                start_world_pos=(current_x, current_z),
                container_bounds_min=container_bounds_min,
                container_bounds_max=container_bounds_max,
            )

        if pose_info is None:
            print("   ERROR 无法找到可达的理想停靠位置")
            return False

        ideal_pos, ideal_yaw = pose_info
        nav_target = ideal_pos
        self.last_plan_docking_debug = dict(getattr(self.pathfinder, "last_docking_debug", {}) or {})
        self.last_chosen_docking_distance_m = self._extract_last_chosen_docking_distance_m()
        if self.last_container_docking_debug:
            self.last_container_docking_debug["actual_docking_distance_m"] = self.last_chosen_docking_distance_m
            self.last_container_docking_debug["exceeds_feedback_threshold"] = bool(
                self.last_chosen_docking_distance_m is not None
                and self.last_chosen_docking_distance_m > CONTAINER_DOCKING_TOO_FAR_THRESHOLD_M
            )
        print(f"   OK 理想停靠位置: ({ideal_pos[0]:.2f}, {ideal_pos[1]:.2f})")
        print(f"      理想朝向: {ideal_yaw:.1f}°")
        print(f"      请求站位距离: {self.last_requested_docking_distance_m*100:.0f}cm")
        if self.last_chosen_docking_distance_m is not None:
            print(f"      实际计算站位距离: {self.last_chosen_docking_distance_m*100:.1f}cm")
        print(f"      内部矩形选边策略: {self.internal_rect_strategy}")

        # Save nav_target for use in navigation loop
        # Save nav_target for use in navigation loop
        # IMPORTANT: Use nav_target for arrival checks, not final_target!
        self.nav_target = nav_target
        self.use_nav_target_for_arrival = True  # Flag to use nav_target instead of final_target

        pathfinder = self._require_valid_pathfinder(caller="plan_navigation.find_path")
        nav_target_grid = pathfinder.world_to_grid(nav_target[0], nav_target[1])
        print(
            f"[NAV][DEBUG] nav_target_grid={nav_target_grid} "
            f"nav_target_hard_obstacle={pathfinder.is_hard_obstacle(*nav_target_grid)}"
        )

        # Plan A* path to ideal docking position (not target position!)
        print("\n[4/5] 规划A*全局路径（到理想停靠位置）...")
        path = pathfinder.find_path((current_x, current_z), nav_target)
        if not path:
            if pathfinder.last_path_debug:
                print(
                    f"[NAV][DEBUG] latest_path_debug="
                    f"{json.dumps(pathfinder.last_path_debug, ensure_ascii=False, sort_keys=True)}"
                )
            print("   ERROR A* path planning failed")
            return False


        # Smooth path
        smoothed_path = self.pathfinder.smooth_path(path, max_angle=25.0)
        self.current_path = smoothed_path
        self.current_waypoint_index = 0

        print(f"   OK 路径规划完成")
        print(f"      路径点数: {len(smoothed_path)}")
        print(f"      最终目标: ({target_x:.2f}, {target_z:.2f})")

        # Update visualizer with planned path
        if VISUALIZER_AVAILABLE:
            update_navigation_state(
                current_pos=(current_x, current_z),
                target_pos=(target_x, target_z),
                planned_path=smoothed_path,
                waypoint_index=0
            )

        # Start navigation
        print("\n[5/5] 开始导航（A*路径 + 8方向移动 + 雷达避障）...")
        print("      每次移动都会检查是否已到达最终目标（阈值:{}m）".format(self.arrival_threshold))
        success = self._navigate_along_path()
        return success

    def _navigate_along_path(self) -> bool:
        """Navigate along the planned path"""
        attempts = 0
        stuck_count = 0
        last_position = None

        while attempts < self.max_attempts:
            attempts += 1

            # Get current position
            camera_info = self.data_loader.load_camera_info()
            if not camera_info:
                continue

            current_x, current_y, current_z = camera_info.position
            current_pitch, current_yaw, current_roll = camera_info.rotation

            # Update visualizer
            if VISUALIZER_AVAILABLE:
                update_navigation_state(
                    current_pos=(current_x, current_z),
                    waypoint_index=self.current_waypoint_index
                )
            # 优先检查：是否已经到达最终目标（不管路径点）
            # Use nav_target if available, otherwise use final_target
            arrival_target = self.nav_target if hasattr(self, 'use_nav_target_for_arrival') and self.use_nav_target_for_arrival else self.final_target
            if arrival_target is not None:
                final_dx = arrival_target[0] - current_x
                final_dz = arrival_target[1] - current_z
                final_distance = math.sqrt(final_dx * final_dx + final_dz * final_dz)

                if final_distance < self.arrival_threshold:
                    print(f"   [{attempts:3d}] OK 到达最终目标！距离={final_distance:.2f}m")
                    print(f"   [{attempts:3d}] 开始两阶段精确调整（位置±20cm + 朝向±5°）")

                    # 三阶段精确调整：位置+朝向+视野居中（v3.7.3: 位置容忍度20cm）
                    success = self._adjust_position_and_orientation_precise(self.final_target, tolerance_pos=0.05, tolerance_yaw=0.05)

                    # 截图并标注目标位置
                    if ENABLE_CAPTURE_AND_ANNOTATE:
                        print(f"\n   [{attempts:3d}] 正在截图并标注目标位置...")
                    try:
                        target_name = getattr(self, 'target_name', 'unknown')
                        try:
                            _anno = None  # screenshot disabled
                        except Exception:
                            _anno = None
                        screenshot_path = _anno(target_name) if _anno else None
                        if screenshot_path:
                            print(f"   [{attempts:3d}] 截图已保存: {screenshot_path}")
                    except Exception as e:
                        print(f"   [{attempts:3d}] WARNING: 截图失败 - {e}")

                    if success:
                        print(f"\n{'='*70}")
                        print(f"  导航成功！")
                        print(f"{'='*70}")
                    else:
                        print(f"\n{'='*70}")
                        print(f"  导航失败 - 未满足精确调整要求")
                        print(f"{'='*70}")

                    return success

            # 检测卡住
            if last_position is not None:
                moved_dist = math.sqrt((current_x - last_position[0])**2 + (current_z - last_position[1])**2)
                if moved_dist < 0.05:
                    stuck_count += 1
                else:
                    stuck_count = 0
            last_position = (current_x, current_z)

            # 卡住超过15次，检查是否接近目标
            if stuck_count > 15:
                # 如果已经接近最终目标（<0.2m），认为到达，不脱困
                arrival_target = self.nav_target if hasattr(self, 'use_nav_target_for_arrival') and self.use_nav_target_for_arrival else self.final_target
                if arrival_target is not None:
                    final_dx = arrival_target[0] - current_x
                    final_dz = arrival_target[1] - current_z
                    final_distance_check = math.sqrt(final_dx * final_dx + final_dz * final_dz)

                    if final_distance_check < 0.2:
                        print(f"   [{attempts:3d}] 卡住但已接近目标{final_distance_check:.2f}m，认为到达")
                        print(f"   [{attempts:3d}] 开始两阶段精确调整（位置±20cm + 朝向±5°）")

                        # 三阶段精确调整：位置+朝向+视野居中（v3.7.3: 位置容忍度20cm）
                        success = self._adjust_position_and_orientation_precise(self.final_target, tolerance_pos=0.05, tolerance_yaw=0.05)

                        # 截图并标注目标位置
                        if ENABLE_CAPTURE_AND_ANNOTATE:
                            print(f"\n   [{attempts:3d}] 正在截图并标注目标位置...")
                        try:
                            target_name = getattr(self, 'target_name', 'unknown')
                            try:
                                _anno = None  # screenshot disabled
                            except Exception:
                                _anno = None
                            screenshot_path = _anno(target_name) if _anno else None
                            if screenshot_path:
                                print(f"   [{attempts:3d}] 截图已保存: {screenshot_path}")
                        except Exception as e:
                            print(f"   [{attempts:3d}] WARNING: 截图失败 - {e}")

                        if success:
                            print(f"\n{'='*70}")
                            print(f"  导航成功！")
                            print(f"{'='*70}")
                        else:
                            print(f"\n{'='*70}")
                            print(f"  导航失败 - 未满足精确调整要求")
                            print(f"{'='*70}")

                        return success

                # 检查是否真的被困（前方180度全部被堵）
                # 获取当前雷达读数
                current_radar_readings = self.radar_reader.get_all_readings()

                if self._is_front_hemisphere_blocked(current_yaw, current_radar_readings):
                    # 前方180度全部被堵，需要后退脱困
                    print(f"   [{attempts:3d}] WARNING 前方被堵死，执行后退脱困...")
                    self._force_unstuck()
                    stuck_count = 0
                else:
                    # 只是暂时卡住，前方还有可通行方向
                    # 重置计数器，让雷达避障逻辑自然地选择转向
                    print(f"   [{attempts:3d}] INFO 暂时卡住但前方有通路，尝试转向绕行...")
                    stuck_count = 0
                    # 不执行force_unstuck，让下一轮的smart_move自然选择方向

                continue

            # 检查路径点索引是否越界（路径点已用完）
            if self.current_waypoint_index >= len(self.current_path):
                # 路径点用完，直接朝向最终目标移动
                if self.final_target is not None:
                    target_x, target_z = self.final_target
                    print(f"   [{attempts:3d}] 路径点已用完，直接朝向最终目标...")
                else:
                    print(f"   [{attempts:3d}] ERROR 路径点用完且无最终目标")
                    return False
            else:
                # 前瞻优化：检查是否已经走过当前路径点，距离后续路径点更近
                # 检查接下来的N个路径点，找到最近的那个
                lookahead_range = min(10, len(self.current_path) - self.current_waypoint_index)
                closest_waypoint_idx = self.current_waypoint_index
                closest_distance = float('inf')

                for i in range(self.current_waypoint_index, self.current_waypoint_index + lookahead_range):
                    if i >= len(self.current_path):
                        break
                    wp_x, wp_z = self.current_path[i]
                    wp_dist = math.sqrt((wp_x - current_x)**2 + (wp_z - current_z)**2)
                    if wp_dist < closest_distance:
                        closest_distance = wp_dist
                        closest_waypoint_idx = i

                # 如果发现更近的后续路径点，直接跳过中间的路径点
                if closest_waypoint_idx > self.current_waypoint_index:
                    skipped = closest_waypoint_idx - self.current_waypoint_index
                    if skipped > 0:
                        print(f"   [{attempts:3d}] 前瞻优化：跳过{skipped}个路径点，"
                              f"从路径点{self.current_waypoint_index}直接到{closest_waypoint_idx}")
                        self.current_waypoint_index = closest_waypoint_idx

                # 当前目标路径点
                target_waypoint = self.current_path[self.current_waypoint_index]
                target_x, target_z = target_waypoint

            # 计算到路径点的距离
            dx = target_x - current_x
            dz = target_z - current_z
            distance = math.sqrt(dx * dx + dz * dz)

            # 检查是否到达路径点
            if distance < self.waypoint_threshold:
                print(f"   [{attempts:3d}] OK 到达路径点 {self.current_waypoint_index}/{len(self.current_path)-1}")
                self.current_waypoint_index += 1

                # 检查是否到达最终目标（路径点用完）
                if self.current_waypoint_index >= len(self.current_path):
                    # 额外验证：确保真的接近最终目标（防止前瞻优化导致的提前判断）
                    arrival_target = self.nav_target if hasattr(self, 'use_nav_target_for_arrival') and self.use_nav_target_for_arrival else self.final_target
                    if arrival_target is not None:
                        final_dx = arrival_target[0] - current_x
                        final_dz = arrival_target[1] - current_z
                        final_distance_check = math.sqrt(final_dx * final_dx + final_dz * final_dz)

                        # 路径点用完后放宽阈值：如果距离<2.0m就认为到达（A*路径已尽力）
                        if final_distance_check >= 2.0:
                            print(f"   [{attempts:3d}] 路径点已用完，但距目标还有{final_distance_check:.2f}m，继续前进...")
                            continue  # 继续导航，不返回
                        elif final_distance_check >= self.arrival_threshold:
                            print(f"   [{attempts:3d}] 路径点已用完，距目标{final_distance_check:.2f}m（<2.0m），认为A*路径完成")

                    print(f"   [{attempts:3d}] OK 到达最终目标！")

                    # 两阶段精确调整：位置+朝向
                    success = False
                    if self.final_target is not None:
                        print(f"   [{attempts:3d}] 开始两阶段精确调整（位置±20cm + 朝向±5°）")
                        success = self._adjust_position_and_orientation_precise(self.final_target, tolerance_pos=0.05, tolerance_yaw=0.05)

                    # 截图并标注目标位置
                    if ENABLE_CAPTURE_AND_ANNOTATE:
                        print(f"\n   [{attempts:3d}] 正在截图并标注目标位置...")
                    try:
                        target_name = getattr(self, 'target_name', 'unknown')
                        try:
                            _anno = None  # screenshot disabled
                        except Exception:
                            _anno = None
                        screenshot_path = _anno(target_name) if _anno else None
                        if screenshot_path:
                            print(f"   [{attempts:3d}] 截图已保存: {screenshot_path}")
                    except Exception as e:
                        print(f"   [{attempts:3d}] WARNING: 截图失败 - {e}")

                    if success:
                        print(f"\n{'='*70}")
                        print(f"  导航成功！")
                        print(f"{'='*70}")
                    else:
                        print(f"\n{'='*70}")
                        print(f"  导航失败 - 未满足精确调整要求")
                        print(f"{'='*70}")

                    return success

                continue

            # 计算到目标的方向
            target_yaw = math.degrees(math.atan2(dx, dz))
            if target_yaw < 0:
                target_yaw += 360

            # 显示进度（包含到最终目标的距离）
            if attempts % 5 == 1:
                final_dist_str = ""
                if self.final_target is not None:
                    final_dx = self.final_target[0] - current_x
                    final_dz = self.final_target[1] - current_z
                    final_distance = math.sqrt(final_dx * final_dx + final_dz * final_dz)
                    final_dist_str = f" | 距最终目标:{final_distance:.2f}m"

                print(f"   [{attempts:3d}] 路径点{self.current_waypoint_index}/{len(self.current_path)-1} | "
                      f"距路径点:{distance:.2f}m{final_dist_str}")

            # 转向目标方向
            self._turn_to_target_yaw(target_yaw, current_yaw)

            # 读取实时雷达
            readings = self.radar_reader.get_all_readings()

            # 根据雷达数据决定移动策略
            move_success = self._smart_move_towards_target(
                current_yaw, target_yaw, readings
            )

            if not move_success:
                print(f"   [{attempts:3d}] WARNING 移动受阻")

            time.sleep(0.10)  # v3.7.3: 从0.05增加到0.10，减少读取频率降低卡顿

            # Update matplotlib visualization
            try:
                import matplotlib.pyplot as plt
                plt.pause(0.001)
            except:
                pass

        if attempts >= self.max_attempts:
            print(f"   ERROR 超过最大尝试次数")
            return False

        return True

    def _smart_move_towards_target(
        self,
        current_yaw: float,
        target_yaw: float,
        radar_readings
    ) -> bool:
        """
        智能移动：根据雷达数据选择最佳移动方向

        策略：
        1. 前方安全：直接前进（W）
        2. 前方不安全，但左前/右前安全：斜向前进（W+A 或 W+D）
        3. 前方和斜前都不安全：侧向移动（A 或 D）
        4. 都不安全：后退（S）
        """
        if not LOCAL_ACTIONS_AVAILABLE:
            return False

        # 检查各个方向的安全性
        front_safe = self._check_direction_safe(current_yaw, 0, radar_readings, check_angle=30)
        front_left_safe = self._check_direction_safe(current_yaw, -45, radar_readings, check_angle=25)
        front_right_safe = self._check_direction_safe(current_yaw, 45, radar_readings, check_angle=25)
        left_safe = self._check_direction_safe(current_yaw, -90, radar_readings, check_angle=25)
        right_safe = self._check_direction_safe(current_yaw, 90, radar_readings, check_angle=25)

        # 计算目标在左还是右
        angle_diff = target_yaw - current_yaw
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360

        target_on_left = angle_diff < 0

        # 决策树
        if front_safe:
            # 前方安全：直接前进
            local_actions.move_forward(duration=self.move_duration)
            return True

        elif target_on_left and front_left_safe:
            # 目标在左，左前方安全：左前斜进
            local_actions.move_forward_left(duration=self.move_duration)
            return True

        elif not target_on_left and front_right_safe:
            # 目标在右，右前方安全：右前斜进
            local_actions.move_forward_right(duration=self.move_duration)
            return True

        elif target_on_left and left_safe:
            # 目标在左，左侧安全：左移
            local_actions.move_left(duration=self.move_duration * 0.8)
            return True

        elif not target_on_left and right_safe:
            # 目标在右，右侧安全：右移
            local_actions.move_right(duration=self.move_duration * 0.8)
            return True

        elif front_left_safe:
            # 左前方安全：左前斜进
            local_actions.move_forward_left(duration=self.move_duration)
            return True

        elif front_right_safe:
            # 右前方安全：右前斜进
            local_actions.move_forward_right(duration=self.move_duration)
            return True

        elif left_safe:
            # 左侧安全：左移
            local_actions.move_left(duration=self.move_duration * 0.8)
            return True

        elif right_safe:
            # 右侧安全：右移
            local_actions.move_right(duration=self.move_duration * 0.8)
            return True

        else:
            # 都不安全：后退
            print(f"      所有方向都不安全，后退")
            local_actions.move_backward(duration=self.move_duration * 0.5)
            return False

    def _check_direction_safe(
        self,
        current_yaw: float,
        relative_angle: float,
        radar_readings,
        check_angle: float = 30.0
    ) -> bool:
        """
        检查某个方向是否安全

        Args:
            current_yaw: 当前朝向
            relative_angle: 相对角度（-180到180，0=正前方，-90=左侧，90=右侧）
            radar_readings: 雷达读数
            check_angle: 检查的角度范围（±度）

        Returns:
            该方向是否安全
        """
        if not radar_readings:
            return True

        # 计算目标方向的绝对角度
        target_direction = (current_yaw + relative_angle) % 360

        # 检查该方向附近的雷达读数
        for reading in radar_readings:
            if reading.distance is None:
                continue

            # 计算角度差
            angle_diff = abs(reading.angle - target_direction)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # 如果在检查范围内且距离过近
            if angle_diff <= check_angle and reading.distance < self.safe_distance:
                return False

        return True

    def _is_front_hemisphere_blocked(self, current_yaw: float, radar_readings) -> bool:
        """
        检查前方180度半圆范围是否被完全堵住

        Args:
            current_yaw: 当前朝向（度）
            radar_readings: 雷达读数

        Returns:
            True=前方180度全部被堵，False=至少有一个方向可通行
        """
        if not radar_readings:
            return False  # 没有雷达数据，假设没被堵

        # 检查前方180度范围：-90°（左）到 +90°（右）
        # 将180度分成7个扇区检查：-90, -60, -30, 0, 30, 60, 90
        check_directions = [-90, -60, -30, 0, 30, 60, 90]

        blocked_count = 0
        for relative_angle in check_directions:
            # 检查每个方向是否安全（±20度范围）
            if not self._check_direction_safe(current_yaw, relative_angle, radar_readings, check_angle=20):
                blocked_count += 1

        # 如果7个方向中有6个或以上被堵（>=85%），认为前方被堵死
        all_blocked = (blocked_count >= 6)

        if all_blocked:
            print(f"      前方180度被堵死: {blocked_count}/7个方向不可通行")
        else:
            print(f"      前方有可通行方向: {7-blocked_count}/7个方向安全")

        return all_blocked

    def _force_unstuck(self):
        """强制脱困：小幅后退+转向"""
        if not LOCAL_ACTIONS_AVAILABLE:
            return

        import random

        print("      执行强制脱困")

        # 小幅后退（减少到0.2秒，约30-40cm）
        local_actions.move_backward(duration=0.2)
        time.sleep(0.2)

        # 中等角度随机转向
        turn_direction = random.choice(['left', 'right'])
        turn_pixels = random.randint(150, 300)  # 减少转向幅度

        if turn_direction == 'left':
            local_actions.look_left(pixels=turn_pixels)
        else:
            local_actions.look_right(pixels=turn_pixels)

        time.sleep(0.3)

        # 尝试前进一点
        local_actions.move_forward(duration=0.2)  # 也减少前进距离
        time.sleep(0.2)

    def _run_container_forward_settle(
        self,
        target_pos: Tuple[float, float],
        *,
        max_steps: int = 6,
        step_duration: float = 0.18,
        post_move_wait_s: float = 0.35,
        blocked_move_threshold_m: float = 0.025,
        blocked_confirm_steps: int = 2,
        overshoot_slack_m: float = 0.03,
    ) -> None:
        """
        容器内目标专用：导航到粗站位后，沿目标方向做短脉冲前顶，
        用真实碰撞补偿地图误差，贴到当前可达最近位置。
        """
        self.last_forward_settle_debug = {}
        if not LOCAL_ACTIONS_AVAILABLE:
            return
        if not isinstance(self.last_container_docking_debug, dict) or not self.last_container_docking_debug:
            return

        before_info = self._load_camera_info_with_retry()
        if not before_info:
            self.last_forward_settle_debug = {
                "enabled": True,
                "attempted": False,
                "reason": "camera_info_unavailable_before_forward_settle",
            }
            return

        before_pos = (float(before_info.position[0]), float(before_info.position[2]))
        initial_dist = math.sqrt((target_pos[0] - before_pos[0])**2 + (target_pos[1] - before_pos[1])**2)
        steps_debug: list[dict] = []
        stop_reason = "max_steps_reached"
        executed_steps = 0
        blocked_streak = 0

        print("   [阶段0/3] 容器内目标前顶贴边开始")
        print(f"            初始距离目标: {initial_dist*100:.1f}cm")

        for step_idx in range(max_steps):
            current_info = self._load_camera_info_with_retry()
            if not current_info:
                stop_reason = "camera_info_unavailable_during_forward_settle"
                break

            current_pos = (float(current_info.position[0]), float(current_info.position[2]))
            current_yaw = float(current_info.rotation[1])
            dx = target_pos[0] - current_pos[0]
            dz = target_pos[1] - current_pos[1]
            dist_before = math.sqrt(dx * dx + dz * dz)
            target_yaw = math.degrees(math.atan2(dx, dz)) % 360

            print(
                f"            [前顶{step_idx+1}/{max_steps}] "
                f"距离目标={dist_before*100:.1f}cm, target_yaw={target_yaw:.1f}°"
            )
            self._turn_to_target_yaw(target_yaw, current_yaw, tolerance=8.0)
            local_actions.move_forward(duration=step_duration)
            time.sleep(post_move_wait_s)

            after_info = self._load_camera_info_with_retry()
            if not after_info:
                executed_steps = step_idx + 1
                stop_reason = "camera_info_unavailable_after_move"
                break

            after_pos = (float(after_info.position[0]), float(after_info.position[2]))
            moved = math.sqrt((after_pos[0] - current_pos[0])**2 + (after_pos[1] - current_pos[1])**2)
            dist_after = math.sqrt((target_pos[0] - after_pos[0])**2 + (target_pos[1] - after_pos[1])**2)
            progress = dist_before - dist_after
            step_debug = {
                "step_index": int(step_idx + 1),
                "pos_before": [round(current_pos[0], 4), round(current_pos[1], 4)],
                "pos_after": [round(after_pos[0], 4), round(after_pos[1], 4)],
                "distance_to_target_before_m": round(float(dist_before), 4),
                "distance_to_target_after_m": round(float(dist_after), 4),
                "moved_distance_m": round(float(moved), 4),
                "progress_towards_target_m": round(float(progress), 4),
                "target_yaw_deg": round(float(target_yaw), 4),
            }
            steps_debug.append(step_debug)
            executed_steps = step_idx + 1

            print(
                f"            [前顶{step_idx+1}/{max_steps}] "
                f"位移={moved*100:.1f}cm, 接近目标={progress*100:.1f}cm, 当前距离={dist_after*100:.1f}cm"
            )

            if moved <= blocked_move_threshold_m:
                blocked_streak += 1
                step_debug["blocked_streak"] = int(blocked_streak)
                if blocked_streak >= max(1, int(blocked_confirm_steps)):
                    stop_reason = "blocked_by_obstacle"
                    break
            else:
                blocked_streak = 0
            if dist_after > dist_before + overshoot_slack_m:
                stop_reason = "moved_away_from_target"
                break

        final_info = self._load_camera_info_with_retry()
        final_pos = None
        final_dist = None
        if final_info:
            final_pos = (float(final_info.position[0]), float(final_info.position[2]))
            final_dist = math.sqrt((target_pos[0] - final_pos[0])**2 + (target_pos[1] - final_pos[1])**2)

        self.last_forward_settle_debug = {
            "enabled": True,
            "attempted": True,
            "max_steps": int(max_steps),
            "step_duration_s": round(float(step_duration), 4),
            "post_move_wait_s": round(float(post_move_wait_s), 4),
            "blocked_move_threshold_m": round(float(blocked_move_threshold_m), 4),
            "blocked_confirm_steps": int(blocked_confirm_steps),
            "stop_reason": stop_reason,
            "executed_steps": int(executed_steps),
            "initial_distance_to_target_m": round(float(initial_dist), 4),
            "final_distance_to_target_m": (None if final_dist is None else round(float(final_dist), 4)),
            "final_pos": (None if final_pos is None else [round(final_pos[0], 4), round(final_pos[1], 4)]),
            "steps": steps_debug,
        }

        if final_dist is not None:
            print(
                f"            前顶贴边结束: reason={stop_reason}, "
                f"steps={executed_steps}, 最终距离目标={final_dist*100:.1f}cm"
            )
        else:
            print(f"            前顶贴边结束: reason={stop_reason}, steps={executed_steps}")

    def _adjust_position_and_orientation_precise(self, target_pos: Tuple[float, float], tolerance_pos: float = 0.05, tolerance_yaw: float = 0.05) -> bool:
        """
        两阶段精确调整位姿（v3.7.3: 位置容忍度从4cm放宽到20cm）：
        1. 位置调整：导航动作微调到距离目标130cm的位置（容忍±20cm）
        2. 朝向调整：直接计算并调整到面向目标的朝向（容忍±5°）

        Args:
            target_pos: 目标物品位置 (x, z)
            tolerance_pos: 位置容忍度（米）
            tolerance_yaw: 朝向容忍度（度）

        Returns:
            是否成功调整
        """
        if not LOCAL_ACTIONS_AVAILABLE:
            return True

        # 计算理想停靠位姿（距离目标130cm，垂直于最近障碍物）
        target_name_norm = str(getattr(self, "target_name", "") or "").strip().lower()
        pose_info = self._fixed_docking_pose_for_special_target(target_name_norm, target_pos)

        prefer_closer_to = None
        _tname = getattr(self, 'target_name', 'unknown')
        _tid = getattr(self, 'target_instance_id', None)
        if _tid is not None:
            target_obj = self.data_loader.find_object_by_name_and_instance_id(_tname, int(_tid))
        else:
            target_obj = self.data_loader.find_object_by_name(_tname)
        if target_obj and getattr(target_obj, "layer", None) == "Refrigerator":
            fridge_obj = self.data_loader.find_object_by_name("refrigerator")
            if fridge_obj:
                fridge_x, _, fridge_z = fridge_obj.position
                prefer_closer_to = (fridge_x, fridge_z)

        current_pose_for_docking = None
        camera_info_for_docking = self._load_camera_info_with_retry()
        if camera_info_for_docking:
            current_pose_for_docking = (
                float(camera_info_for_docking.position[0]),
                float(camera_info_for_docking.position[2]),
            )

        if pose_info is None:
            pathfinder = self._require_valid_pathfinder(caller="precise_adjust.calculate_ideal_docking_pose_v2")
            container_bounds_min, container_bounds_max = self._current_container_bounds_for_docking()
            pose_info = pathfinder.calculate_ideal_docking_pose_v2(
                target_pos,
                safe_distance=self.docking_distance_m,
                search_radius=0.1,
                prefer_closer_to=prefer_closer_to,
                internal_rect_strategy=self.internal_rect_strategy,
                start_world_pos=current_pose_for_docking,
                container_bounds_min=container_bounds_min,
                container_bounds_max=container_bounds_max,
            )

        if pose_info is None:
            print("   WARNING: 无法确定墙壁位姿，使用fallback: 直接面向目标")
            # Fallback: 使用当前位置，但朝向目标
            camera_info = self._load_camera_info_with_retry()
            if not camera_info:
                return False

            current_pos = (camera_info.position[0], camera_info.position[2])

            # 计算朝向目标的yaw
            dx = target_pos[0] - current_pos[0]
            dz = target_pos[1] - current_pos[1]
            ideal_yaw = math.degrees(math.atan2(dx, dz)) % 360

            print(f"   Fallback朝向: {ideal_yaw:.0f}° (玩家→目标)")
            self.last_precise_adjust_debug = {
                "mode": "fallback_face_target",
                "target_pos": [round(target_pos[0], 4), round(target_pos[1], 4)],
                "ideal_pos": None,
                "ideal_yaw": round(float(ideal_yaw), 4),
                "tolerance_pos_m": round(float(tolerance_pos), 4),
                "tolerance_yaw_deg": round(float(tolerance_yaw), 4),
            }

            # 只调整朝向，不调整位置
            orientation_adjusted = self._adjust_orientation_with_feedback(ideal_yaw, tolerance=tolerance_yaw, max_attempts=10)
            return orientation_adjusted

        ideal_pos, ideal_yaw = pose_info
        chosen_docking_distance_m = self._extract_last_chosen_docking_distance_m()
        has_container_special_case = bool(self.last_container_docking_debug)
        total_precise_phases = 3 if has_container_special_case else 2
        self.last_precise_adjust_debug = {
            "mode": "ideal_pose",
            "target_pos": [round(target_pos[0], 4), round(target_pos[1], 4)],
            "ideal_pos": [round(ideal_pos[0], 4), round(ideal_pos[1], 4)],
            "ideal_yaw": round(float(ideal_yaw), 4),
            "tolerance_pos_m": round(float(tolerance_pos), 4),
            "tolerance_yaw_deg": round(float(tolerance_yaw), 4),
            "requested_docking_distance_m": round(float(self.last_requested_docking_distance_m), 4),
        }
        if chosen_docking_distance_m is not None:
            self.last_precise_adjust_debug["chosen_docking_distance_m"] = round(float(chosen_docking_distance_m), 4)

        # ========== 阶段1：位置微调 ==========
        print(f"   [阶段1/{total_precise_phases}] 位置微调开始")
        print(f"            目标物品位置: ({target_pos[0]:.2f}, {target_pos[1]:.2f})")
        print(f"            理想停靠位置: ({ideal_pos[0]:.2f}, {ideal_pos[1]:.2f})")
        print(f"            理想朝向: {ideal_yaw:.1f}°")
        print(f"            请求站位距离: {self.last_requested_docking_distance_m*100:.0f}cm")
        if chosen_docking_distance_m is not None:
            print(f"            计算站位距离: {chosen_docking_distance_m*100:.1f}cm")
        print(f"            容忍度: ±{tolerance_pos*100:.0f}cm，最多100次尝试")

        # 先检查当前距离理想位置的距离
        camera_info_initial = self._load_camera_info_with_retry()
        if camera_info_initial:
            current_pos_initial = (camera_info_initial.position[0], camera_info_initial.position[2])
            dist_to_ideal = math.sqrt((ideal_pos[0] - current_pos_initial[0])**2 + (ideal_pos[1] - current_pos_initial[1])**2)
            dist_to_target = math.sqrt((target_pos[0] - current_pos_initial[0])**2 + (target_pos[1] - current_pos_initial[1])**2)
            self.last_precise_adjust_debug.update(
                {
                    "initial_pos": [round(current_pos_initial[0], 4), round(current_pos_initial[1], 4)],
                    "initial_dist_to_ideal_m": round(float(dist_to_ideal), 4),
                    "initial_dist_to_target_m": round(float(dist_to_target), 4),
                }
            )
            print(f"            当前位置: ({current_pos_initial[0]:.2f}, {current_pos_initial[1]:.2f})")
            print(f"            距离理想位置: {dist_to_ideal*100:.1f}cm")
            print(f"            距离目标: {dist_to_target*100:.1f}cm（应该约{self.docking_distance_m*100:.0f}cm）")

            # 不管距离多远，都必须尝试移动到理想位置
            position_adjusted = False
        else:
            position_adjusted = False

        # 执行位置微调循环（增加到100次尝试）
        for attempt in range(100):
                # 读取当前位置
                camera_info = self._load_camera_info_with_retry()
                if not camera_info:
                    print(f"            [尝试{attempt+1}/100] 无法读取相机信息")
                    break

                current_pos = (camera_info.position[0], camera_info.position[2])

                # 计算位置误差
                pos_error = math.sqrt((current_pos[0] - ideal_pos[0])**2 + (current_pos[1] - ideal_pos[1])**2)

                # 检查是否达标
                if pos_error <= tolerance_pos:
                    # 计算距目标的实际距离
                    dist_to_target_final = math.sqrt((target_pos[0] - current_pos[0])**2 + (target_pos[1] - current_pos[1])**2)
                    self.last_precise_adjust_debug.update(
                        {
                            "position_adjusted": True,
                            "position_adjust_attempts": int(attempt + 1),
                            "position_phase_final_pos": [round(current_pos[0], 4), round(current_pos[1], 4)],
                            "position_phase_final_error_m": round(float(pos_error), 4),
                            "position_phase_final_dist_to_target_m": round(float(dist_to_target_final), 4),
                            "position_phase_within_tolerance": True,
                        }
                    )
                    print(f"            位置调整成功: 误差{pos_error*100:.1f}cm，距目标{dist_to_target_final*100:.1f}cm")
                    position_adjusted = True
                    break

                # 显示进度（每5次显示一次）
                if attempt % 5 == 0 or attempt < 3:
                    print(f"            [尝试{attempt+1}/30] 当前({current_pos[0]:.2f}, {current_pos[1]:.2f})，误差{pos_error*100:.1f}cm")

                # 改进的移动策略：根据距离选择移动方式
                # 精确调整阶段禁用雷达检查（目标可能就在障碍物旁边）
                if pos_error > 0.5:
                    # 距离较远（>50cm），使用转向+前进
                    print(f"            大步移动: 误差{pos_error*100:.1f}cm")
                    if not self._navigate_to_position(current_pos, ideal_pos, skip_radar_check=True):
                        print(f"            WARNING: 大距离移动失败")
                        break
                else:
                    # 距离较近（<50cm），使用小步微调
                    print(f"            小步微调: 误差{pos_error*100:.1f}cm")
                    if not self._micro_move_towards(current_pos, ideal_pos, skip_radar_check=True):
                        print(f"            WARNING: 小步移动失败")
                        break

                time.sleep(0.25)  # 增加等待时间，让动作完成

        if not position_adjusted:
            # 打印最终状态
            camera_info_final_pos = self._load_camera_info_with_retry()
            if camera_info_final_pos:
                final_pos = (camera_info_final_pos.position[0], camera_info_final_pos.position[2])
                final_error = math.sqrt((final_pos[0] - ideal_pos[0])**2 + (final_pos[1] - ideal_pos[1])**2)
                dist_to_target_final = math.sqrt((target_pos[0] - final_pos[0])**2 + (target_pos[1] - final_pos[1])**2)
                self.last_precise_adjust_debug.update(
                    {
                        "position_adjusted": False,
                        "position_phase_final_pos": [round(final_pos[0], 4), round(final_pos[1], 4)],
                        "position_phase_final_error_m": round(float(final_error), 4),
                        "position_phase_final_dist_to_target_m": round(float(dist_to_target_final), 4),
                        "position_phase_within_tolerance": bool(final_error <= tolerance_pos),
                    }
                )
                print(f"            WARNING: 位置微调未完全达标")
                print(f"            最终位置: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
                print(f"            距离理想位置: {final_error*100:.1f}cm")
                print(f"            距离目标物品: {dist_to_target_final*100:.1f}cm（应该约{self.docking_distance_m*100:.0f}cm）")

        self._run_container_forward_settle(target_pos)
        if isinstance(self.last_forward_settle_debug, dict) and self.last_forward_settle_debug:
            self.last_precise_adjust_debug["forward_settle_debug"] = self.last_forward_settle_debug

        # ========== 阶段2：视野中央精确调整（直接用屏幕投影，一步到位）==========
        # 注：之前有两个阶段（3D向量调整 + 屏幕投影调整），做的是重复的事情
        # 现在合并为一个阶段，直接用屏幕投影调整，更直接准确
        print(f"   [阶段2/{total_precise_phases}] 视野中央精确调整（屏幕投影法）")
        print(f"            容忍度: ±10像素，最多100次尝试")
        center_tolerance_pixels = self._center_tolerance_pixels_for_target(default_pixels=10.0)
        center_adjusted = self._adjust_to_center_view_v2(target_pos, tolerance_pixels=center_tolerance_pixels, max_attempts=100)
        self.last_precise_adjust_debug["center_adjusted"] = bool(center_adjusted)

        interaction_probe_debug = None
        if has_container_special_case:
            print(f"   [阶段3/{total_precise_phases}] 交互微探测（Alt+J）")
            print("            已居中后，在上下左右各缓慢探测到35像素；命中目标交互即停止")
            interaction_probe_debug = self._probe_target_interaction_near_center(
                target_obj=target_obj,
                probe_pixels=35,
                settle_s=0.05,
                step_pixels=1,
                max_raw_pixels_per_direction=180,
                max_age_s=0.8,
            )
            self.last_precise_adjust_debug["interaction_probe_debug"] = interaction_probe_debug
            snapshot_available = bool(interaction_probe_debug.get("snapshot_available"))
            if interaction_probe_debug.get("success"):
                print(
                    "            交互探测命中: "
                    f"direction={interaction_probe_debug.get('matched_direction', 'unknown')}"
                )
            elif snapshot_available:
                print("            交互探测未命中目标，保持当前失败反馈路径")
            else:
                print("            交互探测未获得有效 Alt+J 快照，回退到几何验证")

        # ========== 最终验证 ==========
        print(f"\n   [最终验证] 检查是否真正满足所有要求...")

        # 读取最终状态
        camera_info_final = self._load_camera_info_with_retry()
        if not camera_info_final:
            print(f"   [最终验证] FAILED: 无法读取相机信息")
            return False

        # 获取所有同名物品（v3.7.4: 支持多同类物品验证）
        target_name = getattr(self, 'target_name', 'unknown')
        all_matching_objects = self.data_loader.find_all_objects_by_name(target_name)
        if not all_matching_objects:
            print(f"   [最终验证] FAILED: 无法获取目标物品信息")
            return False

        camera_3d = camera_info_final.position
        forward_vec = camera_info_final.forward
        validation_target_distance_m = float(self.docking_distance_m)
        validation_distance_mode = "requested_docking_distance"
        if has_container_special_case and chosen_docking_distance_m is not None:
            validation_target_distance_m = float(chosen_docking_distance_m)
            validation_distance_mode = "chosen_docking_distance"
        validation_tolerance_m = float(self.docking_distance_tolerance_m)
        self.last_precise_adjust_debug["distance_validation_target_m"] = round(validation_target_distance_m, 4)
        self.last_precise_adjust_debug["distance_validation_tolerance_m"] = round(validation_tolerance_m, 4)
        self.last_precise_adjust_debug["distance_validation_mode"] = validation_distance_mode

        # 找最佳匹配的同名物品（综合距离、朝向、视野居中）
        best_obj = None
        best_score = float('inf')  # 越小越好
        best_distance = 0
        best_angle = 0
        best_projection = None
        best_center_dist = float('inf')

        print(f"   [验证] 检查 {len(all_matching_objects)} 个同名物品...")

        # Y坐标补偿：射线检测通常命中物品底部，视觉中心约高5cm
        Y_OFFSET_COMPENSATION = 0.00

        for obj in all_matching_objects:
            obj_pos = obj.position
            # 应用Y补偿用于投影计算
            obj_pos_compensated = (obj_pos[0], obj_pos[1] + Y_OFFSET_COMPENSATION, obj_pos[2])

            # 计算2D距离（X-Z平面，不含高度差）
            dist = math.sqrt(
                (obj_pos[0] - camera_3d[0])**2 +
                (obj_pos[2] - camera_3d[2])**2
            )

            # 计算3D朝向夹角
            dx = obj_pos[0] - camera_3d[0]
            dy = obj_pos[1] - camera_3d[1]
            dz = obj_pos[2] - camera_3d[2]
            dist_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist_3d > 0.01:
                target_dir = (dx / dist_3d, dy / dist_3d, dz / dist_3d)
                dot_product = (forward_vec[0] * target_dir[0] +
                              forward_vec[1] * target_dir[1] +
                              forward_vec[2] * target_dir[2])
                dot_product = max(-1.0, min(1.0, dot_product))
                angle = math.degrees(math.acos(dot_product))
            else:
                angle = 0.0

            # 计算视野投影偏移（使用Y补偿后的坐标）
            projection = self._project_to_screen_v2(obj_pos_compensated, camera_info_final)
            if projection:
                screen_x, screen_y = projection
                center_dist = math.sqrt((screen_x - 800)**2 + (screen_y - 450)**2)  # 1600x900 屏幕中心
            else:
                center_dist = 9999  # 不在视野内，给个大值

            # 综合评分（视野居中权重最大，因为这是最直接的判断标准）
            # 距离偏离130cm的程度 + 角度/10 + 视野偏移/5
            dist_penalty = abs(dist - validation_target_distance_m) * 100  # cm偏离
            score = center_dist / 5 + angle / 10 + dist_penalty

            if score < best_score:
                best_score = score
                best_obj = obj
                best_distance = dist
                best_angle = angle
                best_projection = projection
                best_center_dist = center_dist

        if not best_obj:
            print(f"   [最终验证] FAILED: 没有找到合适的目标物品")
            return False

        target_3d = best_obj.position
        distance_to_target = best_distance
        angle_3d = best_angle

        # 1. 验证距离（2D水平距离，130cm ± 20cm）
        distance_ok = (
            validation_target_distance_m - validation_tolerance_m
            <= distance_to_target
            <= validation_target_distance_m + validation_tolerance_m
        )
        # Avoid Unicode symbols (e.g., ✓/✗) which can crash on GBK consoles.
        print(
            f"   [验证1/3] 2D距离目标: {distance_to_target*100:.1f}cm "
            f"(要求: {(validation_target_distance_m - validation_tolerance_m)*100:.0f}-"
            f"{(validation_target_distance_m + validation_tolerance_m)*100:.0f}cm, "
            f"mode={validation_distance_mode}) "
            f"{'OK' if distance_ok else 'FAIL'}"
        )

        # 2. 验证朝向
        yaw_ok = angle_3d <= 50.0
        print(f"   [验证2/3] 朝向误差(3D夹角): {angle_3d:.1f}° (要求: ≤50°) {'OK' if yaw_ok else 'FAIL'}")

        # 3. 验证视野居中（使用最佳匹配物品）
        if best_projection:
            screen_x, screen_y = best_projection
            center_distance = best_center_dist
            center_tolerance_pixels = self._center_tolerance_pixels_for_target(default_pixels=10.0)
            center_ok = center_distance <= center_tolerance_pixels
            print(f"   [verify3/3] center: projection({screen_x}, {screen_y}) dist_to_center={center_distance:.1f}px (require: <= {center_tolerance_pixels:.0f}px) {'OK' if center_ok else 'FAIL'}")
        else:
            center_ok = False
            center_distance = 9999
            print("   [验证3/3] 视野居中: 目标不在视野内 FAIL")

        interaction_available = False
        interaction_ok = False
        interaction_reason = "not_required"
        if has_container_special_case:
            interaction_state = interaction_probe_debug if isinstance(interaction_probe_debug, dict) else {}
            interaction_available = bool(interaction_state.get("snapshot_available"))
            interaction_ok = bool(interaction_state.get("success"))
            interaction_match_type = str(interaction_state.get("matched_state", {}).get("match_type") or "none")
            if interaction_available:
                interaction_reason = str(interaction_state.get("stop_reason") or "not_detected")
                print(
                    f"   [验证4/4] Alt+J交互命中目标: "
                    f"{'OK' if interaction_ok else 'FAIL'} "
                    f"(reason={interaction_reason}, match_type={interaction_match_type})"
                )
            else:
                interaction_reason = "snapshot_unavailable"
                print("   [验证4/4] Alt+J交互命中目标: SKIP (snapshot unavailable)")

        view_ok = center_ok or (has_container_special_case and interaction_ok)
        interaction_required = bool(has_container_special_case and interaction_available)
        all_ok = distance_ok and yaw_ok and view_ok and (not interaction_required or interaction_ok)

        print(f"\n   [最终判定] {'成功 OK' if all_ok else 'FAILED FAIL'}")
        if all_ok:
            print(f"               距离: {distance_to_target*100:.1f}cm")
            print(f"               朝向误差(3D): {angle_3d:.1f}°")
            if best_projection:
                print(f"               视野投影: ({best_projection[0]}, {best_projection[1]})")
            if has_container_special_case and interaction_available:
                print(f"               Alt+J命中: {interaction_reason}")
        else:
            print(f"               未满足所有条件，导航失败")
            if not distance_ok:
                print(
                    f"               - 距离{distance_to_target*100:.1f}cm不在"
                    f"{(validation_target_distance_m - validation_tolerance_m)*100:.0f}-"
                    f"{(validation_target_distance_m + validation_tolerance_m)*100:.0f}cm范围内"
                )
            if not yaw_ok:
                print(f"               - 朝向误差(3D夹角){angle_3d:.1f}°超过50°")
            if not view_ok:
                if best_projection:
                    print(f"               - 视野偏移{center_distance:.1f}px超过10px")
                else:
                    print(f"               - 目标不在视野内")
            if interaction_required and not interaction_ok:
                if interaction_match_type == "container_contents_only":
                    print("               - Alt+J只显示了容器及其内容，尚未直接显示目标物品唯一名称")
                else:
                    print("               - Alt+J未命中目标物品；容器可能遮挡了容器内物品")

        return all_ok

    def _navigate_to_position(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float], skip_radar_check: bool = False) -> bool:
        """
        大距离移动：转向目标位置，然后前进

        Args:
            current_pos: 当前位置 (x, z)
            target_pos: 目标位置 (x, z)
            skip_radar_check: 是否跳过雷达检查（精确调整阶段使用）

        Returns:
            是否成功移动
        """
        if not LOCAL_ACTIONS_AVAILABLE:
            return True

        # 计算方向
        dx = target_pos[0] - current_pos[0]
        dz = target_pos[1] - current_pos[1]
        distance = math.sqrt(dx*dx + dz*dz)

        if distance < 0.01:
            return True

        # 计算需要的朝向
        target_yaw = math.degrees(math.atan2(dx, dz)) % 360

        # 读取当前朝向
        camera_info = self._load_camera_info_with_retry()
        if not camera_info:
            return False

        current_yaw = camera_info.rotation[1]

        # 转向目标方向
        angle_diff = target_yaw - current_yaw
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        print(f"            [转向] 当前{current_yaw:.1f}° → 目标{target_yaw:.1f}°，需转{angle_diff:.1f}°")
        self._turn_to_target_yaw(target_yaw, current_yaw, tolerance=15.0)

        # 前进（根据距离决定时长）- v3.7.3: 适度提高移动时长
        move_duration = min(0.15, distance / 2.0)  # 最多0.15秒（从0.12提高）
        move_duration = max(0.08, move_duration)   # 最少0.08秒（从0.06提高）

        # 检查前方是否安全（精确调整阶段可跳过）
        if not skip_radar_check:
            radar_readings = self.radar_reader.get_all_readings()
            if not self._check_direction_safe(current_yaw, 0, radar_readings, check_angle=30):
                return False  # 前方不安全

        local_actions.move_forward(duration=move_duration)
        time.sleep(0.20)  # 等待动作完成 - v3.7.1: 从0.35降低到0.20配合更短的移动
        return True

    def _micro_move_towards(self, current_pos: Tuple[float, float], target_pos: Tuple[float, float], skip_radar_check: bool = False) -> bool:
        """
        小幅移动一步（考虑雷达避障）

        修复(2025-12-11): 将世界坐标系方向转换为玩家本地坐标系，
        解决朝向不同时移动方向错误的bug。

        Args:
            current_pos: 当前位置 (x, z)
            target_pos: 目标位置 (x, z)
            skip_radar_check: 是否跳过雷达检查（精确调整阶段使用）

        Returns:
            是否成功移动（False表示被雷达阻止）
        """
        if not LOCAL_ACTIONS_AVAILABLE:
            return True

        # 计算世界坐标系中的方向向量
        world_dx = target_pos[0] - current_pos[0]
        world_dz = target_pos[1] - current_pos[1]
        dist = math.sqrt(world_dx*world_dx + world_dz*world_dz)

        if dist < 0.01:  # 太近了，不需要移动
            return True

        # 归一化方向
        world_dx /= dist
        world_dz /= dist

        # 决定移动方向 - v3.7.3: 从0.03提高到0.05，步子稍大些
        move_duration = 0.05  # 小幅移动，约5-8cm

        # 读取雷达和当前朝向
        # 注意：即使跳过雷达检查，也需要读取朝向来正确计算移动方向！
        camera_info = self._load_camera_info_with_retry()
        if not camera_info:
            return True
        current_yaw = camera_info.rotation[1]

        if not skip_radar_check:
            radar_readings = self.radar_reader.get_all_readings()
        else:
            radar_readings = None

        # ========== 关键修复：世界坐标系 -> 玩家本地坐标系 ==========
        # 玩家朝向 yaw 角度（Unity中：0°=北/Z+, 90°=东/X+, 180°=南/Z-, 270°=西/X-）
        #
        # 世界坐标系中的目标方向 (world_dx, world_dz) 需要转换为玩家视角的方向：
        # - local_forward: 玩家前方分量（正=前进，负=后退）
        # - local_right: 玩家右方分量（正=右移，负=左移）
        #
        # 转换公式（将世界方向旋转 -yaw 度到玩家本地坐标系）：
        # local_forward = world_dx * sin(yaw) + world_dz * cos(yaw)
        # local_right = world_dx * cos(yaw) - world_dz * sin(yaw)

        yaw_rad = math.radians(current_yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        local_forward = world_dx * sin_yaw + world_dz * cos_yaw  # 玩家前方分量
        local_right = world_dx * cos_yaw - world_dz * sin_yaw    # 玩家右方分量

        # 基于玩家本地坐标系决定移动方向
        if abs(local_forward) > abs(local_right):
            # 主要前后移动
            if local_forward > 0:
                # 目标在玩家前方 → 前进
                if not skip_radar_check and not self._check_direction_safe(current_yaw, 0, radar_readings, check_angle=30):
                    return False  # 前方不安全
                local_actions.move_forward(duration=move_duration)
            else:
                # 目标在玩家后方 → 后退
                if not skip_radar_check and not self._check_direction_safe(current_yaw, 180, radar_readings, check_angle=30):
                    return False  # 后方不安全
                local_actions.move_backward(duration=move_duration)
        else:
            # 主要左右移动
            if local_right > 0:
                # 目标在玩家右方 → 右移
                if not skip_radar_check and not self._check_direction_safe(current_yaw, 90, radar_readings, check_angle=30):
                    return False  # 右侧不安全
                local_actions.move_right(duration=move_duration)
            else:
                # 目标在玩家左方 → 左移
                if not skip_radar_check and not self._check_direction_safe(current_yaw, -90, radar_readings, check_angle=30):
                    return False  # 左侧不安全
                local_actions.move_left(duration=move_duration)

        return True

    def _adjust_orientation_with_feedback(self, target_yaw: float, tolerance: float = 15.0, max_attempts: int = 5) -> bool:
        """
        带反馈的精确朝向调整（多次迭代微调）

        Args:
            target_yaw: 目标朝向角度（度）
            tolerance: 容忍度（度）
            max_attempts: 最大尝试次数（自动调整）

        Returns:
            是否成功调整到目标朝向
        """
        if not LOCAL_ACTIONS_AVAILABLE:
            return True

        # 读取初始朝向，根据误差大小调整最大尝试次数
        camera_info_init = self._load_camera_info_with_retry()
        if camera_info_init:
            current_yaw_init = camera_info_init.rotation[1]
            angle_diff_init = target_yaw - current_yaw_init
            while angle_diff_init > 180:
                angle_diff_init -= 360
            while angle_diff_init < -180:
                angle_diff_init += 360

            # 根据初始误差动态调整最大尝试次数
            # 每次转向约25°，需要 abs(angle_diff_init) / 25 次
            required_attempts = int(abs(angle_diff_init) / 25) + 2  # +2作为缓冲
            max_attempts = max(max_attempts, min(required_attempts, 50))  # 上限50次

            print(f"      朝向调整: 初始误差{abs(angle_diff_init):.1f}°，自动设置最多{max_attempts}次尝试")

        for attempt in range(max_attempts):
            # 1. 读取当前朝向
            camera_info = self._load_camera_info_with_retry()
            if not camera_info:
                print(f"      [尝试{attempt+1}/{max_attempts}] 无法读取相机信息")
                return False

            current_yaw = camera_info.rotation[1]

            # 2. 计算角度差（标准化到-180~180）
            angle_diff = target_yaw - current_yaw
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360

            # 3. 检查是否已达标
            if abs(angle_diff) <= tolerance:
                print(f"      朝向调整成功: 当前{current_yaw:.1f}°, 目标{target_yaw:.1f}°, 误差{angle_diff:.1f}°")
                return True

            # 4. 执行转向（逐步逼近）
            print(f"      [尝试{attempt+1}/{max_attempts}] 当前{current_yaw:.1f}° → 目标{target_yaw:.1f}° (差{angle_diff:.1f}°)")

            # 计算转向像素数（使用较小的灵敏度避免过冲）
            pixels = int(abs(angle_diff) * self.turn_sensitivity * 0.8)
            pixels = max(20, min(pixels, 300))

            if angle_diff > 0:
                local_actions.look_right(pixels=pixels)
            else:
                local_actions.look_left(pixels=pixels)

            # 5. 等待视角稳定
            time.sleep(0.15)

        # 最大尝试后仍未达标
        camera_info_final = self._load_camera_info_with_retry()
        if camera_info_final:
            current_yaw_final = camera_info_final.rotation[1]
            angle_diff_final = target_yaw - current_yaw_final
            while angle_diff_final > 180:
                angle_diff_final -= 360
            while angle_diff_final < -180:
                angle_diff_final += 360

            print(f"      WARNING: 朝向调整未完全达标（{max_attempts}次尝试后）")
            print(f"      最终: 当前{current_yaw_final:.1f}°, 目标{target_yaw:.1f}°, 误差{angle_diff_final:.1f}°")

        return False

    def _adjust_orientation_with_forward_feedback(self, target_3d: Tuple[float, float, float], tolerance: float = 5.0, max_attempts: int = 10) -> bool:
        """
        用Forward向量反馈调整朝向（不依赖euler角）

        Args:
            target_3d: 目标3D位置 (x, y, z)
            tolerance: 容忍度（度，3D夹角）
            max_attempts: 最大尝试次数

        Returns:
            是否成功调整
        """
        if not LOCAL_ACTIONS_AVAILABLE:
            return True

        # 震荡检测：记录最近的角度历史
        angle_history = []
        oscillation_count = 0

        for attempt in range(max_attempts):
            # 1. 读取当前相机状态
            camera_info = self._load_camera_info_with_retry()
            if not camera_info:
                return False

            camera_3d = camera_info.position
            forward_vec = camera_info.forward

            # 2. 计算目标方向
            dx = target_3d[0] - camera_3d[0]
            dy = target_3d[1] - camera_3d[1]
            dz = target_3d[2] - camera_3d[2]
            distance_3d = math.sqrt(dx*dx + dy*dy + dz*dz)

            if distance_3d < 0.01:
                return True

            target_dir = (dx / distance_3d, dy / distance_3d, dz / distance_3d)

            # 3. 计算Forward向量和目标方向的夹角
            dot_product = (forward_vec[0] * target_dir[0] +
                          forward_vec[1] * target_dir[1] +
                          forward_vec[2] * target_dir[2])
            dot_product = max(-1.0, min(1.0, dot_product))
            angle_3d = math.degrees(math.acos(dot_product))

            # 震荡检测：记录当前角度
            angle_history.append(angle_3d)
            if len(angle_history) > 10:
                angle_history.pop(0)  # 只保留最近10次

            # 检测震荡：如果最近6次角度基本不变（波动<1.0°）
            # v3.7.3: 从0.5°提高到1.0°，更早检测到震荡
            if len(angle_history) >= 6:
                recent_angles = angle_history[-6:]
                angle_variance = max(recent_angles) - min(recent_angles)
                if angle_variance < 1.0:  # 角度变化很小，说明已经收敛或震荡
                    oscillation_count += 1
                    if oscillation_count >= 2:  # 连续2轮检测到震荡就退出（从3降到2）
                        print(f"      检测到收敛/震荡（角度在{min(recent_angles):.1f}°-{max(recent_angles):.1f}°间），提前退出")
                        print(f"      当前3D夹角={angle_3d:.1f}°，已尽最大努力")
                        return angle_3d <= tolerance * 5  # 放宽5倍容忍度（1°→5°）
                else:
                    oscillation_count = 0

            # 4. 检查是否达标
            if angle_3d <= tolerance:
                print(f"      朝向调整成功: 3D夹角={angle_3d:.1f}°（尝试{attempt+1}次）")
                return True

            # 减少日志输出频率（每20次或前3次才打印）
            if attempt < 3 or (attempt + 1) % 20 == 0:
                print(f"      [尝试{attempt+1}/{max_attempts}] 3D夹角={angle_3d:.1f}°")

            # 5. 计算水平偏差方向（判断应该左转还是右转）
            # 投影到XZ平面（忽略Y）
            forward_xz = (forward_vec[0], forward_vec[2])  # (X, Z)
            target_xz = (dx, dz)

            # 叉乘判断方向：forward × target 的Y分量
            # 如果 > 0，目标在左侧，需要左转
            # 如果 < 0，目标在右侧，需要右转
            cross_y = forward_xz[0] * target_xz[1] - forward_xz[1] * target_xz[0]

            # ========== 关键修复：单独计算水平角度误差 ==========
            # 问题：之前用3D夹角(angle_3d)决定像素步长，但只调yaw
            #       当pitch有偏差时，3D角可能10°，但yaw误差只有2-3°
            #       用50像素步长（对应3.5°）调整2-3°的误差 → 必然震荡
            # 解决：计算水平角度误差，用它来决定yaw的步长
            forward_xz_len = math.sqrt(forward_xz[0]**2 + forward_xz[1]**2)
            target_xz_len = math.sqrt(target_xz[0]**2 + target_xz[1]**2)

            if forward_xz_len > 0.01 and target_xz_len > 0.01:
                # 归一化后计算水平角度
                forward_xz_norm = (forward_xz[0]/forward_xz_len, forward_xz[1]/forward_xz_len)
                target_xz_norm = (target_xz[0]/target_xz_len, target_xz[1]/target_xz_len)
                dot_xz = forward_xz_norm[0]*target_xz_norm[0] + forward_xz_norm[1]*target_xz_norm[1]
                dot_xz = max(-1.0, min(1.0, dot_xz))
                horizontal_angle = math.degrees(math.acos(dot_xz))
            else:
                horizontal_angle = angle_3d  # 退化情况

            # 6. 小幅调整（基于水平角度误差，不是3D角度）
            # v3.7.3: 优化像素步长，减少震荡
            if horizontal_angle > 10.0:
                pixels = 50  # 水平大误差
            elif horizontal_angle > 5.0:
                pixels = 30  # 水平中误差
            elif horizontal_angle > 2.0:
                pixels = 12  # 水平小误差（从15降到12）
            elif horizontal_angle > 0.8:
                pixels = 5   # 水平很小误差（从8降到5）
            else:
                pixels = 1   # 水平极小误差（从3降到1，减少震荡）

            if cross_y > 0:
                local_actions.look_left(pixels=pixels)
            else:
                local_actions.look_right(pixels=pixels)

            # 7. 垂直调整（pitch）- v3.7.2: 启用PITCH调整，避免阶段3大幅调整
            # 计算垂直角度误差
            # forward_vec.y 是相机向前的垂直分量，target_dir.y 是目标方向的垂直分量
            pitch_diff = target_dir[1] - forward_vec[1]  # 需要调整的垂直分量差

            # 将分量差转换为角度估计（近似）
            # forward.y ≈ -sin(pitch)，target.y 也类似
            vertical_angle_error = math.degrees(math.asin(max(-1, min(1, pitch_diff))))

            if abs(vertical_angle_error) > 2.0:  # 垂直误差超过2度才调整
                # 计算垂直调整像素
                if abs(vertical_angle_error) > 15.0:
                    v_pixels = 50
                elif abs(vertical_angle_error) > 8.0:
                    v_pixels = 30
                elif abs(vertical_angle_error) > 4.0:
                    v_pixels = 15
                else:
                    v_pixels = 8

                if pitch_diff > 0:  # 目标在上方，需要抬头
                    local_actions.look_up(pixels=v_pixels)
                else:  # 目标在下方，需要低头
                    local_actions.look_down(pixels=v_pixels)

            time.sleep(0.15)  # v3.7.3: 从0.12增加到0.15，减少读取频率降低卡顿

        # 最大尝试后仍未达标
        print(f"      WARNING: 朝向调整未完全达标（{max_attempts}次尝试后，3D夹角={angle_3d:.1f}°）")
        return False

    def _turn_to_target_yaw(self, target_yaw: float, current_yaw: float, tolerance: float = 15.0) -> bool:
        """
        转向目标方向（快速版，用于导航过程中）

        注意：到达目标后的精确朝向调整请使用 _adjust_orientation_with_feedback()
        """
        if not LOCAL_ACTIONS_AVAILABLE:
            return True

        # 计算角度差
        angle_diff = target_yaw - current_yaw
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360

        # 如果角度差小于容忍度，不转向
        if abs(angle_diff) < tolerance:
            return True

        # 计算转向像素（降低灵敏度）
        pixels = int(abs(angle_diff) * self.turn_sensitivity * 0.5)
        pixels = max(30, min(pixels, 350))

        # 执行转向
        if angle_diff > 0:
            local_actions.look_right(pixels=pixels)
        else:
            local_actions.look_left(pixels=pixels)

        time.sleep(0.08)
        return True

    def _load_camera_info_with_retry(self, max_retries: int = 3):
        """带重试的相机信息读取"""
        for attempt in range(max_retries):
            try:
                camera_info = self.data_loader.load_camera_info()
                if camera_info:
                    return camera_info
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.05)
                    continue
        return None

    def _project_to_screen_v2(self, obj_world_pos, camera_info, screen_width=1600, screen_height=900):
        """
        将世界坐标投影到屏幕坐标（v2.0：使用Forward向量，更可靠）

        参数:
            obj_world_pos: (x, y, z) 物品世界坐标
            camera_info: CameraInfo对象

        返回:
            (screen_x, screen_y) 或 None（如果在视野外）
        """
        camera_world_pos = camera_info.position
        camera_pitch, camera_yaw, camera_roll = camera_info.rotation

        # Use the full camera rotation to avoid basis degeneration near vertical pitch.
        proj = CameraProjection()
        result = proj.project_with_rotation(
            obj_world_pos,
            camera_world_pos,
            camera_yaw,
            camera_pitch,
            camera_roll,
        )

        if result is None:
            return None

        screen_x, screen_y = result

        # 检查是否在屏幕范围内（允许一定的边界外延）
        if screen_x < -screen_width * 0.5 or screen_x > screen_width * 1.5:
            return None
        if screen_y < -screen_height * 0.5 or screen_y > screen_height * 1.5:
            return None

        return (int(screen_x), int(screen_y))

    def _normalize_pitch_degrees(self, pitch_degrees: float) -> float:
        normalized_pitch = float(pitch_degrees)
        if normalized_pitch > 180.0:
            normalized_pitch -= 360.0
        elif normalized_pitch < -180.0:
            normalized_pitch += 360.0
        return normalized_pitch

    def _normalize_angle_delta(self, angle_delta: float) -> float:
        while angle_delta > 180.0:
            angle_delta -= 360.0
        while angle_delta < -180.0:
            angle_delta += 360.0
        return angle_delta

    def _compute_target_yaw(self, target_3d: Tuple[float, float, float], camera_pos: Tuple[float, float, float]) -> float:
        dx = target_3d[0] - camera_pos[0]
        dz = target_3d[2] - camera_pos[2]
        return math.degrees(math.atan2(dx, dz)) % 360.0

    def _prealign_yaw_for_pitch_escape(
        self,
        target_3d: Tuple[float, float, float],
        camera_info,
        yaw_tolerance: float = 10.0,
    ) -> bool:
        if not LOCAL_ACTIONS_AVAILABLE:
            return True

        target_yaw = self._compute_target_yaw(target_3d, camera_info.position)
        current_yaw = float(camera_info.rotation[1])
        angle_diff = self._normalize_angle_delta(target_yaw - current_yaw)
        if abs(angle_diff) <= yaw_tolerance:
            return True

        pixels = int(abs(angle_diff) * self.turn_sensitivity * 0.6)
        pixels = max(30, min(pixels, 220))
        if angle_diff > 0:
            local_actions.look_right(pixels=pixels)
        else:
            local_actions.look_left(pixels=pixels)
        time.sleep(0.08)
        return False

    def _escape_extreme_pitch(
        self,
        normalized_pitch: float,
        target_3d: Tuple[float, float, float],
        camera_info,
        attempt: int,
        cycles: int = 4,
    ) -> None:
        if not LOCAL_ACTIONS_AVAILABLE:
            return

        target_yaw = self._compute_target_yaw(target_3d, camera_info.position)
        current_yaw = float(camera_info.rotation[1])
        yaw_diff = self._normalize_angle_delta(target_yaw - current_yaw)
        vertical_pixels = 95
        horizontal_base = 28
        escape_downward = normalized_pitch < 0.0

        for escape_idx in range(cycles):
            horizontal_pixels = horizontal_base + min(escape_idx * 6, 24)
            if abs(yaw_diff) > 4.0:
                move_right = yaw_diff > 0.0
            else:
                move_right = ((attempt + escape_idx) % 2 == 0)

            if move_right:
                local_actions.look_right(pixels=horizontal_pixels)
            else:
                local_actions.look_left(pixels=horizontal_pixels)
            time.sleep(0.03)

            if escape_downward:
                local_actions.look_down(pixels=vertical_pixels)
            else:
                local_actions.look_up(pixels=vertical_pixels)
            time.sleep(0.05)

    def _adjust_to_center_view_v2(self, target_pos: Tuple[float, float], tolerance_pixels: float = 5, max_attempts: int = 50) -> bool:
        """
        Rotation-based screen-centering adjustment for the target.

        Args:
            target_pos: target world position `(x, z)`
            tolerance_pixels: allowed pixel error from screen center
            max_attempts: maximum adjustment iterations

        Returns:
            True if the target is centered successfully, else False.
        """
        if not LOCAL_ACTIONS_AVAILABLE:
            return True

        _tname = getattr(self, 'target_name', 'unknown')
        _tid = getattr(self, 'target_instance_id', None)
        if _tid is not None:
            target_obj = self.data_loader.find_object_by_name_and_instance_id(_tname, int(_tid))
        else:
            target_obj = self.data_loader.find_object_by_name(_tname)
        if not target_obj:
            print("            ERROR: target object not found")
            return False

        target_3d = (
            target_obj.position[0],
            target_obj.position[1],
            target_obj.position[2],
        )
        center_x, center_y = 800, 450
        safe_pitch_limit = 80.0
        extreme_pitch_limit = 85.0
        is_knife_target = self._is_knife_target()
        knife_min_step_pixels = 1
        knife_axis_deadzone_pixels = 1

        for attempt in range(max_attempts):
            camera_info = self._load_camera_info_with_retry()
            if not camera_info:
                print(f"            [attempt {attempt+1}/{max_attempts}] failed to load camera info")
                return False

            camera_pitch, camera_yaw, camera_roll = camera_info.rotation
            normalized_pitch = self._normalize_pitch_degrees(camera_pitch)

            if abs(normalized_pitch) >= extreme_pitch_limit:
                if not self._prealign_yaw_for_pitch_escape(target_3d, camera_info):
                    print(
                        f"            [attempt {attempt+1}/{max_attempts}] "
                        f"extreme pitch前先水平对准目标"
                    )
                    time.sleep(0.12)
                    continue
                print(
                    f"            [attempt {attempt+1}/{max_attempts}] "
                    f"extreme pitch detected pitch={normalized_pitch:.1f}°, running escape"
                )
                self._escape_extreme_pitch(
                    normalized_pitch,
                    target_3d,
                    camera_info,
                    attempt,
                    cycles=4,
                )
                time.sleep(0.12)
                continue

            projection = self._project_to_screen_v2(target_3d, camera_info)

            if projection is None:
                print(f"            [attempt {attempt+1}/{max_attempts}] target projection unavailable")
                proj = CameraProjection()
                x_in_cam, y_in_cam, _ = proj.get_camera_space_offset_with_rotation(
                    target_3d,
                    camera_info.position,
                    camera_yaw,
                    camera_pitch,
                    camera_roll,
                )

                if abs(x_in_cam) > 0.05:
                    if x_in_cam > 0:
                        local_actions.look_right(pixels=80)
                    else:
                        local_actions.look_left(pixels=80)

                if abs(y_in_cam) > 0.05:
                    if y_in_cam > 0:
                        local_actions.look_down(pixels=80)
                    else:
                        local_actions.look_up(pixels=80)

                time.sleep(0.2)
                continue

            screen_x, screen_y = projection
            offset_x = screen_x - center_x
            offset_y = screen_y - center_y
            distance = math.sqrt(offset_x**2 + offset_y**2)

            if distance <= tolerance_pixels:
                print(f"            Centered: screen=({screen_x}, {screen_y}) distance={distance:.1f}px")
                return True

            if attempt % 20 == 0 or attempt < 3:
                print(
                    f"            [attempt {attempt+1}/{max_attempts}] "
                    f"screen=({screen_x}, {screen_y}) offsetX={offset_x}px offsetY={offset_y}px"
                )

            sensitivity = 0.6 if is_knife_target else 0.5
            move_x = int(offset_x * sensitivity)
            move_y = int(offset_y * sensitivity)
            move_x = max(-100, min(100, move_x))
            move_y = max(-100, min(100, move_y))
            if is_knife_target:
                if abs(offset_x) > knife_axis_deadzone_pixels and move_x == 0:
                    move_x = knife_min_step_pixels if offset_x > 0 else -knife_min_step_pixels
                if abs(offset_y) > knife_axis_deadzone_pixels and move_y == 0:
                    move_y = knife_min_step_pixels if offset_y > 0 else -knife_min_step_pixels

            pushing_deeper_into_vertical_lock = (
                (move_y > 0 and normalized_pitch >= safe_pitch_limit)
                or (move_y < 0 and normalized_pitch <= -safe_pitch_limit)
            )
            if pushing_deeper_into_vertical_lock:
                print(
                    f"            [attempt {attempt+1}/{max_attempts}] "
                    f"pitch={normalized_pitch:.1f}° would push deeper into vertical lock, escape first"
                )
                self._escape_extreme_pitch(
                    normalized_pitch,
                    target_3d,
                    camera_info,
                    attempt,
                    cycles=2,
                )
                time.sleep(0.12)
                continue

            if abs(move_x) > 3 or (is_knife_target and abs(offset_x) > knife_axis_deadzone_pixels and abs(move_x) > 0):
                if move_x > 0:
                    local_actions.look_right(pixels=abs(move_x))
                else:
                    local_actions.look_left(pixels=abs(move_x))

            if abs(move_y) > 3 or (is_knife_target and abs(offset_y) > knife_axis_deadzone_pixels and abs(move_y) > 0):
                if move_y > 0:
                    local_actions.look_down(pixels=abs(move_y))
                else:
                    local_actions.look_up(pixels=abs(move_y))

            time.sleep(0.15)

        print(f"            WARNING: failed to center target within {max_attempts} attempts")
        return False

    def _adjust_to_center_view(self, target_pos: Tuple[float, float], tolerance_pixels: float = 5, max_attempts: int = 50) -> bool:
        """Legacy entrypoint kept for compatibility; routes to the unified rotation-based centering path."""
        return self._adjust_to_center_view_v2(
            target_pos,
            tolerance_pixels=tolerance_pixels,
            max_attempts=max_attempts,
        )

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='简化版A*+雷达导航器')
    parser.add_argument('object_name', help='目标物品名称')
    parser.add_argument(
        '--map',
        default=os.environ.get(
            "COOKGAME_AUTO_NAV_MAP",
            os.path.join("data", "auto_nav_data", "obstacle_map.json"),
        ),
        help='地图文件路径'
    )

    args = parser.parse_args()

    # 检查地图文件
    if not os.path.exists(args.map):
        print(f"\n[错误] 地图文件不存在: {args.map}")
        return

    print(f"\n[使用地图] {os.path.basename(args.map)}")

    # 创建导航器
    navigator = SimpleAstarRadarNavigator(args.map)

    # 初始化
    if not navigator.initialize():
        print("\n[错误] 导航器初始化失败")
        return

    # 执行导航
    success = navigator.navigate_to_object(args.object_name)

    if success:
        print("\n" + "=" * 70)
        print("  导航成功！")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("  导航失败")
        print("=" * 70)


if __name__ == "__main__":
    main()
