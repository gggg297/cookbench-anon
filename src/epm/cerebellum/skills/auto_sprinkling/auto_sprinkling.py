"""
Cooking Simulator - Auto Sprinkling (Spices)

This module mirrors `auto_pouring`'s control flow, but for spices:
- Reads `is_sprinkle_mode` from `realtime_products.json` (CS_CamDump)
- Uses repeated dumping clicks (no tilt control)
- Uses Alt+J interaction feed for grams (best-effort)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import math
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPRINKLE_SCRIPT_VERSION = "2026-01-11a"

# EPM note:
# The upstream script relied on sys.path hacks and `deploy_agent/local_actions.py`.
# In this repo we use `epm.cerebellum.local_actions` and absolute imports so this
# module can be imported safely (e.g. via tool calls).
from epm.cerebellum.skills.auto_sprinkling.config import (  # noqa: E402
    ACTION_DELAY_SECONDS,
    DEFAULT_SPRINKLE_NUM,
    GAME_WINDOW_TITLE,
    GRAMS_TRACKING_ENABLED,
    GRAMS_TRACKING_MODE,
    POST_CLICK_SLEEP_SECONDS,
    REALTIME_PRODUCTS_JSON,
)

from epm.cerebellum.local_actions import (  # noqa: E402
    _activate_window,
    click_mouse,
    enter_spices_sprinkle_mode,
    exit_spices_sprinkle_mode,
    horizontal_movement,
)


POSITION_TOLERANCE_M = 0.015
MAX_ALIGN_ITERATIONS = 25
ACTION_DELAY = float(ACTION_DELAY_SECONDS)
CALIBRATION_PIXELS = 50
CALIBRATION_MIN_DISPLACEMENT_M = 0.001
ALIGN_MAX_STEP_PX = 80
ALIGN_NEAR_MAX_STEP_PX = 30
ALIGN_NEAR_THRESHOLD_M = 0.0006
ALIGN_MOVE_DELAY_S = 0.15
CLICK_DEADLINE_GUARD_S = 0.05
FIRST_CLICK_EXTRA_ALIGN_BUDGET_S = 1.5
SPRINKLE_CLICK_INTERVAL_S = 0.8


_LAST_READ_ERROR: Optional[str] = None


def _read_realtime_products() -> Optional[dict]:
    # Be tolerant to:
    # - BOM / encoding differences (some files may be UTF-8-SIG)
    # - partial writes (JSONDecodeError while the mod is writing)
    encodings = ("utf-8", "utf-8-sig", "gbk")
    global _LAST_READ_ERROR
    _LAST_READ_ERROR = None
    for _ in range(5):
        for enc in encodings:
            try:
                with open(REALTIME_PRODUCTS_JSON, "r", encoding=enc) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                _LAST_READ_ERROR = "JSONDecodeError (file may be mid-write)"
                continue
            except OSError:
                _LAST_READ_ERROR = f"OSError opening file: {REALTIME_PRODUCTS_JSON}"
                return None
            except Exception:
                _LAST_READ_ERROR = "Unknown error while reading/parsing realtime_products.json"
                continue
        time.sleep(0.05)
    return None


def _print_realtime_products_diagnostics(prefix: str) -> None:
    global _LAST_READ_ERROR
    try:
        exists = os.path.exists(REALTIME_PRODUCTS_JSON)
        size = os.path.getsize(REALTIME_PRODUCTS_JSON) if exists else -1
    except Exception:
        exists = False
        size = -1
    print(f"{prefix} realtime_products.json unreadable")
    print(f"  path: {REALTIME_PRODUCTS_JSON}")
    print(f"  exists: {exists}  size: {size}")
    if _LAST_READ_ERROR:
        print(f"  last_error: {_LAST_READ_ERROR}")


def _normalize_name(value: str) -> str:
    return (value or "").strip().lower()


def _is_spices_item(item: dict) -> bool:
    if not item:
        return False
    return _normalize_name(str(item.get("kind", ""))) == "spices"


def _find_held_item(data: dict) -> Optional[dict]:
    if not data:
        return None
    for p in data.get("products", []) or []:
        if p and p.get("is_held"):
            return p
    return None


def _find_sprinkling_spices_bottle(data: dict) -> Optional[dict]:
    """
    Find the spices bottle currently in sprinkle mode.

    We intentionally prefer `is_sprinkle_mode` over `is_held` because `is_held`
    can be unreliable depending on scan timing / mod behavior.
    """
    if not data:
        return None

    candidates = []
    for p in data.get("products", []) or []:
        if not p:
            continue
        if not _is_spices_item(p):
            continue
        if not bool(p.get("is_sprinkle_mode", p.get("is_sprinkle", False))):
            continue
        candidates.append(p)

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    on_screen = [p for p in candidates if p.get("is_on_screen", False)]
    if on_screen:
        return on_screen[0]
    candidates.sort(key=lambda x: float(x.get("distance", 9999) or 9999))
    return candidates[0]


def _find_target_item(data: dict, target_item: str, target_instance_id: Optional[int] = None) -> Optional[dict]:
    if not data:
        return None

    want = _normalize_name(target_item)
    if not want:
        return None

    best = None
    best_dist = None
    for p in data.get("products", []) or []:
        if not p:
            continue
        if target_instance_id is not None:
            try:
                if int(p.get("instance_id")) != int(target_instance_id):  # type: ignore[arg-type]
                    continue
            except Exception:
                continue

        name_en = _normalize_name(p.get("name_en", ""))
        name_cn = _normalize_name(p.get("name_cn", ""))
        game_object = _normalize_name(p.get("game_object", ""))

        if not (name_en == want or name_cn == want or game_object == want):
            continue

        dist = p.get("distance")
        try:
            dist = float(dist) if dist is not None else None
        except Exception:
            dist = None

        is_on_screen = bool(p.get("is_on_screen"))
        if best is None:
            best = p
            best_dist = dist
            continue

        if is_on_screen and not bool(best.get("is_on_screen")):
            best = p
            best_dist = dist
            continue
        if bool(best.get("is_on_screen")) and not is_on_screen:
            continue

        if dist is not None and (best_dist is None or dist < best_dist):
            best = p
            best_dist = dist

    return best


def _format_vec3(pos: dict) -> str:
    if not isinstance(pos, dict):
        return "(?, ?, ?)"
    try:
        return f"({float(pos.get('x', 0)):.3f}, {float(pos.get('y', 0)):.3f}, {float(pos.get('z', 0)):.3f})"
    except Exception:
        return "(?, ?, ?)"


def list_visible_items(limit: int = 20) -> list[dict]:
    data = _read_realtime_products()
    if not data:
        _print_realtime_products_diagnostics("[list] failed:")
        return []

    items = []
    for p in data.get("products", []) or []:
        if not p:
            continue
        if not p.get("is_on_screen"):
            continue
        items.append(p)

    items.sort(key=lambda x: float(x.get("distance", 9999) or 9999))
    items = items[: max(0, int(limit))]

    print(f"\nScreen-visible items (top {len(items)}):")
    for i, item in enumerate(items, 1):
        status = []
        if item.get("is_held"):
            status.append("held")
        if item.get("is_sprinkle_mode") or item.get("is_sprinkle"):
            status.append("sprinkle_mode")
        # `is_pouring_mode` is for liquids; spices should use `is_sprinkle_mode`.
        if item.get("is_pouring_mode") and not _is_spices_item(item):
            status.append("pour_mode")
        s = f" [{' '.join(status)}]" if status else ""
        print(
            f"{i:>2}. {item.get('name_en')} ({item.get('name_cn')})"
            f" kind={item.get('kind')} dist={float(item.get('distance', 0) or 0):.2f}m{s}"
        )
    print("")
    return items


def show_info(target_item: str) -> None:
    data = _read_realtime_products()
    if not data:
        _print_realtime_products_diagnostics("[info] failed:")
        return

    held_any = _find_held_item(data) or {}
    held_spices = _find_sprinkling_spices_bottle(data)
    target = _find_target_item(data, target_item)

    print("\n=== Auto Sprinkle Info ===")
    if held_any:
        print(
            f"Held: {held_any.get('name_en')} ({held_any.get('name_cn')})"
            f" kind={held_any.get('kind')} is_sprinkle_mode={bool(held_any.get('is_sprinkle_mode', held_any.get('is_sprinkle', False)))}"
        )
    else:
        print("Held: (none detected) -> pick up a spices bottle first")

    if held_any and not held_spices:
        print("Sprinkle bottle: not detected (need kind=spices and is_sprinkle_mode=true)")
        print("Tip: enter sprinkle mode first (it should flip is_sprinkle_mode=true).")

    if target:
        print(
            f"Target: {target.get('name_en')} ({target.get('name_cn')})"
            f" kind={target.get('kind')} on_screen={bool(target.get('is_on_screen', False))}"
        )
        print(f"  pos={_format_vec3(target.get('position'))} dist={float(target.get('distance', 0) or 0):.2f}m")
    else:
        print(f"Target: not found by exact name match: '{target_item}'")        
        print("Tip: use `python auto_sprinkling.py list` to see on-screen names.")

    g = _try_read_container_grams_from_interaction()
    if g is not None:
        print(f"Alt+J weight: {g:.2f} g")
    else:
        print("Alt+J weight: (no reading)")
    print("=========================\n")


def _get_json_file_mtime() -> float:
    try:
        return os.path.getmtime(REALTIME_PRODUCTS_JSON)
    except Exception:
        return 0.0


def _wait_for_data_update(pre_mtime: float, timeout: float = 2.0) -> bool:
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if _get_json_file_mtime() > pre_mtime:
            return True
        time.sleep(0.05)
    return False


def _wait_for_next_data_update() -> float:
    """
    Wait until realtime_products.json mtime advances at least once.

    Intentionally has no timeout: the scan/update rate may vary, and we prefer
    synchronizing actions to fresh frames over fixed sleeps.
    """
    pre = _get_json_file_mtime()
    while True:
        now = _get_json_file_mtime()
        if now > pre:
            return now
        time.sleep(0.01)


def _wait_for_is_sprinkle_true() -> None:
    """
    Block until realtime_products.json indicates we're holding a sprinkling spices bottle.
    """
    while True:
        _wait_for_next_data_update()
        data = _read_realtime_products() or {}
        bottle = _find_sprinkling_spices_bottle(data)
        if bottle is not None:
            return


def _move_spices_horizontal(mouse_dx: int, mouse_dy: int) -> None:
    horizontal_movement(int(mouse_dx), int(mouse_dy))
    # Give the game/mod time to apply the move before we read positions again.
    time.sleep(float(ALIGN_MOVE_DELAY_S))


def _get_pos_xz(item: dict) -> tuple[float, float]:
    pos = item.get("position") or {}
    try:
        return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
    except Exception:
        return 0.0, 0.0


def _get_spices_bottle_mouth_xz(item: dict) -> tuple[float, float]:
    # Prefer the scanner-provided spout_position if available.
    try:
        spout = item.get("spout_position") or {}
        if isinstance(spout, dict) and "x" in spout and "z" in spout:
            return float(spout.get("x", 0.0)), float(spout.get("z", 0.0))
    except Exception:
        pass
    return _get_pos_xz(item)


def _calibrate_mouse_to_3d_mapping_spices(verbose: bool = True) -> Optional[tuple[list[float], list[float]]]:
    if verbose:
        print("[*] Calibrating mouse->(x,z) mapping for spices bottle...")

    data = _read_realtime_products()
    bottle = _find_sprinkling_spices_bottle(data or {})
    if bottle is None:
        print("[!] Calibration failed: no sprinkling spices bottle detected (is_sprinkle_mode=true)")
        return None

    # Calibrate using the same reference point that alignment uses (spout/mouth if available).
    init_x, init_z = _get_spices_bottle_mouth_xz(bottle)
    if verbose:
        print(f"  initial (x,z)=({init_x:.4f},{init_z:.4f})")

    def read_bottle_pos(timeout_s: float = 1.2) -> Optional[tuple[float, float]]:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            data = _read_realtime_products()
            b = _find_sprinkling_spices_bottle(data or {})
            if b is not None:
                return _get_spices_bottle_mouth_xz(b)
            time.sleep(0.05)
        return None

    def measure_delta(move_dx: int, move_dy: int) -> Optional[tuple[float, float]]:
        _move_spices_horizontal(move_dx, move_dy)
        _wait_for_next_data_update()
        pos = read_bottle_pos(timeout_s=1.2)
        if pos is None:
            return None
        after_x, after_z = pos
        dx, dz = after_x - init_x, after_z - init_z

        _move_spices_horizontal(-move_dx, -move_dy)
        _wait_for_next_data_update()
        return dx, dz

    d1 = measure_delta(CALIBRATION_PIXELS, 0)
    if d1 is None:
        print("[!] Calibration failed: bottle missing after X move")
        return None
    dx1, dz1 = d1
    if verbose:
        print(f"  mouse X+{CALIBRATION_PIXELS}px -> Δx={dx1:.4f}, Δz={dz1:.4f}")

    # If X-only move doesn't budge enough, fall back to a diagonal probe.
    if (dx1 * dx1 + dz1 * dz1) ** 0.5 < CALIBRATION_MIN_DISPLACEMENT_M:
        d1b = measure_delta(CALIBRATION_PIXELS, CALIBRATION_PIXELS)
        if d1b is not None:
            dx1, dz1 = d1b
            if verbose:
                print(
                    f"  (fallback) mouse ({CALIBRATION_PIXELS},{CALIBRATION_PIXELS})px"
                    f" -> Δx={dx1:.4f}, Δz={dz1:.4f}"
                )

    d2 = measure_delta(0, CALIBRATION_PIXELS)
    if d2 is None:
        print("[!] Calibration failed: bottle missing after Y move")
        return None
    dx2, dz2 = d2
    if verbose:
        print(f"  mouse Y+{CALIBRATION_PIXELS}px -> Δx={dx2:.4f}, Δz={dz2:.4f}")

    total_disp = (dx1 * dx1 + dz1 * dz1) ** 0.5 + (dx2 * dx2 + dz2 * dz2) ** 0.5
    if total_disp < CALIBRATION_MIN_DISPLACEMENT_M * 2:
        print(f"[!] Calibration failed: displacement too small ({total_disp:.4f}m)")
        return None

    x_coeffs = [dx1 / CALIBRATION_PIXELS, dx2 / CALIBRATION_PIXELS]
    z_coeffs = [dz1 / CALIBRATION_PIXELS, dz2 / CALIBRATION_PIXELS]
    if verbose:
        print("  calibration:")
        print(f"    1px mouseX -> Δx={x_coeffs[0]:.6f}, Δz={z_coeffs[0]:.6f}")
        print(f"    1px mouseY -> Δx={x_coeffs[1]:.6f}, Δz={z_coeffs[1]:.6f}")
    return x_coeffs, z_coeffs


def _compute_mouse_movement(err_x: float, err_z: float, x_coeffs: list[float], z_coeffs: list[float]) -> tuple[int, int]:
    a, b = x_coeffs[0], x_coeffs[1]
    c, d = z_coeffs[0], z_coeffs[1]
    det = a * d - b * c
    if abs(det) < 1e-10:
        mouse_dx = int(err_x / a) if abs(a) > 1e-6 else 0
        mouse_dy = int(err_z / d) if abs(d) > 1e-6 else 0
        return mouse_dx, mouse_dy
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    mouse_dx = inv_a * err_x + inv_b * err_z
    mouse_dy = inv_c * err_x + inv_d * err_z
    return int(round(mouse_dx)), int(round(mouse_dy))


def _align_spices_bottle_center_to_target_xz(
    target: dict,
    calibration: Optional[tuple[list[float], list[float]]] = None,
    verbose: bool = False,
) -> bool:
    if not target:
        return False

    target_x, target_z = _get_pos_xz(target)
    if calibration is None:
        calibration = _calibrate_mouse_to_3d_mapping_spices(verbose=verbose)
    if calibration is None:
        return False
    x_coeffs, z_coeffs = calibration

    for _ in range(MAX_ALIGN_ITERATIONS):
        pre = _get_json_file_mtime()
        data = _read_realtime_products()
        bottle = _find_sprinkling_spices_bottle(data or {})
        if bottle is None:
            return False

        # Align using the bottle mouth (spout_position) if available.
        cur_x, cur_z = _get_spices_bottle_mouth_xz(bottle)
        err_x = target_x - cur_x
        err_z = target_z - cur_z

        if abs(err_x) <= POSITION_TOLERANCE_M and abs(err_z) <= POSITION_TOLERANCE_M:
            return True

        mouse_dx, mouse_dy = _compute_mouse_movement(err_x, err_z, x_coeffs, z_coeffs)
        # Prevent huge jumps; use smaller steps when close to the target to reduce jitter.
        max_step = int(ALIGN_NEAR_MAX_STEP_PX if max(abs(err_x), abs(err_z)) <= float(ALIGN_NEAR_THRESHOLD_M) else ALIGN_MAX_STEP_PX)
        mouse_dx = max(-max_step, min(max_step, mouse_dx))
        mouse_dy = max(-max_step, min(max_step, mouse_dy))
        if mouse_dx == 0 and mouse_dy == 0:
            break

        if verbose:
            print(f"[align] err_x={err_x:.3f} err_z={err_z:.3f} -> mouse({mouse_dx},{mouse_dy})")

        _move_spices_horizontal(mouse_dx, mouse_dy)
        # Do not run ahead of the scan: wait for at least one json update after moving.
        _wait_for_next_data_update()
        time.sleep(0.02)

    return False


def _wait_for_sprinkle_mode(expected: bool, timeout_s: float = 2.0, stable_s: float = 0.0) -> bool:
    deadline = time.time() + max(float(timeout_s), float(stable_s))
    stable_since: Optional[float] = None
    while time.time() < deadline:
        data = _read_realtime_products()
        bottle = _find_sprinkling_spices_bottle(data or {})
        matched = (bottle is not None) if bool(expected) else (bottle is None)
        if matched:
            now = time.time()
            if float(stable_s) <= 0.0:
                return True
            if stable_since is None:
                stable_since = now
            elif (now - stable_since) >= float(stable_s):
                return True
        else:
            stable_since = None
        time.sleep(0.05)
    return False


def _enter_sprinkle_mode(target: Optional[dict], action_delay_seconds: float) -> bool:
    enter_spices_sprinkle_mode()
    time.sleep(action_delay_seconds)
    return _wait_for_sprinkle_mode(expected=True, timeout_s=4.5, stable_s=2.0)


def _exit_sprinkle_mode(action_delay_seconds: float) -> bool:
    exit_spices_sprinkle_mode()
    time.sleep(action_delay_seconds)
    return _wait_for_sprinkle_mode(expected=False, timeout_s=2.0)


def _try_read_container_grams_from_interaction() -> Optional[float]:
    if not GRAMS_TRACKING_ENABLED:
        return None
    if GRAMS_TRACKING_MODE != "interaction":
        return None

    try:
        from epm.cerebellum.skills.auto_navigation.interaction_detector import (  # type: ignore
            init_udp_mode,
            read_interaction_info,
        )

        init_udp_mode()  # idempotent
        info = read_interaction_info(use_udp=True)
        if not info:
            return None

        # Prefer parsing pouring/sprinkling popup contents (more reliable for spices than `weight`).
        # Example logs (user):
        #   item=Black Pepper, action=Pouring, containerName=Pan, contents=Black Pepper:7g,Black Pepper:13g
        item_name = (getattr(info, "item_name", "") or "").strip()
        action = (getattr(info, "action", "") or "").strip().lower()
        contents = (getattr(info, "container_contents", "") or "").strip()

        if item_name and action == "pouring" and contents:
            # Find all occurrences like "<item>:<num>g" and use the last one.
            # Allow spaces, and keep it case-insensitive.
            pattern = re.compile(rf"(?:^|,)\s*{re.escape(item_name)}\s*:\s*([0-9]+(?:\\.[0-9]+)?)\s*g\s*(?:,|$)", re.IGNORECASE)
            matches = pattern.findall(contents)
            if matches:
                # Some popups contain "current g" and "target g" for the same spice,
                # e.g. "Black Pepper:7g,Black Pepper:13g". We want the current value.
                return float(min(float(m) for m in matches))

        # Fallback: parse `weight` if it contains grams.
        raw = (getattr(info, "weight", "") or "").strip()
        if raw and "g" in raw.lower():
            m = re.search(r"([0-9.]+)", raw)
            if m:
                return float(m.group(1))

        return None
    except Exception:
        return None


def _read_interaction_info_best_effort():
    """
    Read Alt+J interaction info with a robust fallback order.

    Some environments occasionally miss UDP packets; when that happens we fall
    back to file mode explicitly (the mod writes `realtime_interaction_info.txt`).
    """
    from epm.cerebellum.skills.auto_navigation.interaction_detector import (  # type: ignore
        init_udp_mode,
        read_interaction_info,
    )
    try:
        from epm.cerebellum.skills.auto_navigation.interaction_detector import (  # type: ignore
            read_interaction_info_prefer_pour,
        )
    except Exception:
        read_interaction_info_prefer_pour = None  # type: ignore

    global _UDP_MODE_AVAILABLE
    if _UDP_MODE_AVAILABLE is None:
        try:
            init_udp_mode()  # binds UDP port; may fail if port is already in use
            _UDP_MODE_AVAILABLE = True
        except Exception as exc:
            _UDP_MODE_AVAILABLE = False
            print(f"[sprinkle] udp unavailable -> falling back to file mode: {exc}")

    if _UDP_MODE_AVAILABLE:
        if read_interaction_info_prefer_pour:
            # During sprinkling, non-pouring packets interleave with pouring packets.
            # Use a wider age window to avoid empty packets overwriting PourAmount reads.
            info_udp = read_interaction_info_prefer_pour(use_udp=True, max_age_s=3.0)
        else:
            info_udp = read_interaction_info(use_udp=True)
    else:
        info_udp = None
    info_file = read_interaction_info(use_udp=False)

    udp_pour = (getattr(info_udp, "pour_amount", "") or "").strip() if info_udp else ""
    file_pour = (getattr(info_file, "pour_amount", "") or "").strip() if info_file else ""

    if udp_pour:
        return info_udp
    if file_pour:
        return info_file

    return info_udp or info_file


def _try_read_pour_total_grams_from_interaction() -> Optional[float]:
    """
    Read the current "Pouring xx g" total from Alt+J's `PourAmount` field.

    For spices this is expected to be a running total during the pouring UI,
    not a per-click delta.
    """
    if not GRAMS_TRACKING_ENABLED:
        return None
    if GRAMS_TRACKING_MODE != "interaction":
        return None

    try:
        info = _read_interaction_info_best_effort()
        if not info:
            return None

        raw = (getattr(info, "pour_amount", "") or "").strip()
        raw_l = raw.lower()
        if not raw or ("g" not in raw_l and "kg" not in raw_l):
            return None

        m = re.search(r"([0-9.]+)", raw)
        if not m:
            return None
        val = float(m.group(1))
        if "kg" in raw_l:
            val *= 1000.0
        return val
    except Exception:
        return None


def _try_read_pour_total_grams_from_udp_snapshot() -> Optional[float]:
    """
    Read PourAmount total (g) from the UDP receiver snapshot directly.

    This bypasses any timing races where non-pouring packets overwrite the latest info
    between reads, and avoids file-mode ambiguity.
    """
    if not GRAMS_TRACKING_ENABLED or GRAMS_TRACKING_MODE != "interaction":
        return None
    try:
        from epm.cerebellum.skills.auto_navigation.interaction_detector import get_udp_debug_snapshot  # type: ignore

        snap = get_udp_debug_snapshot()
        if not snap:
            return None
        latest_pour = snap.get("latest_pour", None) or {}
        latest = snap.get("latest", None) or {}

        raw = ""
        for key in ("pour_amount", "pour", "PourAmount", "pourAmount"):
            if key in latest_pour:
                raw = str(latest_pour.get(key, "") or "").strip()
                if raw:
                    break
        if not raw:
            for key in ("pour_amount", "pour", "PourAmount", "pourAmount"):
                if key in latest:
                    raw = str(latest.get(key, "") or "").strip()
                    if raw:
                        break
        raw_l = raw.lower()
        if not raw or ("g" not in raw_l and "kg" not in raw_l):
            return None
        m = re.search(r"([0-9.]+)", raw)
        if not m:
            return None
        val = float(m.group(1))
        if "kg" in raw_l:
            val *= 1000.0
        return val
    except Exception:
        return None


def _wait_for_pour_total_increase(
    baseline: float,
    timeout_s: float = 0.8,
    interval_s: float = 0.08,
    disappear_grace_s: float = 0.25,
) -> float:
    """
    Poll pour-total (g) until it increases over `baseline` or timeout.    

    Additionally, stop early if the pouring UI disappears (i.e. value becomes unreadable)
    for longer than `disappear_grace_s`.

    Returns the best (max) observed value; falls back to baseline when unreadable.
    """
    best = baseline
    deadline = time.time() + float(timeout_s)
    last_readable_at = time.time()
    while time.time() < deadline:
        g = _try_read_pour_total_grams_from_interaction()
        if g is not None:
            last_readable_at = time.time()
            if g > best:
                best = g
            if best > baseline:
                # Keep polling a bit within the time window: some frames might jump.
                # We'll still exit early if the UI disappears.
                pass
        else:
            if time.time() - last_readable_at > float(disappear_grace_s):
                break
        time.sleep(float(interval_s))
    return best


def _wait_for_pour_total_available(timeout_s: float = 0.8, interval_s: float = 0.06) -> Optional[float]:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        g = _try_read_pour_total_grams_from_interaction()
        if g is not None:
            return g
        time.sleep(float(interval_s))
    return None


def _wait_for_pour_total_disappear(timeout_s: float = 1.2, interval_s: float = 0.05, stable_s: float = 0.25) -> bool:
    """
    Wait until PourAmount becomes unreadable (UI disappears) for `stable_s`, or timeout.
    """
    deadline = time.time() + float(timeout_s)
    first_none_at: Optional[float] = None
    while time.time() < deadline:
        g = _try_read_pour_total_grams_from_interaction()
        if g is None:
            if first_none_at is None:
                first_none_at = time.time()
            if time.time() - first_none_at >= float(stable_s):
                return True
        else:
            first_none_at = None
        time.sleep(float(interval_s))
    return False


def _debug_peek_interaction(prefix: str) -> None:
    if not GRAMS_TRACKING_ENABLED or GRAMS_TRACKING_MODE != "interaction":
        return
    try:
        info = _read_interaction_info_best_effort()
        if not info:
            print(f"{prefix} Alt+J info: None")
            return
        item = (getattr(info, "item_name", "") or "").strip()
        action = (getattr(info, "action", "") or "").strip()
        pour = (getattr(info, "pour_amount", "") or "").strip()
        weight = (getattr(info, "weight", "") or "").strip()
        container = (getattr(info, "container_name", "") or "").strip()
        contents = (getattr(info, "container_contents", "") or "").strip()
        contents = contents[:120] + ("..." if len(contents) > 120 else "")
        print(f"{prefix} Alt+J info: item='{item}' action='{action}' pour='{pour}' weight='{weight}' container='{container}' contents='{contents}'")
    except Exception:
        return


def _ensure_udp_started_for_interaction_tracking() -> None:
    """
    Start the UDP receiver early so PourAmount caching can warm up before clicks.
    """
    if not GRAMS_TRACKING_ENABLED or GRAMS_TRACKING_MODE != "interaction":
        return
    try:
        from epm.cerebellum.skills.auto_navigation.interaction_detector import init_udp_mode  # type: ignore

        init_udp_mode()
    except Exception as exc:
        print(f"[sprinkle] udp init failed (will fall back): {exc}")


def _debug_udp_snapshot(prefix: str) -> None:
    if not GRAMS_TRACKING_ENABLED or GRAMS_TRACKING_MODE != "interaction":
        return
    try:
        from epm.cerebellum.skills.auto_navigation.interaction_detector import get_udp_debug_snapshot  # type: ignore

        snap = get_udp_debug_snapshot()
        if not snap:
            print(f"{prefix} udp_snapshot: None")
            return
        age = snap.get("latest_pour_age_s", None)
        latest = snap.get("latest", None) or {}
        latest_pour = snap.get("latest_pour", None) or {}
        print(
            f"{prefix} udp_snapshot: running={snap.get('running')} latest_pour_age_s={age} "
            f"latest.action='{latest.get('action','')}' latest.pour='{latest.get('pour_amount','')}' "
            f"latest_pour.action='{latest_pour.get('action','')}' latest_pour.pour='{latest_pour.get('pour_amount','')}'"
        )
    except Exception:
        return


def sprinkle(
    target_item: str,
    target_instance_id: Optional[int] = None,
    sprinkle_num: int = DEFAULT_SPRINKLE_NUM,
    align_to_target: bool = False,
    exit_after: bool = True,
    action_delay_seconds: float = ACTION_DELAY_SECONDS,
    post_click_sleep_seconds: float = POST_CLICK_SLEEP_SECONDS,
    aim_each_time: bool = True,
) -> int:
    if not target_item or not isinstance(target_item, str):
        raise ValueError("target_item must be a non-empty string")

    sprinkle_num = int(sprinkle_num if sprinkle_num is not None else DEFAULT_SPRINKLE_NUM)
    if sprinkle_num <= 0:
        return 0
    print(f"[sprinkle] script_version: {SPRINKLE_SCRIPT_VERSION}")

    _activate_window(GAME_WINDOW_TITLE)
    time.sleep(0.2)

    _ensure_udp_started_for_interaction_tracking()

    data0 = _read_realtime_products() or {}
    target = _find_target_item(data0, target_item, target_instance_id)

    if align_to_target:
        print("[sprinkle] align_to_target=True is not supported (navigation disabled); ignoring")

    if not target:
        raise RuntimeError(
            f"Target '{target_item}' not found in realtime scan. "
            "Ensure F12 scan is running and the target exists in `realtime_products.json`."
        )

    # Enter sprinkle mode (best-effort) then rely on `is_sprinkle_mode` for detection.
    if not _wait_for_sprinkle_mode(expected=True, timeout_s=2.4, stable_s=2.0):
        _enter_sprinkle_mode(target, action_delay_seconds)
        if not _wait_for_sprinkle_mode(expected=True, timeout_s=4.5, stable_s=2.0):
            raise RuntimeError(
                "No stable sprinkling spices bottle detected for 2 seconds (is_sprinkle_mode=true). "
                "Make sure you are holding a spices bottle and can enter sprinkle mode."
            )

    # Keep behavior consistent with upstream:
    # - always attempt a calibration once
    # - attempt one-time alignment to target before timed clicks
    calibration = _calibrate_mouse_to_3d_mapping_spices(verbose=True)

    grams_total = 0.0
    # If a click fails to produce an increase within the 2s window, stop immediately to avoid
    # extra (wasted) sprinkling clicks.

    # Establish baselines BEFORE the first click (best-effort).
    last_pour_total: Optional[float] = _wait_for_pour_total_available(timeout_s=0.4, interval_s=0.05)
    if last_pour_total is None and GRAMS_TRACKING_ENABLED and GRAMS_TRACKING_MODE == "interaction":
        # print("[sprinkle] warning: Alt+J pour reading unavailable (will retry after first click); grams_total may be 0.00 g")
        _debug_peek_interaction("[sprinkle][debug]")
        _debug_udp_snapshot("[sprinkle][debug]")

    actual = 0
    next_click_at = time.time()

    def _ensure_aligned_before_click(click_index: int) -> None:
        nonlocal calibration
        if not bool(aim_each_time):
            return

        # First try with the current calibration; if that fails, re-calibrate once
        # and retry alignment. Do not sprinkle unless alignment succeeds.
        aligned = _align_spices_bottle_center_to_target_xz(target, calibration=calibration, verbose=False)
        if aligned:
            return

        print(f"[sprinkle] re-calibrating before click {click_index}/{sprinkle_num} ...")
        calibration = _calibrate_mouse_to_3d_mapping_spices(verbose=False)
        if calibration is None:
            raise RuntimeError("sprinkle_alignment_failed:calibration_unavailable")

        aligned = _align_spices_bottle_center_to_target_xz(target, calibration=calibration, verbose=False)
        if not aligned:
            raise RuntimeError("sprinkle_alignment_failed:target_not_aligned")

    # Ensure we're actually in sprinkle mode before starting timed clicks/alignment.
    _wait_for_is_sprinkle_true()
    start_at = time.time()
    next_click_at = start_at + float(SPRINKLE_CLICK_INTERVAL_S)
    for _ in range(sprinkle_num):
        # Re-align before every sprinkle click unless explicitly disabled.
        _ensure_aligned_before_click(actual + 1)

        # Enforce strict sprinkling cadence (0.8s). If we ever fall behind (OS/game lag),
        # resync rather than accumulating drift.
        remaining_to_click = float(next_click_at) - time.time()
        if remaining_to_click > 0:
            time.sleep(float(remaining_to_click))
        else:
            if remaining_to_click < -0.05:
                print(f"[sprinkle] warning: click late by {-remaining_to_click:.3f}s; resyncing cadence")
            next_click_at = time.time()

        # Log each click so it's clear how many inputs we actually send.
        # Note: the game may still ignore a click (focus/mode/UI), but this confirms our loop.
        print(f"[sprinkle] click: {actual + 1}/{sprinkle_num}")
        click_mouse("left")
        next_click_at = float(next_click_at) + float(SPRINKLE_CLICK_INTERVAL_S)
        actual += 1

        # No post-click waits: cadence is enforced at the start of the next iteration.

    # After the final click, wait briefly so the popup can fully update before reading totals.
    if actual > 0:
        time.sleep(1.0)

    # Always print a final summary for downstream scripts/logging.
    if GRAMS_TRACKING_ENABLED and GRAMS_TRACKING_MODE == "interaction":
        # Final best-effort: capture the latest cached total before exiting mode.
        final_total = _try_read_pour_total_grams_from_udp_snapshot()
        if final_total is None:
            final_total = _try_read_pour_total_grams_from_interaction()
        if final_total is not None:
            grams_total = float(final_total)
        elif grams_total == 0.0:
            # If we still ended at 0, show snapshot to diagnose mismatched runtime behavior.
            _debug_udp_snapshot("[sprinkle][final_debug]")
        print(f"[sprinkle] grams_total: {grams_total:.2f} g")
    elif GRAMS_TRACKING_ENABLED:
        print("[sprinkle] grams_total: 0.00 g (tracking misconfigured)")
    else:
        print("[sprinkle] grams_total: 0.00 g (tracking disabled)")
    print(f"[sprinkle] actions_done: {actual}/{sprinkle_num}")

    if exit_after:
        _exit_sprinkle_mode(action_delay_seconds)

    return actual


def _print_usage() -> None:
    print("Usage:")
    print("  python auto_sprinkling.py list")
    print("  python auto_sprinkling.py info \"<target_item>\"")
    print("  python auto_sprinkling.py sprinkle \"<target_item>\" <sprinkle_num>")
    print("")
    print("Notes:")
    print("  - Hold a spices bottle first (kind=spices).")
    print("  - Keep F12 scan running (realtime_products.json).")
    print("  - Script aligns by world XZ only (no screen aiming).")
    print("Examples:")
    print("  python auto_sprinkling.py list")
    print("  python auto_sprinkling.py info \"pan\"")
    print("  python auto_sprinkling.py sprinkle \"pan\" 5")


def main(argv: list[str]) -> int:
    _print_usage()
    print("")
    if len(argv) < 2:
        return 2

    cmd = argv[1].strip().lower()
    if cmd == "list":
        list_visible_items()
        return 0

    if cmd == "info":
        if len(argv) < 3:
            return 2
        show_info(argv[2])
        return 0

    if cmd != "sprinkle":
        return 2

    if len(argv) < 4:
        return 2

    target_item = argv[2]
    sprinkle_num = int(argv[3])

    actual = sprinkle(
        target_item=target_item,
        sprinkle_num=sprinkle_num,
    )
    print(f"[sprinkle] done: {actual}/{sprinkle_num}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
