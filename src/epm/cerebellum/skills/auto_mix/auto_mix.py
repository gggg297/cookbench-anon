from __future__ import annotations

"""
EPM auto_mix (Blender mixing mode) — canonical implementation.

This file was previously named `pot_mix.py` during integration; it is now the
single source of truth for mixing behavior.

CLI usage (recommended via `-m` so package imports work):
  - List visible items (on-screen):
      python -m epm.cerebellum.skills.auto_mix.auto_mix list
  - Show info for a container + blender snapshot:
      python -m epm.cerebellum.skills.auto_mix.auto_mix info "Big Pot"
  - Mix in a container (hold Blender, aim at container first):
      python -m epm.cerebellum.skills.auto_mix.auto_mix mix "Big Pot"
"""

import json
import math
import time
from pathlib import Path
from typing import Any, Optional

from epm.cerebellum.game_hotkeys import ensure_f12_products_scan
from epm.cerebellum.local_actions import (
    blender_downward,
    blender_upward,
    enter_mixing_mode,
    exit_mixing_mode,
    horizontal_movement,
    io_controller,
)
from epm.cerebellum.skills.auto_mix.config import (
    DEFAULT_CALIBRATION_PIXELS,
    DEFAULT_DOWNWARD_STEPS,
    DEFAULT_MIX_LAPS,
    DEFAULT_MOVE_DISTANCE_M,
    DEFAULT_MOVE_INTERVAL_S,
    DEFAULT_MOVE_PIXELS,
)
from epm.cerebellum.skills._shared_paths import userdata_root, window_title
from epm.vision.screen_capture import activate_window


def _read_realtime_products(path: Path, *, retries: int = 6, sleep_s: float = 0.05) -> dict[str, Any]:
    # Be tolerant to BOM + partial writes.
    last_err: Exception | None = None
    for _ in range(max(1, int(retries))):
        try:
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
            return json.loads(raw) if raw.strip() else {}
        except Exception as e:
            last_err = e
            time.sleep(float(sleep_s))
    raise RuntimeError(f"failed_to_read_realtime_products: {last_err!r}")


def _normalize(s: str) -> str:
    return (s or "").strip().lower()


def _name_match(keyword: str, product: dict[str, Any]) -> bool:
    keyword = _normalize(keyword)
    if not keyword:
        return False
    name_en = _normalize(str(product.get("name_en", "") or ""))
    name_cn = _normalize(str(product.get("name_cn", "") or ""))
    go_name = _normalize(str(product.get("game_object", "") or ""))
    return keyword in name_en or keyword in name_cn or keyword in go_name


def _choose_nearest(items: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not items:
        return None
    items.sort(key=lambda p: float(p.get("distance", 9999) or 9999))
    return items[0]


def _find_items_by_name(name: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    products = data.get("products", [])
    if not isinstance(products, list):
        return []
    keyword = _normalize(name)
    if not keyword:
        return []
    out: list[dict[str, Any]] = []
    for p in products:
        if isinstance(p, dict) and _name_match(keyword, p):
            out.append(p)
    return out


def _find_container_by_name(container_name: str, data: dict[str, Any], *, must_be_on_screen: bool) -> Optional[dict[str, Any]]:
    candidates = _find_items_by_name(container_name, data)
    # Prefer kind=container, but fall back to any matching item (some scenes label pots differently).
    containers = [p for p in candidates if _normalize(str(p.get("kind", "") or "")) == "container"]
    if must_be_on_screen:
        containers = [p for p in containers if bool(p.get("is_on_screen"))]
    picked = _choose_nearest(containers)
    if picked is not None:
        return picked
    # Fallback: nearest on-screen match of any kind.
    if must_be_on_screen:
        candidates = [p for p in candidates if bool(p.get("is_on_screen"))]
    return _choose_nearest(candidates)


def _find_item_by_instance_id(data: dict[str, Any], instance_id: int) -> Optional[dict[str, Any]]:
    products = data.get("products", [])
    if not isinstance(products, list):
        return None
    for p in products:
        if not isinstance(p, dict):
            continue
        try:
            if int(p.get("instance_id")) == int(instance_id):  # type: ignore[arg-type]
                return p
        except Exception:
            continue
    return None


def _get_position_xyz(product: dict[str, Any]) -> Optional[tuple[float, float, float]]:
    pos = product.get("position")
    if not isinstance(pos, dict):
        return None
    try:
        return float(pos.get("x")), float(pos.get("y")), float(pos.get("z"))
    except Exception:
        return None


def _get_bounds_min_y(product: dict[str, Any]) -> Optional[float]:
    bmin = product.get("bounds_min")
    if not isinstance(bmin, dict):
        return None
    try:
        return float(bmin.get("y"))
    except Exception:
        return None


def _looks_like_blender(p: dict[str, Any]) -> bool:
    name_en = _normalize(str(p.get("name_en", "") or ""))
    name_cn = str(p.get("name_cn", "") or "")
    if "blender" in name_en or "mixer" in name_en:
        return True
    if "搅拌" in name_cn or "攪拌" in name_cn:
        return True
    # Fallback: only Blender tends to expose this flag in our mod output.
    if "is_mixing_mode" in p:
        return True
    return False


def _find_blender(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    products = data.get("products", [])
    if not isinstance(products, list):
        return None

    candidates: list[dict[str, Any]] = []
    for p in products:
        if not isinstance(p, dict):
            continue
        if not _looks_like_blender(p):
            continue
        candidates.append(p)

    held = [p for p in candidates if bool(p.get("is_held"))]
    if held:
        return held[0]
    mixing = [p for p in candidates if bool(p.get("is_mixing_mode", False))]
    if mixing:
        return mixing[0]
    on_screen = [p for p in candidates if bool(p.get("is_on_screen"))]
    if on_screen:
        on_screen.sort(key=lambda x: float(x.get("distance", 9999) or 9999))
        return on_screen[0]
    return candidates[0] if candidates else None


def _wait_blender_mixing_state(*, realtime_products_path: Path, blender_instance_id: int, desired: bool, timeout_s: float = 2.5) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        data = _read_realtime_products(realtime_products_path, retries=2, sleep_s=0.02)
        products = data.get("products", [])
        if isinstance(products, list):
            for p in products:
                try:
                    match = isinstance(p, dict) and int(p.get("instance_id")) == int(blender_instance_id)  # type: ignore[arg-type]
                except Exception:
                    match = False
                if match:
                    if bool(p.get("is_mixing_mode", False)) is bool(desired):
                        return True
        time.sleep(0.05)
    return False


def _wait_blender_mixing_state_stable(
    *,
    realtime_products_path: Path,
    blender_instance_id: int,
    desired: bool,
    timeout_s: float = 4.5,
    stable_s: float = 2.0,
) -> bool:
    deadline = time.time() + max(float(timeout_s), float(stable_s))
    stable_since: float | None = None
    while time.time() < deadline:
        if _wait_blender_mixing_state(
            realtime_products_path=realtime_products_path,
            blender_instance_id=blender_instance_id,
            desired=desired,
            timeout_s=0.05,
        ):
            now = time.time()
            if stable_since is None:
                stable_since = now
            elif (now - stable_since) >= float(stable_s):
                return True
        else:
            stable_since = None
        time.sleep(0.05)
    return False


def _get_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _wait_products_update(path: Path, prev_mtime: float, timeout_s: float = 2.0) -> bool:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if _get_mtime(path) > float(prev_mtime):
            return True
        time.sleep(0.05)
    return False


def _estimate_pixels_for_distance(
    *,
    realtime_products_path: Path,
    blender_instance_id: int,
    distance_m: float,
    calibration_pixels: int,
) -> Optional[int]:
    """
    Best-effort mapping: how many pixels in horizontal_movement correspond to `distance_m` (in world XZ).
    Mirrors the upstream `obtain_camera_info/auto_mix/auto_mix.py` approach.
    """
    data0 = _read_realtime_products(realtime_products_path, retries=2, sleep_s=0.02)
    b0 = _find_item_by_instance_id(data0, blender_instance_id)
    if not b0:
        return None
    pos0 = _get_position_xyz(b0)
    if not pos0:
        return None

    px = int(max(20, min(400, abs(int(calibration_pixels)))))

    prev = _get_mtime(realtime_products_path)
    horizontal_movement(px, 0)
    _wait_products_update(realtime_products_path, prev, timeout_s=2.0)
    data1 = _read_realtime_products(realtime_products_path, retries=2, sleep_s=0.02)
    b1 = _find_item_by_instance_id(data1, blender_instance_id)
    pos1 = _get_position_xyz(b1) if b1 else None

    # Return to reduce drift.
    prev = _get_mtime(realtime_products_path)
    horizontal_movement(-px, 0)
    _wait_products_update(realtime_products_path, prev, timeout_s=2.0)

    if not pos1:
        return None

    dx = pos1[0] - pos0[0]
    dz = pos1[2] - pos0[2]
    moved_m = math.hypot(dx, dz)
    if moved_m <= 1e-5:
        return None

    meters_per_pixel = moved_m / float(px)
    pixels_for_distance = int(round(float(distance_m) / meters_per_pixel))
    return max(5, min(800, pixels_for_distance))


def _lower_blender_near_container(
    *,
    realtime_products_path: Path,
    blender_instance_id: int,
    container: Optional[dict[str, Any]],
    max_steps: int,
    fallback_steps: int,
    verbose: bool,
) -> dict[str, Any]:
    if not container:
        steps_used = int(max(0, int(fallback_steps)))
        for _ in range(max(0, int(fallback_steps))):
            blender_downward()
            time.sleep(0.03)
        if verbose:
            print(f"[mix] lower: done (no container) steps_used={steps_used}")
        return {"target_y": None, "steps_used": steps_used, "final_y": None}

    target_y = _get_bounds_min_y(container)
    if target_y is None:
        steps_used = int(max(0, int(fallback_steps)))
        for _ in range(max(0, int(fallback_steps))):
            blender_downward()
            time.sleep(0.03)
        if verbose:
            print(f"[mix] lower: done (missing bounds_min.y) steps_used={steps_used}")
        return {"target_y": None, "steps_used": steps_used, "final_y": None}

    target_y = float(target_y) + 0.02
    cname = container.get("name_en") or container.get("name_cn") or container.get("game_object") or "container"

    final_y: Optional[float] = None
    steps_used = 0
    for i in range(max(0, int(max_steps))):
        data = _read_realtime_products(realtime_products_path, retries=2, sleep_s=0.02)
        b = _find_item_by_instance_id(data, blender_instance_id)
        pos = _get_position_xyz(b) if b else None
        if pos:
            final_y = float(pos[1])
            if float(pos[1]) <= target_y:
                break
        blender_downward()
        steps_used += 1
        time.sleep(0.03)

    if verbose:
        print(f"[mix] lower: done container={cname!r} target_y={target_y:.3f} steps_used={steps_used} final_y={final_y}")
    return {"target_y": target_y, "steps_used": steps_used, "final_y": final_y}


def mix_in_container(
    *,
    container_name: str,
    container_instance_id: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Mix inside a container using Blender mixing mode.

    Expected gameplay flow:
    - Hold the Blender in hand.
    - Make sure the target container is on screen / centered (e.g. after `auto_goto(container)`).
    - This function clicks to enter mixing mode, performs mixing motion, and exits.
    """

    realtime_products_path = userdata_root() / "realtime_products.json"
    ensure_f12_products_scan(
        realtime_products_path=realtime_products_path,
        window_title=window_title("CookingSimulator"),
        activate_window=activate_window,
        io_controller=io_controller,
    )

    try:
        activate_window(window_title("CookingSimulator"))
    except Exception:
        pass

    data0 = _read_realtime_products(realtime_products_path)
    blender = _find_blender(data0)
    if blender is None:
        return {"success": False, "error": "Blender not found in realtime scan (F12). Hold the Blender and try again."}

    if not bool(blender.get("is_held")) and not bool(blender.get("is_mixing_mode", False)):
        return {"success": False, "error": "Blender is not held (and not already mixing). Pick it up first."}

    try:
        # Note: some mods/scenes use negative instance_id values; that's fine as long as it's consistent.
        blender_id = int(blender.get("instance_id"))  # type: ignore[arg-type]
    except Exception:
        return {"success": False, "error": "Blender instance_id missing in realtime scan."}

    already_mixing = bool(blender.get("is_mixing_mode", False))
    data_for_container = data0
    if container_instance_id is not None:
        container = _find_item_by_instance_id(data_for_container, int(container_instance_id))
        if container is not None and not _name_match(_normalize(container_name), container):
            return {
                "success": False,
                "error": f"Container instance_id found but name mismatch: container_name={container_name!r} container_instance_id={int(container_instance_id)}",
            }
    else:
        container = _find_container_by_name(container_name, data_for_container, must_be_on_screen=(not already_mixing))
    if container is None and not already_mixing:
        return {
            "success": False,
            "error": f"Container not found or not on screen: {container_name!r}. Aim at it and ensure F12 scan sees it.",
        }

    if verbose:
        bn = blender.get("name_en") or blender.get("name_cn") or "Blender"
        held = bool(blender.get("is_held"))
        mm = bool(blender.get("is_mixing_mode", False))
        print(f"[mix] blender={bn!r} held={held} mixing_mode={mm} id={blender_id}")
        print(f"[mix] container={container_name!r} (assumed aimed/centered on screen)")

    # Enter mixing mode by click (must be aiming at the container).
    if not already_mixing:
        enter_mixing_mode()
        ok = _wait_blender_mixing_state_stable(
            realtime_products_path=realtime_products_path,
            blender_instance_id=blender_id,
            desired=True,
            timeout_s=4.5,
            stable_s=2.0,
        )
        if not ok:
            return {
                "success": False,
                "error": "Failed to enter stable mixing mode for 2 seconds. Aim at the container and click once, or use auto_goto(container) first.",
            }
    else:
        ok = _wait_blender_mixing_state_stable(
            realtime_products_path=realtime_products_path,
            blender_instance_id=blender_id,
            desired=True,
            timeout_s=4.5,
            stable_s=2.0,
        )
        if not ok:
            return {
                "success": False,
                "error": "Mixing mode was not stable for 2 seconds. Re-aim at the container and re-enter mixing mode before auto_mix.",
            }

    # 1) Lower blender near container contents (closed-loop on blender_y vs container bounds_min.y).
    lower = _lower_blender_near_container(
        realtime_products_path=realtime_products_path,
        blender_instance_id=blender_id,
        container=container,
        max_steps=int(DEFAULT_DOWNWARD_STEPS),
        fallback_steps=int(DEFAULT_DOWNWARD_STEPS),
        verbose=bool(verbose),
    )

    # 2) Convert desired world distance to pixels (best-effort calibration in mixing mode).
    px = _estimate_pixels_for_distance(
        realtime_products_path=realtime_products_path,
        blender_instance_id=blender_id,
        distance_m=float(DEFAULT_MOVE_DISTANCE_M),
        calibration_pixels=int(DEFAULT_CALIBRATION_PIXELS),
    )
    if px is None:
        px = int(DEFAULT_MOVE_PIXELS)
    diag_px = max(1, int(round(float(px) / math.sqrt(2))))

    # 3) Mix in 8 directions for N laps.
    moves = [
        (0, -px),
        (diag_px, -diag_px),
        (px, 0),
        (diag_px, diag_px),
        (0, px),
        (-diag_px, diag_px),
        (-px, 0),
        (-diag_px, -diag_px),
    ]
    laps = max(1, int(DEFAULT_MIX_LAPS))
    interval_s = float(DEFAULT_MOVE_INTERVAL_S)
    if verbose:
        print(f"[mix] motion: laps={laps} px={px} interval_s={interval_s}")
    for _ in range(laps):
        for dx, dy in moves:
            horizontal_movement(int(dx), int(dy))
            time.sleep(max(0.01, float(interval_s)))

    blender_upward()
    time.sleep(0.05)

    exit_mixing_mode()
    exit_ok = _wait_blender_mixing_state(realtime_products_path=realtime_products_path, blender_instance_id=blender_id, desired=False, timeout_s=2.5)
    if not exit_ok:
        return {"success": False, "error": "Failed to exit mixing mode (right click)."}

    return {
        "success": True,
        "container_name": container_name,
        "laps": int(laps),
        "px": int(px),
        "blender_id": blender_id,
        "lower_steps_used": int(lower.get("steps_used", 0) or 0),
        "lower_target_y": lower.get("target_y", None),
        "lower_final_y": lower.get("final_y", None),
        "error": "",
    }


def _list_visible_items(data: dict[str, Any], limit: int = 40) -> list[dict[str, Any]]:
    products = data.get("products", [])
    if not isinstance(products, list):
        return []
    visible = [p for p in products if isinstance(p, dict) and bool(p.get("is_on_screen"))]
    visible.sort(key=lambda p: float(p.get("distance", 9999) or 9999))
    return visible[: max(0, int(limit))]


def _print_usage() -> None:
    print("Usage:")
    print("  python -m epm.cerebellum.skills.auto_mix.auto_mix list")
    print("  python -m epm.cerebellum.skills.auto_mix.auto_mix info <container_name>")
    print("  python -m epm.cerebellum.skills.auto_mix.auto_mix mix <container_name>")
    print("")
    print("Notes:")
    print("  - Keep F12 scan running (realtime_products.json).")
    print("  - Hold Blender, aim at container, then run `mix`.")


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv or []
    if not argv:
        _print_usage()
        return 2

    cmd = (argv[0] or "").strip().lower()
    realtime_products_path = userdata_root() / "realtime_products.json"

    if cmd == "list":
        data = _read_realtime_products(realtime_products_path, retries=2, sleep_s=0.02)
        items = _list_visible_items(data, limit=60)
        print(f"Visible items ({len(items)}):")
        for p in items:
            name_en = p.get("name_en")
            name_cn = p.get("name_cn")
            kind = p.get("kind")
            held = p.get("is_held")
            dist = p.get("distance")
            mixing = p.get("is_mixing_mode", None)
            extra = f" mixing_mode={mixing}" if mixing is not None else ""
            print(f"- {name_en} ({name_cn}) kind={kind} held={held} dist={dist}{extra}")
        return 0

    if cmd == "info":
        if len(argv) < 2:
            _print_usage()
            return 2
        name = " ".join(argv[1:]).strip()
        data = _read_realtime_products(realtime_products_path, retries=2, sleep_s=0.02)
        container = _find_container_by_name(name, data, must_be_on_screen=False)
        blender = _find_blender(data)
        print(f"Container: {container.get('name_en')} ({container.get('name_cn')})" if container else "Container: (not found)")
        if container:
            print(f"  on_screen={container.get('is_on_screen')} pos={container.get('position')} bounds_min={container.get('bounds_min')}")
        print(f"Blender:   {blender.get('name_en')} ({blender.get('name_cn')})" if blender else "Blender: (not found)")
        if blender:
            print(f"  held={blender.get('is_held')} mixing_mode={blender.get('is_mixing_mode')} kind={blender.get('kind')} inst={blender.get('instance_id')}")
        return 0

    if cmd == "mix":
        if len(argv) < 2:
            _print_usage()
            return 2
        container_name = " ".join(argv[1:]).strip()
        result = mix_in_container(container_name=container_name, verbose=True)
        ok = bool(isinstance(result, dict) and result.get("success", False))
        if not ok:
            print(f"[!] mix failed: {result.get('error') if isinstance(result, dict) else result}")
        return 0 if ok else 1

    _print_usage()
    return 2


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
