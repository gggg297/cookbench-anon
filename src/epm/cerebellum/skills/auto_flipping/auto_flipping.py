from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_PLACE_RADIUS_M,
    GAME_WINDOW_TITLE,
    POST_FLIP_EXIT_DELAY_S,
    REALTIME_PRODUCTS_JSON,
)

# 运行时可被 main() 覆盖（用于适配不同 Steam 库目录）
CURRENT_JSON_PATH = REALTIME_PRODUCTS_JSON

try:
    # Prefer the consolidated local_actions in this repo (epm.cerebellum.local_actions).
    from epm.cerebellum.local_actions import (  # type: ignore
        Horizontal_movement,
        _activate_window,
        enter_the_flipping_mode,
        exit_the_flipping_mode,
        flip_over,
    )
except Exception:
    Horizontal_movement = None
    _activate_window = None
    enter_the_flipping_mode = None
    exit_the_flipping_mode = None
    flip_over = None


_LAST_ERROR_LINES: List[str] = []


def _reset_last_error() -> None:
    _LAST_ERROR_LINES.clear()


def get_last_error() -> str:
    # De-dup while preserving order
    out: List[str] = []
    seen = set()
    for s in _LAST_ERROR_LINES:
        s = str(s or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    # Keep it single-line for easier downstream prompt parsing/logging.
    return " | ".join(out)


def _push_error(msg: str, *, also_print: bool = True) -> None:
    _LAST_ERROR_LINES.append(str(msg))
    if also_print:
        print(str(msg))

def _print_info(msg: str) -> None:
    # Printed but not treated as a failure reason (so it won't pollute return.error).
    print(str(msg))


def _is_flipping_mode(spatula: Dict[str, Any]) -> bool:
    # Mod field uses legacy typo: is_filp_mode.
    return bool(spatula.get("is_filp_mode") or spatula.get("is_flip_mode") or spatula.get("is_flipping_mode"))

def _any_spatula_in_flipping_mode(data: Dict[str, Any]) -> bool:
    try:
        for s in list_spatulas(data):
            if _is_flipping_mode(s):
                return True
    except Exception:
        return False
    return False


def ensure_flipping_mode(verbose: bool = True) -> bool:
    """
    Best-effort: ensure the player is holding a spatula and is in flipping mode.

    - If a held spatula exists but `is_filp_mode` is False, try `enter_the_flipping_mode()` and verify via realtime scan.
    """
    if enter_the_flipping_mode is None:
        _push_error("[!] 无法进入翻面模式：enter_the_flipping_mode 不可用")
        return False

    data0 = read_realtime_products()
    if data0 is None:
        _push_error("[!] 无法读取 realtime_products.json（无法检测翻面模式）")
        return False

    spatula0 = get_reference_spatula(data0)
    if spatula0 is not None and _is_flipping_mode(spatula0):
        return True
    if _any_spatula_in_flipping_mode(data0):
        return True

    if verbose:
        _print_info("[*] 当前不在翻面模式：尝试进入翻面模式（左键）...")

    prev_mtime = get_json_file_mtime()
    try:
        enter_the_flipping_mode()
    except Exception as e:
        _push_error(f"[!] 进入翻面模式失败：{e}")
        return False

    # Give the game/mod time to switch mode and publish updated scan.
    time.sleep(2.0)
    wait_for_data_update(prev_mtime, timeout=2.0)
    data1 = read_realtime_products()
    if data1 is None:
        _push_error("[!] 进入翻面模式后仍无法读取 realtime_products.json")
        return False
    if not _any_spatula_in_flipping_mode(data1):
        # Best-effort: check reference spatula for debugging context.
        spatula1 = get_reference_spatula(data1)
        if spatula1 is None:
            _push_error("[!] 进入翻面模式失败：未找到 spatula（realtime_products.json 未包含 spatula 条目）")
            return False
        _push_error("[!] 进入翻面模式失败：realtime_products.json 未显示 is_filp_mode=true（请手动进入翻面模式）")
        return False
    return True


# ================================================================
# JSON helpers
# ================================================================

def read_realtime_products(json_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    json_path = json_path or CURRENT_JSON_PATH
    if not os.path.exists(json_path):
        return None
    try:
        # 与 auto_pouring 保持一致：部分文件带 BOM，需要用 utf-8-sig 才能正确解析
        with open(json_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def get_json_file_mtime(json_path: Optional[str] = None) -> float:
    json_path = json_path or CURRENT_JSON_PATH
    try:
        return os.path.getmtime(json_path)
    except Exception:
        return 0.0


def wait_for_data_update(prev_mtime: float,
                         timeout: float = 2.0,
                         interval: float = 0.05,
                         json_path: Optional[str] = None) -> bool:
    json_path = json_path or CURRENT_JSON_PATH
    start = time.time()
    while time.time() - start < timeout:
        if get_json_file_mtime(json_path) > prev_mtime:
            return True
        time.sleep(interval)
    return False


def _name_match(keyword: str, product: Dict[str, Any]) -> bool:
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return False
    name_en = (product.get("name_en") or "").strip().lower()
    name_cn = (product.get("name_cn") or "").strip().lower()
    go_name = (product.get("game_object") or "").strip().lower()
    return keyword in name_en or keyword in name_cn or keyword in go_name


def find_items_by_name(name: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    keyword = (name or "").strip().lower()
    if not keyword:
        return []
    return [p for p in data.get("products", []) if _name_match(keyword, p)]


def choose_nearest(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:    
    if not items:
        return None
    items_sorted = sorted(items, key=lambda p: float(p.get("distance", 9999)))  
    return items_sorted[0]


def find_item_by_instance_id(data: Dict[str, Any], instance_id: int) -> Optional[Dict[str, Any]]:
    for p in data.get("products", []):
        if p.get("instance_id") == instance_id:
            return p
    return None


def get_position_xyz(product: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    pos = product.get("position")
    if not isinstance(pos, dict):
        return None
    try:
        return float(pos.get("x")), float(pos.get("y")), float(pos.get("z"))
    except Exception:
        return None


def choose_closest_to_xyz(items: List[Dict[str, Any]],
                          ref_xyz: Tuple[float, float, float]) -> Optional[Dict[str, Any]]:
    if not items:
        return None
    rx, ry, rz = ref_xyz
    best = None
    best_d2 = None
    for p in items:
        pos = get_position_xyz(p)
        if pos is None:
            continue
        x, y, z = pos
        d2 = (x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2
        if best is None or d2 < best_d2:
            best = p
            best_d2 = d2
    return best if best is not None else choose_nearest(items)


def get_position_xz(product: Dict[str, Any]) -> Optional[Tuple[float, float]]:  
    pos = product.get("position")
    if not isinstance(pos, dict):
        return None
    try:
        return float(pos.get("x")), float(pos.get("z"))
    except Exception:
        return None


def list_visible_items(data: Dict[str, Any], limit: int = 30) -> List[Dict[str, Any]]:
    visible = [p for p in data.get("products", []) if p.get("is_on_screen")]
    visible.sort(key=lambda p: float(p.get("distance", 9999)))
    return visible[: max(0, int(limit))]


def list_spatulas(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    spatulas: List[Dict[str, Any]] = []
    for p in data.get("products", []):
        name_en = (p.get("name_en") or "").lower()
        if "spatula" in name_en:
            spatulas.append(p)
    spatulas.sort(key=lambda p: float(p.get("distance", 9999)))
    return spatulas


def get_reference_spatula(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    spatulas = list_spatulas(data)
    if not spatulas:
        return None
    held = [s for s in spatulas if s.get("is_held")]
    return held[0] if held else spatulas[0]


MAX_MEAT_TO_SPATULA_M = 0.12
MIN_AXIS_SENSITIVITY = 5e-6


def distance_xyz(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    ax, ay, az = a
    bx, by, bz = b
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def select_meat_on_spatula(data: Dict[str, Any],
                           meat_name: str,
                           tracked_instance_id: Optional[int] = None,
                           verbose: bool = False) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    """
    仅选择“在手持 spatula 附近”的那块肉，避免选到冰箱/场景里同名物体导致标定或收敛失败。
    Returns: (meat, spatula, meat_to_spatula_dist_m)
    """
    spatula = get_reference_spatula(data)
    if spatula is None:
        if verbose:
            print("[!] 未找到 spatula")
        return None
    spatula_xyz = get_position_xyz(spatula)
    if spatula_xyz is None:
        if verbose:
            print("[!] spatula 缺少 position.x/y/z")
        return None

    meat = None
    if tracked_instance_id is not None:
        meat = find_item_by_instance_id(data, tracked_instance_id)

    if meat is None:
        candidates = find_items_by_name(meat_name, data)
        meat = choose_closest_to_xyz(candidates, spatula_xyz) if candidates else None

    if meat is None:
        if verbose:
            _push_error(f"[!] 未找到食材: {meat_name}")
        return None

    meat_xyz = get_position_xyz(meat)
    if meat_xyz is None:
        if verbose:
            print("[!] 食材缺少 position.x/y/z")
        return None

    d = distance_xyz(meat_xyz, spatula_xyz)
    if d > MAX_MEAT_TO_SPATULA_M:
        if verbose:
            _push_error(f"[!] 未检测到铲子上的食材（meat-spatula 距离 {d:.3f}m > {MAX_MEAT_TO_SPATULA_M:.3f}m）")
        return None

    return meat, spatula, d


def print_spatula_status(data: Dict[str, Any]) -> None:
    spatulas = list_spatulas(data)
    if not spatulas:
        print("[!] 未找到 Spatula")
        return
    for s in spatulas:
        print(
            f"- {s.get('name_en')} / {s.get('name_cn')} "
            f"(dist={s.get('distance')}, on_screen={s.get('is_on_screen')}, held={s.get('is_held')}, "
            f"is_filp_mode={s.get('is_filp_mode')})"
        )


# ================================================================
# Mouse <-> world calibration (2x2, refer to auto_pouring)
# ================================================================

CALIBRATION_PIXELS = 120


def move_spatula_horizontal(dx: int, dy: int) -> None:
    if Horizontal_movement is None:
        raise RuntimeError("local_actions.Horizontal_movement not available")
    Horizontal_movement(int(dx), int(dy))
    time.sleep(0.05)


def compute_mouse_movement(err_x: float, err_z: float, x_coeffs: List[float], z_coeffs: List[float]) -> Tuple[int, int]:
    """
    2x2 线性模型（与 auto_pouring 同构）：
      Δx = a*dx + b*dy
      Δz = c*dx + d*dy
    求 dx,dy 使 Δx≈err_x, Δz≈err_z
    """
    a, b = x_coeffs
    c, d = z_coeffs
    det = a * d - b * c
    if abs(det) < 1e-9:
        mouse_dx = int(err_x / a) if abs(a) > 1e-9 else 0
        mouse_dy = int(err_z / d) if abs(d) > 1e-9 else 0
        return mouse_dx, mouse_dy

    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det

    mouse_dx = inv_a * err_x + inv_b * err_z
    mouse_dy = inv_c * err_x + inv_d * err_z
    return int(round(mouse_dx)), int(round(mouse_dy))


def compute_mouse_movement_robust(err_x: float,
                                 err_z: float,
                                 x_coeffs: List[float],
                                 z_coeffs: List[float],
                                 ridge: float = 1e-6) -> Tuple[float, float]:
    """
    当 2x2 变换矩阵接近奇异（det≈0）时，用带正则的最小二乘求解更稳：
      min ||A*[dx,dy]^T - e||^2 + ridge*||[dx,dy]||^2
    其中 A = [[a,b],[c,d]], e = [err_x, err_z].
    """
    a, b = x_coeffs
    c, d = z_coeffs

    # (A^T A + λI)^-1 A^T e
    m00 = a * a + c * c + ridge
    m01 = a * b + c * d
    m11 = b * b + d * d + ridge
    det = m00 * m11 - m01 * m01
    if abs(det) < 1e-12:
        return 0.0, 0.0

    # A^T e
    at_e0 = a * err_x + c * err_z
    at_e1 = b * err_x + d * err_z

    inv00 = m11 / det
    inv01 = -m01 / det
    inv10 = -m01 / det
    inv11 = m00 / det

    dx = inv00 * at_e0 + inv01 * at_e1
    dy = inv10 * at_e0 + inv11 * at_e1
    return dx, dy


def calibrate_mouse_to_3d_mapping(meat_name: str, verbose: bool = True) -> Optional[Tuple[List[float], List[float]]]:
    """
    试探标定法：在翻面模式下做两次水平移动(dx,dy)，测量食材(x,z)的变化，建立线性映射。
    """
    data = read_realtime_products()
    if data is None:
        return None

    picked = select_meat_on_spatula(data, meat_name=meat_name, tracked_instance_id=None, verbose=verbose)
    if picked is None:
        return None
    meat, _, meat_to_spatula = picked
    tracked_instance_id = meat.get("instance_id")

    init = get_position_xz(meat)
    if init is None:
        return None

    init_x, init_z = init
    if verbose:
        print(f"[*] 标定起点: meat=({init_x:.4f}, {init_z:.4f})")

    pre_mtime = get_json_file_mtime()
    move_spatula_horizontal(CALIBRATION_PIXELS, 0)
    wait_for_data_update(pre_mtime, timeout=2.0)
    time.sleep(0.15)

    data = read_realtime_products()
    picked = select_meat_on_spatula(data or {}, meat_name=meat_name, tracked_instance_id=tracked_instance_id, verbose=verbose)
    if picked is None:
        return None
    meat, _, _ = picked
    after_x = get_position_xz(meat)
    if after_x is None:
        return None
    dx1 = after_x[0] - init_x
    dz1 = after_x[1] - init_z

    pre_mtime = get_json_file_mtime()
    move_spatula_horizontal(-CALIBRATION_PIXELS, 0)
    wait_for_data_update(pre_mtime, timeout=2.0)
    time.sleep(0.15)

    data = read_realtime_products()
    picked = select_meat_on_spatula(data or {}, meat_name=meat_name, tracked_instance_id=tracked_instance_id, verbose=verbose)
    if picked is None:
        return None
    meat, _, _ = picked
    before_y = get_position_xz(meat)
    if before_y is None:
        return None

    pre_mtime = get_json_file_mtime()
    move_spatula_horizontal(0, CALIBRATION_PIXELS)
    wait_for_data_update(pre_mtime, timeout=2.0)
    time.sleep(0.15)

    data = read_realtime_products()
    picked = select_meat_on_spatula(data or {}, meat_name=meat_name, tracked_instance_id=tracked_instance_id, verbose=verbose)
    if picked is None:
        return None
    meat, _, _ = picked
    after_y = get_position_xz(meat)
    if after_y is None:
        return None

    dx2 = after_y[0] - before_y[0]
    dz2 = after_y[1] - before_y[1]

    pre_mtime = get_json_file_mtime()
    move_spatula_horizontal(0, -CALIBRATION_PIXELS)
    wait_for_data_update(pre_mtime, timeout=2.0)
    time.sleep(0.10)

    x_coeffs = [dx1 / CALIBRATION_PIXELS, dx2 / CALIBRATION_PIXELS]        
    z_coeffs = [dz1 / CALIBRATION_PIXELS, dz2 / CALIBRATION_PIXELS]        

    x_sens = abs(x_coeffs[0]) + abs(x_coeffs[1])
    z_sens = abs(z_coeffs[0]) + abs(z_coeffs[1])
    if x_sens < MIN_AXIS_SENSITIVITY and z_sens < MIN_AXIS_SENSITIVITY:
        if verbose:
            _push_error(f"[!] 标定失败：Δx/Δz 对鼠标移动几乎无响应（x_sens={x_sens:.2e}, z_sens={z_sens:.2e}）")
            return None

    if verbose:
        print(f"[*] 标定结果:")
        print(f"  Δx = {x_coeffs[0]:.6f}*dx + {x_coeffs[1]:.6f}*dy")
        print(f"  Δz = {z_coeffs[0]:.6f}*dx + {z_coeffs[1]:.6f}*dy")
        print(f"  meat_to_spatula={meat_to_spatula:.3f}m, instance_id={tracked_instance_id}")

    return x_coeffs, z_coeffs


def get_move_pixels(distance_m: float) -> int:
    if distance_m >= 0.10:
        return 300
    if distance_m >= 0.05:
        return 200
    return 60


def align_meat_to_put_place(
    meat_name: str,
    put_place_name: str,
    place_radius_m: float,
    verbose: bool = True,
    meat_instance_id: Optional[int] = None,
    put_place_instance_id: Optional[int] = None,
) -> bool:
    data = read_realtime_products()
    if data is None:
        print("[!] 无法读取 realtime_products.json")
        return False

    picked = select_meat_on_spatula(data, meat_name=meat_name, tracked_instance_id=meat_instance_id, verbose=verbose)
    meat = picked[0] if picked is not None else None

    if put_place_instance_id is not None:
        place = find_item_by_instance_id(data, int(put_place_instance_id))
        if place is not None and not _name_match((put_place_name or "").strip().lower(), place):
            print(f"[!] 放置点 instance_id 存在但名称不匹配: put_place={put_place_name!r} put_place_instance_id={int(put_place_instance_id)}")
            return False
    else:
        place_candidates = find_items_by_name(put_place_name, data)
        place = choose_nearest(place_candidates)
    if meat is None:
        _push_error(f"[!] 未找到食材: {meat_name}")
        return False
    if place is None:
        print(f"[!] 未找到放置点: {put_place_name}")
        return False
    tracked_instance_id = meat.get("instance_id")
    if meat_instance_id is not None:
        tracked_instance_id = int(meat_instance_id)

    meat_pos = get_position_xz(meat)
    place_pos = get_position_xz(place)
    if meat_pos is None or place_pos is None:
        print("[!] 缺少 position.x/z")
        return False

    target_x, target_z = place_pos
    if verbose:
        print(f"[*] 目标落点: ({target_x:.3f}, {target_z:.3f}), radius={place_radius_m:.3f}m")

    calib = calibrate_mouse_to_3d_mapping(meat_name=meat_name, verbose=verbose)
    if calib is None:
        print("[!] 标定失败，使用默认映射（可能不准）")
        x_coeffs = [0.001, 0.0]
        z_coeffs = [0.0, -0.001]
    else:
        x_coeffs, z_coeffs = calib
        x_sens = abs(x_coeffs[0]) + abs(x_coeffs[1])
        z_sens = abs(z_coeffs[0]) + abs(z_coeffs[1])
        if x_sens < MIN_AXIS_SENSITIVITY and abs(target_x - meat_pos[0]) > place_radius_m:
            _push_error(f"[!] 当前视角/模式下 X 方向几乎不可控（x_sens={x_sens:.2e}），请调整铲子/相机朝向后重试")
            return False
        if z_sens < MIN_AXIS_SENSITIVITY and abs(target_z - meat_pos[1]) > place_radius_m:
            _push_error(f"[!] 当前视角/模式下 Z 方向几乎不可控（z_sens={z_sens:.2e}），请调整铲子/相机朝向后重试")
            return False

    for iteration in range(120):
        pre_mtime = get_json_file_mtime()
        data = read_realtime_products()
        if data is None:
            return False

        picked = select_meat_on_spatula(data, meat_name=meat_name, tracked_instance_id=tracked_instance_id, verbose=False)
        if picked is None:
            print("[!] 运行中未检测到铲子上的食材（可能掉落/选错目标）")
            return False
        meat, _, meat_to_spatula = picked
        cur = get_position_xz(meat)
        if cur is None:
            return False
        cur_x, cur_z = cur

        err_x = target_x - cur_x
        err_z = target_z - cur_z
        dist = math.sqrt(err_x * err_x + err_z * err_z)

        if verbose and iteration % 3 == 0:
            print(f"  [iter {iteration}] meat=({cur_x:.3f},{cur_z:.3f}) err=({err_x:.3f},{err_z:.3f}) dist={dist:.3f}m")

        if dist <= place_radius_m:
            if verbose:
                print(f"[*] 到位确认: meat=({cur_x:.3f},{cur_z:.3f}) err=({err_x:.3f},{err_z:.3f}) dist={dist:.3f}m meat_to_spatula={meat_to_spatula:.3f}m")
            return True

        raw_dx, raw_dy = compute_mouse_movement_robust(err_x, err_z, x_coeffs, z_coeffs)
        move_x = int(round(raw_dx))
        move_y = int(round(raw_dy))

        max_px = get_move_pixels(dist)
        move_x = max(-max_px, min(max_px, move_x))
        move_y = max(-max_px, min(max_px, move_y))

        # 避免四舍五入为0导致“卡住”：沿标定反解出来的方向轻推一步（而非用 err_x/err_z 符号）
        if move_x == 0 and move_y == 0 and dist > place_radius_m:
            min_px = 5
            if abs(raw_dx) >= abs(raw_dy) and abs(raw_dx) > 1e-9:
                move_x = min_px if raw_dx > 0 else -min_px
            elif abs(raw_dy) > 1e-9:
                move_y = min_px if raw_dy > 0 else -min_px

        move_spatula_horizontal(move_x, move_y)
        wait_for_data_update(pre_mtime, timeout=2.0)

    return False


def auto_flip(
    meat_name: str,
    put_place_name: str,
    meat_instance_id: Optional[int] = None,
    put_place_instance_id: Optional[int] = None,
    place_radius_m: float = DEFAULT_PLACE_RADIUS_M,
    verbose: bool = True,
) -> bool:
    _reset_last_error()
    if _activate_window is None or flip_over is None or exit_the_flipping_mode is None:
        _push_error("[!] local_actions 不可用：无法导入/初始化 epm.cerebellum.local_actions")
        return False

    _activate_window(GAME_WINDOW_TITLE)
    time.sleep(0.2)

    if not ensure_flipping_mode(verbose=bool(verbose)):
        return False

    ok = align_meat_to_put_place(
        meat_name,
        put_place_name,
        place_radius_m=place_radius_m,
        verbose=verbose,
        meat_instance_id=meat_instance_id,
        put_place_instance_id=put_place_instance_id,
    )
    if not ok:
        _push_error("[!] 未能将食材移动到落点")
        return False

    if verbose:
        print(f"[*] 到位，执行翻面(E) + 等待{POST_FLIP_EXIT_DELAY_S:.1f}s + 退出翻面模式(右键)")

    flip_over()
    time.sleep(POST_FLIP_EXIT_DELAY_S)
    exit_the_flipping_mode()
    return True


# ================================================================
# CLI
# ================================================================

def main(argv: List[str]) -> int:
    global CURRENT_JSON_PATH

    # 全局参数：允许通过命令行覆盖 realtime_products.json 路径
    args = list(argv)
    json_override = None
    userdata_override = None
    i = 1
    while i < len(args):
        if args[i] in {"--json", "--json-path"} and i + 1 < len(args):
            json_override = args[i + 1]
            del args[i:i + 2]
            continue
        if args[i] in {"--userdata", "--userdata-path"} and i + 1 < len(args):
            userdata_override = args[i + 1]
            del args[i:i + 2]
            continue
        i += 1

    if json_override:
        CURRENT_JSON_PATH = json_override
    elif userdata_override:
        CURRENT_JSON_PATH = os.path.join(userdata_override, "realtime_products.json")
    else:
        CURRENT_JSON_PATH = REALTIME_PRODUCTS_JSON

    cmd = args[1].lower() if len(args) > 1 else "help"
    known_cmds = {"status", "list", "find", "flip", "-h", "--help", "help"}
    implicit_flip = cmd not in known_cmds and len(args) >= 3

    if cmd in {"-h", "--help", "help"}:
        print("Auto-Flipping")
        print("Usage:")
        print("  python auto_flipping.py [--json <path>|--userdata <UserDataDir>] status")
        print("  python auto_flipping.py [--json <path>|--userdata <UserDataDir>] list [n]")
        print("  python auto_flipping.py [--json <path>|--userdata <UserDataDir>] find <keyword>")
        print("  python auto_flipping.py [--json <path>|--userdata <UserDataDir>] flip <meat> <put_place> [radius_m]")
        print("  python auto_flipping.py [--json <path>|--userdata <UserDataDir>] <meat> <put_place> [radius_m]  (implicit flip)")
        return 0

    data = read_realtime_products()
    if data is None:
        print(f"[!] 读取失败: {CURRENT_JSON_PATH}")
        print("[*] 请确认已在游戏内开启 F12 扫描并生成 realtime_products.json")
        print("[*] 可选：用 --json 指定文件路径，或用 --userdata 指定 CookingSimulator\\UserData 目录")
        print("[*] 或设置环境变量 COOKGAME_USERDATA_PATH / COOKGAME_REALTIME_PRODUCTS_JSON")
        return 2

    if cmd == "status":
        print_spatula_status(data)
        return 0

    if cmd == "list":
        limit = 30
        if len(args) >= 3:
            try:
                limit = int(args[2])
            except ValueError:
                pass
        items = list_visible_items(data, limit=limit)
        for p in items:
            print(f"- {p.get('name_en')} / {p.get('name_cn')} (dist={p.get('distance')})")
        return 0

    if cmd == "find":
        if len(args) < 3:
            print("[!] 用法: python auto_flipping.py find <keyword>")
            return 2
        keyword = " ".join(args[2:])
        hits = find_items_by_name(keyword, data)
        if not hits:
            print("[!] 未找到匹配项")
            return 1
        for p in hits[:30]:
            print(f"- {p.get('name_en')} / {p.get('name_cn')} (go={p.get('game_object')}, dist={p.get('distance')})")
        return 0

    if cmd == "flip" or implicit_flip:
        if implicit_flip:
            meat_name = args[1]
            put_place_name = args[2]
            radius_arg = args[3] if len(args) >= 4 else None
            print("[*] 未检测到子命令，按 implicit flip 模式执行")
        else:
            if len(args) < 4:
                print("[!] 用法: python auto_flipping.py flip <meat> <put_place> [radius_m]")
                return 2
            meat_name = args[2]
            put_place_name = args[3]
            radius_arg = args[4] if len(args) >= 5 else None

        radius_m = DEFAULT_PLACE_RADIUS_M
        if radius_arg is not None:
            try:
                radius_m = float(radius_arg)
            except ValueError:
                pass

        for attempt in range(DEFAULT_MAX_ATTEMPTS):
            if auto_flip(meat_name, put_place_name, place_radius_m=radius_m, verbose=True):
                return 0
            print(f"[*] retry {attempt + 1}/{DEFAULT_MAX_ATTEMPTS}")
            time.sleep(0.2)
        return 1

    print(f"[!] 未知命令: {cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
