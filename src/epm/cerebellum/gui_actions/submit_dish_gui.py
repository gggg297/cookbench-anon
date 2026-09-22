from __future__ import annotations

import ctypes
import ctypes.wintypes
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.skills._shared_paths import repo_root
from epm.cerebellum.skills._shared_paths import userdata_root
from epm.vision.screen_capture import WindowRect, activate_window, capture_screenshot_mss, get_window_rect

MOUSEEVENTF_MOVE = 0x0001


def _try_enable_per_monitor_dpi_awareness() -> None:
    """
    Best-effort: reduce DPI virtualization mismatches on multi-monitor + mixed DPI setups.
    """
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return
    except Exception:
        pass
    try:
        shcore = ctypes.windll.shcore
        # PROCESS_PER_MONITOR_DPI_AWARE = 2
        shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


_try_enable_per_monitor_dpi_awareness()


def _get_cursor_pos() -> tuple[int, int]:
    pt = ctypes.wintypes.POINT()
    ok = int(ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)))
    if not ok:
        return (0, 0)
    return (int(pt.x), int(pt.y))


def _get_clip_cursor_rect() -> Optional[tuple[int, int, int, int]]:
    rect = ctypes.wintypes.RECT()
    ok = int(ctypes.windll.user32.GetClipCursor(ctypes.byref(rect)))
    if not ok:
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def _clip_cursor_rect(left: int, top: int, right: int, bottom: int) -> bool:
    rect = ctypes.wintypes.RECT(int(left), int(top), int(right), int(bottom))
    return bool(int(ctypes.windll.user32.ClipCursor(ctypes.byref(rect))))


def _unclip_cursor() -> bool:
    return bool(int(ctypes.windll.user32.ClipCursor(None)))


def _get_foreground_window_title() -> str:
    try:
        user32 = ctypes.windll.user32
        hwnd = int(user32.GetForegroundWindow())
        if hwnd == 0:
            return ""
        buf = ctypes.create_unicode_buffer(512)
        _ = int(user32.GetWindowTextW(hwnd, buf, int(len(buf))))
        return str(buf.value or "")
    except Exception:
        return ""


def _find_window_hwnd(window_title: str) -> int:
    try:
        return int(ctypes.windll.user32.FindWindowW(None, str(window_title)))
    except Exception:
        return 0


def _get_window_dpi_scale(window_title: str) -> float:
    """
    Return the effective DPI scale for the window's monitor.
    1.0 means 100%, 1.25 means 125%, etc.
    """
    hwnd = _find_window_hwnd(window_title)
    if hwnd:
        try:
            user32 = ctypes.windll.user32
            if hasattr(user32, "GetDpiForWindow"):
                dpi = int(user32.GetDpiForWindow(hwnd))
                if dpi > 0:
                    return max(1.0, float(dpi) / 96.0)
        except Exception:
            pass
        try:
            user32 = ctypes.windll.user32
            shcore = ctypes.windll.shcore
            monitor = int(user32.MonitorFromWindow(hwnd, 2))
            if monitor:
                dpi_x = ctypes.c_uint()
                dpi_y = ctypes.c_uint()
                # MDT_EFFECTIVE_DPI = 0
                hr = int(shcore.GetDpiForMonitor(monitor, 0, ctypes.byref(dpi_x), ctypes.byref(dpi_y)))
                if hr == 0 and int(dpi_x.value) > 0:
                    return max(1.0, float(int(dpi_x.value)) / 96.0)
        except Exception:
            pass
    return 1.0


def _get_ui_dump_screen_size(data: dict[str, Any]) -> Optional[tuple[int, int]]:
    try:
        screen = data.get("screen") if isinstance(data, dict) else None
        if not isinstance(screen, dict):
            return None
        width = int(screen.get("width"))
        height = int(screen.get("height"))
        if width > 0 and height > 0:
            return width, height
    except Exception:
        pass
    return None


def _is_key_down(vk: int) -> bool:
    try:
        return bool(int(ctypes.windll.user32.GetAsyncKeyState(int(vk))) & 0x8000)
    except Exception:
        return False


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    encodings = ("utf-8-sig", "utf-8", "gbk")
    last: Optional[Exception] = None
    for _ in range(8):
        for enc in encodings:
            try:
                raw = path.read_text(encoding=enc, errors="ignore")
                data = json.loads(raw)
                return data if isinstance(data, dict) else {"data": data}
            except json.JSONDecodeError as e:
                last = e
                continue
            except Exception as e:
                last = e
                break
        time.sleep(0.05)
    raise RuntimeError(f"read_json_failed:{path}: {last}")


def _trigger_ui_dump(
    *,
    io: RawInputController,
    dump_path: Path,
    wait_timeout_s: float,
) -> dict[str, Any]:
    prev_mtime = _mtime(dump_path)
    io.key_down("alt")
    io.key_press("y")
    io.key_up("alt")

    deadline = time.time() + float(wait_timeout_s)
    while time.time() < deadline:
        m = _mtime(dump_path)
        if m and m != prev_mtime:
            break
        time.sleep(0.05)

    if not dump_path.exists():
        raise RuntimeError(f"ui_targets_dump_missing:{str(dump_path)!r}")
    return _read_json(dump_path)


def _normalize_ui_text(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = s.strip().lower()
    return re.sub(r"\s+", " ", s)


def _ui_dump_has_submit_tooltip(data: dict[str, Any]) -> bool:
    if not isinstance(data, dict):
        return False
    for bucket_name in ("texts", "tmpTexts"):
        values = data.get(bucket_name)
        if not isinstance(values, list):
            continue
        for it in values:
            if not isinstance(it, dict):
                continue
            if not bool(it.get("activeInHierarchy", False)):
                continue
            txt = _normalize_ui_text(str(it.get("text", "")))
            if txt in ("serve dish.", "serve dish", "served"):
                return True
            if "serve dish" in txt:
                return True
    return False


def _slow_seek_submit_tooltip(
    *,
    io: RawInputController,
    dump_path: Path,
    click_x: int,
    click_y: int,
    step_px: int,
    max_seek_steps: int,
    wait_ui_dump_timeout_s: float,
    step_sleep_s: float,
    verbose: bool,
) -> dict[str, Any]:
    initial_data = _trigger_ui_dump(io=io, dump_path=dump_path, wait_timeout_s=wait_ui_dump_timeout_s)
    if _ui_dump_has_submit_tooltip(initial_data):
        cur_x, cur_y = _get_cursor_pos()
        io.click("left")
        return {
            "success": True,
            "click": {"x": int(click_x), "y": int(click_y)},
            "move": {
                "mode": "submit_tooltip_seek",
                "reason": "tooltip_already_active",
                "steps_taken": 0,
                "tooltip_found": True,
                "cursor_before": {"x": int(cur_x), "y": int(cur_y)},
                "cursor_after": {"x": int(cur_x), "y": int(cur_y)},
                "net_move": {"dx": 0, "dy": 0},
            },
            "target": {"x": int(click_x), "y": int(click_y), "src": "tooltip_seek"},
            "error": "",
        }

    cursor_before = _get_cursor_pos()
    dx_total = int(click_x) - int(cursor_before[0])
    dy_total = int(click_y) - int(cursor_before[1])
    direction = -1 if dx_total <= 0 else 1
    seek_step_px = max(8, int(step_px))
    total_steps = max(1, int(max_seek_steps))
    base_dy = int(dy_total / total_steps)
    rem_dy = int(dy_total - base_dy * total_steps)

    tooltip_found = False
    dump_after: Optional[dict[str, Any]] = None
    steps_taken = 0
    moved_dx = 0
    moved_dy = 0
    for i in range(total_steps):
        cur_dx = int(direction * seek_step_px)
        cur_dy = int(base_dy)
        if i < abs(rem_dy):
            cur_dy += 1 if rem_dy > 0 else -1
        _mouse_move_relative_quiet(cur_dx, cur_dy)
        moved_dx += int(cur_dx)
        moved_dy += int(cur_dy)
        steps_taken = i + 1
        if float(step_sleep_s) > 0:
            time.sleep(float(step_sleep_s))
        try:
            dump_after = _trigger_ui_dump(io=io, dump_path=dump_path, wait_timeout_s=wait_ui_dump_timeout_s)
        except Exception:
            dump_after = None
        if _ui_dump_has_submit_tooltip(dump_after or {}):
            tooltip_found = True
            break

    cursor_after = _get_cursor_pos()
    move = {
        "mode": "submit_tooltip_seek",
        "reason": "tooltip_found" if tooltip_found else "tooltip_not_found",
        "steps_taken": int(steps_taken),
        "step_px": int(seek_step_px),
        "max_seek_steps": int(total_steps),
        "step_sleep_s": float(step_sleep_s),
        "tooltip_found": bool(tooltip_found),
        "cursor_before": {"x": int(cursor_before[0]), "y": int(cursor_before[1])},
        "cursor_after": {"x": int(cursor_after[0]), "y": int(cursor_after[1])},
        "net_move": {"dx": int(moved_dx), "dy": int(moved_dy)},
        "cursor_delta": {"dx": int(cursor_after[0] - cursor_before[0]), "dy": int(cursor_after[1] - cursor_before[1])},
        "requested_target": {"x": int(click_x), "y": int(click_y)},
    }
    if verbose:
        print(
            f"[submit] seek_left tooltip_found={bool(tooltip_found)} steps={int(steps_taken)} "
            f"step_px={int(seek_step_px)} max_seek_steps={int(total_steps)} "
            f"cursor_before=({int(cursor_before[0])},{int(cursor_before[1])}) "
            f"cursor_after=({int(cursor_after[0])},{int(cursor_after[1])}) "
            f"net_move=({int(moved_dx)},{int(moved_dy)})"
        )

    if not tooltip_found:
        return {
            "success": False,
            "click": {"x": int(click_x), "y": int(click_y)},
            "move": move,
            "target": {"x": int(click_x), "y": int(click_y), "src": "tooltip_seek"},
            "error": "submit_tooltip_not_found_after_left_seek",
        }

    io.click("left")
    return {
        "success": True,
        "click": {"x": int(click_x), "y": int(click_y)},
        "move": move,
        "target": {"x": int(click_x), "y": int(click_y), "src": "tooltip_seek"},
        "error": "",
    }


def _pil_to_bgr(img) -> np.ndarray:
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _imread_gray(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def _imread_gray_with_mask(path: Path) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None, None
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None
        if img.ndim == 2:
            return img, None
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
            return gray, mask
        if img.ndim == 3 and img.shape[2] >= 3:
            gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            return gray, None
    except Exception:
        pass
    return None, None


def _match_best_with_scales(
    screenshot_bgr: np.ndarray,
    template_path: Path,
    *,
    scales: tuple[float, ...] = (0.85, 0.925, 1.0, 1.075, 1.15),
) -> tuple[Optional[dict[str, int | float]], float]:
    if not template_path.exists():
        return None, 0.0
    tpl = _imread_gray(template_path)
    if tpl is None:
        return None, 0.0
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape[:2]
    best: Optional[dict[str, int | float]] = None
    best_score = 0.0
    for scale in scales:
        if abs(scale - 1.0) < 1e-6:
            scaled = tpl
        else:
            scaled = cv2.resize(tpl, dsize=None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_LINEAR)
        th, tw = scaled.shape[:2]
        if th <= 0 or tw <= 0 or th > gh or tw > gw:
            continue
        res = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
        if float(max_val) >= best_score:
            best_score = float(max_val)
            best = {
                "x": int(max_loc[0]),
                "y": int(max_loc[1]),
                "w": int(tw),
                "h": int(th),
                "scale": float(scale),
            }
    return best, float(best_score)


def _match_best_with_scales_masked(
    screenshot_bgr: np.ndarray,
    template_path: Path,
    *,
    scales: tuple[float, ...],
) -> tuple[Optional[dict[str, int | float]], float]:
    if not template_path.exists():
        return None, 0.0
    tpl, mask = _imread_gray_with_mask(template_path)
    if tpl is None:
        return None, 0.0
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape[:2]
    best: Optional[dict[str, int | float]] = None
    best_score = 0.0
    for scale in scales:
        if abs(scale - 1.0) < 1e-6:
            scaled = tpl
            scaled_mask = mask
        else:
            scaled = cv2.resize(tpl, dsize=None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_LINEAR)
            scaled_mask = None
            if mask is not None:
                scaled_mask = cv2.resize(mask, dsize=None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_NEAREST)
        th, tw = scaled.shape[:2]
        if th <= 0 or tw <= 0 or th > gh or tw > gw:
            continue
        if scaled_mask is not None:
            if cv2.countNonZero(scaled_mask) <= 0:
                continue
            res = cv2.matchTemplate(gray, scaled, cv2.TM_CCORR_NORMED, mask=scaled_mask)
        else:
            res = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
        _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
        if float(max_val) >= best_score:
            best_score = float(max_val)
            best = {
                "x": int(max_loc[0]),
                "y": int(max_loc[1]),
                "w": int(tw),
                "h": int(th),
                "scale": float(scale),
            }
    return best, float(best_score)


def _clip_roi(
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    max_w: int,
    max_h: int,
) -> Optional[tuple[int, int, int, int]]:
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(int(max_w), int(x + w))
    y1 = min(int(max_h), int(y + h))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _crop_bgr(
    screenshot_bgr: np.ndarray,
    *,
    roi: Optional[tuple[int, int, int, int]],
) -> tuple[np.ndarray, tuple[int, int]]:
    if roi is None:
        return screenshot_bgr, (0, 0)
    x, y, w, h = roi
    return screenshot_bgr[y : y + h, x : x + w], (x, y)


def _tablet_inner_roi_from_match(
    *,
    tablet_match: Optional[dict[str, int | float]],
    screenshot_shape: tuple[int, ...],
) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(tablet_match, dict):
        return None
    try:
        x = int(tablet_match.get("x", 0))
        y = int(tablet_match.get("y", 0))
        w = int(tablet_match.get("w", 0))
        h = int(tablet_match.get("h", 0))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    # Inner visible panel coordinates in submit-tablet.png after hollowing.
    left_ratio = 87.0 / 918.0
    top_ratio = 60.0 / 568.0
    right_ratio = 821.0 / 918.0
    bottom_ratio = 513.0 / 568.0
    inner_x = x + int(round(w * left_ratio))
    inner_y = y + int(round(h * top_ratio))
    inner_w = int(round(w * (right_ratio - left_ratio)))
    inner_h = int(round(h * (bottom_ratio - top_ratio)))
    return _clip_roi(
        x=inner_x,
        y=inner_y,
        w=inner_w,
        h=inner_h,
        max_w=int(screenshot_shape[1]),
        max_h=int(screenshot_shape[0]),
    )


def _center_focus_roi(
    screenshot_shape: tuple[int, ...],
    *,
    x_ratio: float = 0.72,
    y_ratio: float = 0.78,
) -> Optional[tuple[int, int, int, int]]:
    h = int(screenshot_shape[0])
    w = int(screenshot_shape[1])
    roi_w = max(1, int(round(float(w) * float(x_ratio))))
    roi_h = max(1, int(round(float(h) * float(y_ratio))))
    x = max(0, int(round((w - roi_w) * 0.5)))
    y = max(0, int(round((h - roi_h) * 0.5)))
    return _clip_roi(x=x, y=y, w=roi_w, h=roi_h, max_w=w, max_h=h)


def _relative_roi_from_shape(
    screenshot_shape: tuple[int, ...],
    *,
    left_ratio: float,
    top_ratio: float,
    width_ratio: float,
    height_ratio: float,
) -> Optional[tuple[int, int, int, int]]:
    h = int(screenshot_shape[0])
    w = int(screenshot_shape[1])
    return _clip_roi(
        x=int(round(float(w) * float(left_ratio))),
        y=int(round(float(h) * float(top_ratio))),
        w=int(round(float(w) * float(width_ratio))),
        h=int(round(float(h) * float(height_ratio))),
        max_w=w,
        max_h=h,
    )


def _relative_roi_from_base(
    base_roi: Optional[tuple[int, int, int, int]],
    screenshot_shape: tuple[int, ...],
    *,
    left_ratio: float,
    top_ratio: float,
    width_ratio: float,
    height_ratio: float,
) -> Optional[tuple[int, int, int, int]]:
    if base_roi is None:
        return None
    base_x, base_y, base_w, base_h = base_roi
    return _clip_roi(
        x=int(round(float(base_x) + float(base_w) * float(left_ratio))),
        y=int(round(float(base_y) + float(base_h) * float(top_ratio))),
        w=int(round(float(base_w) * float(width_ratio))),
        h=int(round(float(base_h) * float(height_ratio))),
        max_w=int(screenshot_shape[1]),
        max_h=int(screenshot_shape[0]),
    )


@dataclass(frozen=True)
class SubmitFlowStageSnapshot:
    stage: str
    submit_icon: bool
    submit_icon_score: float
    tablet_frame: bool
    tablet_frame_score: float
    stage3_popup: bool
    stage3_popup_score: float


def _submit_stage_templates() -> dict[str, list[dict[str, Any]]]:
    base = (repo_root() / "data" / "figure" / "computer").resolve()
    return {
        "submit_icon": [
            {
                "path": (base / "submit.png").resolve(),
                "min_score": 0.88,
                "scales": (0.70, 0.85, 1.0, 1.15, 1.30),
            },
        ],
        "tablet_frame": [
            {
                "path": (base / "submit-tablet.png").resolve(),
                "min_score": 0.70,
                "scales": (0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15),
            },
        ],
        "stage3_popup": [
            {
                "path": (base / "submit-dish-step-3.png").resolve(),
                "min_score": 0.90,
                "scales": (0.60, 0.75, 0.90, 1.0, 1.10, 1.25),
            },
        ],
    }


def _best_stage_match(
    *,
    screenshot_bgr: np.ndarray,
    candidates: list[dict[str, Any]],
) -> tuple[bool, float, Optional[dict[str, int | float]]]:
    best_score = 0.0
    matched = False
    best_match: Optional[dict[str, int | float]] = None
    for cfg in candidates:
        path = cfg.get("path")
        if not isinstance(path, Path) or (not path.exists()):
            continue
        match, score = _match_best_with_scales_masked(
            screenshot_bgr,
            path,
            scales=tuple(cfg.get("scales") or (1.0,)),
        )
        if score >= best_score:
            best_score = float(score)
            best_match = match
        if match is not None and score >= float(cfg.get("min_score", 0.8) or 0.8):
            matched = True
    return matched, float(best_score), best_match


def detect_submit_flow_stage(
    *,
    window_title: str = "CookingSimulator",
    activate: bool = False,
    verbose: bool = True,
) -> SubmitFlowStageSnapshot:
    if activate:
        try:
            activate_window(window_title)
        except Exception:
            pass
    rect = get_window_rect(window_title)
    img = capture_screenshot_mss(region=rect.to_mss_region())
    bgr = _pil_to_bgr(img)
    templates = _submit_stage_templates()
    tablet_focus_roi = _center_focus_roi(bgr.shape)
    tablet_focus_bgr, tablet_focus_offset = _crop_bgr(bgr, roi=tablet_focus_roi)
    tablet_frame, tablet_frame_score, tablet_frame_match_local = _best_stage_match(
        screenshot_bgr=tablet_focus_bgr,
        candidates=templates["tablet_frame"],
    )
    tablet_frame_match: Optional[dict[str, int | float]] = None
    if isinstance(tablet_frame_match_local, dict):
        off_x, off_y = tablet_focus_offset
        tablet_frame_match = {
            "x": int(tablet_frame_match_local.get("x", 0)) + int(off_x),
            "y": int(tablet_frame_match_local.get("y", 0)) + int(off_y),
            "w": int(tablet_frame_match_local.get("w", 0)),
            "h": int(tablet_frame_match_local.get("h", 0)),
            "scale": float(tablet_frame_match_local.get("scale", 1.0)),
        }
    # Use a fixed inner-tablet ROI relative to the game window; it is more stable
    # than chasing the hollow-frame template after cursor/UI transitions.
    tablet_inner_roi = _relative_roi_from_shape(
        bgr.shape,
        left_ratio=0.26,
        top_ratio=0.226,
        width_ratio=0.459,
        height_ratio=0.503,
    )
    stage_inner_bgr, _stage_inner_offset = _crop_bgr(bgr, roi=tablet_inner_roi)
    submit_icon_roi = _relative_roi_from_base(
        tablet_inner_roi,
        bgr.shape,
        left_ratio=0.24,
        top_ratio=0.70,
        width_ratio=0.52,
        height_ratio=0.24,
    )
    submit_icon_bgr, _submit_icon_offset = _crop_bgr(bgr, roi=submit_icon_roi)
    stage3_panel_roi = _relative_roi_from_base(
        tablet_inner_roi,
        bgr.shape,
        left_ratio=0.0,
        top_ratio=0.0,
        width_ratio=0.42,
        height_ratio=1.0,
    )
    stage3_panel_bgr, _stage3_panel_offset = _crop_bgr(bgr, roi=stage3_panel_roi)
    submit_icon = False
    submit_icon_score = 0.0
    if submit_icon_roi is not None:
        submit_icon, submit_icon_score, _submit_icon_match = _best_stage_match(
            screenshot_bgr=submit_icon_bgr,
            candidates=templates["submit_icon"],
        )
    stage3_popup = False
    stage3_popup_score = 0.0
    if stage3_panel_roi is not None:
        stage3_popup, stage3_popup_score, _stage3_popup_match = _best_stage_match(
            screenshot_bgr=stage3_panel_bgr,
            candidates=templates["stage3_popup"],
        )
    if stage3_popup and (not submit_icon):
        stage = "stage_3_feedback_popup"
    elif submit_icon and tablet_frame:
        stage = "stage_1_submit_panel"
    elif tablet_frame:
        stage = "stage_2_dish_grid"
    else:
        stage = "unknown"
    snapshot = SubmitFlowStageSnapshot(
        stage=stage,
        submit_icon=bool(submit_icon),
        submit_icon_score=float(submit_icon_score),
        tablet_frame=bool(tablet_frame),
        tablet_frame_score=float(tablet_frame_score),
        stage3_popup=bool(stage3_popup),
        stage3_popup_score=float(stage3_popup_score),
    )
    if verbose:
        print(
            "[submit_stage] "
            f"stage={snapshot.stage} "
            f"submit_icon={snapshot.submit_icon} submit_icon_score={snapshot.submit_icon_score:.3f} "
            f"tablet_frame={snapshot.tablet_frame} tablet_frame_score={snapshot.tablet_frame_score:.3f} "
            f"stage3_popup={snapshot.stage3_popup} stage3_popup_score={snapshot.stage3_popup_score:.3f} "
            f"tablet_focus_roi={tablet_focus_roi!r} tablet_inner_roi={tablet_inner_roi!r} "
            f"submit_icon_roi={submit_icon_roi!r} stage3_panel_roi={stage3_panel_roi!r}"
        )
    return snapshot


def wait_for_submit_flow_stage(
    *,
    expected_stage: str,
    window_title: str = "CookingSimulator",
    timeout_s: float = 2.0,
    poll_s: float = 0.25,
    stable_hits: int = 2,
    verbose: bool = True,
) -> dict[str, Any]:
    deadline = time.time() + max(0.1, float(timeout_s))
    expected = str(expected_stage or "").strip()
    hits = 0
    last_stage = ""
    last_snapshot: Optional[SubmitFlowStageSnapshot] = None
    while time.time() < deadline:
        snap = detect_submit_flow_stage(window_title=window_title, activate=False, verbose=verbose)
        last_snapshot = snap
        if last_stage and last_stage != snap.stage and verbose:
            print(f"[submit_stage_transition] from={last_stage} to={snap.stage}")
        last_stage = snap.stage
        if snap.stage == expected:
            hits += 1
            if hits >= max(1, int(stable_hits)):
                return {"success": True, "stage": snap.stage, "snapshot": snap}
        else:
            hits = 0
        time.sleep(max(0.05, float(poll_s)))
    return {
        "success": False,
        "stage": (last_snapshot.stage if last_snapshot is not None else "unknown"),
        "snapshot": last_snapshot,
        "error": f"submit_stage_timeout:expected={expected!r}",
    }


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", (s or "").strip()).casefold()


def _norm_key(s: str) -> str:
    """
    Unicode-tolerant matching key for dish text:
    - casefold
    - strip diacritics (e.g., é -> e, ä -> a)
    - remove non-alnum separators/punctuation
    """
    t = _norm(s)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^0-9a-z]+", "", t)
    return t


def _rect_area(rect_tl: dict[str, Any]) -> float:
    try:
        left = float(rect_tl.get("left", 0))
        right = float(rect_tl.get("right", 0))
        top = float(rect_tl.get("top", 0))
        bottom = float(rect_tl.get("bottom", 0))
        return max(0.0, right - left) * max(0.0, bottom - top)
    except Exception:
        return 0.0


def _sorted_text_hits(
    items: list[dict[str, Any]],
    *,
    dish_name: str,
    window_width: int,
    window_height: int,
) -> list[dict[str, Any]]:
    want_text = _norm(dish_name)
    want_key = _norm_key(dish_name)
    if not want_text and not want_key:
        return []

    hits: list[dict[str, Any]] = []
    exact_key_hits: list[dict[str, Any]] = []
    for it in items:
        text_raw = str(it.get("text", ""))
        text = _norm(text_raw)
        key = _norm_key(text_raw)
        if text == want_text:
            hits.append(it)
        elif key and key == want_key:
            exact_key_hits.append(it)

    if not hits:
        hits = exact_key_hits

    if not hits:
        for it in items:
            text_raw = str(it.get("text", ""))
            text = _norm(text_raw)
            key = _norm_key(text_raw)
            if (want_text and want_text in text) or (want_key and (want_key in key or key in want_key)):
                hits.append(it)

    def _norm_valid(it: dict[str, Any]) -> int:
        cn = it.get("center_norm") if isinstance(it.get("center_norm"), dict) else None
        if not isinstance(cn, dict):
            return 0
        try:
            nx = float(cn.get("x"))
            ny = float(cn.get("y"))
        except Exception:
            return 0
        return 1 if (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0) else 0

    def _window_rect_valid(it: dict[str, Any]) -> int:
        rt = it.get("rect_tl") if isinstance(it.get("rect_tl"), dict) else None
        if not isinstance(rt, dict):
            return 0
        try:
            cx = float(rt.get("centerX"))
            cy = float(rt.get("centerY"))
        except Exception:
            return 0
        return 1 if (0.0 <= cx <= float(window_width) and 0.0 <= cy <= float(window_height)) else 0

    def key(it: dict[str, Any]) -> tuple[int, int, int, float]:
        active = 1 if bool(it.get("activeInHierarchy", True)) else 0
        rect_ok = _window_rect_valid(it)
        norm_ok = _norm_valid(it)
        rect_tl = it.get("rect_tl") if isinstance(it.get("rect_tl"), dict) else {}
        area = _rect_area(rect_tl if isinstance(rect_tl, dict) else {})
        return (active, rect_ok, norm_ok, area)

    hits.sort(key=key, reverse=True)
    return hits


def _filter_checkout_stand_entries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix = "details/work stands/checkout stand/tablet/tablet gui/background/checkoutdishchoice/scroll view/viewport/content/tabletrecipebutton(clone)"
    out: list[dict[str, Any]] = []
    for it in items:
        path = _norm(str(it.get("path", "")))
        if path.startswith(prefix):
            out.append(it)
    return out


def _is_checkout_dish_name_entry(it: dict[str, Any]) -> bool:
    path = _norm(str(it.get("path", "")))
    name = _norm(str(it.get("name", "")))
    text = str(it.get("text", "")).strip()
    if not text:
        return False
    if "/dish name bg/name" not in path:
        return False
    if name not in ("name", "latin", "tmp"):
        return False
    return True


def _checkout_slot_sort_key(it: dict[str, Any]) -> tuple[float, float]:
    rt = it.get("rect_tl") if isinstance(it.get("rect_tl"), dict) else {}
    try:
        return (float(rt.get("centerY")), float(rt.get("centerX")))
    except Exception:
        pass
    cn = it.get("center_norm") if isinstance(it.get("center_norm"), dict) else {}
    try:
        return (1.0 - float(cn.get("y")), float(cn.get("x")))
    except Exception:
        return (1e9, 1e9)


def _visible_checkout_dish_entries(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [it for it in items if bool(it.get("activeInHierarchy", True)) and _is_checkout_dish_name_entry(it)]
    out.sort(key=_checkout_slot_sort_key)
    return out


def _hit_to_abs_click_xy(
    hit: dict[str, Any],
    *,
    window_left: int,
    window_top: int,
    window_width: int,
    window_height: int,
    dpi_scale: float = 1.0,
) -> Optional[tuple[int, int, str]]:
    """
    Convert a single Alt+Y hit record into virtual-desktop absolute click coords.
    Returns (x, y, source) or None if coordinates are unusable.
    """
    scale = float(dpi_scale) if float(dpi_scale) > 0 else 1.0

    try:
        cn = hit.get("center_norm") if isinstance(hit.get("center_norm"), dict) else None
        if isinstance(cn, dict) and cn.get("x") is not None and cn.get("y") is not None:
            nx = float(cn.get("x"))
            ny = float(cn.get("y"))
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                x = int(window_left + nx * float(window_width))
                y = int(window_top + (1.0 - ny) * float(window_height))
                if abs(scale - 1.0) > 0.01:
                    return x, y, "center_norm_dpi_aware"
                return x, y, "center_norm"
    except Exception:
        pass

    rect_tl = hit.get("rect_tl") if isinstance(hit.get("rect_tl"), dict) else {}
    try:
        cx = float(rect_tl.get("centerX"))
        cy = float(rect_tl.get("centerY"))
        scaled_cx = float(cx) * scale
        scaled_cy = float(cy) * scale
        if 0.0 <= scaled_cx <= float(window_width) and 0.0 <= scaled_cy <= float(window_height):
            return int(window_left + scaled_cx), int(window_top + scaled_cy), ("rect_tl_scaled" if abs(scale - 1.0) > 0.01 else "rect_tl")
        if 0.0 <= cx <= float(window_width) and 0.0 <= cy <= float(window_height):
            return int(window_left + cx), int(window_top + cy), "rect_tl"
    except Exception:
        pass

    return None


def _get_window_rect_win32(window_title: str) -> Optional[WindowRect]:
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_title)
        if hwnd == 0:
            return None
        rect = ctypes.wintypes.RECT()
        ok = int(user32.GetWindowRect(hwnd, ctypes.byref(rect)))
        if not ok:
            return None
        left = int(rect.left)
        top = int(rect.top)
        right = int(rect.right)
        bottom = int(rect.bottom)
        return WindowRect(left=left, top=top, width=max(0, right - left), height=max(0, bottom - top))
    except Exception:
        return None


def _mouse_move_relative_quiet(dx: int, dy: int) -> None:
    try:
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
    except Exception:
        # Fall back to RawInputController which uses the same API but prints debug.
        RawInputController().mouse_move_relative(int(dx), int(dy))


def _move_relative_in_steps(*, dx: int, dy: int, steps: int, step_sleep_s: float) -> None:
    n = int(steps) if int(steps) > 0 else 1
    step_dx = int(dx / n)
    step_dy = int(dy / n)
    rem_dx = int(dx - step_dx * n)
    rem_dy = int(dy - step_dy * n)
    for i in range(n):
        cur_dx = step_dx + (1 if i < abs(rem_dx) and rem_dx > 0 else 0) + (-1 if i < abs(rem_dx) and rem_dx < 0 else 0)
        cur_dy = step_dy + (1 if i < abs(rem_dy) and rem_dy > 0 else 0) + (-1 if i < abs(rem_dy) and rem_dy < 0 else 0)
        if cur_dx != 0 or cur_dy != 0:
            _mouse_move_relative_quiet(int(cur_dx), int(cur_dy))
        if float(step_sleep_s) > 0:
            time.sleep(float(step_sleep_s))


def _move_relative_converge_to_target(
    *,
    io: RawInputController,
    target_x: int,
    target_y: int,
    tolerance_px: int = 8,
    max_iters: int = 36,
    gain: float = 0.58,
    min_step_px: int = 1,
    max_step_px: int = 10,
    step_sleep_s: float = 0.006,
) -> dict[str, Any]:
    """
    Closed-loop relative movement: repeatedly read current cursor position and
    move a fraction of the remaining delta so the cursor converges to target.
    """
    tol = max(1, int(tolerance_px))
    n_iters = max(1, int(max_iters))
    g = float(gain)
    if not (0.05 <= g <= 0.95):
        g = 0.58
    min_step = max(1, int(min_step_px))
    max_step = max(min_step, int(max_step_px))
    sleep_s = max(0.0, float(step_sleep_s))

    start = _get_cursor_pos()
    cur = start
    reached = False
    reason = "max_iters_reached"
    moved_iters = 0

    for i in range(n_iters):
        dx = int(target_x) - int(cur[0])
        dy = int(target_y) - int(cur[1])
        if abs(dx) <= tol and abs(dy) <= tol:
            reached = True
            reason = "within_tolerance"
            break

        step_dx = int(round(float(dx) * g))
        step_dy = int(round(float(dy) * g))
        if step_dx == 0 and dx != 0:
            step_dx = min_step if dx > 0 else -min_step
        if step_dy == 0 and dy != 0:
            step_dy = min_step if dy > 0 else -min_step

        if abs(step_dx) > abs(dx):
            step_dx = dx
        if abs(step_dy) > abs(dy):
            step_dy = dy

        # Cap per-iteration move distance so the cursor does not overshoot heavily
        # on the first few iterations when the remaining error is large.
        step_len = (float(step_dx) * float(step_dx) + float(step_dy) * float(step_dy)) ** 0.5
        if step_len > float(max_step) and step_len > 0.0:
            k = float(max_step) / step_len
            step_dx = int(round(float(step_dx) * k))
            step_dy = int(round(float(step_dy) * k))
            if step_dx == 0 and dx != 0:
                step_dx = 1 if dx > 0 else -1
            if step_dy == 0 and dy != 0:
                step_dy = 1 if dy > 0 else -1

        if step_dx == 0 and step_dy == 0:
            reason = "zero_step"
            break

        io.mouse_move_relative(int(step_dx), int(step_dy))
        moved_iters = i + 1
        if sleep_s > 0:
            time.sleep(sleep_s)

        nxt = _get_cursor_pos()
        if nxt == cur:
            reason = "cursor_not_moving"
            cur = nxt
            break
        cur = nxt

    final_dx = int(target_x) - int(cur[0])
    final_dy = int(target_y) - int(cur[1])
    if (not reached) and abs(final_dx) <= tol and abs(final_dy) <= tol:
        reached = True
        reason = "within_tolerance_after_loop"

    return {
        "reached": bool(reached),
        "reason": str(reason),
        "iters": int(moved_iters),
        "cursor_before": {"x": int(start[0]), "y": int(start[1])},
        "cursor_after": {"x": int(cur[0]), "y": int(cur[1])},
        "remaining": {"dx": int(final_dx), "dy": int(final_dy)},
        "tolerance_px": int(tol),
    }


def _probe_cursor_movable(io: RawInputController) -> bool:
    try:
        probe_before = _get_cursor_pos()
        io.mouse_move_relative(1, 0)
        time.sleep(0.002)
        probe_after = _get_cursor_pos()
        io.mouse_move_relative(-1, 0)
        return (probe_after[0] != probe_before[0]) or (probe_after[1] != probe_before[1])
    except Exception:
        return False


def rewind_submit_cursor_to_window_center(
    *,
    submit_result: dict[str, Any],
    window_title: str = "CookingSimulator",
    steps: int = 18,
    step_sleep_s: float = 0.006,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    After clicking the submit panel, undo the recorded submit-seek cursor drift
    by moving back with the exact opposite delta. The submit flow starts from
    the checkout 3x3 grid center, so this inverse move should bring the cursor
    back to that center reference before the dish-entry click.
    """
    move = submit_result.get("move") if isinstance(submit_result, dict) else {}
    move = move if isinstance(move, dict) else {}
    net_move = move.get("net_move") if isinstance(move.get("net_move"), dict) else {}

    try:
        activate_window(window_title)
    except Exception:
        pass

    out: dict[str, Any] = {
        "success": True,
        "undo_move": None,
        "error": "",
    }

    try:
        undo_dx = -int(net_move.get("dx"))
        undo_dy = -int(net_move.get("dy"))
        io = RawInputController()
        if undo_dx != 0 or undo_dy != 0:
            _move_relative_in_steps(
                dx=int(undo_dx),
                dy=int(undo_dy),
                steps=max(1, int(steps)),
                step_sleep_s=float(step_sleep_s),
            )
        out["undo_move"] = {
            "dx": int(undo_dx),
            "dy": int(undo_dy),
            "source_net_move": {
                "dx": int(net_move.get("dx") or 0),
                "dy": int(net_move.get("dy") or 0),
            },
        }
    except Exception:
        out["undo_move"] = {
            "dx": 0,
            "dy": 0,
            "skipped": True,
            "reason": "missing_submit_net_move",
        }

    if verbose:
        print(
            f"[submit] rewind_to_center success={bool(out.get('success'))} "
            f"undo={out.get('undo_move')} "
            f"error={out.get('error')!r}"
        )
    return out


def click_submit_icon(
    *,
    window_title: str = "CookingSimulator",
    strategy: str = "template_match",
    relative_dx: int = -400,
    relative_dy: int = 0,
    steps: int = 12,
    step_sleep_s: float = 0.008,
    min_score: float = 0.80,
    retries: int = 6,
    retry_sleep_s: float = 0.20,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Click the submit entry point in the Checkout Stand UI.

    Strategy:
    - `template_match`: detect `epm/data/figure/computer/submit.png` and click its center.
    - `relative_offset`: keep the legacy hardcoded offset path for rollback/debugging.
    """
    strat = (strategy or "").strip().lower()
    if strat not in ("template", "template_match", "match", "relative", "relative_offset", "offset"):
        return {"success": False, "best": 0.0, "error": f"unsupported_submit_click_strategy:{strategy!r}"}

    try:
        activate_window(window_title)
    except Exception:
        pass

    io = RawInputController()
    if strat in ("template", "template_match", "match"):
        try:
            rect = get_window_rect(window_title)
        except Exception as e:
            return {"success": False, "best": 0.0, "error": f"get_window_rect_failed:{e}"}

        submit_tpl = (repo_root() / "data" / "figure" / "computer" / "submit.png").resolve()
        best_match: Optional[dict[str, int | float]] = None
        best_score = 0.0
        for attempt in range(max(1, int(retries))):
            try:
                img = capture_screenshot_mss(region=rect.to_mss_region())
            except Exception as e:
                return {"success": False, "best": 0.0, "error": f"capture_screenshot_failed:{e}"}
            bgr = _pil_to_bgr(img)
            match, score = _match_best_with_scales(bgr, submit_tpl)
            if score >= best_score:
                best_match = match
                best_score = float(score)
            if match is not None and score >= float(min_score):
                cx = rect.left + int(match["x"]) + int(match["w"]) // 2
                cy = rect.top + int(match["y"]) + int(match["h"]) // 2
                target_win_x = int(match["x"]) + int(match["w"]) // 2
                target_win_y = int(match["y"]) + int(match["h"]) // 2
                cursor_movable = _probe_cursor_movable(io)
                if cursor_movable:
                    move_res = _move_relative_converge_to_target(
                        io=io,
                        target_x=int(cx),
                        target_y=int(cy),
                        tolerance_px=8,
                        max_iters=36,
                        gain=0.58,
                        min_step_px=1,
                        max_step_px=10,
                        step_sleep_s=0.006,
                    )
                    move_ok = bool(move_res.get("reached", False))
                    move_mode = "cursor_feedback"
                    if (not move_ok) and str(move_res.get("reason") or "") == "cursor_not_moving":
                        base_x = int(rect.width * 0.5)
                        base_y = int(rect.height * 0.5)
                        rel_dx = int(target_win_x - base_x)
                        rel_dy = int(target_win_y - base_y)
                        _move_relative_in_steps(dx=int(rel_dx), dy=int(rel_dy), steps=18, step_sleep_s=0.006)
                        move_res = {
                            "reached": True,
                            "reason": "window_center_fallback_after_cursor_stall",
                            "iters": 18,
                            "cursor_before": move_res.get("cursor_before"),
                            "cursor_after": move_res.get("cursor_after"),
                            "remaining": {"dx": 0, "dy": 0},
                            "tolerance_px": 8,
                            "relative_move": {"dx": int(rel_dx), "dy": int(rel_dy)},
                            "initial_feedback_move": move_res,
                        }
                        move_ok = True
                        move_mode = "window_center_fallback"
                else:
                    base_x = int(rect.width * 0.5)
                    base_y = int(rect.height * 0.5)
                    rel_dx = int(target_win_x - base_x)
                    rel_dy = int(target_win_y - base_y)
                    _move_relative_in_steps(dx=int(rel_dx), dy=int(rel_dy), steps=18, step_sleep_s=0.006)
                    move_res = {
                        "reached": True,
                        "reason": "window_center_fallback",
                        "iters": 18,
                        "cursor_before": None,
                        "cursor_after": None,
                        "remaining": {"dx": 0, "dy": 0},
                        "tolerance_px": 8,
                        "relative_move": {"dx": int(rel_dx), "dy": int(rel_dy)},
                    }
                    move_ok = True
                    move_mode = "window_center_fallback"
                if verbose:
                    print(
                        f"[submit] strategy=template_match score={score:.3f} scale={float(match['scale']):.3f} "
                        f"click=({cx},{cy}) move_mode={move_mode} move_reason={move_res.get('reason')} "
                        f"reached={move_res.get('reached')}"
                    )
                if not bool(move_ok):
                    best_match = {
                        "x": int(match["x"]),
                        "y": int(match["y"]),
                        "w": int(match["w"]),
                        "h": int(match["h"]),
                        "scale": float(match["scale"]),
                    }
                    if attempt + 1 < max(1, int(retries)):
                        time.sleep(float(retry_sleep_s))
                        continue
                    return {
                        "success": False,
                        "best": float(score),
                        "click": {"x": int(cx), "y": int(cy)},
                        "move": {"mode": move_mode, **move_res},
                        "match": {
                            "src": str(submit_tpl),
                            "x": int(match["x"]),
                            "y": int(match["y"]),
                            "w": int(match["w"]),
                            "h": int(match["h"]),
                            "scale": float(match["scale"]),
                        },
                        "error": f"submit_cursor_not_reached:{move_res.get('reason')}",
                    }
                time.sleep(0.05)
                io.click("left")
                return {
                    "success": True,
                    "best": float(score),
                    "click": {"x": int(cx), "y": int(cy)},
                    "move": {"mode": move_mode, **move_res},
                    "match": {
                        "src": str(submit_tpl),
                        "x": int(match["x"]),
                        "y": int(match["y"]),
                        "w": int(match["w"]),
                        "h": int(match["h"]),
                        "scale": float(match["scale"]),
                    },
                    "error": "",
                }
            if attempt + 1 < max(1, int(retries)):
                time.sleep(float(retry_sleep_s))
        return {
            "success": False,
            "best": float(best_score),
            "click": None,
            "move": None,
            "match": best_match,
            "error": f"submit_template_not_found:min_score={float(min_score):.3f}",
        }

    dx = int(relative_dx)
    dy = int(relative_dy)
    if verbose:
        print(
            f"[submit] strategy=relative_offset move(dx={dx},dy={dy}) steps={int(steps)} step_sleep_s={float(step_sleep_s):.3f}"
        )
    _move_relative_in_steps(dx=dx, dy=dy, steps=int(steps), step_sleep_s=float(step_sleep_s))
    time.sleep(0.03)
    io.click("left")
    return {
        "success": True,
        "best": 1.0,
        "click": None,
        "move": None,
        "match": {"src": "relative_offset", "relative_dx": dx, "relative_dy": dy, "steps": int(steps)},
        "error": "",
    }


def _click_ui_dump_hit(
    *,
    io: RawInputController,
    hit: dict[str, Any],
    click_xy: tuple[int, int, str],
    window_rect: WindowRect,
    game_screen_size: Optional[tuple[int, int]],
    move_strategy: str,
    submit_anchor_move: Optional[dict[str, Any]],
    grid_slot_index: Optional[int],
    grid_visible_count: int,
    relative_scale: float,
    relative_steps: int,
    relative_step_sleep_s: float,
    require_move_within_tolerance: bool,
    move_tolerance_px: int,
    verbose: bool,
    debug_label: str,
) -> dict[str, Any]:
    click_x, click_y, click_src = click_xy
    target_win_x: Optional[int] = None
    target_win_y: Optional[int] = None
    try:
        rect_tl = hit.get("rect_tl") if isinstance(hit.get("rect_tl"), dict) else {}
        target_win_x = int(float(rect_tl.get("centerX")))
        target_win_y = int(float(rect_tl.get("centerY")))
    except Exception:
        try:
            cn = hit.get("center_norm") if isinstance(hit.get("center_norm"), dict) else None
            if isinstance(cn, dict):
                nx = float(cn.get("x"))
                ny = float(cn.get("y"))
                target_win_x = int(nx * float(window_rect.width))
                target_win_y = int((1.0 - ny) * float(window_rect.height))
        except Exception:
            target_win_x = None
            target_win_y = None

    strat = (move_strategy or "").strip().lower()
    if strat not in ("auto", "absolute", "relative_window_center", "checkout_grid_slot", "checkout_grid_locked_cursor"):
        return {"success": False, "error": f"unknown_move_strategy:{move_strategy!r}"}

    if strat == "checkout_grid_locked_cursor":
        if grid_slot_index is None or grid_slot_index < 0 or grid_slot_index > 8:
            return {
                "success": False,
                "error": f"{debug_label}_grid_slot_invalid:{grid_slot_index}",
                "target": {"x": int(click_x), "y": int(click_y), "src": str(click_src)},
            }
        anchor = submit_anchor_move if isinstance(submit_anchor_move, dict) else {}
        try:
            anchor_dx = int(anchor.get("dx"))
        except Exception:
            anchor_dx = 0
        cell_dx = abs(int(anchor_dx))
        if cell_dx <= 0:
            return {
                "success": False,
                "error": f"{debug_label}_grid_anchor_missing",
                "target": {"x": int(click_x), "y": int(click_y), "src": str(click_src)},
            }
        cell_dy = max(1, int(round(float(cell_dx) * 3.0 / 4.0)))
        col = int(grid_slot_index % 3)
        row = int(grid_slot_index // 3)
        dx = int((col - 1) * cell_dx)
        dy = int((row - 1) * cell_dy)
        move_reason = "game_screen_center:grid_locked"
        if row == 2:
            dx = int(round(float(dx) * 0.75))
            dy = int(round(float(dy) * 0.75))
            move_reason = f"{move_reason}:bottom_row*0.75"
        move_reason = f"{move_reason}:grid_slot[{row},{col}]"
        try:
            activate_window("CookingSimulator")
            time.sleep(0.05)
        except Exception:
            pass
        try:
            # Small nudge helps the game UI consume the relative-input path.
            io.mouse_move_relative(1, 0)
            time.sleep(0.01)
            io.mouse_move_relative(-1, 0)
            time.sleep(0.01)
        except Exception:
            pass
        _move_relative_in_steps(dx=int(dx), dy=int(dy), steps=int(relative_steps), step_sleep_s=float(relative_step_sleep_s))
        time.sleep(0.03)
        io.click("left")
        return {
            "success": True,
            "click": {"x": int(click_x), "y": int(click_y)},
            "move": {
                "mode": "checkout_grid_locked_cursor",
                "reached": True,
                "reason": str(move_reason),
                "iters": int(relative_steps),
                "cursor_before": None,
                "cursor_after": None,
                "remaining": {"dx": 0, "dy": 0},
                "tolerance_px": int(move_tolerance_px),
                "relative_move": {"dx": int(dx), "dy": int(dy)},
                "grid_slot_index": int(grid_slot_index),
                "grid_visible_count": int(grid_visible_count),
            },
            "target": {"x": int(click_x), "y": int(click_y), "src": str(click_src)},
            "error": "",
        }

    cursor_before = _get_cursor_pos()
    clip_before = _get_clip_cursor_rect()
    moved_ok = False
    move_result: Optional[dict[str, Any]] = None
    if strat == "absolute":
        restore_clip: Optional[tuple[int, int, int, int]] = None
        if clip_before is not None:
            l, t, r2, b2 = clip_before
            clip_w = int(r2 - l)
            clip_h = int(b2 - t)
            target_inside = (l <= int(click_x) <= r2) and (t <= int(click_y) <= b2)
            if (clip_w <= 2 and clip_h <= 2) or (not target_inside):
                restore_clip = clip_before
                _unclip_cursor()
                time.sleep(0.003)

        io.mouse_move_absolute(int(click_x), int(click_y))
        time.sleep(0.05)
        io.click("left")
        if restore_clip is not None:
            _clip_cursor_rect(*restore_clip)
        time.sleep(0.10)

        cursor_after = _get_cursor_pos()
        moved_ok = (abs(cursor_after[0] - int(click_x)) <= 4) and (abs(cursor_after[1] - int(click_y)) <= 4)
        move_result = {
            "mode": "absolute",
            "cursor_before": {"x": int(cursor_before[0]), "y": int(cursor_before[1])},
            "cursor_after": {"x": int(cursor_after[0]), "y": int(cursor_after[1])},
        }

    if not moved_ok and strat in ("auto", "relative_window_center", "checkout_grid_slot"):
        if target_win_x is None or target_win_y is None:
            return {
                "success": False,
                "error": f"{debug_label}_relative_missing_window_coords",
                "target": {"x": int(click_x), "y": int(click_y), "src": str(click_src)},
            }

        cursor_rel_base = _get_cursor_pos()
        rel_src = "cursor_now"
        cursor_movable = False
        if strat == "checkout_grid_slot":
            if game_screen_size is not None:
                screen_w, screen_h = game_screen_size
                cursor_rel_base = (int(window_rect.left + screen_w * 0.5), int(window_rect.top + screen_h * 0.5))
                rel_src = "game_screen_center"
            else:
                cursor_rel_base = (int(window_rect.left + window_rect.width * 0.5), int(window_rect.top + window_rect.height * 0.5))
                rel_src = "window_center"
            if grid_slot_index is None or grid_slot_index < 0 or grid_slot_index > 8:
                return {
                    "success": False,
                    "error": f"{debug_label}_grid_slot_invalid:{grid_slot_index}",
                    "target": {"x": int(click_x), "y": int(click_y), "src": str(click_src)},
                }
            anchor = submit_anchor_move if isinstance(submit_anchor_move, dict) else {}
            try:
                anchor_dx = int(anchor.get("dx"))
            except Exception:
                anchor_dx = 0
            cell_dx = abs(int(anchor_dx))
            if cell_dx <= 0:
                return {
                    "success": False,
                    "error": f"{debug_label}_grid_anchor_missing",
                    "target": {"x": int(click_x), "y": int(click_y), "src": str(click_src)},
                }
            cell_dy = max(1, int(round(float(cell_dx) * 3.0 / 4.0)))
            col = int(grid_slot_index % 3)
            row = int(grid_slot_index // 3)
            dx = int((col - 1) * cell_dx)
            dy = int((row - 1) * cell_dy)
            if row == 2:
                dx = int(round(float(dx) * 0.75))
                dy = int(round(float(dy) * 0.75))
                rel_src = f"{rel_src}:bottom_row*0.75"
            rel_src = f"{rel_src}:grid_slot[{row},{col}]"
        elif strat == "relative_window_center":
            if game_screen_size is not None:
                screen_w, screen_h = game_screen_size
                cursor_rel_base = (int(window_rect.left + screen_w * 0.5), int(window_rect.top + screen_h * 0.5))
                rel_src = "game_screen_center"
            else:
                cursor_rel_base = (int(window_rect.left + window_rect.width * 0.5), int(window_rect.top + window_rect.height * 0.5))
                rel_src = "window_center"
        else:
            try:
                probe_before = cursor_rel_base
                io.mouse_move_relative(1, 0)
                time.sleep(0.002)
                probe_after = _get_cursor_pos()
                io.mouse_move_relative(-1, 0)
                cursor_movable = (probe_after[0] != probe_before[0]) or (probe_after[1] != probe_before[1])
            except Exception:
                cursor_movable = False

        if (not cursor_movable) and strat == "auto":
            cursor_rel_base = (int(window_rect.left + window_rect.width * 0.5), int(window_rect.top + window_rect.height * 0.5))
            rel_src = "window_center_fallback"

        if strat != "checkout_grid_slot":
            dx = int(int(click_x) - int(cursor_rel_base[0]))
            dy = int(int(click_y) - int(cursor_rel_base[1]))

        if not cursor_movable:
            scale = float(relative_scale)
            if not (0.05 <= scale <= 20.0):
                scale = 1.0
            if abs(scale - 1.0) > 1e-6:
                dx = int(round(float(dx) * scale))
                dy = int(round(float(dy) * scale))
                rel_src = f"{rel_src}*{scale:g}"
        else:
            rel_src = "cursor_feedback"

        clip_now = _get_clip_cursor_rect()
        restore_clip_rel: Optional[tuple[int, int, int, int]] = None
        if clip_now is not None:
            l2, t2, r2, b2 = clip_now
            clip_w2 = int(r2 - l2)
            clip_h2 = int(b2 - t2)
            target_inside2 = (l2 <= int(click_x) <= r2) and (t2 <= int(click_y) <= b2)
            if (clip_w2 <= 2 and clip_h2 <= 2) or (not target_inside2):
                restore_clip_rel = clip_now
                _unclip_cursor()
                time.sleep(0.003)

        if cursor_movable:
            converge_result = _move_relative_converge_to_target(
                io=io,
                target_x=int(click_x),
                target_y=int(click_y),
                tolerance_px=int(move_tolerance_px),
                max_iters=max(12, int(relative_steps) * 2),
                step_sleep_s=float(relative_step_sleep_s),
            )
            move_result = {"mode": "cursor_feedback", **(converge_result or {})}
            moved_ok = bool((converge_result or {}).get("reached", False))
            if bool(require_move_within_tolerance) and not moved_ok and str((converge_result or {}).get("reason") or "") == "cursor_not_moving":
                cursor_movable = False

        if not cursor_movable:
            _move_relative_in_steps(dx=int(dx), dy=int(dy), steps=int(relative_steps), step_sleep_s=float(relative_step_sleep_s))
            move_result = {
                "mode": "window_center_fallback",
                "reached": True,
                "reason": str(rel_src),
                "iters": int(relative_steps),
                "cursor_before": {"x": int(cursor_before[0]), "y": int(cursor_before[1])},
                "cursor_after": None,
                "remaining": {"dx": 0, "dy": 0},
                "tolerance_px": int(move_tolerance_px),
                "relative_move": {"dx": int(dx), "dy": int(dy)},
                "grid_slot_index": int(grid_slot_index) if grid_slot_index is not None else None,
                "grid_visible_count": int(grid_visible_count),
            }
            moved_ok = True

        try:
            activate_window("CookingSimulator")
            time.sleep(0.05)
        except Exception:
            pass
        try:
            io.mouse_move_relative(1, 0)
            time.sleep(0.01)
            io.mouse_move_relative(-1, 0)
            time.sleep(0.01)
        except Exception:
            pass
        io.click("left")
        if restore_clip_rel is not None:
            _clip_cursor_rect(*restore_clip_rel)

    if not moved_ok:
        return {
            "success": False,
            "error": f"{debug_label}_click_not_converged",
            "target": {"x": int(click_x), "y": int(click_y), "src": str(click_src)},
            "move": move_result or {},
        }

    return {
        "success": True,
        "click": {"x": int(click_x), "y": int(click_y)},
        "move": move_result or {},
        "target": {"x": int(click_x), "y": int(click_y), "src": str(click_src)},
        "error": "",
    }


def click_submit_panel_from_ui_dump(
    *,
    window_title: str = "CookingSimulator",
    wait_ui_dump_timeout_s: float = 2.0,
    move_strategy: str = "relative_window_center",
    relative_scale: float = 1.0,
    relative_steps: int = 18,
    relative_step_sleep_s: float = 0.006,
    require_move_within_tolerance: bool = True,
    move_tolerance_px: int = 8,
    verbose: bool = True,
) -> dict[str, Any]:
    try:
        activate_window(window_title)
    except Exception:
        pass

    io = RawInputController()
    ud = userdata_root()
    dump_path = ud / "ui_targets_dump_latest.json"

    try:
        data = _trigger_ui_dump(io=io, dump_path=dump_path, wait_timeout_s=float(wait_ui_dump_timeout_s))
    except Exception as e:
        return {"success": False, "error": str(e)}
    game_screen_size = _get_ui_dump_screen_size(data)

    window_rect = _get_window_rect_win32(window_title) or None
    if window_rect is None:
        try:
            window_rect = get_window_rect(window_title)
        except Exception as e:
            return {"success": False, "error": f"get_window_rect_failed:{e}"}
    dpi_scale = _get_window_dpi_scale(window_title)

    target_path = _norm("Details/Work Stands/Checkout Stand/Tablet/Tablet GUI/BackGround/MainScreen/PanelDish")
    fallback_suffix = _norm("/MainScreen/PanelDish")
    candidates: list[dict[str, Any]] = []
    for bucket, values in data.items():
        if not isinstance(values, list):
            continue
        for it in values:
            if not isinstance(it, dict):
                continue
            path = _norm(str(it.get("path", "")))
            name = _norm(str(it.get("name", "")))
            if not bool(it.get("activeInHierarchy", False)):
                continue
            if path == target_path:
                candidates.append(it)
                continue
            if path.endswith(fallback_suffix) and name == _norm("PanelDish"):
                candidates.append(it)

    if not candidates:
        return {"success": False, "error": "submit_paneldish_not_found_in_ui_dump"}

    hit = candidates[0]
    panel_width = 0
    try:
        rect_tl = hit.get("rect_tl") if isinstance(hit.get("rect_tl"), dict) else {}
        panel_width = int(round(float(rect_tl.get("right")) - float(rect_tl.get("left"))))
    except Exception:
        panel_width = 0
    submit_step_px = max(12, min(32, int(round(panel_width * 0.25)))) if panel_width > 0 else 24
    max_seek_steps = max(24, int(round(min(float(window_rect.width) * 0.40, 1200.0) / float(submit_step_px))))
    conv = _hit_to_abs_click_xy(
        hit,
        window_left=int(window_rect.left),
        window_top=int(window_rect.top),
        window_width=int(window_rect.width),
        window_height=int(window_rect.height),
        dpi_scale=float(dpi_scale),
    )
    if conv is None:
        return {"success": False, "error": "submit_paneldish_coord_invalid", "hit": hit}
    click_x, click_y, _click_src = conv
    click_res = _slow_seek_submit_tooltip(
        io=io,
        dump_path=dump_path,
        click_x=int(click_x),
        click_y=int(click_y),
        step_px=int(submit_step_px),
        max_seek_steps=int(max_seek_steps),
        wait_ui_dump_timeout_s=float(wait_ui_dump_timeout_s),
        step_sleep_s=max(0.025, float(relative_step_sleep_s)),
        verbose=bool(verbose),
    )
    if not bool(click_res.get("success", False)):
        return click_res

    if bool(verbose):
        print(
            f"[submit] strategy=ui_dump_paneldish move_mode={((click_res.get('move') or {}).get('mode'))} "
            f"click=({int((click_res.get('click') or {}).get('x', 0))},{int((click_res.get('click') or {}).get('y', 0))}) "
            f"path={str(hit.get('path',''))!r} dpi_scale={float(dpi_scale):.3f} "
            f"game_screen={game_screen_size}"
        )
    return {
        "success": True,
        "click": click_res.get("click"),
        "move": click_res.get("move") or {},
        "hit": {"name": hit.get("name", ""), "path": hit.get("path", "")},
        "dpi_scale": float(dpi_scale),
        "game_screen_size": game_screen_size,
        "match": {
            "src": "ui_dump_paneldish",
            "path": hit.get("path", ""),
            "x": int((click_res.get("click") or {}).get("x", 0)),
            "y": int((click_res.get("click") or {}).get("y", 0)),
        },
        "best": 1.0,
        "error": "",
    }


def click_dish_entry_from_ui_dump(
    *,
    dish_name: str,
    window_title: str = "CookingSimulator",
    wait_ui_dump_timeout_s: float = 2.0,
    move_strategy: str = "auto",
    submit_anchor_move: Optional[dict[str, Any]] = None,
    relative_scale: float = 1.0,
    relative_steps: int = 18,
    relative_step_sleep_s: float = 0.006,
    require_move_within_tolerance: bool = True,
    move_tolerance_px: int = 8,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Click a dish entry using Alt+Y UI dump (`UserData/ui_targets_dump_latest.json`).
    """
    if not dish_name or not str(dish_name).strip():
        return {"success": False, "error": "missing_dish_name"}

    try:
        activate_window(window_title)
    except Exception:
        pass

    io = RawInputController()
    ud = userdata_root()
    dump_path = ud / "ui_targets_dump_latest.json"
    prev_mtime = _mtime(dump_path)

    io.key_down("alt")
    io.key_press("y")
    io.key_up("alt")

    deadline = time.time() + float(wait_ui_dump_timeout_s)
    while time.time() < deadline:
        m = _mtime(dump_path)
        if m and m != prev_mtime:
            break
        time.sleep(0.05)

    if not dump_path.exists():
        return {"success": False, "error": f"ui_targets_dump_missing:{str(dump_path)!r}"}

    try:
        data = _read_json(dump_path)
    except Exception as e:
        return {"success": False, "error": str(e)}
    game_screen_size = _get_ui_dump_screen_size(data)

    window_rect = _get_window_rect_win32(window_title) or None
    if window_rect is None:
        try:
            window_rect = get_window_rect(window_title)
        except Exception as e:
            return {"success": False, "error": f"get_window_rect_failed:{e}"}
    dpi_scale = _get_window_dpi_scale(window_title)

    raw_items: list[dict[str, Any]] = []
    for k in ("dishCandidates", "texts", "tmpTexts"):
        v = data.get(k)
        if isinstance(v, list):
            raw_items = [x for x in v if isinstance(x, dict)]
            if raw_items:
                break

    raw_items = _filter_checkout_stand_entries(raw_items)
    if not raw_items:
        return {"success": False, "error": "checkout_dish_list_not_found"}
    visible_dish_entries = _visible_checkout_dish_entries(raw_items)

    hits = _sorted_text_hits(
        raw_items,
        dish_name=str(dish_name),
        window_width=int(window_rect.width),
        window_height=int(window_rect.height),
    )
    if not hits:
        return {"success": False, "error": f"dish_text_not_found:{dish_name!r}", "dish_name": str(dish_name)}

    chosen_hit: Optional[dict[str, Any]] = None
    chosen_click: Optional[tuple[int, int, str]] = None
    for it in hits:
        conv = _hit_to_abs_click_xy(
            it,
            window_left=int(window_rect.left),
            window_top=int(window_rect.top),
            window_width=int(window_rect.width),
            window_height=int(window_rect.height),
            dpi_scale=float(dpi_scale),
        )
        if conv is not None:
            chosen_hit = it
            chosen_click = conv
            break

    if chosen_hit is None or chosen_click is None:
        preview: list[dict[str, Any]] = []
        for it in hits[:6]:
            rt = it.get("rect_tl") if isinstance(it.get("rect_tl"), dict) else {}
            cn = it.get("center_norm") if isinstance(it.get("center_norm"), dict) else {}
            preview.append(
                {
                    "active": bool(it.get("activeInHierarchy", True)),
                    "path": it.get("path", ""),
                    "text": it.get("text", ""),
                    "rect_tl_center": {"x": rt.get("centerX"), "y": rt.get("centerY")},
                    "center_norm": {"x": cn.get("x"), "y": cn.get("y")},
                }
            )
        return {
            "success": False,
            "error": "dish_click_coord_invalid",
            "dish_name": str(dish_name),
            "hit_preview": preview,
            "window_rect": {"left": window_rect.left, "top": window_rect.top, "width": window_rect.width, "height": window_rect.height},
            "dpi_scale": float(dpi_scale),
            "game_screen_size": game_screen_size,
            "visible_dish_count": int(len(visible_dish_entries)),
        }

    hit = chosen_hit
    grid_slot_index: Optional[int] = None
    if str(move_strategy).strip().lower() in ("checkout_grid_slot", "checkout_grid_locked_cursor"):
        for idx, candidate in enumerate(visible_dish_entries):
            if candidate is hit:
                grid_slot_index = int(idx)
                break
        if grid_slot_index is None:
            hit_path = str(hit.get("path", ""))
            hit_text = str(hit.get("text", ""))
            for idx, candidate in enumerate(visible_dish_entries):
                if str(candidate.get("path", "")) == hit_path and str(candidate.get("text", "")) == hit_text:
                    grid_slot_index = int(idx)
                    break
        if grid_slot_index is None:
            return {
                "success": False,
                "error": "checkout_grid_slot_not_resolved",
                "dish_name": str(dish_name),
                "visible_dish_count": int(len(visible_dish_entries)),
            }
    click_x, click_y, click_src = chosen_click
    click_res = _click_ui_dump_hit(
        io=io,
        hit=hit,
        click_xy=chosen_click,
        window_rect=window_rect,
        game_screen_size=game_screen_size,
        move_strategy=str(move_strategy),
        submit_anchor_move=submit_anchor_move,
        grid_slot_index=grid_slot_index,
        grid_visible_count=int(len(visible_dish_entries)),
        relative_scale=float(relative_scale),
        relative_steps=int(relative_steps),
        relative_step_sleep_s=float(relative_step_sleep_s),
        require_move_within_tolerance=bool(require_move_within_tolerance),
        move_tolerance_px=int(move_tolerance_px),
        verbose=bool(verbose),
        debug_label="dish_click",
    )
    if not bool(click_res.get("success", False)):
        out = dict(click_res)
        out["dish_name"] = str(dish_name)
        out["window_rect"] = {"left": window_rect.left, "top": window_rect.top, "width": window_rect.width, "height": window_rect.height}
        out["game_screen_size"] = game_screen_size
        out["visible_dish_count"] = int(len(visible_dish_entries))
        out["grid_slot_index"] = grid_slot_index
        return out
    print(
        f"[serve] dish_click: requested={str(dish_name)!r} hit_text={str(hit.get('text',''))!r} "
        f"path={str(hit.get('path',''))!r} click=({int(click_x)},{int(click_y)}) src={str(click_src)!r} "
        f"window_rect=({window_rect.left},{window_rect.top},{window_rect.width},{window_rect.height}) "
        f"dpi_scale={float(dpi_scale):.3f} game_screen={game_screen_size} "
        f"grid_slot_index={grid_slot_index} visible_dish_count={int(len(visible_dish_entries))}"
    )

    return {
        "success": True,
        "dish_name": str(dish_name),
        "dish_click": {
            "x": int(click_x),
            "y": int(click_y),
            "src": str(click_src),
            "strategy": str(((click_res.get("move") or {}).get("mode")) or ("absolute" if str(move_strategy) == "absolute" else "relative")),
        },
        "hit": {"text": hit.get("text", ""), "path": hit.get("path", "")},
        "window_rect": {"left": window_rect.left, "top": window_rect.top, "width": window_rect.width, "height": window_rect.height},
        "game_screen_size": game_screen_size,
        "visible_dish_count": int(len(visible_dish_entries)),
        "grid_slot_index": grid_slot_index,
        "error": "",
    }
