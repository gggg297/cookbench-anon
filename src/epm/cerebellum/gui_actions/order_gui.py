from __future__ import annotations
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.clipboard_win32 import set_clipboard_text
from epm.cerebellum.skills._shared_paths import repo_root
from epm.vision.screen_capture import activate_window, capture_screenshot_mss, get_window_rect

SEARCH_FRAME_MIN_SCORE = 0.70
# Dish cards vary slightly across UI states (hover/selected/scale). Use a slightly lower
# threshold, then validate via a nearby `order-button` match.
DISH_MIN_SCORE = 0.82
ORDER_BUTTON_MIN_SCORE = 0.78
SECOND_ORDER_RESULT_DISHES = {
    "beef stroganoff over buttered fusilli",
    "pasta alla genovese",
    "caldo verde",
    "pumpkin soup",
    "ukrainian borscht",
    "ratatouille",
    "shakshuka",
    "chinese egg drop soup",
    "chicken pumpkin stew",
    "fruit salad",
}


def _resume_cmd() -> str:
    return "python epm/scripts/run_episode_dashboard.py --config epm/configs/pipelines/planner_executor.json --dish-id 1 --resume --refresh-memory"


@dataclass(frozen=True)
class MatchRect:
    x: int
    y: int
    w: int
    h: int


def _pil_to_bgr(img) -> np.ndarray:
    arr = np.array(img)
    # PIL Image is RGB; cv2 expects BGR.
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def _imread_gray(path: Path) -> Optional[np.ndarray]:
    """
    Unicode-safe image loader for Windows.

    `cv2.imread(str(path))` can fail when the path contains non-ASCII characters.
    Use `np.fromfile + cv2.imdecode` instead.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def _match_best(screenshot_bgr: np.ndarray, template_path: Path) -> tuple[Optional[MatchRect], float]:
    """
    Return (best_rect, best_score) for template matching.

    `best_score` is useful for debugging when we fail threshold checks.
    """
    if not template_path.exists():
        return None, 0.0
    tpl = _imread_gray(template_path)
    if tpl is None:
        return None, 0.0
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    x, y = int(max_loc[0]), int(max_loc[1])
    return MatchRect(x=x, y=y, w=int(tpl.shape[1]), h=int(tpl.shape[0])), float(max_val)


def _match_best_multi(screenshot_bgr: np.ndarray, template_paths: list[Path]) -> tuple[Optional[MatchRect], float, Optional[Path]]:
    best_rect: Optional[MatchRect] = None
    best_score = 0.0
    best_path: Optional[Path] = None
    for p in template_paths:
        rect, score = _match_best(screenshot_bgr, p)
        if score >= best_score:
            best_rect = rect
            best_score = score
            best_path = p
    return best_rect, float(best_score), best_path


def _find_first(screenshot_bgr: np.ndarray, template_path: Path, *, threshold: float) -> Optional[MatchRect]:
    rect, best = _match_best(screenshot_bgr, template_path)
    if rect is None:
        return None
    return rect if best >= float(threshold) else None


def _match_best_in_roi(screenshot_gray: np.ndarray, template_gray: np.ndarray, *, roi: MatchRect) -> float:
    """
    Match template within a ROI in the screenshot (both must be grayscale).
    Returns best score (0..1).
    """
    h, w = screenshot_gray.shape[:2]
    x1 = max(0, int(roi.x))
    y1 = max(0, int(roi.y))
    x2 = min(w, int(roi.x + roi.w))
    y2 = min(h, int(roi.y + roi.h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = screenshot_gray[y1:y2, x1:x2]
    if sub.shape[0] < template_gray.shape[0] or sub.shape[1] < template_gray.shape[1]:
        return 0.0
    res = cv2.matchTemplate(sub, template_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)
    return float(max_val)


def _find_all_template_matches(
    screenshot_bgr: np.ndarray,
    template_path: Path,
    *,
    threshold: float,
) -> list[tuple[MatchRect, float]]:
    """
    Return de-duplicated template matches sorted top-to-bottom, then left-to-right.
    """
    if not template_path.exists():
        return []
    tpl = _imread_gray(template_path)
    if tpl is None:
        return []
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= float(threshold))
    raw: list[tuple[MatchRect, float]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        score = float(res[y, x])
        raw.append((MatchRect(x=int(x), y=int(y), w=int(tpl.shape[1]), h=int(tpl.shape[0])), score))
    raw.sort(key=lambda item: item[1], reverse=True)

    deduped: list[tuple[MatchRect, float]] = []
    min_dx = max(8, int(tpl.shape[1] * 0.5))
    min_dy = max(8, int(tpl.shape[0] * 0.5))
    for rect, score in raw:
        keep = True
        for kept, _kept_score in deduped:
            if abs(rect.x - kept.x) < min_dx and abs(rect.y - kept.y) < min_dy:
                keep = False
                break
        if keep:
            deduped.append((rect, score))
    deduped.sort(key=lambda item: (item[0].y, item[0].x))
    return deduped


def _click_rect(io: RawInputController, rect, m: MatchRect) -> None:
    cx = rect.left + m.x + m.w // 2
    cy = rect.top + m.y + m.h // 2
    io.mouse_move_absolute(int(cx), int(cy))
    time.sleep(0.05)
    io.click("left")


def _should_click_second_order_result(dish_name: str) -> bool:
    return str(dish_name or "").strip().lower() in SECOND_ORDER_RESULT_DISHES


def _save_debug_snap(*, stage: str, bgr: np.ndarray) -> str:
    out_dir = (repo_root() / "runs" / "_debug" / "order_gui").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    path = out_dir / f"{ts}_{stage}.png"
    try:
        ok, enc = cv2.imencode(".png", bgr)
        if bool(ok):
            enc.tofile(str(path))
            return str(path)
    except Exception:
        pass
    return ""


def _apply_decorations(*, io: RawInputController, rect, window_title: str) -> dict:
    """
    Best-effort decoration flow:
    - Click decorations
    - Click events
    - Click None
    - If in-use -> no-op; if install -> click and wait 3s
    """

    base = repo_root() / "data" / "figure" / "computer"
    deco_btn = base / "decorations.png"
    deco_dir = base / "decorations"
    events_btn = deco_dir / "events.png"
    none_btn = deco_dir / "None.png"
    # UI state can change text/button style (selected/hovered). Keep grayscale matching,
    # but try a small set of template variants before declaring failure.
    events_candidates = [
        events_btn,
        deco_dir / "events_selected.png",
        deco_dir / "events-active.png",
    ]
    none_candidates = [
        none_btn,
        deco_dir / "None_selected.png",
        deco_dir / "None-active.png",
        deco_dir / "none_selected.png",
    ]
    in_use = deco_dir / "in-use.png"
    install = deco_dir / "install.png"

    if not deco_btn.exists():
        return {"success": False, "error": f"missing_template:{str(deco_btn)!r}"}
    if not events_btn.exists() or not none_btn.exists():
        return {"success": False, "error": f"missing_template:{str(events_btn)!r} or {str(none_btn)!r}"}
    if not in_use.exists() or not install.exists():
        return {"success": False, "error": f"missing_template:{str(in_use)!r} or {str(install)!r}"}

    try:
        activate_window(window_title)
    except Exception:
        pass

    def _snap() -> np.ndarray:
        img = capture_screenshot_mss(region=rect.to_mss_region())
        return _pil_to_bgr(img)

    def _cap_meta(bgr: np.ndarray) -> str:
        h, w = bgr.shape[:2]
        return (
            f"rect=({int(rect.left)},{int(rect.top)},{int(rect.width)},{int(rect.height)}) "
            f"snap=({int(w)}x{int(h)})"
        )

    bgr = _snap()
    match, best = _match_best(bgr, deco_btn)
    if match is None or best < 0.78:
        dbg = _save_debug_snap(stage="decorations_button_not_found", bgr=bgr)
        return {
            "success": False,
            "error": f"decorations_button_not_found (best={best:.3f}) {_cap_meta(bgr)} debug={dbg}",
        }
    _click_rect(io, rect, match)
    time.sleep(0.35)

    bgr = _snap()
    match, best, used_events_tpl = _match_best_multi(bgr, events_candidates)
    if match is None or best < 0.78:
        dbg = _save_debug_snap(stage="decorations_events_not_found", bgr=bgr)
        used = used_events_tpl.name if used_events_tpl is not None else "none"
        return {
            "success": False,
            "error": f"decorations_events_not_found (best={best:.3f} tpl={used}) {_cap_meta(bgr)} debug={dbg}",
        }
    _click_rect(io, rect, match)
    time.sleep(0.30)

    bgr = _snap()
    match, best, used_none_tpl = _match_best_multi(bgr, none_candidates)
    if match is None or best < 0.72:
        # Fallback for UI theme/scale drift: click the first item row in the right list.
        # In the Events tab this row is expected to be "None".
        fx = int(rect.left + rect.width * 0.86)
        fy = int(rect.top + rect.height * 0.27)
        io.mouse_move_absolute(fx, fy)
        time.sleep(0.05)
        io.click("left")
        time.sleep(0.30)

        bgr_fb = _snap()
        in_use_fb, in_use_best_fb = _match_best(bgr_fb, in_use)
        install_fb, install_best_fb = _match_best(bgr_fb, install)
        if in_use_fb is not None and in_use_best_fb >= 0.76 and in_use_best_fb >= install_best_fb:
            return {"success": True, "installed": False, "error": "", "fallback": "none_row_geometry"}
        if install_fb is not None and install_best_fb >= 0.76:
            _click_rect(io, rect, install_fb)
            time.sleep(3.0)
            return {"success": True, "installed": True, "error": "", "fallback": "none_row_geometry"}

        dbg = _save_debug_snap(stage="decorations_none_not_found", bgr=bgr_fb)
        used = used_none_tpl.name if used_none_tpl is not None else "none"
        return {
            "success": False,
            "error": (
                "decorations_none_not_found "
                f"(best={best:.3f} tpl={used} min=0.72 fallback=none_row_geometry "
                f"in_use_best={in_use_best_fb:.3f} install_best={install_best_fb:.3f}) "
                f"{_cap_meta(bgr_fb)} debug={dbg}"
            ),
        }
    _click_rect(io, rect, match)
    time.sleep(0.30)

    bgr = _snap()
    in_use_match, in_use_best = _match_best(bgr, in_use)
    install_match, install_best = _match_best(bgr, install)

    if in_use_match is not None and in_use_best >= 0.78 and in_use_best >= install_best:
        print("[decorations] already in use")
        return {"success": True, "installed": False, "error": ""}
    if install_match is not None and install_best >= 0.78:
        print("[decorations] installing")
        _click_rect(io, rect, install_match)
        time.sleep(3.0)
        return {"success": True, "installed": True, "error": ""}

    return {
        "success": False,
        "error": (
            "decorations_state_unknown "
            f"(in_use_best={in_use_best:.3f} install_best={install_best:.3f})"
        ),
        "debug_screenshot": _save_debug_snap(stage="decorations_state_unknown", bgr=bgr),
    }


def _try_focus_and_clear_search_box(
    *,
    io: RawInputController,
    rect,
    bgr: np.ndarray,
    search_rect: MatchRect,
    search_button_tpl: Path,
) -> None:
    """
    Best-effort: focus the search box and clear it (Ctrl+A + Backspace).

    We do:
    - click magnifier icon (if found) to bring the UI into "search" state
    - click inside the input area
    - clear
    """
    if search_button_tpl.exists():
        sb = _find_first(bgr, search_button_tpl, threshold=0.84)
        if sb is not None:
            sx = rect.left + sb.x + sb.w // 2
            sy = rect.top + sb.y + sb.h // 2
            io.mouse_move_absolute(int(sx), int(sy))
            time.sleep(0.05)
            io.click("left")
            time.sleep(0.10)

    # Click inside the text input area (frame center may land on an icon/button).
    cx = rect.left + search_rect.x + int(search_rect.w * 0.22)
    cy = rect.top + search_rect.y + int(search_rect.h * 0.50)
    io.mouse_move_absolute(int(cx), int(cy))
    time.sleep(0.05)
    io.click("left")
    time.sleep(0.15)

    io.key_down("ctrl")
    io.key_press("a")
    io.key_up("ctrl")
    time.sleep(0.05)
    io.key_press("backspace")
    time.sleep(0.03)
    io.key_press("delete")
    time.sleep(0.03)
    # Extra backspaces (some UIs swallow the first one if focus just changed)
    io.key_press("backspace")
    io.key_press("backspace")
    time.sleep(0.03)


def _clear_text_box(io: RawInputController) -> None:
    io.key_down("ctrl")
    io.key_press("a")
    io.key_up("ctrl")
    time.sleep(0.05)
    io.key_press("backspace")
    time.sleep(0.03)
    io.key_press("delete")
    time.sleep(0.03)
    io.key_press("backspace")
    io.key_press("backspace")


def _resolve_order_assets_dir() -> Path:
    # <repo>/epm/data/figure/computer/order
    return repo_root() / "data" / "figure" / "computer" / "order"

def _resolve_computer_assets_dir() -> Path:
    # <repo>/epm/data/figure/computer
    return repo_root() / "data" / "figure" / "computer"


def _resolve_dish_template(order_dir: Path, dish_name: str) -> Optional[Path]:
    # Exact filename first.
    p = order_dir / f"{dish_name}.png"
    if p.exists():
        return p
    # Fallback: case-insensitive stem match.
    want = (dish_name or "").strip().lower()
    if not want:
        return None
    for cand in order_dir.glob("*.png"):
        if cand.stem.strip().lower() == want:
            return cand
    return None


def detect_computer_ui_state(*, window_title: str = "CookingSimulator") -> dict:
    """
    Lightweight check for whether the in-game computer UI is already open.

    We intentionally reuse the same templates as `order_dish_by_search`:
    - `order-manager-button` means the computer main UI is visible
    - `search-button` / `search-frame` mean the order page is visible
    """
    computer_dir = _resolve_computer_assets_dir()
    order_dir = _resolve_order_assets_dir()
    order_manager_button = computer_dir / "order-manager-button.png"
    search_button = order_dir / "search-button.png"
    search_frame = order_dir / "search-frame.png"
    chosen_search_frame = order_dir / "chosen-search-frame.png"

    try:
        activate_window(window_title)
        rect = get_window_rect(window_title)
        img = capture_screenshot_mss(region=rect.to_mss_region())
        bgr = _pil_to_bgr(img)
    except Exception as e:
        return {"is_open": False, "error": f"capture_failed:{e}"}

    om_rect, om_best = _match_best(bgr, order_manager_button)
    sb_rect, sb_best = _match_best(bgr, search_button)
    sf_rect, sf_best, sf_tpl = _match_best_multi(bgr, [chosen_search_frame, search_frame])

    matched_by = ""
    matched_score = 0.0
    if om_rect is not None and om_best >= 0.80:
        matched_by = "order_manager_button"
        matched_score = float(om_best)
    elif sb_rect is not None and sb_best >= 0.78:
        matched_by = "search_button"
        matched_score = float(sb_best)
    elif sf_rect is not None and sf_best >= SEARCH_FRAME_MIN_SCORE:
        matched_by = f"search_frame:{sf_tpl.name if sf_tpl is not None else 'unknown'}"
        matched_score = float(sf_best)

    return {
        "is_open": bool(matched_by),
        "matched_by": matched_by,
        "matched_score": matched_score,
        "scores": {
            "order_manager_button": float(om_best),
            "search_button": float(sb_best),
            "search_frame": float(sf_best),
        },
    }


def order_dish_by_search(*, dish_name: str, window_title: str = "CookingSimulator") -> dict:
    """
    GUI action: search dish name on the in-game computer UI and click the Order button.

    Requires:
    - In-game computer/menu screen is open
    - Templates exist under `epm/data/figure/computer/order/`
    """

    # Preflight: enable "Steady hands" perk automatically (best-effort).
    steady_hands: dict = {"success": None, "already_enabled": None, "error": ""}
    try:
        from epm.cerebellum.gui_actions.perks_gui import enable_steady_hands_perk

        steady_hands = enable_steady_hands_perk(window_title=window_title)
        if not bool(steady_hands.get("success", False)):
            print(f"[gui_order] warning: enable_steady_hands failed: {steady_hands.get('error')}")
    except Exception as e:
        steady_hands = {"success": False, "already_enabled": None, "error": str(e)}
        print(f"[gui_order] warning: enable_steady_hands crashed: {e}")

    order_dir = _resolve_order_assets_dir()
    computer_dir = _resolve_computer_assets_dir()
    order_manager_button = computer_dir / "order-manager-button.png"
    search_frame = order_dir / "search-frame.png"
    chosen_search_frame = order_dir / "chosen-search-frame.png"
    search_button = order_dir / "search-button.png"
    order_button = order_dir / "order-button.png"
    dish_tpl = _resolve_dish_template(order_dir, dish_name)
    if dish_tpl is None:
        return {
            "success": False,
            "error": f"dish_template_not_found:{dish_name!r} in {str(order_dir)!r}",
            "steady_hands": steady_hands,
        }
    if not search_frame.exists() or not order_button.exists():
        return {"success": False, "error": f"missing_order_ui_templates in {str(order_dir)!r}", "steady_hands": steady_hands}

    io = RawInputController()
    try:
        activate_window(window_title)
    except Exception as e:
        return {"success": False, "error": f"activate_window_failed:{e}", "steady_hands": steady_hands}

    rect = get_window_rect(window_title)

    # Preflight: decoration flow (strict fail if not matched).
    decorations: dict = {"success": None, "installed": None, "error": ""}
    try:
        decorations = _apply_decorations(io=io, rect=rect, window_title=window_title)
        if not bool(decorations.get("success", False)):
            return {
                "success": False,
                "error": f"decorations_preflight_failed:{decorations.get('error')}",
                "steady_hands": steady_hands,
                "decorations": decorations,
                "resume_cmd": _resume_cmd(),
            }
    except Exception as e:
        decorations = {"success": False, "installed": None, "error": str(e)}
        return {
            "success": False,
            "error": f"decorations_preflight_crashed:{e}",
            "steady_hands": steady_hands,
            "decorations": decorations,
            "resume_cmd": _resume_cmd(),
        }

    img = capture_screenshot_mss(region=rect.to_mss_region())
    bgr = _pil_to_bgr(img)

    # Step 0: enter the "Order manager" page (best-effort).
    # User preference: always attempt this first.
    if order_manager_button.exists():
        om, om_best = _match_best(bgr, order_manager_button)
        if om is not None and om_best >= 0.80:
            ox = rect.left + om.x + om.w // 2
            oy = rect.top + om.y + om.h // 2
            io.mouse_move_absolute(int(ox), int(oy))
            time.sleep(0.05)
            io.click("left")
            time.sleep(0.35)
            img = capture_screenshot_mss(region=rect.to_mss_region())
            bgr = _pil_to_bgr(img)

    # IMPORTANT: click the magnifier/search icon first.
    # On some UI layouts the search box frame only appears after clicking it.
    sb = None
    sb_best = 0.0
    if search_button.exists():
        sb, sb_best = _match_best(bgr, search_button)
        if sb is not None and sb_best >= 0.78:
            sx = rect.left + sb.x + sb.w // 2
            sy = rect.top + sb.y + sb.h // 2
            io.mouse_move_absolute(int(sx), int(sy))
            time.sleep(0.05)
            io.click("left")
            time.sleep(0.25)
            img = capture_screenshot_mss(region=rect.to_mss_region())
            bgr = _pil_to_bgr(img)
        else:
            # If the magnifier is not confidently found, the search box may still contain text
            # and later frame matching can fail. Provide explicit feedback.
            return {
                "success": False,
                "error": (
                    "search_button_not_found_or_low_confidence "
                    f"(best={sb_best:.3f} min=0.78) - the search box may still contain text; "
                    "click the magnifier and clear the input manually once."
                ),
                "steady_hands": steady_hands,
            }

    search, search_best, used_tpl = _match_best_multi(bgr, [chosen_search_frame, search_frame])
    if search is None or search_best < float(SEARCH_FRAME_MIN_SCORE):
        # Fallback: if the magnifier is found, click a fixed offset to the right to focus the input.
        # This allows us to proceed even when `search-frame` templates drift across UI versions.
        if sb is not None and sb_best >= 0.78:
            base_x = rect.left + sb.x + sb.w
            base_y = rect.top + sb.y + sb.h // 2
            for dx in (80, 120, 160, 200):
                io.mouse_move_absolute(int(base_x + dx), int(base_y))
                time.sleep(0.03)
                io.click("left")
                time.sleep(0.08)
                _clear_text_box(io)
                break
            print(
                "[gui_order] search-frame template low; used magnifier-offset fallback "
                f"(frame_best={search_best:.3f})"
            )
        else:
            tpl_name = used_tpl.name if used_tpl is not None else "none"
            return {
                "success": False,
                "error": (
                    "search_box_not_found (template match failed, "
                    f"best={search_best:.3f} min={SEARCH_FRAME_MIN_SCORE:.2f} tpl={tpl_name} "
                    f"search_btn_best={sb_best:.3f})"
                ),
                "steady_hands": steady_hands,
            }
    else:
        # Clear once before search (and again after ordering).
        _try_focus_and_clear_search_box(io=io, rect=rect, bgr=bgr, search_rect=search, search_button_tpl=search_button)
    # At this point we should have focus on the search input and it's cleared (best-effort).

    # Type/paste and search.
    # In some UI states, unicode typing may be flaky if focus is slightly off.
    # We do: type_text first, then enforce via clipboard paste.
    print(f"[gui_order] input dish_name={dish_name!r}")
    io.type_text(dish_name)
    time.sleep(0.05)
    paste_err = ""
    try:
        set_clipboard_text(dish_name)
        io.key_down("ctrl")
        io.key_press("a")
        io.key_press("v")
        io.key_up("ctrl")
        time.sleep(0.05)
        print("[gui_order] pasted via clipboard (Ctrl+V)")
    except Exception as e:
        paste_err = str(e)
        print(f"[gui_order] clipboard paste failed: {paste_err}")
    io.key_press("enter")
    time.sleep(0.35)

    # Some UI variants require clicking the search icon/button.
    if search_button.exists():
        img_btn = capture_screenshot_mss(region=rect.to_mss_region())
        bgr_btn = _pil_to_bgr(img_btn)
        sb = _find_first(bgr_btn, search_button, threshold=0.86)
        if sb is not None:
            sx = rect.left + sb.x + sb.w // 2
            sy = rect.top + sb.y + sb.h // 2
            io.mouse_move_absolute(int(sx), int(sy))
            time.sleep(0.05)
            io.click("left")

    time.sleep(1.2)

    # After search: find dish card then click its order button.
    img2 = capture_screenshot_mss(region=rect.to_mss_region())
    bgr2 = _pil_to_bgr(img2)

    if _should_click_second_order_result(dish_name):
        print(f"[gui_order] ambiguous dish special-case hit: {dish_name!r}; clicking the second Order result")
        order_matches = _find_all_template_matches(
            bgr2,
            order_button,
            threshold=float(ORDER_BUTTON_MIN_SCORE),
        )
        if len(order_matches) < 2:
            return {
                "success": False,
                "error": (
                    f"second_order_button_not_found_for_ambiguous_dish:{dish_name!r} "
                    f"(matches={len(order_matches)} min_required=2)"
                ),
                "steady_hands": steady_hands,
            }
        second_button, second_score = order_matches[1]
        bx = rect.left + second_button.x + second_button.w // 2
        by = rect.top + second_button.y + second_button.h // 2
        io.mouse_move_absolute(int(bx), int(by))
        time.sleep(0.08)
        io.click("left")

        try:
            img3 = capture_screenshot_mss(region=rect.to_mss_region())
            bgr3 = _pil_to_bgr(img3)
            search3, _best3, _used3 = _match_best_multi(bgr3, [chosen_search_frame, search_frame])
            if search3 is not None:
                _try_focus_and_clear_search_box(io=io, rect=rect, bgr=bgr3, search_rect=search3, search_button_tpl=search_button)
        except Exception:
            pass

        return {
            "success": True,
            "dish_name": str(dish_name),
            "error": "",
            "steady_hands": steady_hands,
            "order_click_strategy": "second_order_button_for_ambiguous_dish",
            "order_button_match_score": float(second_score),
            "order_button_match_count": int(len(order_matches)),
        }

    dish, dish_best = _match_best(bgr2, dish_tpl)
    if dish is None or dish_best < float(DISH_MIN_SCORE):
        return {
            "success": False,
            "error": (
                f"dish_not_found_after_search:{dish_name!r} "
                f"(best={dish_best:.3f} min={DISH_MIN_SCORE:.2f} paste_err={paste_err!r})"
            ),
            "steady_hands": steady_hands,
        }

    btn_tpl = _imread_gray(order_button)
    if btn_tpl is None:
        return {"success": False, "error": "order_button_template_unreadable", "steady_hands": steady_hands}
    btn_w, btn_h = int(btn_tpl.shape[1]), int(btn_tpl.shape[0])

    # Heuristic from the legacy script: order button appears at dish card's lower-right edge.
    order_x = dish.x + dish.w - btn_w
    order_y = dish.y + dish.h

    # Validate: the order button should be visible near the predicted location.
    # This reduces risk when lowering dish threshold.
    gray2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2GRAY)
    roi = MatchRect(
        x=int(order_x) - 12,
        y=int(order_y) - 12,
        w=int(btn_w) + 24,
        h=int(btn_h) + 24,
    )
    btn_best = _match_best_in_roi(gray2, btn_tpl, roi=roi)
    if btn_best < float(ORDER_BUTTON_MIN_SCORE):
        return {
            "success": False,
            "error": (
                f"order_button_not_found_near_dish (dish_best={dish_best:.3f} "
                f"btn_best={btn_best:.3f} min={ORDER_BUTTON_MIN_SCORE:.2f})"
            ),
            "steady_hands": steady_hands,
        }

    bx = rect.left + int(order_x) + btn_w // 2
    by = rect.top + int(order_y) + btn_h // 2

    io.mouse_move_absolute(bx, by)
    time.sleep(0.08)
    io.click("left")

    # Clear again after ordering (double insurance for next query).
    try:
        img3 = capture_screenshot_mss(region=rect.to_mss_region())
        bgr3 = _pil_to_bgr(img3)
        search3, _best3, _used3 = _match_best_multi(bgr3, [chosen_search_frame, search_frame])
        if search3 is not None:
            _try_focus_and_clear_search_box(io=io, rect=rect, bgr=bgr3, search_rect=search3, search_button_tpl=search_button)
    except Exception:
        pass

    return {"success": True, "dish_name": str(dish_name), "error": "", "steady_hands": steady_hands}
