from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.skills._shared_paths import repo_root
from epm.vision.screen_capture import activate_window, capture_screenshot_mss, get_window_rect


@dataclass(frozen=True)
class MatchRect:
    x: int
    y: int
    w: int
    h: int


def _pil_to_bgr(img) -> np.ndarray:
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _imread_bgr(path: Path) -> Optional[np.ndarray]:
    """
    Unicode-safe BGR image loader for Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imread_gray(path: Path) -> Optional[np.ndarray]:
    """
    Unicode-safe image loader for Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def _match_best(screenshot_bgr: np.ndarray, template_path: Path) -> tuple[Optional[MatchRect], float]:
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


def _match_best_color(screenshot_bgr: np.ndarray, template_path: Path) -> float:
    """
    Color template match score (0..1). Useful when templates differ mainly by background color.
    """
    tpl = _imread_bgr(template_path)
    if tpl is None:
        return 0.0
    if screenshot_bgr.shape[0] < tpl.shape[0] or screenshot_bgr.shape[1] < tpl.shape[1]:
        return 0.0
    res = cv2.matchTemplate(screenshot_bgr, tpl, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)
    return float(max_val)


def _click_rect(io: RawInputController, rect, m: MatchRect) -> None:
    cx = rect.left + m.x + m.w // 2
    cy = rect.top + m.y + m.h // 2
    io.mouse_move_absolute(int(cx), int(cy))
    time.sleep(0.05)
    io.click("left")


def enable_steady_hands_perk(*, window_title: str = "CookingSimulator") -> dict:
    """
    GUI action: enable perk "Steady hands".

    Steps:
    1) Click computer/perks.png (enter perks page)
    2) If perks/steady hands-chosed.png is already visible -> done
    3) Else click perks/steady hands.png and verify it becomes chosen
    """

    base = repo_root() / "data" / "figure" / "computer"
    perks_btn = base / "perks.png"
    steady = base / "perks" / "steady hands.png"
    steady_chosen = base / "perks" / "steady hands-chosed.png"

    print("[perks] target=steady_hands")

    if not perks_btn.exists():
        return {"success": False, "error": f"missing_template:{str(perks_btn)!r}"}
    if not steady.exists() or not steady_chosen.exists():
        return {"success": False, "error": f"missing_template:{str(steady)!r} or {str(steady_chosen)!r}"}

    io = RawInputController()
    try:
        activate_window(window_title)
    except Exception as e:
        return {"success": False, "error": f"activate_window_failed:{e}"}

    rect = get_window_rect(window_title)

    # Step 1: open perks page
    img0 = capture_screenshot_mss(region=rect.to_mss_region())
    bgr0 = _pil_to_bgr(img0)
    perks_match, perks_best = _match_best(bgr0, perks_btn)
    if perks_match is None or perks_best < 0.78:
        return {"success": False, "error": f"perks_button_not_found (best={perks_best:.3f})"}
    _click_rect(io, rect, perks_match)
    time.sleep(0.45)

    # Step 2: already chosen?
    img1 = capture_screenshot_mss(region=rect.to_mss_region())
    bgr1 = _pil_to_bgr(img1)
    # IMPORTANT: chosen vs unchosen may only differ by background color.
    # Use color matching for state detection and require a margin to avoid false positives.
    chosen_best = _match_best_color(bgr1, steady_chosen)
    unchosen_best = _match_best_color(bgr1, steady)
    print(f"[perks] state_match color: chosen_best={chosen_best:.3f} unchosen_best={unchosen_best:.3f}")

    margin = 0.03
    if chosen_best >= 0.78 and (chosen_best - unchosen_best) >= margin:
        print("[perks] steady_hands already_enabled=True (no click needed)")
        return {"success": True, "already_enabled": True, "error": ""}
    if unchosen_best >= 0.78 and (unchosen_best - chosen_best) >= margin:
        # clearly not chosen -> proceed to click
        pass
    else:
        # ambiguous; proceed to click once (best-effort) rather than claiming enabled.
        print("[perks] state ambiguous -> will click steady_hands once to ensure enabled")

    # Step 3: click steady hands, then verify chosen
    print("[perks] steady_hands already_enabled=False -> clicking to enable")
    steady_match, steady_best = _match_best(bgr1, steady)
    if steady_match is None or steady_best < 0.78:
        return {"success": False, "error": f"steady_hands_not_found (best={steady_best:.3f})"}
    _click_rect(io, rect, steady_match)
    time.sleep(0.35)

    img2 = capture_screenshot_mss(region=rect.to_mss_region())
    bgr2 = _pil_to_bgr(img2)
    chosen_best2 = _match_best_color(bgr2, steady_chosen)
    unchosen_best2 = _match_best_color(bgr2, steady)
    print(f"[perks] verify color: chosen_best={chosen_best2:.3f} unchosen_best={unchosen_best2:.3f}")
    if not (chosen_best2 >= 0.78 and (chosen_best2 - unchosen_best2) >= margin):
        return {
            "success": False,
            "error": (
                f"steady_hands_enable_failed (chosen_best={chosen_best2:.3f} "
                f"unchosen_best={unchosen_best2:.3f} margin={margin:.2f})"
            ),
        }

    print("[perks] steady_hands enabled OK")
    return {"success": True, "already_enabled": False, "error": ""}
