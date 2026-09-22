from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import ctypes

from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.skills._shared_paths import repo_root
from epm.vision.screen_capture import activate_window, capture_screenshot_mss, get_window_rect


def _pil_to_bgr(img) -> np.ndarray:
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


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


def _match_best(screenshot_bgr: np.ndarray, template_path: Path) -> tuple[Optional[tuple[int, int, int, int]], float]:
    if not template_path.exists():
        return None, 0.0
    tpl = _imread_gray(template_path)
    if tpl is None:
        return None, 0.0
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    x, y = int(max_loc[0]), int(max_loc[1])
    return (x, y, int(tpl.shape[1]), int(tpl.shape[0])), float(max_val)


def _is_template_visible(
    *,
    rect,
    template: Path,
    threshold: float = 0.78,
) -> tuple[bool, float, Optional[tuple[int, int, int, int]]]:
    img = capture_screenshot_mss(region=rect.to_mss_region())
    bgr = _pil_to_bgr(img)
    match, best = _match_best(bgr, template)
    visible = match is not None and best >= float(threshold)
    return bool(visible), float(best), match


def _click_rect(io: RawInputController, rect, m: tuple[int, int, int, int]) -> None:
    x, y, w, h = m
    cx = rect.left + x + w // 2
    cy = rect.top + y + h // 2
    io.mouse_move_absolute(int(cx), int(cy))
    time.sleep(0.05)
    io.click("left")


def _find_and_click(
    *,
    io: RawInputController,
    rect,
    template: Path,
    name: str,
    threshold: float = 0.78,
) -> Optional[str]:
    img = capture_screenshot_mss(region=rect.to_mss_region())
    bgr = _pil_to_bgr(img)
    match, best = _match_best(bgr, template)
    if match is None or best < threshold:
        return f"{name}_not_found (best={best:.3f})"
    _click_rect(io, rect, match)
    return None


def _find_only(
    *,
    rect,
    template: Path,
    name: str,
    threshold: float = 0.78,
) -> Optional[str]:
    img = capture_screenshot_mss(region=rect.to_mss_region())
    bgr = _pil_to_bgr(img)
    match, best = _match_best(bgr, template)
    if match is None or best < threshold:
        return f"{name}_not_found (best={best:.3f})"
    return None


def _wait_for_template(
    *,
    rect,
    template: Path,
    name: str,
    threshold: float = 0.78,
    timeout_s: float = 60.0,
    poll_s: float = 0.5,
) -> Optional[str]:
    deadline = time.time() + max(0.1, float(timeout_s))
    best_seen = 0.0
    while time.time() < deadline:
        img = capture_screenshot_mss(region=rect.to_mss_region())
        bgr = _pil_to_bgr(img)
        match, best = _match_best(bgr, template)
        best_seen = max(best_seen, float(best))
        if match is not None and best >= threshold:
            print(f"[restart_env] ready stage={name} best={best:.3f}")
            return None
        time.sleep(max(0.05, float(poll_s)))
    return f"{name}_timeout_not_found (best={best_seen:.3f} timeout_s={float(timeout_s):.1f})"


def _wait_for_template_hidden(
    *,
    rect,
    template: Path,
    name: str,
    threshold: float = 0.78,
    timeout_s: float = 60.0,
    poll_s: float = 0.5,
) -> Optional[str]:
    deadline = time.time() + max(0.1, float(timeout_s))
    best_seen = 0.0
    while time.time() < deadline:
        visible, best, _match = _is_template_visible(rect=rect, template=template, threshold=threshold)
        best_seen = max(best_seen, float(best))
        if not visible:
            print(f"[restart_env] verified_hidden stage={name} last_best={best:.3f}")
            return None
        time.sleep(max(0.05, float(poll_s)))
    return f"{name}_timeout_not_hidden (best={best_seen:.3f} timeout_s={float(timeout_s):.1f})"


def _wait_and_click(
    *,
    io: RawInputController,
    rect,
    template: Path,
    name: str,
    threshold: float = 0.78,
    timeout_s: float = 60.0,
    poll_s: float = 0.5,
) -> Optional[str]:
    deadline = time.time() + max(0.1, float(timeout_s))
    best_seen = 0.0
    while time.time() < deadline:
        img = capture_screenshot_mss(region=rect.to_mss_region())
        bgr = _pil_to_bgr(img)
        match, best = _match_best(bgr, template)
        best_seen = max(best_seen, float(best))
        if match is not None and best >= threshold:
            print(f"[restart_env] click stage={name} best={best:.3f}")
            _click_rect(io, rect, match)
            return None
        time.sleep(max(0.05, float(poll_s)))
    return f"{name}_timeout_not_found (best={best_seen:.3f} timeout_s={float(timeout_s):.1f})"


def _click_with_verification(
    *,
    io: RawInputController,
    rect,
    template: Path,
    name: str,
    threshold: float = 0.78,
    timeout_s: float = 60.0,
    poll_s: float = 0.5,
    verify_visible_template: Optional[Path] = None,
    verify_visible_name: str = "",
    verify_visible_threshold: float = 0.78,
    verify_hidden_template: Optional[Path] = None,
    verify_hidden_name: str = "",
    verify_hidden_threshold: float = 0.78,
    post_click: Optional[Callable[[], None]] = None,
) -> Optional[str]:
    deadline = time.time() + max(0.1, float(timeout_s))
    best_seen = 0.0
    while time.time() < deadline:
        visible, best, match = _is_template_visible(rect=rect, template=template, threshold=threshold)
        best_seen = max(best_seen, float(best))
        if visible and match is not None:
            print(f"[restart_env] click stage={name} best={best:.3f}")
            _click_rect(io, rect, match)
            time.sleep(0.15)
            if post_click is not None:
                post_click()
            remaining = max(0.5, deadline - time.time())
            verify_timeout = min(8.0, remaining)
            if verify_visible_template is not None:
                err = _wait_for_template(
                    rect=rect,
                    template=verify_visible_template,
                    name=(verify_visible_name or f"{name}_verify_visible"),
                    threshold=float(verify_visible_threshold),
                    timeout_s=verify_timeout,
                    poll_s=poll_s,
                )
                if err is None:
                    return None
                print(f"[restart_env] verification_retry stage={name} reason={err}")
            elif verify_hidden_template is not None:
                err = _wait_for_template_hidden(
                    rect=rect,
                    template=verify_hidden_template,
                    name=(verify_hidden_name or f"{name}_verify_hidden"),
                    threshold=float(verify_hidden_threshold),
                    timeout_s=verify_timeout,
                    poll_s=poll_s,
                )
                if err is None:
                    return None
                print(f"[restart_env] verification_retry stage={name} reason={err}")
            else:
                return None
        time.sleep(max(0.05, float(poll_s)))
    return f"{name}_timeout_click_or_verify_failed (best={best_seen:.3f} timeout_s={float(timeout_s):.1f})"


def _send_key_to_window(*, window_title: str, vk_code: int) -> bool:
    hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
    if hwnd == 0:
        return False
    WM_KEYDOWN = 0x0100
    WM_KEYUP = 0x0101
    ctypes.windll.user32.PostMessageW(hwnd, WM_KEYDOWN, vk_code, 0)
    time.sleep(0.05)
    ctypes.windll.user32.PostMessageW(hwnd, WM_KEYUP, vk_code, 0)
    return True


def restart_environment(*, window_title: str = "CookingSimulator") -> dict:
    """
    Restart a new environment (assumes we are on evaluation page).

    Flow:
    - click ok.png
    - press Escape
    - click back-to-menu.png
    - press Space, wait 3s
    - click new-game.png
    - click career-mode.png
    - scroll down 20 times
    - click sandbox-mod.png
    - verify classic-sandbox.png exists (success if visible)
    - click sandbox-play.png
    - wait 10s, success
    """

    base = repo_root() / "data" / "figure" / "computer"
    new_dir = base / "newGame"

    ok_btn = base / "ok.png"
    back_to_menu = new_dir / "back-to-menu.png"
    new_game = new_dir / "new-game.png"
    career_mode = new_dir / "career-mode.png"
    sandbox_mod = new_dir / "sandbox-mod.png"
    classic_sandbox = new_dir / "classic-sandbox.png"
    sandbox_play = new_dir / "sandbox-play.png"

    for p in (ok_btn, back_to_menu, new_game, career_mode, sandbox_mod, classic_sandbox, sandbox_play):
        if not p.exists():
            return {"success": False, "error": f"missing_template:{str(p)!r}"}

    io = RawInputController()
    try:
        activate_window(window_title)
    except Exception as e:
        return {"success": False, "error": f"activate_window_failed:{e}"}

    rect = get_window_rect(window_title)

    print("[restart_env] begin")

    err = _click_with_verification(
        io=io,
        rect=rect,
        template=ok_btn,
        name="ok",
        threshold=0.78,
        timeout_s=60.0,
        poll_s=0.5,
        verify_hidden_template=ok_btn,
        verify_hidden_name="ok_hidden",
        verify_hidden_threshold=0.78,
    )
    if err:
        return {"success": False, "error": err}
    time.sleep(0.25)

    if not _send_key_to_window(window_title=window_title, vk_code=0x1B):
        io.key_press("escape")
    time.sleep(0.35)

    def _confirm_back_to_menu() -> None:
        io.key_press("space")
        time.sleep(3.0)

    err = _click_with_verification(
        io=io,
        rect=rect,
        template=back_to_menu,
        name="back_to_menu",
        timeout_s=60.0,
        poll_s=0.5,
        verify_visible_template=new_game,
        verify_visible_name="new_game_after_back_to_menu",
        verify_visible_threshold=0.78,
        post_click=_confirm_back_to_menu,
    )
    if err:
        return {"success": False, "error": err}
    time.sleep(0.25)

    err = _click_with_verification(
        io=io,
        rect=rect,
        template=new_game,
        name="new_game",
        timeout_s=60.0,
        poll_s=0.5,
        verify_visible_template=career_mode,
        verify_visible_name="career_mode_after_new_game",
        verify_visible_threshold=0.78,
    )
    if err:
        return {"success": False, "error": err}
    time.sleep(0.35)

    def _scroll_to_sandbox_mod() -> None:
        for _ in range(20):
            io.scroll_wheel(-1)
            time.sleep(0.02)

    err = _click_with_verification(
        io=io,
        rect=rect,
        template=career_mode,
        name="career_mode",
        timeout_s=60.0,
        poll_s=0.5,
        verify_visible_template=sandbox_mod,
        verify_visible_name="sandbox_mod_after_career_mode",
        verify_visible_threshold=0.78,
        post_click=_scroll_to_sandbox_mod,
    )
    if err:
        return {"success": False, "error": err}
    time.sleep(0.35)

    err = _click_with_verification(
        io=io,
        rect=rect,
        template=sandbox_mod,
        name="sandbox_mod",
        timeout_s=60.0,
        poll_s=0.5,
        verify_visible_template=classic_sandbox,
        verify_visible_name="classic_sandbox_after_sandbox_mod",
        verify_visible_threshold=0.78,
    )
    if err:
        return {"success": False, "error": err}
    time.sleep(0.35)

    err = _wait_for_template(rect=rect, template=classic_sandbox, name="classic_sandbox", timeout_s=60.0, poll_s=0.5)
    if err:
        return {"success": False, "error": err}

    err = _click_with_verification(
        io=io,
        rect=rect,
        template=sandbox_play,
        name="sandbox_play",
        timeout_s=60.0,
        poll_s=0.5,
        verify_hidden_template=sandbox_play,
        verify_hidden_name="sandbox_play_hidden_after_click",
        verify_hidden_threshold=0.78,
    )
    if err:
        return {"success": False, "error": err}

    time.sleep(10.0)
    return {"success": True, "error": ""}


if __name__ == "__main__":
    # Simple debug run (assumes game is focused and on evaluation page).
    print(restart_environment())
