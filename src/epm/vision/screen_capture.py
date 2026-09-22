from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import mss
import pygetwindow as gw
from PIL import Image
import ctypes
import ctypes.wintypes

from epm.cerebellum.raw_input_controller import RawInputController


SW_RESTORE = 9


@dataclass(frozen=True)
class WindowRect:
    left: int
    top: int
    width: int
    height: int

    def to_mss_region(self) -> Dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


_io = RawInputController()


def _try_enable_per_monitor_dpi_awareness() -> None:
    """
    Best-effort: reduce DPI virtualization mismatch on multi-monitor setups.
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


def activate_window(window_title: str = "CookingSimulator") -> gw.Win32Window:
    hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
    if hwnd == 0:
        raise RuntimeError(f"Window not found: {window_title!r}")
    ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)
    ctypes.windll.user32.SetForegroundWindow(hwnd)
    time.sleep(0.2)

    game_window = gw.Win32Window(hwnd)
    print("[*] Nudge mouse to activate game input...")
    _io.mouse_move_relative(dx=1, dy=0)
    time.sleep(0.05)
    _io.mouse_move_relative(dx=-1, dy=0)
    return game_window


def get_window_rect(window_title: str = "CookingSimulator") -> WindowRect:
    # Prefer client-area rect (exclude window borders/title bar) to align with in-game screen coords.
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, window_title)
    if hwnd:
        rect = ctypes.wintypes.RECT()
        if user32.GetClientRect(hwnd, ctypes.byref(rect)):
            pt = ctypes.wintypes.POINT(rect.left, rect.top)  # (0,0) in client coords
            if user32.ClientToScreen(hwnd, ctypes.byref(pt)):
                width = int(rect.right - rect.left)
                height = int(rect.bottom - rect.top)
                if width > 0 and height > 0:
                    return WindowRect(left=int(pt.x), top=int(pt.y), width=width, height=height)
        # Fallback for same hwnd: window outer rect (if client rect path fails)
        rect2 = ctypes.wintypes.RECT()
        if user32.GetWindowRect(hwnd, ctypes.byref(rect2)):
            width2 = int(rect2.right - rect2.left)
            height2 = int(rect2.bottom - rect2.top)
            if width2 > 0 and height2 > 0:
                return WindowRect(left=int(rect2.left), top=int(rect2.top), width=width2, height=height2)

    # Fallback: window outer rect (may include borders)
    wins = gw.getWindowsWithTitle(window_title)
    if not wins:
        raise RuntimeError(f"Window not found: {window_title!r}")
    w = wins[0]
    return WindowRect(left=int(w.left), top=int(w.top), width=int(w.width), height=int(w.height))


def capture_screenshot_mss(*, region: Optional[Dict[str, int]] = None) -> Image.Image:
    with mss.mss() as sct:
        monitor = region if region else sct.monitors[1]
        sct_img = sct.grab(monitor)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def screenshot_base64_png(*, window_title: str = "CookingSimulator", activate: bool = True) -> str:
    if activate:
        activate_window(window_title)
    rect = get_window_rect(window_title)
    img = capture_screenshot_mss(region=rect.to_mss_region())
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _normalize_image_format(image_format: str) -> str:
    fmt = str(image_format or "jpeg").strip().lower()
    if fmt in {"jpg", "jpeg"}:
        return "jpeg"
    if fmt == "png":
        return "png"
    return "jpeg"


def _normalized_screenshot_path(*, out_dir: Path, step_id: int, filename: str | None, image_format: str) -> Path:
    suffix = ".jpg" if _normalize_image_format(image_format) == "jpeg" else ".png"
    if filename:
        raw = out_dir / str(filename)
        return raw.with_suffix(suffix) if raw.suffix.lower() != suffix else raw
    return out_dir / f"step_{step_id:06d}{suffix}"


def save_step_screenshot(
    *,
    step_id: int,
    out_dir: str | Path,
    window_title: str = "CookingSimulator",
    activate: bool = True,
    filename: str | None = None,
    image_format: str = "jpeg",
    jpeg_quality: int = 70,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if activate:
        activate_window(window_title)
    rect = get_window_rect(window_title)
    img = capture_screenshot_mss(region=rect.to_mss_region())
    fmt = _normalize_image_format(image_format)
    path = _normalized_screenshot_path(out_dir=out, step_id=step_id, filename=filename, image_format=fmt)
    if fmt == "jpeg":
        if img.mode != "RGB":
            img = img.convert("RGB")
        quality = max(1, min(95, int(jpeg_quality or 70)))
        img.save(path, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(path, format="PNG")
    return path
