"""
RawInputController

Minimal Windows keyboard/mouse controller used by `epm.cerebellum.local_actions`.

Design goals for this repo:
- Avoid AutoHotkey (AHK) dependency
- Keep call sites stable (`mouse_move_relative`, `mouse_move_absolute`, `click`, `key_press`, ...)
- Provide a reliable `type_text()` for GUI actions (menu/store search boxes)
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import os
import time
import threading
from loguru import logger as logging

# Windows API constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_SCANCODE = 0x0008

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

# Virtual desktop system metrics (multi-monitor aware)
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# Hotkey constants (global stop)
WM_HOTKEY = 0x0312
VK_ESCAPE = 0x1B


class StopRequested(RuntimeError):
    pass


# Minimal VK mapping used in the project.
VK_CODES: dict[str, int] = {
    "w": 0x57,
    "s": 0x53,
    "a": 0x41,
    "d": 0x44,
    "q": 0x51,
    "e": 0x45,
    "r": 0x52,
    "t": 0x54,
    "y": 0x59,
    "j": 0x4A,
    "space": 0x20,
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "enter": 0x0D,
    "escape": 0x1B,
    "tab": 0x09,
    "backspace": 0x08,
    "delete": 0x2E,
    "f11": 0x7A,  # radar scan (mod)
    "f12": 0x7B,  # products scan (mod)
    "0": 0x30,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "5": 0x35,
    "6": 0x36,
    "7": 0x37,
    "8": 0x38,
    "9": 0x39,
}


class RawInputController:
    _stop_requested = False
    _hotkey_started = False
    _hotkey_lock = threading.Lock()
    _esc_was_down = False
    _hotkey_mode_logged = False

    def __init__(self) -> None:
        self.user32 = ctypes.windll.user32
        self._ensure_stop_hotkey_support()
        logging.debug("[RawInput] RawInputController initialized")
        self._use_sendinput_scancode = str(os.environ.get("COOKGAME_KEYBOARD_SENDINPUT", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
        }

    @classmethod
    def request_stop(cls) -> None:
        cls._stop_requested = True

    @classmethod
    def check_stop(cls) -> None:
        cls._check_stop()

    @classmethod
    def _check_stop(cls) -> None:
        cls._poll_stop_hotkey()
        if cls._stop_requested:
            raise StopRequested("global_stop_requested")

    @classmethod
    def _use_stop_hotkey_thread(cls) -> bool:
        return str(os.environ.get("COOKGAME_STOP_HOTKEY_MODE", "poll")).strip().lower() == "thread"

    @classmethod
    def _ensure_stop_hotkey_support(cls) -> None:
        if not cls._hotkey_mode_logged:
            cls._hotkey_mode_logged = True
            mode = "thread" if cls._use_stop_hotkey_thread() else "poll"
            logging.info(f"[STOP] hotkey_mode={mode}")
        if cls._use_stop_hotkey_thread():
            cls._ensure_stop_hotkey_thread()

    @classmethod
    def _poll_stop_hotkey(cls) -> None:
        if cls._use_stop_hotkey_thread():
            return
        try:
            esc_down = bool(int(ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE)) & 0x8000)
        except Exception:
            return
        if esc_down and not cls._esc_was_down:
            cls.request_stop()
            logging.warning("[STOP] Esc pressed; stop requested.")
        cls._esc_was_down = esc_down

    @classmethod
    def _ensure_stop_hotkey_thread(cls) -> None:
        with cls._hotkey_lock:
            if cls._hotkey_started:
                return
            cls._hotkey_started = True

        def _hotkey_loop() -> None:
            user32 = ctypes.windll.user32
            # Register Esc as global stop
            if not user32.RegisterHotKey(None, 1, 0, VK_ESCAPE):
                return

            msg = ctypes.wintypes.MSG()
            while True:
                res = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if res == 0:
                    break
                if res == -1:
                    break
                if msg.message == WM_HOTKEY:
                    RawInputController.request_stop()
                    logging.warning("[STOP] Esc pressed; stop requested.")

            user32.UnregisterHotKey(None, 1)

        t = threading.Thread(target=_hotkey_loop, name="global_stop_hotkey", daemon=True)
        t.start()

    def _get_cursor_pos(self) -> tuple[int, int]:
        pt = ctypes.wintypes.POINT()
        ok = int(self.user32.GetCursorPos(ctypes.byref(pt)))
        if not ok:
            return (0, 0)
        return (int(pt.x), int(pt.y))

    def mouse_move_absolute(self, x: int, y: int) -> None:
        """
        Move mouse to absolute *virtual desktop* screen coordinates (pixels).

        Note: when the game window is on a secondary monitor, `x` can be larger than the primary
        monitor width. `SetCursorPos` accepts virtual desktop coordinates directly (and supports
        negative coords), so it is the most reliable option on multi-monitor setups.
        """
        self._check_stop()
        tx = int(x)
        ty = int(y)

        # 1) Try SetCursorPos and verify the cursor actually moved.
        try:
            _ = int(self.user32.SetCursorPos(tx, ty))
            time.sleep(0.001)
            cx, cy = self._get_cursor_pos()
            if abs(cx - tx) <= 2 and abs(cy - ty) <= 2:
                return
        except Exception:
            cx, cy = self._get_cursor_pos()

        # 2) Fallback: emulate absolute movement via relative steps.
        # This is slower but reliable when SetCursorPos is blocked but the cursor is movable.
        cx, cy = self._get_cursor_pos()
        max_steps = 90
        step_cap = 220
        for _i in range(max_steps):
            dx = int(tx - cx)
            dy = int(ty - cy)
            if abs(dx) <= 2 and abs(dy) <= 2:
                return
            sx = max(-step_cap, min(step_cap, dx))
            sy = max(-step_cap, min(step_cap, dy))
            self.user32.mouse_event(MOUSEEVENTF_MOVE, int(sx), int(sy), 0, 0)
            time.sleep(0.001)
            cx, cy = self._get_cursor_pos()

        # 3) Last resort: mouse_event absolute mapping using virtual desktop bounds.
        vx = int(self.user32.GetSystemMetrics(SM_XVIRTUALSCREEN))
        vy = int(self.user32.GetSystemMetrics(SM_YVIRTUALSCREEN))
        vw = int(self.user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)) or 1
        vh = int(self.user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)) or 1

        nx = max(0.0, min(1.0, (float(tx) - float(vx)) / float(vw)))
        ny = max(0.0, min(1.0, (float(ty) - float(vy)) / float(vh)))
        abs_x = int(nx * 65535.0)
        abs_y = int(ny * 65535.0)
        self.user32.mouse_event(MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE, abs_x, abs_y, 0, 0)
        return

    def mouse_move_relative(self, dx: int = 0, dy: int = 0) -> None:
        """Move mouse relative (pixels)."""
        self._check_stop()
        logging.debug(f"[RawInput] mouse_move_relative dx={dx} dy={dy}")
        self.user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

    def click(self, button: str = "left") -> None:
        """Click a mouse button."""
        self._check_stop()
        if button == "left":
            self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.05)
            self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        elif button == "right":
            self.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
            time.sleep(0.05)
            self.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        elif button == "middle":
            self.user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
            time.sleep(0.05)
            self.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)

    def click_sendinput(self, button: str = "left") -> None:
        """
        Click via `SendInput` (can be more reliable for some games than `mouse_event`).
        """
        self._check_stop()
        btn = (button or "").strip().lower()
        if btn not in ("left", "right", "middle"):
            btn = "left"

        INPUT_MOUSE = 0
        flags_down, flags_up = 0x0002, 0x0004  # left
        if btn == "right":
            flags_down, flags_up = 0x0008, 0x0010
        elif btn == "middle":
            flags_down, flags_up = 0x0020, 0x0040

        # Use correct Win32 field types. In WinUser.h: dwExtraInfo is ULONG_PTR (not a pointer).
        try:
            _ULONG_PTR = ctypes.wintypes.ULONG_PTR  # type: ignore[attr-defined]
        except Exception:
            _ULONG_PTR = ctypes.c_size_t

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.wintypes.LONG),
                ("dy", ctypes.wintypes.LONG),
                ("mouseData", ctypes.wintypes.DWORD),
                ("dwFlags", ctypes.wintypes.DWORD),
                ("time", ctypes.wintypes.DWORD),
                ("dwExtraInfo", _ULONG_PTR),
            ]

        class INPUT_I(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.wintypes.DWORD), ("ii", INPUT_I)]

        down = INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(0, 0, 0, int(flags_down), 0, _ULONG_PTR(0))))
        up = INPUT(type=INPUT_MOUSE, ii=INPUT_I(mi=MOUSEINPUT(0, 0, 0, int(flags_up), 0, _ULONG_PTR(0))))

        arr = (INPUT * 2)(down, up)
        sent = int(self.user32.SendInput(2, ctypes.byref(arr), ctypes.sizeof(INPUT)))
        if sent != 2:
            # Best-effort fallback.
            self.click(btn)

    def mouse_down(self, button: str = "left") -> None:
        self._check_stop()
        if button == "left":
            self.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        elif button == "right":
            self.user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        elif button == "middle":
            self.user32.mouse_event(MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)

    def mouse_up(self, button: str = "left") -> None:
        self._check_stop()
        if button == "left":
            self.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        elif button == "right":
            self.user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        elif button == "middle":
            self.user32.mouse_event(MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)

    def scroll_wheel(self, distance: float) -> None:
        """Scroll wheel (distance in 'notches', 1 notch = 120)."""
        self._check_stop()
        wheel_delta = int(float(distance) * 120)
        self.user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, wheel_delta, 0)

    def key_press(self, key: str) -> None:
        self._check_stop()
        vk_code = self._get_vk_code(key)
        if vk_code is None:
            return
        if self._use_sendinput_scancode and self._sendinput_key(vk_code, is_keyup=False):
            time.sleep(0.05)
            if self._sendinput_key(vk_code, is_keyup=True):
                return
        self.user32.keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.05)
        self.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)

    def key_down(self, key: str) -> None:
        self._check_stop()
        vk_code = self._get_vk_code(key)
        if vk_code is None:
            return
        if self._use_sendinput_scancode and self._sendinput_key(vk_code, is_keyup=False):
            return
        self.user32.keybd_event(vk_code, 0, 0, 0)

    def key_up(self, key: str) -> None:
        self._check_stop()
        vk_code = self._get_vk_code(key)
        if vk_code is None:
            return
        if self._use_sendinput_scancode and self._sendinput_key(vk_code, is_keyup=True):
            return
        self.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)

    def _sendinput_key(self, vk_code: int, *, is_keyup: bool) -> bool:
        try:
            scan = int(self.user32.MapVirtualKeyW(int(vk_code), 0))
            if scan == 0:
                return False

            flags = int(KEYEVENTF_SCANCODE)
            if is_keyup:
                flags |= int(KEYEVENTF_KEYUP)

            try:
                _ULONG_PTR = ctypes.wintypes.ULONG_PTR  # type: ignore[attr-defined]
            except Exception:
                _ULONG_PTR = ctypes.c_size_t

            class MOUSEINPUT(ctypes.Structure):
                _fields_ = [
                    ("dx", ctypes.wintypes.LONG),
                    ("dy", ctypes.wintypes.LONG),
                    ("mouseData", ctypes.wintypes.DWORD),
                    ("dwFlags", ctypes.wintypes.DWORD),
                    ("time", ctypes.wintypes.DWORD),
                    ("dwExtraInfo", _ULONG_PTR),
                ]

            class KEYBDINPUT(ctypes.Structure):
                _fields_ = [
                    ("wVk", ctypes.wintypes.WORD),
                    ("wScan", ctypes.wintypes.WORD),
                    ("dwFlags", ctypes.wintypes.DWORD),
                    ("time", ctypes.wintypes.DWORD),
                    ("dwExtraInfo", _ULONG_PTR),
                ]

            class INPUT_I(ctypes.Union):
                _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]

            class INPUT(ctypes.Structure):
                _fields_ = [("type", ctypes.wintypes.DWORD), ("ii", INPUT_I)]

            inp = INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(0, scan, flags, 0, _ULONG_PTR(0))))
            sent = int(self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)))
            return sent == 1
        except Exception:
            return False

    def type_text(self, text: str, *, interval_s: float = 0.0) -> None:
        """
        Type unicode text via `SendInput` (supports spaces/mixed-case without manual shift).
        The game/window must already have focus.
        """
        self._check_stop()
        if text is None:
            return
        s = str(text)
        if not s:
            return

        try:
            _ULONG_PTR = ctypes.wintypes.ULONG_PTR  # type: ignore[attr-defined]
        except Exception:
            _ULONG_PTR = ctypes.c_size_t

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.wintypes.LONG),
                ("dy", ctypes.wintypes.LONG),
                ("mouseData", ctypes.wintypes.DWORD),
                ("dwFlags", ctypes.wintypes.DWORD),
                ("time", ctypes.wintypes.DWORD),
                ("dwExtraInfo", _ULONG_PTR),
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.wintypes.WORD),
                ("wScan", ctypes.wintypes.WORD),
                ("dwFlags", ctypes.wintypes.DWORD),
                ("time", ctypes.wintypes.DWORD),
                ("dwExtraInfo", _ULONG_PTR),
            ]

        class INPUT_I(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", ctypes.wintypes.DWORD), ("ii", INPUT_I)]

        for ch in s:
            code = ord(ch)
            down = INPUT(type=INPUT_KEYBOARD, ii=INPUT_I(ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE, 0, _ULONG_PTR(0))))
            up = INPUT(
                type=INPUT_KEYBOARD,
                ii=INPUT_I(ki=KEYBDINPUT(0, code, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, _ULONG_PTR(0))),
            )
            self.user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(INPUT))
            self.user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(INPUT))
            if interval_s and interval_s > 0:
                time.sleep(float(interval_s))

    def _get_vk_code(self, key: str) -> int | None:
        k = (key or "").strip().lower()
        if k in VK_CODES:
            return VK_CODES[k]
        if len(k) == 1:
            return ord(k.upper())
        print(f"[!] 警告: 未知按键 {key!r}")
        return None


if __name__ == "__main__":
    RawInputController()
    print("RawInputController smoke test OK")
