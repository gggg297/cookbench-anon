from __future__ import annotations

"""
Windows clipboard helpers (no external deps).

Used as a fallback for GUI actions when direct typing is flaky.
"""

import ctypes
import ctypes.wintypes


CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


def set_clipboard_text(text: str) -> None:
    """
    Set unicode text into Windows clipboard.

    Raises RuntimeError on failure.
    """
    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    # Define Win32 prototypes to avoid 64-bit handle truncation.
    kernel32.GlobalAlloc.argtypes = (ctypes.wintypes.UINT, ctypes.c_size_t)
    kernel32.GlobalAlloc.restype = ctypes.wintypes.HGLOBAL
    kernel32.GlobalFree.argtypes = (ctypes.wintypes.HGLOBAL,)
    kernel32.GlobalFree.restype = ctypes.wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = (ctypes.wintypes.HGLOBAL,)
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = (ctypes.wintypes.HGLOBAL,)
    kernel32.GlobalUnlock.restype = ctypes.wintypes.BOOL

    user32.OpenClipboard.argtypes = (ctypes.wintypes.HWND,)
    user32.OpenClipboard.restype = ctypes.wintypes.BOOL
    user32.CloseClipboard.argtypes = ()
    user32.CloseClipboard.restype = ctypes.wintypes.BOOL
    user32.EmptyClipboard.argtypes = ()
    user32.EmptyClipboard.restype = ctypes.wintypes.BOOL
    user32.SetClipboardData.argtypes = (ctypes.wintypes.UINT, ctypes.wintypes.HANDLE)
    user32.SetClipboardData.restype = ctypes.wintypes.HANDLE

    s = str(text or "")
    # Clipboard can be briefly locked by other apps; retry a bit.
    for _ in range(12):
        if user32.OpenClipboard(None):
            break
        import time

        time.sleep(0.02)
    else:
        raise RuntimeError("OpenClipboard failed (clipboard busy)")
    try:
        if not user32.EmptyClipboard():
            raise RuntimeError("EmptyClipboard failed")

        # Allocate global memory for the UTF-16LE string (+ null terminator).
        data = s.encode("utf-16le") + b"\x00\x00"
        hglob = kernel32.GlobalAlloc(GMEM_MOVEABLE, ctypes.c_size_t(len(data)))
        if not hglob:
            raise RuntimeError("GlobalAlloc failed")
        locked = kernel32.GlobalLock(hglob)
        if not locked:
            kernel32.GlobalFree(hglob)
            raise RuntimeError("GlobalLock failed")
        try:
            ctypes.memmove(locked, data, ctypes.c_size_t(len(data)))
        finally:
            kernel32.GlobalUnlock(hglob)

        if not user32.SetClipboardData(CF_UNICODETEXT, hglob):
            kernel32.GlobalFree(hglob)
            raise RuntimeError("SetClipboardData failed")
        # On success, the system owns hglob.
    finally:
        user32.CloseClipboard()
