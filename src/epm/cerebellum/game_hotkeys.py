"""
Game hotkey bootstrap helpers (F11/F12/Alt+J).

These hotkeys are handled by the C# mod (CS_CamDump):
- F12: realtime products scan -> `realtime_products.json`
- F11: realtime radar scan -> `realtime_radar_scan.txt`
- Alt+J: realtime interaction info -> `realtime_interaction_info.txt` (and/or UDP stream)

We avoid blindly pressing toggles by checking for the expected output files and
their freshness first.
"""

from __future__ import annotations

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _fresh_enough(path: Path, *, max_age_s: float) -> bool:
    if not path.exists():
        return False
    age = time.time() - _mtime(path)
    return age >= 0 and age <= max_age_s


def _press_alt_j(io_controller) -> None:
    # Alt+J combo: hold Alt, press J, release Alt.
    io_controller.key_down("alt")
    time.sleep(0.03)
    io_controller.key_press("j")
    time.sleep(0.03)
    io_controller.key_up("alt")


def _parse_kv_text(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or ":" not in s:
            continue
        k, v = s.split(":", 1)
        k = k.strip()
        v = v.strip()
        if k:
            out[k] = v
    return out


def _parse_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    return None


def _parse_status_timestamp(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    # Format emitted by CS_CamDump: "yyyy-MM-dd HH:mm:ss.fff"
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return None


def _epm_config_path() -> Path:
    # <repo>/epm/src/epm/cerebellum/game_hotkeys.py -> parents[3] == <repo>/epm
    return Path(__file__).resolve().parents[3] / "epm_config.json"


def _write_hotkey_status_to_config(*, status: dict[str, Any]) -> None:
    """
    Best-effort: persist the hotkey status file path into epm/epm_config.json.
    Runtime enable checks should still rely on the status file / UDP.
    """
    cfg_path = _epm_config_path()
    if not cfg_path.exists():
        return
    try:
        raw = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            return
    except Exception:
        return

    # User preference: keep this under `paths` (not `runtime.hotkeys`), and avoid writing noisy telemetry.
    paths = raw.get("paths")
    if not isinstance(paths, dict):
        paths = {}
        raw["paths"] = paths
    if status.get("hotkeys_status_path"):
        paths["hotkeys_status_path"] = status.get("hotkeys_status_path")

    # Cleanup legacy key if present.
    runtime = raw.get("runtime")
    if isinstance(runtime, dict) and "hotkeys" in runtime:
        try:
            runtime.pop("hotkeys", None)
        except Exception:
            pass

    try:
        tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
        tmp.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(cfg_path)
    except Exception:
        return


def _read_interaction_status(*, userdata_root: Path) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Read `realtime_interaction_status.txt` produced by CS_CamDump.
    Returns (status_dict, error_str).
    """
    path = userdata_root / "realtime_interaction_status.txt"
    if not path.exists():
        return None, "missing"
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return None, f"unreadable:{e}"

    kv = _parse_kv_text(txt)
    ts = _parse_status_timestamp(kv.get("Timestamp"))
    age_s: Optional[float] = None
    if ts is not None:
        try:
            age_s = time.time() - ts.timestamp()
        except Exception:
            age_s = None

    status: dict[str, Any] = {
        "hotkeys_status_path": str(path),
        "hotkeys_status_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if ts is not None else None,
        "hotkeys_status_age_s": age_s,
        "alt_j_enabled": _parse_bool(kv.get("AltJEnabled")),
        "f10_enabled": _parse_bool(kv.get("F10Enabled")),
        "f11_enabled": _parse_bool(kv.get("F11Enabled")),
        "f12_enabled": _parse_bool(kv.get("F12Enabled")),
        "use_udp_mode": _parse_bool(kv.get("UseUdpMode")),
        "use_file_mode": _parse_bool(kv.get("UseFileMode")),
        "hotkeys_status_read_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "hotkeys_status_source": "realtime_interaction_status.txt",
    }
    return status, None


def ensure_f12_products_scan(
    *,
    realtime_products_path: Path,
    window_title: str,
    activate_window,
    io_controller,
    max_age_s: float = 3.0,
    wait_after_press_s: float = 0.6,
    rearm_if_enabled: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Ensure F12 scan is producing/updating realtime_products.json.
    Returns True if the file exists and is reasonably fresh.
    """
    # Optional one-shot rearm: for some restart flows, status may still report
    # F12 enabled while the scan stream is ineffective until toggled off/on.
    if bool(rearm_if_enabled):
        try:
            status, _status_err = _read_interaction_status(userdata_root=realtime_products_path.parent)
        except Exception:
            status = None
        if isinstance(status, dict):
            enabled = status.get("f12_enabled", None)
            age = status.get("hotkeys_status_age_s", None)
            try:
                age_f = float(age) if age is not None else None
            except Exception:
                age_f = None
            if bool(enabled) and age_f is not None and age_f <= 1.5:
                try:
                    activate_window(window_title)
                except Exception:
                    pass
                if verbose:
                    print(f"[hotkeys] F12 rearm: status_enabled=True age_s={age_f:.2f} (press twice)")
                io_controller.key_press("f12")
                time.sleep(0.20)
                io_controller.key_press("f12")
                time.sleep(float(wait_after_press_s))

    if _fresh_enough(realtime_products_path, max_age_s=max_age_s):
        if verbose:
            age = time.time() - _mtime(realtime_products_path)
            print(f"[hotkeys] F12 ok (fresh): age_s={age:.2f} path={str(realtime_products_path)!r}")
        return True

    try:
        activate_window(window_title)
    except Exception:
        pass

    before = _mtime(realtime_products_path)
    if verbose:
        print(f"[hotkeys] F12 press: before_mtime={before} path={str(realtime_products_path)!r}")
    io_controller.key_press("f12")
    time.sleep(float(wait_after_press_s))
    after = _mtime(realtime_products_path)
    ok = bool(realtime_products_path.exists() and after > before)
    if verbose:
        age = time.time() - _mtime(realtime_products_path)
        print(f"[hotkeys] F12 {'ok' if ok else 'fail'}: after_mtime={after} age_s={age:.2f}")
    return ok


def set_f12_products_scan_enabled(
    *,
    realtime_products_path: Path,
    window_title: str,
    activate_window,
    io_controller,
    enabled: bool,
    wait_after_press_s: float = 0.5,
    verbose: bool = False,
) -> bool:
    """
    Best-effort enable/disable for the F12 realtime-products scan.

    - `enabled=True`: ensure scan is running and producing fresh `realtime_products.json`.
    - `enabled=False`: prefer the interaction-status heartbeat to determine whether a
      toggle press is needed; if status is unavailable, do not blindly press.
    """
    want_enabled = bool(enabled)
    if want_enabled:
        return ensure_f12_products_scan(
            realtime_products_path=realtime_products_path,
            window_title=window_title,
            activate_window=activate_window,
            io_controller=io_controller,
            wait_after_press_s=float(wait_after_press_s),
            verbose=bool(verbose),
        )

    try:
        status, status_err = _read_interaction_status(userdata_root=realtime_products_path.parent)
    except Exception as e:
        status, status_err = None, f"unreadable:{e}"

    status_enabled = None
    status_age = None
    if isinstance(status, dict):
        status_enabled = _parse_bool(status.get("f12_enabled"))
        try:
            status_age = float(status.get("hotkeys_status_age_s")) if status.get("hotkeys_status_age_s") is not None else None
        except Exception:
            status_age = None
        if status_enabled is False and status_age is not None and status_age <= 1.5:
            if verbose:
                print(f"[hotkeys] F12 already disabled: age_s={status_age:.2f}")
            return True

    if status_enabled is not True or status_age is None or status_age > 1.5:
        if verbose:
            print(f"[hotkeys] F12 disable skipped: status_err={status_err!r} enabled={status_enabled!r} age_s={status_age!r}")
        return False

    try:
        activate_window(window_title)
    except Exception:
        pass
    if verbose:
        print(f"[hotkeys] F12 disable press: age_s={status_age:.2f}")
    io_controller.key_press("f12")
    time.sleep(float(wait_after_press_s))

    try:
        status2, _ = _read_interaction_status(userdata_root=realtime_products_path.parent)
    except Exception:
        status2 = None
    if isinstance(status2, dict):
        enabled2 = _parse_bool(status2.get("f12_enabled"))
        try:
            age2 = float(status2.get("hotkeys_status_age_s")) if status2.get("hotkeys_status_age_s") is not None else None
        except Exception:
            age2 = None
        if enabled2 is False and age2 is not None and age2 <= 1.5:
            if verbose:
                print(f"[hotkeys] F12 disabled confirmed: age_s={age2:.2f}")
            return True

    return False


def ensure_f11_radar_scan(
    *,
    userdata_root: Path,
    window_title: str,
    activate_window,
    io_controller,
    max_age_s: float = 3.0,
    wait_after_press_s: float = 0.6,
    verbose: bool = False,
) -> bool:
    radar_path = userdata_root / "realtime_radar_scan.txt"
    if _fresh_enough(radar_path, max_age_s=max_age_s):
        if verbose:
            age = time.time() - _mtime(radar_path)
            print(f"[hotkeys] F11 ok (fresh): age_s={age:.2f} path={str(radar_path)!r}")
        return True

    try:
        activate_window(window_title)
    except Exception:
        pass

    before = _mtime(radar_path)
    if verbose:
        print(f"[hotkeys] F11 press: before_mtime={before} path={str(radar_path)!r}")
    io_controller.key_press("f11")
    time.sleep(float(wait_after_press_s))
    after = _mtime(radar_path)
    ok = bool(radar_path.exists() and after > before)
    if verbose:
        age = time.time() - _mtime(radar_path)
        print(f"[hotkeys] F11 {'ok' if ok else 'fail'}: after_mtime={after} age_s={age:.2f}")
    return ok


def check_camera_info_fresh(
    *,
    camera_info_path: Path,
    max_age_s: float = 2.5,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Check whether realtime camera pose file is being updated recently.
    Freshness is determined ONLY by the in-file `Timestamp` entry.
    This function does not press any hotkeys.
    """
    exists = bool(camera_info_path.exists())
    age = None
    age_source = "missing_file"
    camera_timestamp = None

    # Strict mode: rely on in-file Timestamp only (no mtime fallback).
    if exists:
        try:
            raw = camera_info_path.read_text(encoding="utf-8", errors="ignore")
            kv = _parse_kv_text(raw)
            ts = _parse_status_timestamp(kv.get("Timestamp"))
            if ts is not None:
                camera_timestamp = ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                age = time.time() - ts.timestamp()
                age_source = "camera_timestamp"
            else:
                age_source = "missing_or_unparseable_timestamp"
        except Exception:
            age_source = "timestamp_read_failed"

    ok = bool(exists and age is not None and age >= 0 and age <= float(max_age_s))
    out: dict[str, Any] = {
        "ok": ok,
        "path": str(camera_info_path),
        "exists": exists,
        "age_s": age,
        "max_age_s": float(max_age_s),
        "age_source": age_source,
        "camera_timestamp": camera_timestamp,
    }
    if verbose:
        if ok:
            print(
                f"[hotkeys] camera ok (fresh): age_s={age:.2f} "
                f"source={age_source} path={str(camera_info_path)!r}"
            )
        else:
            if exists and age is not None:
                print(
                    f"[hotkeys] camera stale: age_s={age:.2f} > {float(max_age_s):.2f} "
                    f"source={age_source} path={str(camera_info_path)!r}"
                )
            elif exists:
                print(
                    f"[hotkeys] camera stale: no valid Timestamp entry "
                    f"source={age_source} path={str(camera_info_path)!r}"
                )
            else:
                print(f"[hotkeys] camera missing: path={str(camera_info_path)!r}")
    return out


def ensure_alt_j_interaction(
    *,
    userdata_root: Path,
    window_title: str,
    activate_window,
    io_controller,
    max_age_s: float = 3.0,
    wait_after_press_s: float = 0.6,
    verbose: bool = False,
) -> bool:
    # Prefer UDP-based detection: the file can persist and may only update on specific events,
    # so its mtime/existence is not a reliable "enabled" signal.
    try:
        from epm.cerebellum.skills.auto_navigation.interaction_detector import get_udp_debug_snapshot, init_udp_mode
    except Exception:  # pragma: no cover
        get_udp_debug_snapshot = None  # type: ignore[assignment]
        init_udp_mode = None  # type: ignore[assignment]

    def _udp_enabled() -> bool:
        if get_udp_debug_snapshot is None:
            return False
        snap = get_udp_debug_snapshot()
        if not isinstance(snap, dict) or not snap.get("running"):
            return False
        age = snap.get("latest_age_s", None)
        try:
            return age is not None and float(age) <= float(max_age_s)
        except Exception:
            return False

    info_path = userdata_root / "realtime_interaction_info.txt"

    # Prefer the dedicated status heartbeat file (1Hz) for deterministic enable checks.
    # Require freshness <= 1.5s before trusting it.
    status, status_err = _read_interaction_status(userdata_root=userdata_root)
    if isinstance(status, dict):
        age = status.get("hotkeys_status_age_s", None)
        enabled = status.get("alt_j_enabled", None)
        try:
            age_f = float(age) if age is not None else None
        except Exception:
            age_f = None

        if age_f is not None and age_f <= 1.5 and enabled is not None:
            if verbose:
                print(f"[hotkeys] Alt+J ok (status): enabled={bool(enabled)} age_s={age_f:.2f} path={status.get('hotkeys_status_path')!r}")
            _write_hotkey_status_to_config(status=status)

            # Fresh status provides ground truth; only press Alt+J when it explicitly says disabled.
            if bool(enabled):
                return True

            try:
                activate_window(window_title)
            except Exception:
                pass
            if verbose:
                print("[hotkeys] Alt+J status says disabled; pressing Alt+J to enable...")
            _press_alt_j(io_controller)

            t0 = time.time()
            while time.time() - t0 < max(2.0, float(wait_after_press_s)):
                st2, _ = _read_interaction_status(userdata_root=userdata_root)
                if isinstance(st2, dict):
                    _write_hotkey_status_to_config(status=st2)
                    try:
                        age2 = float(st2.get("hotkeys_status_age_s")) if st2.get("hotkeys_status_age_s") is not None else None
                    except Exception:
                        age2 = None
                    en2 = st2.get("alt_j_enabled", None)
                    if age2 is not None and age2 <= 1.5 and bool(en2):
                        if verbose:
                            print(f"[hotkeys] Alt+J ok (status_after_press): age_s={age2:.2f}")
                        return True
                time.sleep(0.05)

            if verbose:
                print("[hotkeys] Alt+J enable not confirmed by status heartbeat; falling back to UDP/file heuristics")
        else:
            if verbose:
                print(f"[hotkeys] Alt+J status stale/unknown: err={status_err!r} age_s={age!r} enabled={enabled!r}")
    elif verbose:
        print(f"[hotkeys] Alt+J status missing/unreadable: err={status_err!r}")

    try:
        activate_window(window_title)
    except Exception:
        pass

    if init_udp_mode is not None:
        try:
            init_udp_mode()  # idempotent
        except Exception:
            pass

    if _udp_enabled():
        if verbose and get_udp_debug_snapshot is not None:
            snap = get_udp_debug_snapshot()
            age = snap.get("latest_age_s") if isinstance(snap, dict) else None
            print(f"[hotkeys] Alt+J ok (udp): latest_age_s={age}")
        return True

    # If UDP isn't producing packets, the stream may still be enabled (the mod can be event-driven),
    # and Alt+J is a toggle (pressing it again can disable the stream / freeze some setups).
    # If the output file already exists, assume Alt+J is enabled and do NOT press the toggle.
    if info_path.exists():
        if verbose:
            age = time.time() - _mtime(info_path)
            print(f"[hotkeys] Alt+J warning: UDP not confirmed (no packets), but file exists. Assuming enabled to avoid toggling. age_s={age:.2f} path={str(info_path)!r}")
        return True

    # Otherwise, try enabling via Alt+J (toggle) once.
    before = _mtime(info_path)
    if verbose:
        print(f"[hotkeys] Alt+J press: before_mtime={before} path={str(info_path)!r}")
    _press_alt_j(io_controller)
    time.sleep(float(wait_after_press_s))
    after = _mtime(info_path)
    # After toggling, prefer seeing UDP traffic to confirm it's enabled.
    t0 = time.time()
    while time.time() - t0 < max(0.2, float(wait_after_press_s)):
        if _udp_enabled():
            if verbose and get_udp_debug_snapshot is not None:
                snap = get_udp_debug_snapshot()
                age = snap.get("latest_age_s") if isinstance(snap, dict) else None
                print(f"[hotkeys] Alt+J ok (udp_after_press): latest_age_s={age}")
            return True
        time.sleep(0.05)

    # Fallback: if UDP can't be confirmed (no packets), accept file creation as a weak signal.
    ok = bool(info_path.exists())
    if verbose:
        age = time.time() - _mtime(info_path)
        print(f"[hotkeys] Alt+J {'ok' if ok else 'fail'}: after_mtime={after} age_s={age:.2f}")
    return ok


def bootstrap_all(
    *,
    realtime_products_path: Path,
    window_title: str,
    userdata_root: Optional[Path] = None,
    activate_window=None,
    io_controller=None,
) -> dict[str, bool]:
    """
    Best-effort bootstrap for common mod hotkeys.

    Returns:
      {"f12": bool, "f11": bool, "alt_j": bool}
    """
    if userdata_root is None:
        userdata_root = realtime_products_path.parent
    if activate_window is None or io_controller is None:
        raise ValueError("bootstrap_all requires activate_window and io_controller")

    return {
        "f12": ensure_f12_products_scan(
            realtime_products_path=realtime_products_path,
            window_title=window_title,
            activate_window=activate_window,
            io_controller=io_controller,
            rearm_if_enabled=True,
        ),
        "f11": ensure_f11_radar_scan(
            userdata_root=userdata_root,
            window_title=window_title,
            activate_window=activate_window,
            io_controller=io_controller,
        ),
        "alt_j": ensure_alt_j_interaction(
            userdata_root=userdata_root,
            window_title=window_title,
            activate_window=activate_window,
            io_controller=io_controller,
        ),
    }
