"""
Shared path/config helpers for vendored `auto_*` skills.

Goal: keep all hard-coded CookingSimulator UserData paths out of the skill code,
and avoid ambiguous imports like `from config import ...` which can collide across
packages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from epm.core.settings import load_raw_settings


def repo_root() -> Path:
    """Locate the repository root.

    Walks up from this file until a directory containing both ``data/`` and
    ``src/`` is found, so the helper keeps working whether the package is
    checked out at the repo root (``<repo>/src/epm/...``) or nested under a
    subdirectory (``<repo>/epm/src/epm/...``). Falls back to the historical
    fixed depth if no marker directory is found.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "data").is_dir() and (parent / "src").is_dir():
            return parent
    # <repo>/src/epm/cerebellum/skills/_shared_paths.py -> parents[4] == <repo>
    return here.parents[4]


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def load_epm_config() -> dict[str, Any]:
    # Only read explicit run config if provided.
    run_cfg = (os.environ.get("EPM_CONFIG") or "").strip()
    if not run_cfg:
        raise RuntimeError("EPM_CONFIG not set. Please pass --config to run_episode.py.")
    cfg_path = Path(run_cfg).expanduser().resolve()
    if cfg_path.is_dir():
        cfg_path = (cfg_path / "epm_config.json").resolve()
    try:
        cfg = load_raw_settings(cfg_path)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return _load_config_file(cfg_path)


def load_epm_paths() -> dict[str, Any]:
    raw = load_epm_config()
    paths = raw.get("paths") or {}
    return paths if isinstance(paths, dict) else {}


def userdata_root() -> Path:
    """
    Resolve CookingSimulator `UserData` directory.

    Priority:
      1) env `COOKGAME_USERDATA_ROOT` / `COOKGAME_USERDATA_PATH`
      2) epm_config.json -> paths.game_userdata_root
      3) epm_config.json -> paths.realtime_products_path parent dir
      4) fallback to a common path in this repo
    """
    env = (os.environ.get("COOKGAME_USERDATA_ROOT") or os.environ.get("COOKGAME_USERDATA_PATH") or "").strip()
    if env:
        return Path(env).expanduser().resolve()

    ud = load_epm_paths().get("game_userdata_root")
    if not (isinstance(ud, str) and ud.strip()):
        # Backward compatibility: old key name.
        ud = load_epm_paths().get("userdata_root")
    if isinstance(ud, str) and ud.strip():
        p = Path(ud.strip())
        if not p.is_absolute():
            p = (repo_root() / p).resolve()
        return p

    rp = load_epm_paths().get("realtime_products_path")
    if isinstance(rp, str) and rp.strip():
        p = Path(rp.strip())
        if not p.is_absolute():
            p = (repo_root() / p).resolve()
        return p.parent

    return Path.home() / "CookingSimulator" / "UserData"


def realtime_products_json() -> Path:
    env = (os.environ.get("COOKGAME_REALTIME_PRODUCTS_JSON") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return userdata_root() / "realtime_products.json"


def hotkeys_status_path() -> Path:
    env = (os.environ.get("COOKGAME_HOTKEYS_STATUS_PATH") or "").strip()
    if env:
        return Path(env).expanduser().resolve()

    value = load_epm_paths().get("hotkeys_status_path")
    if isinstance(value, str) and value.strip():
        p = Path(value.strip())
        if not p.is_absolute():
            p = (repo_root() / p).resolve()
        return p

    return userdata_root() / "realtime_interaction_status.txt"


def realtime_interaction_info_path() -> Path:
    env = (os.environ.get("COOKGAME_REALTIME_INTERACTION_INFO_PATH") or "").strip()
    if env:
        return Path(env).expanduser().resolve()

    value = load_epm_paths().get("realtime_interaction_info_path")
    if isinstance(value, str) and value.strip():
        p = Path(value.strip())
        if not p.is_absolute():
            p = (repo_root() / p).resolve()
        return p

    return userdata_root() / "realtime_interaction_info.txt"


def realtime_ui_info_path() -> Path:
    env = (os.environ.get("COOKGAME_REALTIME_UI_INFO_PATH") or "").strip()
    if env:
        return Path(env).expanduser().resolve()

    value = load_epm_paths().get("realtime_ui_info_path")
    if isinstance(value, str) and value.strip():
        p = Path(value.strip())
        if not p.is_absolute():
            p = (repo_root() / p).resolve()
        return p

    return userdata_root() / "realtime_ui_info.json"


def realtime_radar_scan_path() -> Path:
    rr = load_epm_paths().get("realtime_radar_scan_path")
    if isinstance(rr, str) and rr.strip():
        p = Path(rr.strip())
        if not p.is_absolute():
            p = (repo_root() / p).resolve()
        return p

    return userdata_root() / "realtime_radar_scan.txt"


def camera_info_path() -> Path:
    env = (os.environ.get("COOKGAME_CAMERA_INFO_PATH") or "").strip()
    if env:
        return Path(env).expanduser().resolve()

    ci = load_epm_paths().get("camera_info_path")
    if isinstance(ci, str) and ci.strip():
        p = Path(ci.strip())
        if not p.is_absolute():
            p = (repo_root() / p).resolve()
        return p

    return userdata_root() / "realtime_camera_info.txt"


def window_title(default: str = "CookingSimulator") -> str:
    return (os.environ.get("COOKGAME_WINDOW_TITLE") or default).strip() or default
