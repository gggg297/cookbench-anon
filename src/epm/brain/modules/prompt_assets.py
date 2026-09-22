from __future__ import annotations

from pathlib import Path


def asset_root(asset_group: str) -> Path:
    group = str(asset_group or "").strip().replace("\\", "/").strip("/")
    root = Path(__file__).resolve().parents[4] / "memory"
    if not group:
        return root
    path = root
    for part in group.split("/"):
        part = str(part or "").strip()
        if part:
            path = path / part
    return path


def load_asset(asset_group: str, asset_name: str, fallback: str) -> str:
    path = asset_root(asset_group) / asset_name
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
    except Exception:
        pass
    return str(fallback or "").strip()
