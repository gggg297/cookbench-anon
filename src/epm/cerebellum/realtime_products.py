from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


def read_realtime_products(path: Path, *, retries: int = 5, sleep_s: float = 0.05) -> dict[str, Any]:
    """
    Read `realtime_products.json` with Windows-friendly tolerance:
    - UTF-8 BOM (`utf-8-sig`)
    - partial writes (retry on JSONDecodeError)
    """
    last: Optional[Exception] = None
    for _ in range(max(1, int(retries))):
        try:
            raw = path.read_text(encoding="utf-8-sig")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {"objects": data}
        except json.JSONDecodeError as e:
            last = e
            time.sleep(float(sleep_s))
        except Exception as e:
            last = e
            break
    raise RuntimeError(f"failed_to_read_realtime_products:{path}: {last}")


def extract_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(data.get("products"), list):
        return [_normalize_item(p) for p in (data.get("products") or []) if isinstance(p, dict)]
    if isinstance(data.get("objects"), list):
        return [_normalize_item(p) for p in (data.get("objects") or []) if isinstance(p, dict)]
    return []


def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    name_en = str(item.get("name_en") or item.get("name") or "").strip().lower()
    name_cn = str(item.get("name_cn") or "").strip()
    if ("food processor container" in name_en) or ("料理机容器" in name_cn) or ("食物处理机容器" in name_cn):
        out = dict(item)
        out["kind"] = "container"
        return out
    return item


def _norm(s: Any) -> str:
    return (str(s or "")).strip().lower()


def item_display_name(item: dict[str, Any]) -> str:
    for k in ("name_en", "name_cn", "name", "label", "game_object"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def best_match_by_name(items: list[dict[str, Any]], target_name: str) -> Optional[dict[str, Any]]:
    """
    Find the best match for `target_name` using exact name matching (case-insensitive)
    on common keys: name_en/name_cn/game_object.

    Preference:
    1) on-screen items
    2) smaller distance (if available)
    """
    want = _norm(target_name)
    if not want:
        return None

    def _score_pick(candidate: dict[str, Any], current: Optional[dict[str, Any]], current_dist: Optional[float]) -> tuple[Optional[dict[str, Any]], Optional[float]]:
        is_on_screen = bool(candidate.get("is_on_screen", False))
        dist_raw = candidate.get("distance", None)
        dist: Optional[float]
        try:
            dist = float(dist_raw) if dist_raw is not None else None
        except Exception:
            dist = None

        if current is None:
            return candidate, dist

        if is_on_screen and not bool(current.get("is_on_screen", False)):
            return candidate, dist
        if bool(current.get("is_on_screen", False)) and not is_on_screen:
            return current, current_dist

        if dist is not None and (current_dist is None or dist < current_dist):
            return candidate, dist
        return current, current_dist

    # Pass 1: exact match.
    best: Optional[dict[str, Any]] = None
    best_dist: Optional[float] = None
    for it in items:
        keys = (it.get("name_en"), it.get("name_cn"), it.get("game_object"), it.get("name"))
        if not any(_norm(k) == want for k in keys):
            continue
        best, best_dist = _score_pick(it, best, best_dist)
    if best is not None:
        return best

    # Pass 2: substring match (useful when planners use aliases like "pot").
    for it in items:
        keys = (it.get("name_en"), it.get("name_cn"), it.get("game_object"), it.get("name"))
        if not any(want in _norm(k) for k in keys):
            continue
        best, best_dist = _score_pick(it, best, best_dist)

    return best


def any_mode(items: list[dict[str, Any]], *, field: str) -> bool:
    return any(bool(it.get(field, False)) for it in items)
