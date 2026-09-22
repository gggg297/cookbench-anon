from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from epm.cerebellum.cookbench_api import ActionResult
from epm.cerebellum.realtime_products import extract_items as extract_realtime_items, read_realtime_products


@dataclass(frozen=True)
class PerceptionArgs:
    realtime_products_path: Path
    only_on_screen: bool = True
    max_items: int = 50


def _read_realtime(path: Path) -> Dict[str, Any]:
    """
    Read realtime snapshot with short polling to bridge producer refresh gaps.
    - max wait: ~1.0s (20 * 0.05s)
    - return as soon as a valid payload is read
    """
    data = read_realtime_products(path, retries=20, sleep_s=0.05)
    return data if isinstance(data, dict) else {"objects": data}


def _extract_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(extract_realtime_items(data))


def _name_of(item: Dict[str, Any]) -> str:
    for k in ("name_en", "name", "name_cn", "label"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def list_visible_items(args: PerceptionArgs) -> List[Dict[str, Any]]:
    """
    "透视观察"：从 realtime_products.json 中列出视野内物品（优先用 is_on_screen）。
    """
    data = _read_realtime(args.realtime_products_path)
    items = _extract_items(data)

    visible: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        on_screen = item.get("is_on_screen")
        if args.only_on_screen and isinstance(on_screen, bool) and not on_screen:
            continue
        visible.append(item)

    visible.sort(key=lambda x: float(x.get("distance", 999.0)) if isinstance(x.get("distance", 999.0), (int, float)) else 999.0)
    max_items = int(args.max_items) if args.max_items is not None else 0
    if max_items <= 0:
        sliced = visible
    else:
        sliced = visible[:max_items]
    out = []
    for item in sliced:
        pos = item.get("position") if isinstance(item.get("position"), dict) else {}
        out.append(
            {
                "name": _name_of(item),
                "name_en": item.get("name_en", None),
                "kind": item.get("kind", None),
                "name_cn": item.get("name_cn", None),
                "distance": item.get("distance", None),
                "is_on_screen": item.get("is_on_screen", None),
                "is_held": item.get("is_held", None),
                "position": {"x": pos.get("x"), "y": pos.get("y"), "z": pos.get("z")},
                "screen_x": item.get("screen_x", None),
                "screen_y": item.get("screen_y", None),
                "container": item.get("container", None),
                "instance_id": item.get("instance_id", None),
            }
        )
    return out


def run(args: PerceptionArgs) -> ActionResult:
    """
    Returns an ActionResult payload for uniform logging.
    """
    try:
        items = list_visible_items(args)
        return ActionResult(True, raw={"visible_items": items})
    except Exception as e:
        return ActionResult(False, raw={}, error=str(e))
