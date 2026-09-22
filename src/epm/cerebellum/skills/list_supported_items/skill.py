from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from epm.cerebellum.cookbench_api import ActionResult
from epm.cerebellum.skills._instance_resolution import _load_object_mapping, _mapping_path


@dataclass(frozen=True)
class ListSupportedItemsArgs:
    """
    List environment-supported item names (NOT limited to what currently exists in the scene).

    Typical usage:
      - list_supported_items()
      - list_supported_items(query="steak")

    Notes:
      - This skill is read-only: it does NOT send any keyboard/mouse events.
      - To list *scene instances* (with concrete `instance_id`s), use:
          query_scene_objects(query="lemon")
    """

    query: str = ""
    include_mapping: bool = True
    include_put_place: bool = True
    include_tool_points: bool = True


def _load_put_place_list(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"missing:{path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception as e:
        return [], f"unreadable:{e}"
    if not isinstance(data, dict):
        return [], "invalid_json_root"
    platforms = data.get("platforms")
    if not isinstance(platforms, list):
        return [], None
    out: list[dict[str, Any]] = []
    for p in platforms:
        if isinstance(p, dict):
            out.append(p)
    return out, None


def _load_tool_interaction_point_list(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], f"missing:{path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
    except Exception as e:
        return [], f"unreadable:{e}"
    if not isinstance(data, dict):
        return [], "invalid_json_root"
    pts = data.get("tool_points")
    if not isinstance(pts, list):
        return [], None
    out: list[dict[str, Any]] = []
    for p in pts:
        if isinstance(p, dict):
            out.append(p)
    return out, None


def run(*, args: ListSupportedItemsArgs) -> ActionResult:
    q = (args.query or "").strip().lower()
    raw: Dict[str, Any] = {"query": args.query}

    if args.include_mapping:
        entries, mapping_error = _load_object_mapping()
        raw["mapping_path"] = str(_mapping_path())
        raw["mapping_error"] = mapping_error

        items: List[Dict[str, Any]] = []
        for e in entries:
            item = {
                "object_id": e.object_id,
                "name_en": e.name_en,
                "name_cn": e.name_cn,
                "category": e.category,
            }
            if q:
                hay = " ".join(
                    [
                        str(item.get("name_en") or "").lower(),
                        str(item.get("name_cn") or "").lower(),
                        str(item.get("category") or "").lower(),
                        str(item.get("object_id") or ""),
                    ]
                )
                if q not in hay:
                    continue
            items.append(item)

        items.sort(key=lambda it: (str(it.get("category") or ""), str(it.get("name_en") or "")))
        raw["items"] = items

    if bool(args.include_put_place):
        put_place_path = Path(__file__).resolve().parents[5] / "epm" / "data" / "put_place_list.json"
        platforms, err = _load_put_place_list(put_place_path)
        raw["put_place_list_path"] = str(put_place_path)
        raw["put_place_list_error"] = err
        if q:
            filtered: list[dict[str, Any]] = []
            for p in platforms:
                cat = str(p.get("category") or "")
                tmpl = str(p.get("name_template") or "")
                if q in cat.lower() or q in tmpl.lower():
                    filtered.append(p)
            raw["platforms"] = filtered
        else:
            raw["platforms"] = platforms

    if args.include_tool_points:
        tool_points_path = Path(__file__).resolve().parents[5] / "epm" / "data" / "tool_interaction_point_list.json"
        tool_points, err = _load_tool_interaction_point_list(tool_points_path)
        raw["tool_interaction_point_list_path"] = str(tool_points_path)
        raw["tool_interaction_point_list_error"] = err
        if q:
            filtered: list[dict[str, Any]] = []
            for p in tool_points:
                cat = str(p.get("category") or "")
                tmpl = str(p.get("name_template") or "")
                if q in cat.lower() or q in tmpl.lower():
                    filtered.append(p)
            raw["tool_points"] = filtered
        else:
            raw["tool_points"] = tool_points

    return ActionResult(True, raw=raw)
