from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from epm.cerebellum.cookbench_api import ActionResult
from epm.cerebellum.game_hotkeys import ensure_f12_products_scan
from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.realtime_products import extract_items as extract_realtime_items
from epm.cerebellum.skills._shared_paths import window_title
from epm.vision.screen_capture import activate_window


@dataclass(frozen=True)
class QuerySceneObjectsArgs:
    query: str = ""
    only_on_screen: bool = False
    # -1 means "no limit"
    max_items: int = -1
    # <=0 means "no distance limit"
    max_distance: float = -1.0


def _read_realtime(path: Path) -> Dict[str, Any]:
    # Accept UTF-8 with BOM (common on Windows when written by external tools/mods).
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, dict) else {"objects": data}


def _extract_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(extract_realtime_items(data))


def _name_of(item: Dict[str, Any]) -> str:
    for k in ("name_en", "name", "name_cn", "label"):
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "unknown"


def _parse_query_terms(query: str) -> List[str]:
    raw = str(query or "").strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            seen: set[str] = set()
            out: list[str] = []
            for it in obj:
                txt = str(it or "").strip()
                low = txt.lower()
                if not txt or low in seen:
                    continue
                seen.add(low)
                out.append(txt)
            if out:
                return out
    except Exception:
        pass

    parts = [p.strip() for p in re.split(r"[;,\n]+", raw) if p.strip()]
    if not parts:
        parts = [raw]
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        low = part.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(part)
    return out


def _query_single(
    *,
    items: List[Dict[str, Any]],
    query: str,
    only_on_screen: bool,
    max_items: int,
    max_distance: float,
) -> List[Dict[str, Any]]:
    q = str(query or "").strip().lower()
    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if only_on_screen:
            on_screen = item.get("is_on_screen")
            if isinstance(on_screen, bool) and not on_screen:
                continue
        name = _name_of(item)
        if q and q not in name.lower():
            continue
        dist = item.get("distance", None)
        if isinstance(dist, (int, float)):
            if max_distance > 0 and float(dist) > max_distance:
                continue
        out.append(item)

    def _dist_key(it: Dict[str, Any]) -> float:
        d = it.get("distance", None)
        try:
            return float(d) if d is not None else 9999.0
        except Exception:
            return 9999.0

    out.sort(key=_dist_key)
    if max_items >= 0:
        return out[:max_items]
    return out


def query_scene_objects(*, realtime_products_path: Path, args: QuerySceneObjectsArgs) -> Dict[str, Any]:
    """
    Query realtime_products.json for items, focusing on:
    - is_on_screen == True (optional)
    - distance <= max_distance (optional; <=0 disables)
    - name contains `query` (case-insensitive; optional)

    Returns:
      A list of raw item dicts from realtime_products.json (no wrapping).
    """
    data = _read_realtime(realtime_products_path)
    items = _extract_items(data)

    only_on_screen = bool(args.only_on_screen)
    max_items = int(args.max_items)
    max_distance = float(args.max_distance)
    queries = _parse_query_terms(args.query)
    if not queries:
        queries = [""]

    results_by_query: Dict[str, List[Dict[str, Any]]] = {}
    merged: List[Dict[str, Any]] = []
    seen_ids: set[Any] = set()
    seen_fallback: set[str] = set()
    for q in queries:
        cur = _query_single(
            items=items,
            query=q,
            only_on_screen=only_on_screen,
            max_items=max_items,
            max_distance=max_distance,
        )
        results_by_query[q] = cur
        for item in cur:
            instance_id = item.get("instance_id")
            if instance_id is not None:
                key = ("id", instance_id)
                if key in seen_ids:
                    continue
                seen_ids.add(key)
            else:
                fallback = json.dumps(
                    {
                        "name": _name_of(item),
                        "distance": item.get("distance"),
                        "container": item.get("container"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if fallback in seen_fallback:
                    continue
                seen_fallback.add(fallback)
            merged.append(item)

    return {
        "queries": queries,
        "results": merged,
        "results_by_query": results_by_query,
    }


def run(*, realtime_products_path: Path, args: QuerySceneObjectsArgs) -> ActionResult:
    try:
        # Best-effort: ensure F12 scan is fresh before reading realtime_products.json.
        try:
            io = RawInputController()
            f12_ok = ensure_f12_products_scan(
                realtime_products_path=realtime_products_path,
                window_title=window_title(),
                activate_window=activate_window,
                io_controller=io,
                verbose=True,
            )
        except Exception:
            f12_ok = None
        query_payload = query_scene_objects(realtime_products_path=realtime_products_path, args=args)
        raw = {
            "query": args.query,
            "queries": query_payload.get("queries", []),
            "results": query_payload.get("results", []),
            "results_by_query": query_payload.get("results_by_query", {}),
        }
        if f12_ok is not None:
            raw["preflight"] = {"f12_ok": bool(f12_ok)}
        return ActionResult(True, raw=raw)
    except Exception as e:
        return ActionResult(False, raw={}, error=str(e))
