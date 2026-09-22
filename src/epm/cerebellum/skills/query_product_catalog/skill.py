from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epm.cerebellum.cookbench_api import ActionResult
from epm.cerebellum.skills._shared_paths import repo_root


@dataclass(frozen=True)
class QueryProductCatalogArgs:
    # Product metadata lookup: search static product info from products_en.json.
    query: str = ""
    item_type: str = ""
    max_items: int = 20


def _products_en_path() -> Path:
    return repo_root() / "data" / "products_en.json"


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _contains(hay: Any, needle: str) -> bool:
    if not needle:
        return True
    if isinstance(hay, list):
        return any(needle in _norm(x) for x in hay)
    return needle in _norm(hay)


def run(args: QueryProductCatalogArgs) -> ActionResult:
    path = _products_en_path()
    if not path.exists():
        return ActionResult(False, raw={}, error=f"products_en_not_found:{path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return ActionResult(False, raw={}, error=f"products_en_parse_failed:{e}")
    if not isinstance(obj, list):
        return ActionResult(False, raw={}, error="products_en_invalid_schema:not_list")

    q = _norm(args.query)
    item_type = _norm(args.item_type)
    max_items = int(args.max_items)

    keep_keys = (
        "item_name",
        "item_type",
        "weight",
        "forms",
        "relative_position",
    )

    rows: list[dict[str, Any]] = []
    for it in obj:
        if not isinstance(it, dict):
            continue
        if q and not any(
            _contains(it.get(k), q)
            for k in ("item_name", "item_type", "forms", "relative_position", "action2API")
        ):
            continue
        if item_type and not _contains(it.get("item_type"), item_type):
            continue
        rows.append({k: it.get(k) for k in keep_keys if k in it})

    total = len(rows)
    if max_items >= 0:
        rows = rows[:max_items]
    return ActionResult(
        True,
        raw={
            "query": args.query,
            "item_type": args.item_type,
            "total": total,
            "results": rows,
            "notes": {
                "weight": "Static initial weight metadata from products_en.json. It does not change with later in-game operations.",
                "relative_position": "Static initial placement/spawn description from products_en.json. If the item moves later, trust later realtime observations instead.",
                "forms": "Supported form variations the item can change into.",
            },
        },
    )
