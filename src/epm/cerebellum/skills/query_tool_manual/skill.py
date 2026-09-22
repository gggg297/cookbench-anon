from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epm.cerebellum.cookbench_api import ActionResult
from epm.cerebellum.skills._shared_paths import repo_root


@dataclass(frozen=True)
class QueryToolManualArgs:
    # Kitchen tool manual query: search tool usage/specs from tool_en.json.
    query: str = ""
    tool_type: str = ""
    container_type: str = ""
    tool_function: str = ""
    max_items: int = 20


def _tool_en_path() -> Path:
    return repo_root() / "data" / "tool_en.json"


def _norm(v: Any) -> str:
    return str(v or "").strip().lower()


def _contains(hay: Any, needle: str) -> bool:
    if not needle:
        return True
    return needle in _norm(hay)


def run(args: QueryToolManualArgs) -> ActionResult:
    path = _tool_en_path()
    if not path.exists():
        return ActionResult(False, raw={}, error=f"tool_en_not_found:{path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as e:
        return ActionResult(False, raw={}, error=f"tool_en_parse_failed:{e}")
    if not isinstance(obj, list):
        return ActionResult(False, raw={}, error="tool_en_invalid_schema:not_list")

    q = _norm(args.query)
    ttype = _norm(args.tool_type)
    ctype = _norm(args.container_type)
    tfunc = _norm(args.tool_function)
    max_items = int(args.max_items)

    keep_keys = (
        "tool_name",
        "tool_type",
        "container_type",
        "tool_function",
        "capacity",
        "supported_operation_states",
    )

    rows: list[dict[str, Any]] = []
    for it in obj:
        if not isinstance(it, dict):
            continue
        if q and not any(_contains(it.get(k), q) for k in ("tool_name", "tool_type", "container_type", "tool_function")):
            continue
        if ttype and not _contains(it.get("tool_type"), ttype):
            continue
        if ctype and not _contains(it.get("container_type"), ctype):
            continue
        if tfunc and not _contains(it.get("tool_function"), tfunc):
            continue
        rows.append({k: it.get(k) for k in keep_keys if k in it})

    total = len(rows)
    if max_items >= 0:
        rows = rows[:max_items]
    return ActionResult(
        True,
        raw={
            "query": args.query,
            "tool_type": args.tool_type,
            "container_type": args.container_type,
            "tool_function": args.tool_function,
            "total": total,
            "results": rows,
        },
    )

