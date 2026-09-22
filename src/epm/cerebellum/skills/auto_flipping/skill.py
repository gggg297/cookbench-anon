from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from epm.cerebellum.cookbench_api import ActionAPI, ActionResult
from epm.cerebellum.skills._instance_resolution import resolve_instance_id
from epm.cerebellum.skills._shared_paths import realtime_products_json


@dataclass(frozen=True)
class FlipArgs:
    meat: str
    put_place: str
    meat_instance_id: int | None = None
    put_place_instance_id: int | None = None


def _put_place_list_path() -> Path:
    # <repo>/epm/src/epm/cerebellum/skills/auto_flipping/skill.py -> parents[5] == <repo>/epm
    return Path(__file__).resolve().parents[5] / "data" / "put_place_list.json"


def _load_put_place_platforms() -> list[dict]:
    p = _put_place_list_path()
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8-sig"))
        plats = raw.get("platforms")
        if isinstance(plats, list):
            return [x for x in plats if isinstance(x, dict)]
    except Exception:
        return []
    return []

_PLATFORM_SUFFIX_RE = re.compile(r".*-\d+$")


def _try_expand_platform_point_name(*, name: str, index: int | None) -> tuple[str | None, str | None]:
    """
    Compatibility behavior for put_place/target:
    - If `name` is a platform category (e.g. "Side Table") and `index` is provided (e.g. 3),
      expand to "Side Table-3" using epm/data/put_place_list.json.

    Returns (expanded_name, error_reason).
    """
    n = (name or "").strip()
    # Tolerate accidental trailing '-' or whitespace, e.g. "Griddle Surface-".
    n = n.rstrip().rstrip("-").rstrip()
    if not n or _PLATFORM_SUFFIX_RE.match(n):
        return None, None
    if index is None:
        return None, None

    platforms = _load_put_place_platforms()
    if not platforms:
        return None, "put_place_list_missing"

    want = n.lower()
    hit = None
    for p in platforms:
        cat = str(p.get("category") or "").strip()
        if cat.lower() == want:
            hit = p
            break
    if hit is None:
        return None, None

    tmpl = str(hit.get("name_template") or "").strip()
    if "{index}" not in tmpl:
        return None, "put_place_template_invalid"

    # Validate index range if available.
    ranges = hit.get("index_ranges")
    ok = False
    if isinstance(ranges, list) and ranges:
        for rr in ranges:
            if isinstance(rr, list) and len(rr) == 2:
                try:
                    lo, hi = int(rr[0]), int(rr[1])
                except Exception:
                    continue
                if lo <= int(index) <= hi:
                    ok = True
                    break
        if not ok:
            return None, "platform_index_out_of_range"

    return tmpl.format(index=int(index)), None


def run(api: ActionAPI, args: FlipArgs) -> ActionResult:
    """
    High-level flipping skill wrapper.

    Delegates to the consolidated action/tool interface:
      auto_flip(meat="<meat>", put_place="<put_place>")
    """

    meat_resolved, err = resolve_instance_id(
        realtime_products_path=realtime_products_json(),
        name=str(args.meat),
        instance_id=(int(args.meat_instance_id) if args.meat_instance_id is not None else None),
        action="auto_filp",
        name_arg="meat",
        instance_arg="meat_instance_id",
    )
    if err is not None:
        return err

    # Compatibility: for platform points, allow passing the platform index via put_place_instance_id
    # when put_place is given as a category without "-<index>" suffix.
    put_place_name = str(args.put_place or "").strip()
    expanded, expand_err = _try_expand_platform_point_name(name=put_place_name, index=args.put_place_instance_id)
    if expanded:
        put_place_name = expanded
        put_place_instance_id: int | None = None  # let resolver auto-fill from the unique expanded name
    else:
        put_place_instance_id = (int(args.put_place_instance_id) if args.put_place_instance_id is not None else None)
        if expand_err in {"put_place_list_missing", "put_place_template_invalid", "platform_index_out_of_range"}:
            raw = {
                "action": "auto_filp",
                "name": str(args.put_place),
                "put_place": str(args.put_place),
                "put_place_instance_id": args.put_place_instance_id,
                "put_place_platforms": _load_put_place_platforms(),
                "reason": expand_err,
                "selection_prompt": (
                    "Invalid platform-point input. For placement targets, `put_place` should be a platform category "
                    "(e.g. 'Side Table') and `put_place_instance_id` should be the platform index (e.g. 3), "
                    "which will be expanded to 'Side Table-3'. Ensure the index is within the listed ranges."
                ),
            }
            return ActionResult(False, raw=raw, error="platform_point_index_invalid")

    place_resolved, err = resolve_instance_id(
        realtime_products_path=realtime_products_json(),
        name=str(put_place_name),
        instance_id=put_place_instance_id,
        action="auto_filp",
        name_arg="put_place",
        instance_arg="put_place_instance_id",
        kind_required="platform_point",
    )
    if err is not None:
        raw = dict(err.raw or {})
        raw["put_place_kind_required"] = "platform_point"
        raw["put_place_platforms"] = _load_put_place_platforms()
        # Strengthen prompt: make it explicit that put_place must be a platform point.
        sel = raw.get("selection_prompt")
        extra = (
            "NOTE: `put_place` must refer to a placement platform point in realtime_products.json "
            "with kind='platform_point'. The trailing number in names like 'Side Table-3' is an index, "
            "NOT the instance_id.\n"
            "Compatibility: you may pass `put_place='Side Table'` and `put_place_instance_id=3` (index), "
            "and the skill will expand it to `put_place='Side Table-3'` automatically.\n"
            "If multiple candidates still exist, then choose `put_place_instance_id` the same way as normal items: "
            "pick a concrete `instance_id` from candidates.\n"
            "For reference, `put_place_platforms` summarizes platform categories and index ranges "
            "(from epm/data/put_place_list.json)."
        )
        if isinstance(sel, str) and sel.strip():
            raw["selection_prompt"] = sel.rstrip() + "\n\n" + extra
        else:
            raw["selection_prompt"] = extra
        return ActionResult(False, raw=raw, error=str(err.error or ""))

    return api.call_action(
        "auto_flip",
        meat=str(args.meat),
        put_place=str(put_place_name),
        meat_instance_id=(meat_resolved.instance_id if meat_resolved is not None else None),
        put_place_instance_id=(place_resolved.instance_id if place_resolved is not None else None),
    )
