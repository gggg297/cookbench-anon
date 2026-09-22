from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from dataclasses import MISSING
from pathlib import Path
from typing import Any, Dict, List

from epm.cerebellum.cookbench_api import ActionAPI, ActionResult
from epm.cerebellum.skills.auto_cutting.skill import CutArgs as AutoCutArgs, run as run_auto_cutting
from epm.cerebellum.skills.auto_flipping.skill import FlipArgs as AutoFlipArgs, run as run_auto_flipping
from epm.cerebellum.skills.auto_mix.skill import MixArgs as AutoMixArgs, run as run_auto_mix
from epm.cerebellum.skills.auto_navigation.skill import NavigateArgs as AutoNavigateArgs, run as run_auto_navigation
from epm.cerebellum.skills.auto_perception.skill import PerceptionArgs, run as run_auto_perception
from epm.cerebellum.skills.auto_pouring.skill import PourArgs as AutoPourArgs, run as run_auto_pouring
from epm.cerebellum.skills.auto_sprinkling.skill import SprinkleArgs as AutoSprinkleArgs, run as run_auto_sprinkling
from epm.cerebellum.skills.query_product_catalog.skill import QueryProductCatalogArgs, run as run_query_product_catalog
from epm.cerebellum.skills.query_scene_objects.skill import QuerySceneObjectsArgs, run as run_query_scene_objects
from epm.cerebellum.skills.query_tool_manual.skill import QueryToolManualArgs, run as run_query_tool_manual


def _maybe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _maybe_bool(v: Any, *, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off", ""}:
            return False
    return bool(v)


def _first_present(args: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in args:
            return args.get(k)
    return None


def _looks_like_multi_value(v: Any) -> bool:
    if isinstance(v, str):
        s = v.strip()
        return (";" in s) or ("," in s)
    if isinstance(v, (list, tuple)):
        return len(v) > 1
    return False


@dataclass(frozen=True)
class GuiBuyNewItemArgs:
    item: str


@dataclass(frozen=True)
class GuiOrderDishViaComputerArgs:
    dish_name: str
    computer_target: str = "Computer"


@dataclass(frozen=True)
class GuiSubmitDishViaCheckoutStandArgs:
    dish_name: str
    checkout_target: str = "Checkout Stand"


SKILL_ARG_CLASSES: dict[str, type] = {
    # Preferred short names (match action names / CLI ergonomics)
    "auto_cut": AutoCutArgs,
    "auto_filp": AutoFlipArgs,  # keep legacy typo for compatibility with existing prompts
    "auto_flip": AutoFlipArgs,
    "auto_mix": AutoMixArgs,
    "auto_pour": AutoPourArgs,
    "auto_sprinkle": AutoSprinkleArgs,

    # Long-form skill names (kept for backward compatibility)
    "auto_cutting": AutoCutArgs,
    "auto_flipping": AutoFlipArgs,
    "auto_navigation": AutoNavigateArgs,
    # auto_perception has a realtime path injected by runtime; we expose only simple toggles.
    "auto_perception": PerceptionArgs,
    "auto_pouring": AutoPourArgs,
    "auto_sprinkling": AutoSprinkleArgs,
    "query_scene_objects": QuerySceneObjectsArgs,
    "query_product_catalog": QueryProductCatalogArgs,
    "query_tool_manual": QueryToolManualArgs,
    "gui_buy_new_item": GuiBuyNewItemArgs,
    "gui_order_dish_via_computer": GuiOrderDishViaComputerArgs,
    "gui_submit_dish_via_checkout_stand": GuiSubmitDishViaCheckoutStandArgs,
}

SKILL_DROP_FIELDS: dict[str, set[str]] = {
    # never ask the model to provide file paths; runtime injects these.
    "auto_perception": {"realtime_products_path"},
}

SKILL_DESCRIPTIONS: dict[str, str] = {
    "auto_cut": (
        "Purpose: Automatically cut an ingredient `item_name` for `cut_num` cuts.\n"
        "When to call: The target ingredient is interactable and the cutting interaction is available.\n"
        "Inputs: `item_name` (str), `cut_num` (int).\n"
        "Behavior: Align/enter cutting -> perform `cut_num` cuts -> exit/cleanup.\n"
        "Side effects: Sends keyboard/mouse events.\n"
        "Returns: ActionResult(success, error, raw)."
    ),
    "auto_cutting": (
        "Purpose: Automatically cut an ingredient `item_name` for `cut_num` cuts.\n"
        "When to call: The target ingredient is interactable and the cutting interaction is available.\n"
        "Inputs: `item_name` (str), `cut_num` (int).\n"
        "Behavior: Align/enter cutting -> perform `cut_num` cuts -> exit/cleanup.\n"
        "Side effects: Sends keyboard/mouse events.\n"
        "Returns: ActionResult(success, error, raw)."
    ),
    "auto_filp": "Purpose: Flip `meat` and place it onto `put_place` (legacy spelling).",
    "auto_flip": "Purpose: Flip `meat` and place it onto `put_place`.",
    "auto_flipping": "Purpose: Flip `meat` and place it onto `put_place` (legacy long-form name).",
    "auto_mix": "Purpose: Mix contents inside a container using the Blender.",
    "auto_navigation": (
        "Purpose: Navigate to world object `target` using A* + radar avoidance.\n"
        "When to call: `target` exists in realtime products and camera/radar inputs are fresh.\n"
        "Inputs: `target` (str), optional `map_file`, `impl`, `auto_activate_window`.\n"
        "Behavior: Resolve target -> plan path -> move with avoidance -> dock + fine-tune pose/view.\n"
        "Side effects: Sends keyboard/mouse events.\n"
        "Returns: ActionResult(success, error, raw) where `raw` includes preflight and nav diagnostics."
    ),
    "auto_perception": (
        "Purpose: Read oracle perception from `realtime_products.json` and return visible items.\n"
        "When to call: The realtime products file has been refreshed (runtime injects the path).\n"
        "Inputs: `only_on_screen` (bool), `max_items` (int).\n"
        "Behavior: Parse file -> filter/sort -> return structured item list.\n"
        "Side effects: None (read-only).\n"
        "Returns: ActionResult(success, error, raw) where `raw` contains the item list."
    ),
    "auto_pour": (
        "Purpose: Pour liquid into `container_name` until reaching `target_ml`.\n"
        "When to call: The container can be identified/aligned and pouring interaction is available.\n"
        "Inputs: `container_name` (str), `target_ml` (float).\n"
        "Behavior: Align container -> pour -> iterate check/adjust -> stop near target.\n"
        "Side effects: Sends keyboard/mouse events.\n"
        "Returns: ActionResult(success, error, raw)."
    ),
    "auto_pouring": (
        "Purpose: Pour liquid into `container_name` until reaching `target_ml`.\n"
        "When to call: The container can be identified/aligned and pouring interaction is available.\n"
        "Inputs: `container_name` (str), `target_ml` (float).\n"
        "Behavior: Align container -> pour -> iterate check/adjust -> stop near target.\n"
        "Side effects: Sends keyboard/mouse events.\n"
        "Returns: ActionResult(success, error, raw)."
    ),
    "auto_sprinkle": (
        "Purpose: Sprinkle onto `target_item` for `sprinkle_num` times.\n"
        "When to call: Target is interactable and a sprinkling tool/action is available.\n"
        "Inputs: `target_item` (str), `sprinkle_num` (int).\n"
        "Behavior: Align target -> perform `sprinkle_num` sprinkles -> exit/cleanup.\n"
        "Side effects: Sends keyboard/mouse events.\n"
        "Returns: ActionResult(success, error, raw)."
    ),
    "auto_sprinkling": (
        "Purpose: Sprinkle onto `target_item` for `sprinkle_num` times.\n"
        "When to call: Target is interactable and a sprinkling tool/action is available.\n"
        "Inputs: `target_item` (str), `sprinkle_num` (int).\n"
        "Behavior: Align target -> perform `sprinkle_num` sprinkles -> exit/cleanup.\n"
        "Side effects: Sends keyboard/mouse events.\n"
        "Returns: ActionResult(success, error, raw)."
    ),
    "query_scene_objects": (
        "Purpose: Query live scene instances from `realtime_products.json` by name substring + visibility + distance.\n"
        "When to call: You need the current in-scene objects, their live `instance_id`, visibility, container, or distance for execution/disambiguation. "
        "Do not use this skill for static catalog/reference metadata.\n"
        "Inputs: `query` (str), `only_on_screen` (bool), `max_items` (int), `max_distance` (float).\n"
        "Behavior: Filter by `query` -> optional on-screen filter -> distance + count truncation.\n"
        "Side effects: None (read-only).\n"
        "Returns: ActionResult(success, error, raw) where `raw` contains the matches."
    ),
    "query_product_catalog": (
        "Purpose: Query static product reference metadata from `epm/data/products_en.json`.\n"
        "When to call: You need catalog/reference info such as initial weight, supported forms, or initial relative position. "
        "Do not use this skill for current scene state, live `instance_id`, or current object position; use `query_scene_objects` for that.\n"
        "Inputs: `query` (str), `item_type` (str), `max_items` (int).\n"
        "Behavior: Read products_en -> filter -> return compact rows with `item_name`, `item_type`, `weight`, `forms`, `relative_position`.\n"
        "Notes: `weight` is only the initial static weight metadata; it does not change with later in-game operations. "
        "`relative_position` is only the initial static placement description; if the item moves later, trust later realtime observations instead. "
        "`forms` lists the supported form variations the item can change into.\n"
        "Side effects: None (read-only).\n"
        "Returns: ActionResult(success, error, raw) where `raw.results` contains the filtered product metadata rows."
    ),
    "query_tool_manual": (
        "Purpose: Kitchen tool manual lookup. Query `epm/data/tool_en.json` by tool name/type/function.\n"
        "When to call: You need cookware usage/specifications without injecting large tool catalogs into prompt.\n"
        "Inputs: `query` (str), `tool_type` (str), `container_type` (str), `tool_function` (str), `max_items` (int).\n"
        "Behavior: Read tool_en -> filter -> return compact rows with key fields.\n"
        "Side effects: None (read-only).\n"
        "Returns: ActionResult(success, error, raw) where `raw.results` is the filtered tool manual rows."
    ),
    "gui_buy_new_item": (
        "购买物品技能：空手状态下，自动导航至 Carton Box 并打开商店 UI，自动购买指定物品。"
        "购买成功后，智能体会手持该物品；再次购买前需先放置物品以保持空手，否则可能失败。"
    ),
    "gui_order_dish_via_computer": (
        "下订单技能：空手状态下，自动导航至 Computer 并打开电脑 UI，自动下单指定菜品。"
        "下单完成后智能体应保持空手；再次下单前需确保空手，否则可能失败。"
    ),
    "gui_submit_dish_via_checkout_stand": (
        "上菜评价技能：手持菜品状态下，自动导航至 Checkout Stand 并打开上菜 UI，提交指定菜品并读取反馈/评分。"
        "适合在任务结束时调用；提交前需确保手持正确菜品，否则可能出现反馈菜名不一致。"
    ),
}


def list_skills() -> list[str]:
    return [
        "auto_cut",
        "auto_filp",
        "auto_mix",
        "auto_navigation",
        "auto_perception",
        "auto_pour",
        "auto_sprinkle",
        "query_scene_objects",
        "query_product_catalog",
        "query_tool_manual",
        "gui_buy_new_item",
        "gui_order_dish_via_computer",
        "gui_submit_dish_via_checkout_stand",
    ]


def run_skill(api: ActionAPI, *, name: str, args: Dict[str, Any], realtime_products_path: Path) -> ActionResult:
    if name in {"auto_cut", "auto_cutting"}:
        item = str(args.get("item_name") or args.get("object") or args.get("target") or "").strip()
        item_instance_id = _maybe_int(_first_present(args, "item_instance_id", "object_instance_id"))
        cut_num = int(args.get("cut_num", args.get("cuts", args.get("cut_count", 3))))
        cut_args = AutoCutArgs(item_name=item, item_instance_id=item_instance_id, cut_num=cut_num)
        return run_auto_cutting(api, cut_args)

    if name in {"auto_flipping", "auto_flip", "auto_filp"}:
        meat = str(args.get("meat_name") or args.get("meat") or "").strip()
        put_place = str(args.get("put_place_name") or args.get("put_place") or "").strip()
        flip_args = AutoFlipArgs(
            meat=meat,
            put_place=put_place,
            meat_instance_id=_maybe_int(args.get("meat_instance_id")),
            put_place_instance_id=_maybe_int(args.get("put_place_instance_id")),
        )
        return run_auto_flipping(api, flip_args)

    if name == "auto_mix":
        container = str(args.get("container_name") or args.get("container") or args.get("target") or "").strip()
        mix_args = AutoMixArgs(container_name=container, container_instance_id=_maybe_int(args.get("container_instance_id")))
        return run_auto_mix(api, mix_args)

    if name == "auto_navigation":
        nav_args = AutoNavigateArgs(
            target=str(args.get("target", "")).strip(),
            target_instance_id=_maybe_int(args.get("target_instance_id")),
            map_file=(str(args.get("map_file")).strip() if args.get("map_file") is not None else None),
            auto_activate_window=_maybe_bool(args.get("auto_activate_window", True), default=True),
            impl=str(args.get("impl", "simple_astar_radar")).strip() or "simple_astar_radar",
            docking_distance_m=float(args.get("docking_distance_m", 1.5)),
            docking_distance_tolerance_m=float(args.get("docking_distance_tolerance_m", 0.2)),
            internal_rect_strategy=str(args.get("internal_rect_strategy", "normalized_ratio")).strip() or "normalized_ratio",
        )
        return run_auto_navigation(api, nav_args)

    if name == "auto_perception":
        p_args = PerceptionArgs(
            realtime_products_path=realtime_products_path,
            only_on_screen=_maybe_bool(args.get("only_on_screen", True), default=True),
            max_items=int(args.get("max_items", 50)),
        )
        return run_auto_perception(p_args)

    if name in {"auto_pouring", "auto_pour"}:
        container = str(args.get("container_name") or args.get("container") or args.get("target") or "").strip()
        target_raw = args.get("target_ml", args.get("pour_ml", 50.0))
        if target_raw in (None, ""):
            target_raw = 50.0
        container_iid_raw = args.get("container_instance_id")
        if _looks_like_multi_value(container) or _looks_like_multi_value(container_iid_raw) or _looks_like_multi_value(target_raw):
            return ActionResult(
                False,
                raw={
                    "container_name": container,
                    "container_instance_id": container_iid_raw,
                    "target_ml": target_raw,
                },
                error=(
                    "invalid_auto_pour_args:multi_value_not_supported; "
                    "auto_pour expects exactly one held liquid, one target container, and one scalar target_ml"
                ),
            )
        try:
            target_ml = float(target_raw)
        except Exception:
            return ActionResult(
                False,
                raw={
                    "container_name": container,
                    "container_instance_id": container_iid_raw,
                    "target_ml": target_raw,
                },
                error=f"invalid_auto_pour_args:target_ml_not_float:{target_raw!r}",
            )
        pour_args = AutoPourArgs(
            container_name=container,
            container_instance_id=_maybe_int(container_iid_raw),
            target_ml=target_ml,
        )
        return run_auto_pouring(api, pour_args)

    if name in {"auto_sprinkling", "auto_sprinkle"}:
        target_item = str(args.get("target_item") or args.get("container") or args.get("target") or "").strip()
        sprinkle_num = int(args.get("sprinkle_num", args.get("num", 3)))
        sprinkle_args = AutoSprinkleArgs(
            target_item=target_item,
            target_instance_id=_maybe_int(args.get("target_instance_id")),
            sprinkle_num=sprinkle_num,
        )
        return run_auto_sprinkling(api, sprinkle_args)

    if name == "query_scene_objects":
        q_args = QuerySceneObjectsArgs(
            query=str(args.get("query", "")).strip(),
            only_on_screen=_maybe_bool(args.get("only_on_screen", False), default=False),
            # Defaults: no item limit; no distance limit. (Caller can constrain explicitly.)
            max_items=int(args.get("max_items", -1)),
            max_distance=float(args.get("max_distance", 0.0)),
        )
        return run_query_scene_objects(realtime_products_path=realtime_products_path, args=q_args)

    if name == "query_product_catalog":
        q_args = QueryProductCatalogArgs(
            query=str(args.get("query", "")).strip(),
            item_type=str(args.get("item_type", "")).strip(),
            max_items=int(args.get("max_items", 20)),
        )
        return run_query_product_catalog(q_args)

    if name == "query_tool_manual":
        q_args = QueryToolManualArgs(
            query=str(args.get("query", "")).strip(),
            tool_type=str(args.get("tool_type", "")).strip(),
            container_type=str(args.get("container_type", "")).strip(),
            tool_function=str(args.get("tool_function", "")).strip(),
            max_items=int(args.get("max_items", 20)),
        )
        return run_query_tool_manual(q_args)

    if name == "gui_buy_new_item":
        item = str(args.get("item") or "").strip()
        return api.call_action("gui_buy_new_item", item=item)

    if name == "gui_order_dish_via_computer":
        dish_name = str(args.get("dish_name") or "").strip()
        computer_target = str(args.get("computer_target") or "Computer").strip() or "Computer"
        return api.call_action("gui_order_dish_via_computer", dish_name=dish_name, computer_target=computer_target)

    if name == "gui_submit_dish_via_checkout_stand":
        dish_name = str(args.get("dish_name") or "").strip()
        checkout_target = str(args.get("checkout_target") or "Checkout Stand").strip() or "Checkout Stand"
        debug_submit_only = _maybe_bool(args.get("_debug_submit_only", False), default=False)
        debug_skip_holding_check = _maybe_bool(args.get("_debug_skip_holding_check", False), default=False)
        return api.call_action(
            "gui_submit_dish_via_checkout_stand",
            dish_name=dish_name,
            checkout_target=checkout_target,
            _debug_submit_only=debug_submit_only,
            _debug_skip_holding_check=debug_skip_holding_check,
        )

    return ActionResult(False, raw={}, error=f"unknown_skill:{name}")


def skill_specs_to_prompt_text() -> str:
    """
    Render builtin skills (type=skill) as an interface reference.

    This is separate from `skills_catalog.json`, which is used as "skill cards" for grouping actions.
    """

    def _sig(cls: type) -> str:
        parts: list[str] = []
        for f in dataclass_fields(cls):
            t = getattr(f.type, "__name__", str(f.type))
            if f.default is not MISSING:
                parts.append(f"{f.name}: {t} = {f.default!r}")
            elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
                parts.append(f"{f.name}: {t} = <factory>")
            else:
                parts.append(f"{f.name}: {t}")
        return "(" + ", ".join(parts) + ")"

    lines: list[str] = []
    lines.append("Builtin Skills (call with type=skill):")
    lines.append("")
    lines.append(f"- auto_cutting{_sig(AutoCutArgs)}")
    lines.append(f"- auto_navigation{_sig(AutoNavigateArgs)}")
    lines.append("- auto_perception(only_on_screen: bool = true, max_items: int = 50)  # reads realtime_products.json (path injected by runtime)")
    lines.append(f"- auto_pouring{_sig(AutoPourArgs)}")
    lines.append(f"- auto_sprinkling{_sig(AutoSprinkleArgs)}")
    lines.append(f"- query_scene_objects{_sig(QuerySceneObjectsArgs)}  # reads realtime_products.json")
    lines.append(f"- query_product_catalog{_sig(QueryProductCatalogArgs)}  # static product metadata lookup from epm/data/products_en.json")
    lines.append(f"- query_tool_manual{_sig(QueryToolManualArgs)}  # kitchen tool manual lookup from epm/data/tool_en.json")
    return "\n".join(lines).strip() + "\n"
