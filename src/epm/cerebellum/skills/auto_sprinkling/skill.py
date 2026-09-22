from __future__ import annotations

from dataclasses import dataclass

from epm.cerebellum.cookbench_api import ActionAPI, ActionResult
from epm.cerebellum.skills._instance_resolution import resolve_instance_id
from epm.cerebellum.skills._shared_paths import realtime_products_json


@dataclass(frozen=True)
class SprinkleArgs:
    target_item: str
    target_instance_id: int | None = None
    sprinkle_num: int = 3


def run(api: ActionAPI, args: SprinkleArgs) -> ActionResult:
    """
    High-level sprinkling skill wrapper.

    Delegates to the consolidated action/tool interface:
      auto_sprinkle(target="<target_item>", sprinkle_num=<sprinkle_num>)
    """

    resolved, err = resolve_instance_id(
        realtime_products_path=realtime_products_json(),
        name=str(args.target_item),
        instance_id=(int(args.target_instance_id) if args.target_instance_id is not None else None),
        action="auto_sprinkling",
        name_arg="target_item",
        instance_arg="target_instance_id",
    )
    if err is not None:
        return err

    return api.call_action(
        "auto_sprinkle",
        target=str(args.target_item),
        target_instance_id=(resolved.instance_id if resolved is not None else None),
        sprinkle_num=int(args.sprinkle_num),
    )
