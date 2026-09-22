from __future__ import annotations

from dataclasses import dataclass

from epm.cerebellum.cookbench_api import ActionAPI, ActionResult
from epm.cerebellum.skills._instance_resolution import resolve_instance_id
from epm.cerebellum.skills._shared_paths import realtime_products_json


@dataclass(frozen=True)
class CutArgs:
    item_name: str
    item_instance_id: int | None = None
    cut_num: int = 3


def run(api: ActionAPI, args: CutArgs) -> ActionResult:
    """
    High-level cutting skill wrapper.

    Delegates to the consolidated action/tool interface:
      auto_cut(object="<item_name>", cut_num=<cut_num>)
    """

    resolved, err = resolve_instance_id(
        realtime_products_path=realtime_products_json(),
        name=str(args.item_name),
        instance_id=(int(args.item_instance_id) if args.item_instance_id is not None else None),
        action="auto_cutting",
        name_arg="item_name",
        instance_arg="item_instance_id",
    )
    if err is not None:
        return err

    return api.call_action(
        "auto_cut",
        object=str(args.item_name),
        object_instance_id=(resolved.instance_id if resolved is not None else None),
        cut_num=int(args.cut_num),
    )
