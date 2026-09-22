from __future__ import annotations

from dataclasses import dataclass

from epm.cerebellum.cookbench_api import ActionAPI, ActionResult
from epm.cerebellum.skills._instance_resolution import resolve_instance_id
from epm.cerebellum.skills._shared_paths import realtime_products_json


@dataclass(frozen=True)
class MixArgs:
    container_name: str
    container_instance_id: int | None = None


def run(api: ActionAPI, args: MixArgs) -> ActionResult:
    """
    High-level mixing skill wrapper.

    This delegates to the consolidated action/tool interface:
      auto_mix(container="<container_name>")
    """

    resolved, err = resolve_instance_id(
        realtime_products_path=realtime_products_json(),
        name=str(args.container_name),
        instance_id=(int(args.container_instance_id) if args.container_instance_id is not None else None),
        action="auto_mix",
        name_arg="container_name",
        instance_arg="container_instance_id",
    )
    if err is not None:
        return err

    return api.call_action(
        "auto_mix",
        container=str(args.container_name),
        container_instance_id=(resolved.instance_id if resolved is not None else None),
    )
