from __future__ import annotations

from dataclasses import dataclass

from epm.cerebellum.cookbench_api import ActionAPI, ActionResult
from epm.cerebellum.game_hotkeys import ensure_alt_j_interaction
from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.skills._instance_resolution import resolve_instance_id
from epm.cerebellum.skills._shared_paths import realtime_products_json, userdata_root, window_title
from epm.vision.screen_capture import activate_window


@dataclass(frozen=True)
class PourArgs:
    container_name: str
    container_instance_id: int | None = None
    target_ml: float = 50.0


def run(api: ActionAPI, args: PourArgs) -> ActionResult:
    """
    High-level pouring skill wrapper.

    Delegates to the consolidated action/tool interface:
      auto_pour(container="<container_name>", pour_ml=<target_ml>)
    """

    resolved, err = resolve_instance_id(
        realtime_products_path=realtime_products_json(),
        name=str(args.container_name),
        instance_id=(int(args.container_instance_id) if args.container_instance_id is not None else None),
        action="auto_pouring",
        name_arg="container_name",
        instance_arg="container_instance_id",
    )
    if err is not None:
        return err

    # Preflight: make sure Alt+J interaction stream is enabled (for pour amount monitoring).
    io = RawInputController()
    alt_j_ok = ensure_alt_j_interaction(
        userdata_root=userdata_root(),
        window_title=window_title("CookingSimulator"),
        activate_window=activate_window,
        io_controller=io,
        verbose=True,
    )
    if not alt_j_ok:
        return ActionResult(
            False,
            raw={},
            error="alt_j_not_enabled (press Alt+J in-game to enable realtime_interaction_info.txt)",
        )

    return api.call_action(
        "auto_pour",
        container=str(args.container_name),
        container_instance_id=(resolved.instance_id if resolved is not None else None),
        pour_ml=float(args.target_ml),
    )
