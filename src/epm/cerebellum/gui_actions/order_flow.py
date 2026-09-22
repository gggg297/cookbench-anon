from __future__ import annotations

import time
from typing import Any

from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.skills.auto_navigation.skill import NavigateArgs, run as run_nav
from epm.cerebellum.gui_actions.order_gui import detect_computer_ui_state, order_dish_by_search
from epm.vision.screen_capture import activate_window


def _compact_nav_debug(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    debug: dict[str, Any] = {}
    for key in (
        "resolved_target",
        "resolved_target_instance_id",
        "navigation_failure_summary",
        "navigation_failure_signals",
        "navigator_stdout_tail",
        "docking_debug",
        "precise_adjust_debug",
        "posture_decision",
        "blocked_by_mode",
        "hint",
    ):
        value = raw.get(key)
        if value not in (None, "", [], {}):
            debug[key] = value
    return debug


def order_dish_via_computer(
    *,
    dish_name: str,
    computer_target: str = "Computer",
    window_title: str = "CookingSimulator",
) -> dict[str, Any]:
    """
    Higher-level GUI flow:
    1) Navigate to the in-world Computer
    2) Click to enter the computer UI
    3) Use `gui_order_dish` flow to place an order
    4) Right click to exit the computer UI

    Returns a structured dict for action feedback.
    """

    if not dish_name or not str(dish_name).strip():
        return {"success": False, "error": "missing_dish_name"}

    io = RawInputController()
    try:
        activate_window(window_title)
    except Exception:
        pass

    nav_res = run_nav(None, NavigateArgs(target=str(computer_target)))  # type: ignore[arg-type]
    preflight = nav_res.raw.get("preflight") if isinstance(nav_res.raw, dict) else None
    f12_ok = bool(preflight.get("f12_ok")) if isinstance(preflight, dict) else None
    f11_ok = bool(preflight.get("f11_ok")) if isinstance(preflight, dict) else None
    nav_debug = _compact_nav_debug(getattr(nav_res, "raw", None))
    if not bool(nav_res.success):
        computer_ui = detect_computer_ui_state(window_title=window_title)
        if not bool(computer_ui.get("is_open")):
            return {
                "success": False,
                "error": f"goto_computer_failed:{nav_res.error}",
                "computer_target": str(computer_target),
                "nav_error": str(nav_res.error or ""),
                "nav_debug": nav_debug,
                "computer_ui": computer_ui,
                "f12_ok": bool(f12_ok),
                "f11_ok": bool(f11_ok),
            }
    else:
        computer_ui = {"is_open": False, "matched_by": "", "matched_score": 0.0}

    # If navigation ended in failure but the computer UI is already open, trust the UI state
    # and proceed directly to ordering. Otherwise, enter the computer UI now.
    bypassed_by_open_ui = (not bool(nav_res.success)) and bool(computer_ui.get("is_open"))
    if not bypassed_by_open_ui:
        io.click("left")
        time.sleep(0.6)

    order_res = order_dish_by_search(dish_name=str(dish_name), window_title=window_title)
    if not bool(order_res.get("success", False)):
        return {
            "success": False,
            "error": f"gui_order_dish_failed:{order_res.get('error')}",
            "computer_target": str(computer_target),
            "dish_name": str(dish_name),
            "order_result": order_res,
            "nav_error": str(nav_res.error or ""),
            "nav_debug": nav_debug,
            "computer_ui": computer_ui,
            "nav_bypassed_by_open_ui": bool(bypassed_by_open_ui),
            "f12_ok": bool(f12_ok),
            "f11_ok": bool(f11_ok),
        }

    # Exit computer UI.
    io.click("right")
    time.sleep(0.2)

    return {
        "success": True,
        "computer_target": str(computer_target),
        "dish_name": str(dish_name),
        "order_result": order_res,
        "nav_error": str(nav_res.error or ""),
        "nav_debug": nav_debug,
        "computer_ui": computer_ui,
        "nav_bypassed_by_open_ui": bool(bypassed_by_open_ui),
        "f12_ok": bool(f12_ok),
        "f11_ok": bool(f11_ok),
        "error": "",
    }
