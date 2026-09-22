import time
from typing import Any

from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.skills.auto_navigation.skill import NavigateArgs, run as run_nav
from epm.vision.screen_capture import activate_window


def submit_dish_and_evaluate(
    *,
    dish_name: str,
    window_title: str = "CookingSimulator",
    click_submit_template: bool = True,
    submit_click_xy: dict[str, Any] | None = None,
    wait_ui_dump_timeout_s: float = 2.0,
    wait_before_submit_match_s: float = 0.0,
    wait_after_submit_s: float = 0.6,
    strict_feedback_match: bool = True,
) -> dict[str, Any]:
    """
    GUI action: select a dish entry by pixel coordinates from `ui_targets_dump_latest.json`,
    optionally click a submit-like UI element, then trigger Alt+K evaluation dump.

    Note:
    - The working submit click in the Checkout Stand UI uses relative mouse movement.
    - `submit_click_xy` is kept for backwards compatibility but is currently ignored.
    """
    if not dish_name or not str(dish_name).strip():
        return {"success": False, "error": "missing_dish_name"}

    try:
        activate_window(window_title)
    except Exception:
        pass

    from epm.cerebellum.gui_actions.submit_dish_gui import click_dish_entry_from_ui_dump, click_submit_icon

    dish_res = click_dish_entry_from_ui_dump(
        dish_name=str(dish_name),
        window_title=window_title,
        wait_ui_dump_timeout_s=float(wait_ui_dump_timeout_s),
    )
    if not bool(dish_res.get("success", False)):
        return {
            "success": False,
            "error": f"dish_click_failed:{dish_res.get('error')}",
            "dish_name": str(dish_name),
            "dish_result": dish_res,
        }

    submitted = False
    submit_warning: str | None = None
    submit_result: dict[str, Any] | None = None

    if bool(click_submit_template):
        if submit_click_xy:
            submit_warning = "submit_click_xy_ignored(use_relative_submit_click)"
        if float(wait_before_submit_match_s) > 0:
            time.sleep(float(wait_before_submit_match_s))

        submit_result = click_submit_icon(window_title=window_title, strategy="relative_offset", verbose=True)
        submitted = bool(submit_result.get("success", False))
        if not submitted:
            submit_warning = str(submit_result.get("error") or "submit_not_completed")
        else:
            time.sleep(float(wait_after_submit_s))

    from epm.cerebellum.gui_actions.feedback_gui import get_latest_recipe_feedback

    if bool(click_submit_template) and not submitted:
        return {
            "success": False,
            "dish_name": str(dish_name),
            "dish_result": dish_res,
            "submitted": False,
            "submit_result": submit_result,
            "error": submit_warning or "submit_not_completed",
        }

    fb = get_latest_recipe_feedback(window_title=window_title)

    out: dict[str, Any] = {
        "success": True,
        "dish_name": str(dish_name),
        "dish_result": dish_res,
        "submitted": bool(submitted),
        "submit_result": submit_result,
        "feedback": fb,
        "error": "",
    }
    if submit_warning:
        out["warning"] = submit_warning

    try:
        got = str((fb or {}).get("dishName") or "").strip()
        want = str(dish_name).strip()
        if got and want and want.casefold() not in got.casefold():
            out["warning_feedback_dish_mismatch"] = f"requested={want!r} feedback_dishName={got!r}"
            if bool(strict_feedback_match):
                out["success"] = False
                out["error"] = "feedback_dish_mismatch"
    except Exception:
        pass

    return out


def submit_dish_via_checkout_stand(
    *,
    dish_name: str,
    checkout_target: str = "Checkout Stand",
    window_title: str = "CookingSimulator",
    _debug_submit_only: bool = False,
) -> dict[str, Any]:
    """
    Higher-level flow:
      1) goto(checkout_target)
      2) left click to enter the serving/submit UI
      3) wait 3s
      4) click `submit.png`
      5) wait 2s
      6) Alt+Y dump -> click dish entry
      7) wait 2s
      8) Alt+K feedback parse
      9) right click to exit the UI
    """
    if not dish_name or not str(dish_name).strip():
        return {"success": False, "error": "missing_dish_name"}

    io = RawInputController()

    def _exit_submit_ui_twice() -> None:
        for _ in range(2):
            try:
                io.click("right")
            except Exception:
                pass
            time.sleep(0.6)

    try:
        activate_window(window_title)
    except Exception:
        pass

    nav_res = run_nav(None, NavigateArgs(target=str(checkout_target)))  # type: ignore[arg-type]
    preflight = nav_res.raw.get("preflight") if isinstance(nav_res.raw, dict) else None
    f12_ok = bool(preflight.get("f12_ok")) if isinstance(preflight, dict) else None
    f11_ok = bool(preflight.get("f11_ok")) if isinstance(preflight, dict) else None
    if f12_ok is False:
        print("[gui_serve_flow] warning: F12 products scan not confirmed fresh; continuing anyway")
    if f11_ok is False:
        print("[gui_serve_flow] warning: F11 radar scan not confirmed fresh; continuing anyway")
    if not bool(nav_res.success):
        return {
            "success": False,
            "error": f"goto_checkout_failed:{nav_res.error}",
            "checkout_target": str(checkout_target),
            "f12_ok": bool(f12_ok),
            "f11_ok": bool(f11_ok),
        }

    from epm.cerebellum.gui_actions.submit_dish_gui import (
        click_dish_entry_from_ui_dump,
        click_submit_panel_from_ui_dump,
        rewind_submit_cursor_to_window_center,
    )

    # Temporary rollback: stage image matching is too noisy right now, so keep
    # the stable UI-dump-driven submit flow and skip image-based stage checks.
    enable_submit_stage_detection = False

    submit_res: dict[str, Any] = {}
    submit_attempts: list[dict[str, Any]] = []
    stage1_check: dict[str, Any] = {
        "success": False,
        "stage": "skipped",
        "error": "submit_stage_detection_disabled",
    }
    for attempt_idx in range(2):
        io.click("left")
        time.sleep(3.0)
        if enable_submit_stage_detection:
            from epm.cerebellum.gui_actions.submit_dish_gui import wait_for_submit_flow_stage

            stage1_check = wait_for_submit_flow_stage(
                expected_stage="stage_1_submit_panel",
                window_title=window_title,
                timeout_s=1.2,
                poll_s=0.25,
                stable_hits=1,
                verbose=True,
            )
            if not bool(stage1_check.get("success", False)):
                print(
                    f"[gui_serve_flow] warning: submit_stage_expected='stage_1_submit_panel' "
                    f"observed={stage1_check.get('stage')!r} error={stage1_check.get('error')!r}"
                )
        submit_res = click_submit_panel_from_ui_dump(
            window_title=window_title,
            move_strategy="relative_window_center",
            verbose=True,
        )
        submit_attempts.append(
            {
                "attempt": int(attempt_idx + 1),
                "success": bool(submit_res.get("success", False)),
                "error": str(submit_res.get("error", "")),
            }
        )
        print(
            f"[gui_serve_flow] submit attempt={int(attempt_idx + 1)} success={bool(submit_res.get('success'))} "
            f"best={submit_res.get('best')} src={(submit_res.get('match') or {}).get('src')} "
            f"error={submit_res.get('error')!r}"
        )
        if bool(submit_res.get("success", False)):
            break
        try:
            io.click("right")
            time.sleep(1.0)
        except Exception:
            pass

    if not bool(submit_res.get("success", False)):
        return {
            "success": False,
            "error": f"submit_icon_not_found:{submit_res.get('error')}",
            "checkout_target": str(checkout_target),
            "dish_name": str(dish_name),
            "submit": submit_res,
            "submit_attempts": submit_attempts,
            "f12_ok": bool(f12_ok),
            "f11_ok": bool(f11_ok),
        }

    if bool(_debug_submit_only):
        return {
            "success": True,
            "checkout_target": str(checkout_target),
            "dish_name": str(dish_name),
            "submit": submit_res,
            "submit_attempts": submit_attempts,
            "submit_only": True,
            "f12_ok": bool(f12_ok),
            "f11_ok": bool(f11_ok),
            "error": "",
        }

    time.sleep(2.0)
    stage2_check: dict[str, Any] = {
        "success": False,
        "stage": "skipped",
        "error": "submit_stage_detection_disabled",
    }
    if enable_submit_stage_detection:
        from epm.cerebellum.gui_actions.submit_dish_gui import wait_for_submit_flow_stage

        stage2_check = wait_for_submit_flow_stage(
            expected_stage="stage_2_dish_grid",
            window_title=window_title,
            timeout_s=2.5,
            poll_s=0.25,
            stable_hits=1,
            verbose=True,
        )
        if not bool(stage2_check.get("success", False)):
            print(
                f"[gui_serve_flow] warning: submit_stage_expected='stage_2_dish_grid' "
                f"observed={stage2_check.get('stage')!r} error={stage2_check.get('error')!r}"
            )
    submit_cursor_reset = rewind_submit_cursor_to_window_center(
        submit_result=submit_res,
        window_title=window_title,
        steps=18,
        step_sleep_s=0.006,
        verbose=True,
    )

    checkout_dish_recovery: dict[str, Any] = {
        "used": False,
        "trigger_error": "",
        "submit_retry": None,
    }
    dish_attempts: list[dict[str, Any]] = []
    dish_res: dict[str, Any] = {}
    for dish_attempt_idx in range(2):
        dish_res = click_dish_entry_from_ui_dump(
            dish_name=str(dish_name),
            window_title=window_title,
            wait_ui_dump_timeout_s=2.0,
            move_strategy="checkout_grid_locked_cursor",
            submit_anchor_move=((submit_res.get("move") or {}).get("net_move") if isinstance(submit_res.get("move"), dict) else None),
            relative_steps=18,
            relative_step_sleep_s=0.006,
        )
        dish_error = str(dish_res.get("error") or "")
        dish_attempts.append(
            {
                "attempt": int(dish_attempt_idx + 1),
                "success": bool(dish_res.get("success", False)),
                "error": dish_error,
            }
        )
        if bool(dish_res.get("success", False)):
            break
        if dish_attempt_idx == 0 and dish_error == "checkout_dish_list_not_found":
            checkout_dish_recovery["used"] = True
            checkout_dish_recovery["trigger_error"] = dish_error
            _exit_submit_ui_twice()
            io.click("left")
            time.sleep(3.0)
            retry_submit_res = click_submit_panel_from_ui_dump(
                window_title=window_title,
                move_strategy="relative_window_center",
                verbose=True,
            )
            checkout_dish_recovery["submit_retry"] = {
                "success": bool(retry_submit_res.get("success", False)),
                "error": str(retry_submit_res.get("error", "")),
            }
            submit_attempts.append(
                {
                    "attempt": int(len(submit_attempts) + 1),
                    "success": bool(retry_submit_res.get("success", False)),
                    "error": str(retry_submit_res.get("error", "")),
                    "recovery_retry": True,
                }
            )
            if not bool(retry_submit_res.get("success", False)):
                return {
                    "success": False,
                    "error": f"submit_icon_not_found_after_checkout_retry:{retry_submit_res.get('error')}",
                    "checkout_target": str(checkout_target),
                    "dish_name": str(dish_name),
                    "submit": retry_submit_res,
                    "submit_attempts": submit_attempts,
                    "dish_result": dish_res,
                    "dish_attempts": dish_attempts,
                    "checkout_dish_recovery": checkout_dish_recovery,
                    "f12_ok": bool(f12_ok),
                    "f11_ok": bool(f11_ok),
                }
            submit_res = retry_submit_res
            time.sleep(2.0)
            submit_cursor_reset = rewind_submit_cursor_to_window_center(
                submit_result=submit_res,
                window_title=window_title,
                steps=18,
                step_sleep_s=0.006,
                verbose=True,
            )
            continue
        break

    if not bool(dish_res.get("success", False)):
        dish_error = str(dish_res.get("error") or "")
        if checkout_dish_recovery["used"] and dish_error == "checkout_dish_list_not_found":
            dish_error = "checkout_dish_list_not_found_after_recovery"
        return {
            "success": False,
            "error": f"dish_click_failed:{dish_error}",
            "checkout_target": str(checkout_target),
            "dish_name": str(dish_name),
            "submit": submit_res,
            "submit_attempts": submit_attempts,
            "submit_cursor_reset": submit_cursor_reset,
            "dish_result": dish_res,
            "dish_attempts": dish_attempts,
            "checkout_dish_recovery": checkout_dish_recovery,
            "f12_ok": bool(f12_ok),
            "f11_ok": bool(f11_ok),
        }

    time.sleep(2.0)


    from epm.cerebellum.gui_actions.feedback_gui import get_latest_recipe_feedback

    stage3_check: dict[str, Any] = {
        "success": False,
        "stage": "skipped",
        "error": "submit_stage_detection_disabled",
    }
    print("[gui_serve_flow] submit_stage_detection=disabled strategy=legacy_alt_k_then_left_click")
    _ = get_latest_recipe_feedback(window_title=window_title)
    io.click("left")
    fb = get_latest_recipe_feedback(window_title=window_title)
    got = str((fb or {}).get("dishName") or "").strip()
    want = str(dish_name).strip()
    if got and want and want.casefold() not in got.casefold():
        return {
            "success": False,
            "error": "feedback_dish_mismatch",
            "checkout_target": str(checkout_target),
            "dish_name": str(dish_name),
            "submit": submit_res,
            "submit_stage_2": stage2_check,
            "submit_stage_3": stage3_check,
            "submit_cursor_reset": submit_cursor_reset,
            "dish_result": dish_res,
            "feedback": fb,
            "f12_ok": bool(f12_ok),
            "f11_ok": bool(f11_ok),
        }

    io.click("right")
    time.sleep(0.2)

    return {
        "success": True,
        "checkout_target": str(checkout_target),
        "dish_name": str(dish_name),
        "submit": submit_res,
        "submit_stage_2": stage2_check,
        "submit_stage_3": stage3_check,
        "submit_cursor_reset": submit_cursor_reset,
        "dish_result": dish_res,
        "dish_attempts": dish_attempts,
        "checkout_dish_recovery": checkout_dish_recovery,
        "feedback": fb,
        "f12_ok": bool(f12_ok),
        "f11_ok": bool(f11_ok),
        "error": "",
    }
