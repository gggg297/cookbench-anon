from __future__ import annotations

import time
import unicodedata
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from epm.cerebellum.game_hotkeys import set_f12_products_scan_enabled
from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.skills.auto_navigation.skill import NavigateArgs, run as run_nav
from epm.cerebellum.skills._shared_paths import repo_root, userdata_root
from epm.vision.screen_capture import activate_window, capture_screenshot_mss, get_window_rect


def _norm_key(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    for ch in (" ", "_", "-", ".", "'", "\""):
        s = s.replace(ch, "")
    return s


def _load_object_mapping(mapping_path: Path) -> list[dict]:
    if not mapping_path.exists():
        return []
    out: list[dict] = []
    content = mapping_path.read_text(encoding="utf-8-sig", errors="ignore")
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        out.append({"id": parts[0], "cn": parts[1], "en": parts[2], "category": parts[3]})
    return out


def _validate_item_supported(item: str) -> tuple[bool, list[str], Path]:
    mapping_path = (repo_root() / "data" / "object_en_ch_mapping.txt").resolve()
    rows = _load_object_mapping(mapping_path=mapping_path)
    want = _norm_key(item)
    if not want:
        return False, [], mapping_path
    for r in rows:
        if _norm_key(str(r.get("en", ""))) == want or _norm_key(str(r.get("cn", ""))) == want:
            return True, [], mapping_path
    suggestions: list[str] = []
    for r in rows:
        en = str(r.get("en", "") or "").strip()
        cn = str(r.get("cn", "") or "").strip()
        key_en = _norm_key(en)
        key_cn = _norm_key(cn)
        if want in key_en or want in key_cn or key_en in want or key_cn in want:
            if en and en not in suggestions:
                suggestions.append(en)
            elif cn and cn not in suggestions:
                suggestions.append(cn)
        if len(suggestions) >= 10:
            break
    return False, suggestions, mapping_path


def _resume_cmd() -> str:
    return "python epm/scripts/run_episode_dashboard.py --config epm/configs/pipelines/planner_executor.json --dish-id 1 --resume --refresh-memory"


def _plate_stack_source_hint(item: str) -> Optional[str]:
    raw = str(item or "").strip()
    tokens = [t.strip() for t in raw.replace(";", ",").split(",") if t.strip()]
    if not tokens:
        tokens = [raw]
    plate_like = {
        "baketray",
        "largeplate",
        "smallplate",
        "plate",
        "squareplate",
        "deepplate",
        "casserole",
        "bowl",
    }
    hit: Optional[str] = None
    for tok in tokens:
        if _norm_key(tok) in plate_like:
            hit = tok
            break
    if hit is None:
        return None
    item_clean = str(hit or raw or "Plate").strip()
    return f"{item_clean} Stack"


def _imread_gray(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def _grocery_visible(*, window_title: str, threshold: float = 0.84) -> tuple[bool, float]:
    tpl = (repo_root() / "data" / "figure" / "computer" / "grocery.png").resolve()
    if not tpl.exists():
        return False, 0.0
    tpl_g = _imread_gray(tpl)
    if tpl_g is None:
        return False, 0.0
    rect = get_window_rect(window_title)
    img = capture_screenshot_mss(region=rect.to_mss_region())
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < tpl_g.shape[0] or gray.shape[1] < tpl_g.shape[1]:
        return False, 0.0
    res = cv2.matchTemplate(gray, tpl_g, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)
    best = float(max_val)
    return best >= float(threshold), best


def buy_new_item(
    *,
    item: str,
    category: str | None = None,
    window_title: str = "CookingSimulator",
    store_target: str = "Carton Box",
    store_target_instance_id: int | None = -33328,
    wait_after_open_s: float = 2.0,
    max_scrolls: int = 50,
) -> dict:
    """
    GUI flow: navigate to the store (`Carton Box`) -> open store UI -> buy item.

    This wraps `store_gui.buy_new_item()` by adding the world-level navigation + UI open step.
    """
    stack_hint = _plate_stack_source_hint(str(item))
    if stack_hint is not None:
        return {
            "success": False,
            "error": "plate_item_must_use_stack_source",
            "item": str(item),
            "stack_source_hint": stack_hint,
            "hint": f"Plate-like items must be obtained from stack source, e.g. query_scene_objects('{stack_hint}') then navigate/pick.",
            "resume_cmd": _resume_cmd(),
        }

    # Validate item against supported mapping (unless direct template path is provided).
    if item and not Path(str(item)).exists():
        ok, suggestions, mapping_path = _validate_item_supported(str(item))
        if not ok:
            return {
                "success": False,
                "error": "unsupported_item",
                "item": str(item),
                "mapping_path": str(mapping_path),
                "mapping_suggestions": suggestions,
                "resume_cmd": _resume_cmd(),
            }

    io = RawInputController()
    try:
        activate_window(window_title)
    except Exception:
        pass

    rt_products_path = userdata_root() / "realtime_products.json"
    try:
        set_f12_products_scan_enabled(
            realtime_products_path=rt_products_path,
            window_title=window_title,
            activate_window=activate_window,
            io_controller=io,
            enabled=False,
            verbose=False,
        )
    except Exception:
        pass

    def _pick_candidate_id(candidates: list[dict]) -> int | None:
        best_id: int | None = None
        best_key: tuple[int, float] | None = None
        for c in candidates:
            if not isinstance(c, dict):
                continue
            try:
                iid = int(c.get("instance_id"))
            except Exception:
                continue
            on_screen = bool(c.get("is_on_screen"))
            try:
                dist = float(c.get("distance"))
            except Exception:
                dist = float("inf")
            key = (0 if on_screen else 1, dist)
            if best_key is None or key < best_key:
                best_key = key
                best_id = iid
        return best_id

    def _retry_nav_from_candidates(nav_res: object) -> object:
        raw = getattr(nav_res, "raw", None)
        if not isinstance(raw, dict):
            return nav_res
        candidates = raw.get("candidates")
        if not isinstance(candidates, list):
            return nav_res
        picked = _pick_candidate_id(candidates)
        if picked is None:
            return nav_res
        return run_nav(None, NavigateArgs(target=str(store_target), target_instance_id=int(picked)))  # type: ignore[arg-type]

    try:
        nav_res = run_nav(
            None,
            NavigateArgs(
                target=str(store_target),
                target_instance_id=(int(store_target_instance_id) if store_target_instance_id is not None else None),
            ),
        )  # type: ignore[arg-type]
        if not bool(nav_res.success):
            err = str(getattr(nav_res, "error", "") or "")
            if err in {"instance_id_required", "instance_id_not_found"}:
                nav_res = _retry_nav_from_candidates(nav_res)
            if not bool(getattr(nav_res, "success", False)):
                return {"success": False, "error": f"goto_store_failed:{getattr(nav_res, 'error', '')}", "store_target": str(store_target)}

        # Open the store UI.
        io.click("left")
        time.sleep(max(0.0, float(wait_after_open_s)))

        from epm.cerebellum.gui_actions.store_gui import buy_new_item as _buy_new_item_ui

        def _attempt_buy(*, ms: int, extra_passes: int, step: float) -> dict:
            return _buy_new_item_ui(
                item=str(item),
                category=category,
                window_title=window_title,
                max_scrolls=int(ms),
                scroll_step=float(step),
                extra_full_passes=int(extra_passes),
            )

        # Retry policy:
        # - first attempt: normal search
        # - on not-found or grocery-stuck symptom: broaden scrolling range/passes and retry
        attempt_specs: list[tuple[int, int, float, str]] = [
            (int(max_scrolls), 2, 0.50, "base"),
            (max(int(max_scrolls), 100), 4, 0.35, "retry_small_step_1"),
            (max(int(max_scrolls), 150), 6, 0.25, "retry_small_step_2"),
        ]

        attempts: list[dict] = []
        last_res: dict = {"success": False, "error": "buy_not_started"}
        last_visible = False
        last_best = 0.0
        for ms, extra_passes, step, label in attempt_specs:
            last_res = _attempt_buy(ms=ms, extra_passes=extra_passes, step=step)
            try:
                last_visible, last_best = _grocery_visible(window_title=window_title)
            except Exception:
                last_visible, last_best = False, 0.0
            attempts.append(
                {
                    "label": label,
                    "max_scrolls": int(ms),
                    "extra_full_passes": int(extra_passes),
                    "scroll_step": float(step),
                    "success": bool(last_res.get("success", False)),
                    "error": str(last_res.get("error", "")),
                    "grocery_visible": bool(last_visible),
                    "grocery_best": float(last_best),
                    "boundary_name": last_res.get("boundary_name"),
                    "boundary_seen": bool(last_res.get("boundary_seen", False)),
                    "direction": last_res.get("direction"),
                }
            )
            if bool(last_res.get("success", False)):
                # If buy succeeds but grocery marker is still visible, try to close UI first.
                if last_visible:
                    try:
                        io.click("right")
                        time.sleep(0.12)
                        io.click("right")
                        time.sleep(0.12)
                    except Exception:
                        pass
                    try:
                        last_visible, last_best = _grocery_visible(window_title=window_title)
                    except Exception:
                        pass
                # Accept success once UI is closed.
                if not last_visible:
                    last_res.setdefault("store_target", str(store_target))
                    last_res.setdefault("attempts", attempts)
                    return last_res
            # Otherwise keep retrying with broader scan; slight cooldown for frame hitch.
            time.sleep(0.15)

        # Final close attempt (best-effort), then decide failure with clear retry hint.
        try:
            io.click("right")
            time.sleep(0.12)
            io.click("right")
            time.sleep(0.12)
        except Exception:
            pass
        try:
            last_visible, last_best = _grocery_visible(window_title=window_title)
        except Exception:
            pass

        return {
            "success": False,
            "error": (
                "buy_new_item_failed_retry_recommended "
                f"(last_error={str(last_res.get('error', ''))!r} grocery_visible={bool(last_visible)} best={float(last_best):.3f})"
            ),
            "item": str(item),
            "store_target": str(store_target),
            "attempts": attempts,
            "hint": "Purchase may have failed due to list hitch/visibility. Please retry.",
            "resume_cmd": _resume_cmd(),
        }
    finally:
        try:
            set_f12_products_scan_enabled(
                realtime_products_path=rt_products_path,
                window_title=window_title,
                activate_window=activate_window,
                io_controller=io,
                enabled=True,
                verbose=False,
            )
        except Exception:
            pass
