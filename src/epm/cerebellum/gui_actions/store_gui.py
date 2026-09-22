from __future__ import annotations

import csv
import unicodedata
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.skills._shared_paths import repo_root
from epm.vision.screen_capture import activate_window, capture_screenshot_mss, get_window_rect


@dataclass(frozen=True)
class MatchRect:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class StoreOrderEntry:
    index: int
    name: str
    key: str
    template_path: Path


@dataclass(frozen=True)
class VisibleOrderRange:
    min_index: int
    max_index: int
    matches: int


def _pil_to_bgr(img) -> np.ndarray:
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _imread_bgr(path: Path) -> Optional[np.ndarray]:
    """
    Unicode-safe BGR image loader for Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _imread_gray(path: Path) -> Optional[np.ndarray]:
    """
    Unicode-safe image loader for Windows (see order_gui.py).
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception:
        return None


def _match_best(screenshot_bgr: np.ndarray, template_path: Path) -> tuple[Optional[MatchRect], float]:
    if not template_path.exists():
        return None, 0.0
    tpl = _imread_gray(template_path)
    if tpl is None:
        return None, 0.0
    gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    x, y = int(max_loc[0]), int(max_loc[1])
    return MatchRect(x=x, y=y, w=int(tpl.shape[1]), h=int(tpl.shape[0])), float(max_val)


def _match_best_gray(gray: np.ndarray, template_gray: np.ndarray) -> tuple[Optional[MatchRect], float]:
    if gray.shape[0] < template_gray.shape[0] or gray.shape[1] < template_gray.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    x, y = int(max_loc[0]), int(max_loc[1])
    return MatchRect(x=x, y=y, w=int(template_gray.shape[1]), h=int(template_gray.shape[0])), float(max_val)


def _find_first(screenshot_bgr: np.ndarray, template_path: Path, *, threshold: float) -> Optional[MatchRect]:
    rect, best = _match_best(screenshot_bgr, template_path)
    if rect is None:
        return None
    return rect if best >= float(threshold) else None


def _resolve_store_dir() -> Path:
    return repo_root() / "data" / "figure" / "store"


def _norm_key(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    # Normalize accents (e.g. Jalapeño -> Jalapeno)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # Remove separators.
    for ch in (" ", "_", "-", ".", "'", "\""):
        s = s.replace(ch, "")
    return s


def _load_object_mapping(*, mapping_path: Path) -> list[dict]:
    if not mapping_path.exists():
        return []
    out: list[dict] = []
    # File can have BOM and weird encodings; be tolerant.
    content = mapping_path.read_text(encoding="utf-8-sig", errors="ignore")
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        # format: id, cn, en, category
        out.append({"id": parts[0], "cn": parts[1], "en": parts[2], "category": parts[3]})
    return out


def _infer_category_and_name(store_dir: Path, item_name: str) -> tuple[Optional[str], str]:
    mapping_path = (repo_root() / "data" / "object_en_ch_mapping.txt").resolve()
    rows = _load_object_mapping(mapping_path=mapping_path)
    want = _norm_key(item_name)
    if want:
        for r in rows:
            if _norm_key(str(r.get("en", ""))) == want or _norm_key(str(r.get("cn", ""))) == want:
                cat = str(r.get("category", "") or "").strip().lower() or None
                en = str(r.get("en", "") or "").strip() or item_name
                return cat, en

    # Fallback: try to find by filename under store categories.
    for cat in ("products", "spices", "liquids", "utensils", "miscellaneous"):
        base = store_dir / cat
        if not base.exists():
            continue
        for cand in base.rglob("*.png"):
            if _norm_key(cand.stem) == want:
                return cat, cand.stem
    return None, item_name


def _resolve_item_template(store_dir: Path, item_name_or_path: str, *, category: Optional[str]) -> Optional[Path]:
    p = Path(item_name_or_path)
    if p.exists():
        return p

    want = (item_name_or_path or "").strip()
    if not want:
        return None

    key = _norm_key(want)
    if not key:
        return None

    # Prefer exact Unicode filename match first (keeps diacritics/locale chars authoritative).
    if category:
        base = store_dir / str(category).strip().lower()
        if base.exists():
            exact = base / f"{want}.png"
            if exact.exists():
                return exact
    # Fallback exact lookup across all category folders.
    for base in store_dir.iterdir():
        if not base.is_dir():
            continue
        exact = base / f"{want}.png"
        if exact.exists():
            return exact

    # Then allow normalized matching for tolerance (e.g. user omits accents).
    # Prefer category folder if provided.
    if category:
        base = store_dir / str(category).strip().lower()
        if base.exists():
            for cand in base.rglob("*.png"):
                if _norm_key(cand.stem) == key:
                    return cand

    # Search all category folders.
    for base in store_dir.iterdir():
        if not base.is_dir():
            continue
        for cand in base.rglob("*.png"):
            if _norm_key(cand.stem) == key:
                return cand
    return None


def _load_store_order_entries(store_dir: Path, category: str) -> list[StoreOrderEntry]:
    final_cat = (category or "").strip().lower()
    if not final_cat:
        return []

    base = store_dir / final_cat
    if not base.exists():
        return []

    files = sorted(
        (cand for cand in base.glob("*.png") if cand.is_file()),
        key=lambda cand: (_norm_key(cand.stem), cand.name.lower()),
    )
    return [
        StoreOrderEntry(
            index=index,
            name=cand.stem,
            key=_norm_key(cand.stem),
            template_path=cand,
        )
        for index, cand in enumerate(files)
    ]


def _match_best_color_in_roi(screenshot_bgr: np.ndarray, template_bgr: np.ndarray, *, roi: MatchRect) -> float:
    h, w = screenshot_bgr.shape[:2]
    x1 = max(0, int(roi.x))
    y1 = max(0, int(roi.y))
    x2 = min(w, int(roi.x + roi.w))
    y2 = min(h, int(roi.y + roi.h))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    sub = screenshot_bgr[y1:y2, x1:x2]
    if sub.shape[0] < template_bgr.shape[0] or sub.shape[1] < template_bgr.shape[1]:
        return 0.0
    res = cv2.matchTemplate(sub, template_bgr, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, _max_loc = cv2.minMaxLoc(res)
    return float(max_val)


def _is_item_selected(screenshot_bgr: np.ndarray, *, item_rect: MatchRect, chosen_border_tpl: Path) -> tuple[bool, float]:
    tpl = _imread_bgr(chosen_border_tpl)
    if tpl is None:
        return False, 0.0
    # Search a padded region around the item icon.
    pad = 60
    roi = MatchRect(
        x=int(item_rect.x) - pad,
        y=int(item_rect.y) - pad,
        w=int(item_rect.w) + pad * 2,
        h=int(item_rect.h) + pad * 2,
    )
    best = _match_best_color_in_roi(screenshot_bgr, tpl, roi=roi)
    return bool(best >= 0.78), float(best)


def _match_best_color(screenshot_bgr: np.ndarray, template_path: Path) -> tuple[Optional[MatchRect], float]:
    tpl = _imread_bgr(template_path)
    if tpl is None:
        return None, 0.0
    if screenshot_bgr.shape[0] < tpl.shape[0] or screenshot_bgr.shape[1] < tpl.shape[1]:
        return None, 0.0
    res = cv2.matchTemplate(screenshot_bgr, tpl, cv2.TM_CCOEFF_NORMED)
    _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(res)
    x, y = int(max_loc[0]), int(max_loc[1])
    return MatchRect(x=x, y=y, w=int(tpl.shape[1]), h=int(tpl.shape[0])), float(max_val)


def _buy_button_selected_state(
    screenshot_bgr: np.ndarray,
    *,
    buy_chosen_tpl: Path,
    buy_not_chosen_tpl: Path,
    margin: float = 0.03,
) -> tuple[Optional[bool], float, float]:
    """
    Determine whether the currently selected store item is "active" by comparing the Buy button background.

    Returns:
      (selected_state|None, chosen_best, not_chosen_best)

    - `None` means ambiguous (scores too close).
    """
    chosen_rect, chosen_best = _match_best_color(screenshot_bgr, buy_chosen_tpl)
    _not_rect, not_best = _match_best_color(screenshot_bgr, buy_not_chosen_tpl)

    if chosen_rect is None:
        return None, float(chosen_best), float(not_best)

    if chosen_best >= 0.78 and (chosen_best - not_best) >= float(margin):
        return True, float(chosen_best), float(not_best)
    if not_best >= 0.78 and (not_best - chosen_best) >= float(margin):
        return False, float(chosen_best), float(not_best)
    return None, float(chosen_best), float(not_best)


def buy_new_item(
    *,
    item: str,
    category: str | None = None,
    window_title: str = "CookingSimulator",
    max_scrolls: int = 50,
    scroll_step: float = 0.5,
    extra_full_passes: int = 2,
) -> dict:
    """
    GUI action: select an item in the store list and click the Buy button.

    Args:
      item: item name (e.g. "Lemon") or a direct template path to a store icon png.
      category: one of "products|spices|liquids|utensils|miscellaneous" (best-effort tab click).
    """

    store_dir = _resolve_store_dir()
    inferred_cat, canonical_name = _infer_category_and_name(store_dir, item)
    final_cat = (category or inferred_cat or "products").strip().lower()

    item_tpl = _resolve_item_template(store_dir, canonical_name, category=final_cat)
    if item_tpl is None:
        return {
            "success": False,
            "error": f"store_item_template_not_found:{item!r} (canonical={canonical_name!r} category={final_cat!r})",
        }

    # Assets present in this repo (filenames are authoritative).
    tabs = {
        "products": store_dir / "products-button.png",
        "spices": store_dir / "spices-button.png",
        "liquids": store_dir / "liquids-button.png",
        "utensils": store_dir / "utensils.png",
        "miscellaneous": store_dir / "miscellaneous.png",
    }
    buy_btn = store_dir / "buy-button.png"
    buy_btn_not = store_dir / "buy-button-not-chosen.png"
    top_scroll = store_dir / "ingredient-top-scroll.png"
    bottom_scroll = store_dir / "ingredient-bottom-scroll.png"
    chosen_border = store_dir / "store-object-chosen.png"

    io = RawInputController()
    try:
        activate_window(window_title)
    except Exception as e:
        return {"success": False, "error": f"activate_window_failed:{e}"}

    rect = get_window_rect(window_title)

    # Best-effort: click category tab if template exists & matches.
    tab_tpl = tabs.get(final_cat, tabs["products"])
    img0 = capture_screenshot_mss(region=rect.to_mss_region())
    bgr0 = _pil_to_bgr(img0)
    tab = _find_first(bgr0, tab_tpl, threshold=0.88)
    if tab is not None:
        tx = rect.left + tab.x + tab.w // 2
        ty = rect.top + tab.y + tab.h // 2
        io.mouse_move_absolute(tx, ty)
        time.sleep(0.05)
        io.click("left")
        time.sleep(0.35)

    def at_bottom(screenshot_bgr: np.ndarray) -> bool:
        return _find_first(screenshot_bgr, bottom_scroll, threshold=0.90) is not None

    def at_top(screenshot_bgr: np.ndarray) -> bool:
        return _find_first(screenshot_bgr, top_scroll, threshold=0.90) is not None

    def _capture_bgr() -> np.ndarray:
        img = capture_screenshot_mss(region=rect.to_mss_region())
        return _pil_to_bgr(img)

    ordered_entries = _load_store_order_entries(store_dir, final_cat)
    target_key = _norm_key(canonical_name)
    target_indices = sorted(entry.index for entry in ordered_entries if entry.key == target_key)
    first_order_entry = ordered_entries[0] if ordered_entries else None
    last_order_entry = ordered_entries[-1] if ordered_entries else None
    search_debug: dict[str, object] = {
        "boundary_name": None,
        "boundary_seen": False,
        "direction": None,
    }
    ordered_templates: list[tuple[StoreOrderEntry, np.ndarray]] = []
    for entry in ordered_entries:
        tpl_gray = _imread_gray(entry.template_path)
        if tpl_gray is None:
            continue
        ordered_templates.append((entry, tpl_gray))

    def _click_item_and_buy(*, hit: MatchRect) -> dict:
        # Always click target once; click again only if buy button indicates unselected.
        cx = rect.left + hit.x + hit.w // 2
        cy = rect.top + hit.y + hit.h // 2
        io.mouse_move_absolute(cx, cy)
        time.sleep(0.05)
        io.click("left")
        time.sleep(0.22)

        bgr_sel = _capture_bgr()
        buy_state, buy_best, buy_not_best = _buy_button_selected_state(
            bgr_sel, buy_chosen_tpl=buy_btn, buy_not_chosen_tpl=buy_btn_not
        )
        if buy_state is False:
            io.mouse_move_absolute(cx, cy)
            time.sleep(0.05)
            io.click("left")
            time.sleep(0.22)
            bgr_sel2 = _capture_bgr()
            buy_state, buy_best, buy_not_best = _buy_button_selected_state(
                bgr_sel2, buy_chosen_tpl=buy_btn, buy_not_chosen_tpl=buy_btn_not
            )

        if buy_state is not True:
            return {
                "success": False,
                "error": (
                    f"buy_button_state_ambiguous_or_not_selected (state={buy_state!r} "
                    f"buy_best={buy_best:.3f} buy_not_best={buy_not_best:.3f})"
                ),
            }

        bgr_buy = _capture_bgr()
        buy = _find_first(bgr_buy, buy_btn, threshold=0.88)
        if buy is None:
            return {"success": False, "error": "buy_button_not_found (template match failed)"}
        bx = rect.left + buy.x + buy.w // 2
        by = rect.top + buy.y + buy.h // 2
        io.mouse_move_absolute(bx, by)
        time.sleep(0.05)
        io.click("left")
        return {
            "success": True,
            "item": str(item),
            "canonical_name": str(canonical_name),
            "category": str(final_cat),
            "selected": True,
            "buy_best": float(buy_best),
            "buy_not_best": float(buy_not_best),
            "error": "",
            "boundary_name": search_debug.get("boundary_name"),
            "boundary_seen": bool(search_debug.get("boundary_seen", False)),
            "direction": search_debug.get("direction"),
        }

    def _order_boundary_visible(screenshot_bgr: np.ndarray, *, toward: str) -> bool:
        entry = first_order_entry if toward == "up" else last_order_entry
        if entry is None:
            search_debug["boundary_name"] = None
            search_debug["boundary_seen"] = False
            search_debug["direction"] = toward
            return False
        seen = _find_first(screenshot_bgr, entry.template_path, threshold=0.86) is not None
        search_debug["boundary_name"] = entry.name
        search_debug["boundary_seen"] = bool(seen)
        search_debug["direction"] = toward
        return seen

    def _boundary_reached(screenshot_bgr: np.ndarray, *, toward: str) -> bool:
        return _order_boundary_visible(screenshot_bgr, toward=toward)

    def _seek_top() -> bool:
        confirm_hits = 0
        for _ in range(max(1, int(max_scrolls)) * 2):
            bgr = _capture_bgr()
            if _boundary_reached(bgr, toward="up"):
                confirm_hits += 1
                if confirm_hits >= 2:
                    return True
                time.sleep(0.06)
                continue
            confirm_hits = 0
            io.scroll_wheel(float(scroll_step))
            time.sleep(0.14)
        return False

    def _seek_bottom() -> bool:
        confirm_hits = 0
        for _ in range(max(1, int(max_scrolls)) * 2):
            bgr = _capture_bgr()
            if _boundary_reached(bgr, toward="down"):
                confirm_hits += 1
                if confirm_hits >= 2:
                    return True
                time.sleep(0.06)
                continue
            confirm_hits = 0
            io.scroll_wheel(-float(scroll_step))
            time.sleep(0.14)
        return False

    def _probe_item_hit(*, probes: int = 2) -> Optional[MatchRect]:
        for probe in range(max(1, int(probes))):
            bgr = _capture_bgr()
            hit = _find_first(bgr, item_tpl, threshold=0.86)
            if hit is not None:
                return hit
            if probe + 1 < probes:
                time.sleep(0.06)
        return None

    def _detect_visible_order_range(screenshot_bgr: np.ndarray) -> Optional[VisibleOrderRange]:
        if not ordered_templates:
            return None
        gray = cv2.cvtColor(screenshot_bgr, cv2.COLOR_BGR2GRAY)
        matched_hits: list[tuple[int, float]] = []
        for entry, tpl_gray in ordered_templates:
            _rect, best = _match_best_gray(gray, tpl_gray)
            if best >= 0.86:
                matched_hits.append((int(entry.index), float(best)))
        if not matched_hits:
            return None
        matched_hits.sort(key=lambda item: item[1], reverse=True)
        focus_hits = matched_hits[: min(8, len(matched_hits))]
        matched_indices = [idx for idx, _score in focus_hits]
        return VisibleOrderRange(
            min_index=min(matched_indices),
            max_index=max(matched_indices),
            matches=len(matched_indices),
        )

    def _scroll_toward_target(*, screenshot_bgr: np.ndarray, direction: str, amount: float) -> bool:
        if direction == "up":
            if _boundary_reached(screenshot_bgr, toward="up"):
                return False
            io.scroll_wheel(float(amount))
        else:
            if _boundary_reached(screenshot_bgr, toward="down"):
                return False
            io.scroll_wheel(-float(amount))
        time.sleep(0.18)
        return True

    def _scan_up_once(*, pass_name: str) -> dict | None:
        for _ in range(max(1, int(max_scrolls))):
            last_bgr: np.ndarray | None = None
            for probe in range(2):
                bgr = _capture_bgr()
                last_bgr = bgr
                hit = _find_first(bgr, item_tpl, threshold=0.86)
                if hit is not None:
                    res = _click_item_and_buy(hit=hit)
                    if isinstance(res, dict):
                        res.setdefault("scan_pass", pass_name)
                    return res
                if probe == 0:
                    time.sleep(0.06)

            if last_bgr is not None and _boundary_reached(last_bgr, toward="up"):
                return None
            io.scroll_wheel(float(scroll_step))
            time.sleep(0.18)
        return None

    def _edge_anchored_search() -> dict | None:
        if not target_indices or not ordered_entries:
            return None

        target_min = min(target_indices)
        total_entries = max(1, len(ordered_entries))
        target_ratio = target_min / max(1, total_entries - 1)

        if target_ratio <= 0.5:
            search_debug["direction"] = "down"
            search_debug["boundary_name"] = first_order_entry.name if first_order_entry is not None else None
            _seek_top()
            res = _scan_down_once(pass_name="edge_anchor_top_to_bottom")
            if isinstance(res, dict):
                res.setdefault("search_strategy", "edge_anchor")
                res.setdefault("boundary_name", search_debug.get("boundary_name"))
                res.setdefault("boundary_seen", bool(search_debug.get("boundary_seen", False)))
                res.setdefault("direction", search_debug.get("direction"))
                return res
            return None

        search_debug["direction"] = "up"
        search_debug["boundary_name"] = last_order_entry.name if last_order_entry is not None else None
        _seek_bottom()
        res = _scan_up_once(pass_name="edge_anchor_bottom_to_top")
        if isinstance(res, dict):
            res.setdefault("search_strategy", "edge_anchor")
            res.setdefault("boundary_name", search_debug.get("boundary_name"))
            res.setdefault("boundary_seen", bool(search_debug.get("boundary_seen", False)))
            res.setdefault("direction", search_debug.get("direction"))
            return res
        return None

    def _guided_order_search() -> dict | None:
        if not target_indices or not ordered_templates:
            return None

        target_min = min(target_indices)
        target_max = max(target_indices)
        target_mid = (target_min + target_max) / 2.0
        total_entries = max(1, len(ordered_entries))
        target_rank_ratio = target_mid / max(1, total_entries - 1)
        range_unknown_steps = 0
        max_guided_steps = max(8, max(1, int(max_scrolls)) * 2)
        locked_direction: str = "up" if target_rank_ratio <= 0.5 else "down"
        in_range_miss_streak = 0

        for step_idx in range(max_guided_steps):
            hit = _probe_item_hit(probes=2)
            if hit is not None:
                res = _click_item_and_buy(hit=hit)
                if isinstance(res, dict):
                    res.setdefault("search_strategy", "guided_order")
                    res.setdefault("scan_pass", "guided_order")
                    res.setdefault("guided_step", step_idx)
                return res

            bgr = _capture_bgr()
            visible = _detect_visible_order_range(bgr)
            if visible is None:
                range_unknown_steps += 1
                if range_unknown_steps >= 3:
                    break
                if not _scroll_toward_target(
                    screenshot_bgr=bgr,
                    direction=locked_direction,
                    amount=max(0.25, float(scroll_step) * 0.5),
                ):
                    break
                continue

            range_unknown_steps = 0
            if target_max < visible.min_index:
                locked_direction = "up"
                in_range_miss_streak = 0
                if not _scroll_toward_target(screenshot_bgr=bgr, direction="up", amount=max(0.35, float(scroll_step))):
                    break
                continue
            if target_min > visible.max_index:
                locked_direction = "down"
                in_range_miss_streak = 0
                if not _scroll_toward_target(screenshot_bgr=bgr, direction="down", amount=max(0.35, float(scroll_step))):
                    break
                continue

            local_hit = _probe_item_hit(probes=3)
            if local_hit is not None:
                res = _click_item_and_buy(hit=local_hit)
                if isinstance(res, dict):
                    res.setdefault("search_strategy", "guided_order")
                    res.setdefault("scan_pass", "guided_order_local")
                    res.setdefault("guided_step", step_idx)
                return res

            in_range_miss_streak += 1
            direction = locked_direction
            if in_range_miss_streak >= 3:
                direction = "up" if target_mid <= ((visible.min_index + visible.max_index) / 2.0) else "down"
                locked_direction = direction
                in_range_miss_streak = 0
            if not _scroll_toward_target(
                screenshot_bgr=bgr,
                direction=direction,
                amount=max(0.2, float(scroll_step) * 0.5),
            ):
                break

        return None

    def _scan_down_once(*, pass_name: str) -> dict | None:
        for _ in range(max(1, int(max_scrolls))):
            # Two-frame probe at each scroll position to reduce miss caused by short frame hitches.
            last_bgr: np.ndarray | None = None
            for probe in range(2):
                bgr = _capture_bgr()
                last_bgr = bgr
                hit = _find_first(bgr, item_tpl, threshold=0.86)
                if hit is not None:
                    res = _click_item_and_buy(hit=hit)
                    if isinstance(res, dict):
                        res.setdefault("scan_pass", pass_name)
                    return res
                if probe == 0:
                    time.sleep(0.06)

            if last_bgr is not None and at_bottom(last_bgr):
                return None
            io.scroll_wheel(-float(scroll_step))
            time.sleep(0.18)
        return None

    edge_res = _edge_anchored_search()
    if isinstance(edge_res, dict):
        return edge_res

    # Keep old strategies in code for regression/rollback, but do not use them by default.
    # guided_res = _guided_order_search()
    # if isinstance(guided_res, dict):
    #     return guided_res
    #
    # res0 = _scan_down_once(pass_name="current_to_bottom")
    # if isinstance(res0, dict):
    #     res0.setdefault("search_strategy", "fallback_scan")
    #     return res0
    #
    # total_full_passes = 1 + max(0, int(extra_full_passes))
    # for p in range(total_full_passes):
    #     _seek_top()
    #     res = _scan_down_once(pass_name=f"top_to_bottom_pass_{p + 1}")
    #     if isinstance(res, dict):
    #         res.setdefault("search_strategy", "fallback_scan")
    #         return res

    return {
        "success": False,
        "error": (
            f"item_not_found_in_store:{item!r} (canonical={canonical_name!r} category={final_cat!r}) "
            "after_scan_passes=edge_anchor_only"
        ),
        "boundary_name": search_debug.get("boundary_name"),
        "boundary_seen": bool(search_debug.get("boundary_seen", False)),
        "direction": search_debug.get("direction"),
    }
