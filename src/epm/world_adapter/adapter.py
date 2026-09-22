from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from epm.core.epm_types import Observation
from epm.vision.screen_capture import save_step_screenshot


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class WorldAdapterConfig:
    realtime_products_path: Path
    screenshot_dir: Optional[Path] = None
    window_title: str = "CookingSimulator"
    activate_window_each_step: bool = True
    screenshot_format: str = "jpeg"
    screenshot_jpeg_quality: int = 70
    camera_dump_path: Optional[Path] = None


class WorldAdapter:
    """
    Minimal WorldAdapter: reads `realtime_products.json` and optional screenshot path.

    In later iterations this can be extended to:
    - read camera dump
    - produce prompt snippets (VLM/LLM)
    - compute spatial memory updates
    """

    def __init__(self, cfg: WorldAdapterConfig) -> None:
        self.cfg = cfg

    @staticmethod
    def _name_of(item: Dict[str, Any]) -> str:
        for k in ("name_en", "name", "name_cn", "label"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "unknown"

    @staticmethod
    def _extract_on_screen(objects: List[Dict[str, Any]], *, max_items: int = 50) -> List[Dict[str, Any]]:
        visible = []
        for o in objects:
            if not isinstance(o, dict):
                continue
            on_screen = o.get("is_on_screen")
            if isinstance(on_screen, bool) and not on_screen:
                continue
            visible.append(o)
        visible.sort(
            key=lambda x: float(x.get("distance", 999.0))
            if isinstance(x.get("distance", 999.0), (int, float))
            else 999.0
        )
        out: List[Dict[str, Any]] = []
        for o in visible[: max(1, int(max_items))]:
            pos = o.get("position") if isinstance(o.get("position"), dict) else {}
            out.append(
                {
                    "name": WorldAdapter._name_of(o),
                    "distance": o.get("distance", None),
                    "is_on_screen": o.get("is_on_screen", None),
                    "is_held": o.get("is_held", None),
                    "position": {"x": pos.get("x"), "y": pos.get("y"), "z": pos.get("z")},
                }
            )
        return out

    def observe(self, *, frame_id: str, screenshot_filename: Optional[str] = None) -> Observation:
        objects: List[Dict[str, Any]] = []
        state: Dict[str, Any] = {}
        events: List[Dict[str, Any]] = []
        screenshot_path: Optional[str] = None

        if self.cfg.screenshot_dir is not None:
            try:
                step_id = int(frame_id)
            except Exception:
                step_id = 0
            try:
                shot = save_step_screenshot(
                    step_id=step_id,
                    out_dir=self.cfg.screenshot_dir,
                    window_title=self.cfg.window_title,
                    activate=self.cfg.activate_window_each_step,
                    filename=screenshot_filename,
                    image_format=self.cfg.screenshot_format,
                    jpeg_quality=self.cfg.screenshot_jpeg_quality,
                )
                screenshot_path = str(shot)
            except Exception:
                screenshot_path = None

        if self.cfg.realtime_products_path.exists():
            try:
                # Accept UTF-8 with BOM (common on Windows when written by external tools/mods).
                data = json.loads(self.cfg.realtime_products_path.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict):
                    state = data.get("state", {}) if isinstance(data.get("state"), dict) else {}
                    try:
                        if "screen_width" in data:
                            state["_screen_width"] = int(data.get("screen_width"))
                        if "screen_height" in data:
                            state["_screen_height"] = int(data.get("screen_height"))
                    except Exception:
                        pass
                    objs = data.get("objects")
                    prods = data.get("products")
                    if isinstance(objs, list) and len(objs) > 0:
                        objects = objs
                    elif isinstance(prods, list):
                        # CookingSimulator realtime feed commonly exposes `products`.
                        # Use it as observation objects when `objects` is absent/empty.
                        objects = prods
                    else:
                        objects = []
                    events = data.get("events", []) if isinstance(data.get("events"), list) else []
                elif isinstance(data, list):
                    objects = data
            except Exception:
                pass

        # Provide "on-screen" feedback to the brain (for success judgement and planning).
        try:
            on_screen = self._extract_on_screen(objects, max_items=50)
            state["_on_screen_objects"] = on_screen
            state["_on_screen_names"] = [x.get("name") for x in on_screen if isinstance(x, dict) and isinstance(x.get("name"), str)][:50]
        except Exception:
            pass

        return Observation(
            time=_utc_now_iso(),
            frame_id=frame_id,
            screenshot_path=screenshot_path,
            state=state,
            objects=objects,
            events=events,
        )
