from __future__ import annotations

import json
import time
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from epm.cerebellum.raw_input_controller import RawInputController
from epm.cerebellum.skills._shared_paths import userdata_root
from epm.vision.screen_capture import activate_window


@dataclass(frozen=True)
class RecipeFeedback:
    dish_name: str
    taste_score: Optional[float]
    complaints: Any
    file_age_s: float


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _read_json(path: Path) -> dict[str, Any]:
    encodings = ("utf-8-sig", "utf-8", "gbk")
    last: Optional[Exception] = None
    for _ in range(6):
        for enc in encodings:
            try:
                raw = path.read_text(encoding=enc, errors="ignore")
                data = json.loads(raw)
                return data if isinstance(data, dict) else {"data": data}
            except json.JSONDecodeError as e:
                last = e
                continue
            except Exception as e:
                last = e
                break
        time.sleep(0.05)
    raise RuntimeError(f"read_recipe_feedback_failed:{path}: {last}")


def _copy_feedback_to_run(path: Path) -> None:
    run_root = os.environ.get("EPM_RUN_ROOT", "").strip()
    if not run_root:
        return
    dst_dir = Path(run_root)
    if not dst_dir.exists():
        return
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    dst = dst_dir / f"recipe_feedback_{ts}.json"
    if dst.exists():
        dst = dst_dir / f"recipe_feedback_{ts}_{time.time_ns()}.json"
    try:
        shutil.copy2(path, dst)
    except Exception:
        return


def _parse_feedback(data: dict[str, Any], *, file_age_s: float) -> RecipeFeedback:
    dish_name = str(data.get("dishName") or data.get("dish_name") or "").strip()
    taste = data.get("taste") if isinstance(data.get("taste"), dict) else {}
    score = taste.get("score") if isinstance(taste, dict) else None
    taste_score: Optional[float]
    try:
        taste_score = float(score) if score is not None else None
    except Exception:
        taste_score = None
    complaints = data.get("complaints")
    return RecipeFeedback(
        dish_name=dish_name,
        taste_score=taste_score,
        complaints=complaints,
        file_age_s=float(file_age_s),
    )


def get_latest_recipe_feedback(
    *,
    window_title: str = "CookingSimulator",
    max_age_s: float = 2.0,
    wait_update_timeout_s: float = 2.0,
) -> dict[str, Any]:
    """
    GUI action: on the evaluation/feedback screen, press Alt+K to dump the latest recipe feedback,
    then parse `recipe_feedback_latest.json`.

    Reads:
      - dishName
      - taste.score
      - taste.components

    Freshness:
      - If the file is older than `max_age_s`, returns a warning (and may still succeed if parseable).
    """

    try:
        activate_window(window_title)
    except Exception as e:
        return {"success": False, "error": f"activate_window_failed:{e}"}

    io = RawInputController()

    path = userdata_root() / "recipe_feedback_latest.json"
    prev_mtime = _mtime(path)

    # Trigger dump
    io.key_down("alt")
    io.key_press("k")
    io.key_up("alt")

    # Wait for file to update (best-effort).
    deadline = time.time() + float(wait_update_timeout_s)
    while time.time() < deadline:
        m = _mtime(path)
        if m and m != prev_mtime:
            break
        time.sleep(0.05)

    # Read/parse
    if not path.exists():
        return {"success": False, "error": f"recipe_feedback_file_missing:{str(path)!r}"}

    age_s = time.time() - _mtime(path)
    try:
        data = _read_json(path)
        fb = _parse_feedback(data, file_age_s=age_s)
    except Exception as e:
        return {"success": False, "error": str(e)}

    # Best-effort copy into run folder for archival.
    _copy_feedback_to_run(path)

    out: dict[str, Any] = {
        "success": True,
        "dishName": fb.dish_name,
        "taste_score": fb.taste_score,
        "complaints": fb.complaints,
        "file_age_s": fb.file_age_s,
        "source_file": str(path),
        "error": "",
    }

    print(
        f"[feedback] dishName={fb.dish_name!r} taste_score={fb.taste_score} "
        f"age_s={fb.file_age_s:.2f} source={str(path.name)!r}"
    )

    if fb.file_age_s > float(max_age_s):
        out["warning"] = (
            f"{path.name} seems stale (age={fb.file_age_s:.2f}s > max_age_s={float(max_age_s):.2f}s). "
            "Make sure the evaluation screen is open and Alt+K is supported by the mod."
        )

    return out
