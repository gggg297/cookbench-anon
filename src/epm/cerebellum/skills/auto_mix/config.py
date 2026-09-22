"""
auto_mix configuration (EPM-adapted).
"""

from __future__ import annotations

import os

from epm.cerebellum.skills._shared_paths import realtime_products_json, userdata_root, window_title

USERDATA_PATH: str = str(userdata_root())
REALTIME_PRODUCTS_JSON: str = str(realtime_products_json())
GAME_WINDOW_TITLE: str = window_title("CookingSimulator")

SKIP_ACTIVATE_WINDOW = os.environ.get("COOKGAME_SKIP_ACTIVATE_WINDOW", "0").strip().lower() in ("1", "true", "yes")
ACTIVATE_TIMEOUT_S = float(os.environ.get("COOKGAME_ACTIVATE_TIMEOUT_S", "3.0"))
ABORT_ON_ACTIVATE_FAIL = os.environ.get("COOKGAME_ABORT_ON_ACTIVATE_FAIL", "0").strip().lower() in ("1", "true", "yes")

SCREEN_COORDS_ORIGIN = os.environ.get("COOKGAME_SCREEN_ORIGIN", "unity").strip().lower()

DEFAULT_MIX_SECONDS = float(os.environ.get("COOKGAME_MIX_SECONDS", "6.0"))
DEFAULT_DOWNWARD_STEPS = int(os.environ.get("COOKGAME_MIX_DOWNWARD_STEPS", "12"))
DEFAULT_MOVE_PIXELS = int(os.environ.get("COOKGAME_MIX_MOVE_PIXELS", "60"))
DEFAULT_MOVE_INTERVAL_S = float(os.environ.get("COOKGAME_MIX_MOVE_INTERVAL_S", "0.06"))

DEFAULT_MOVE_DISTANCE_M = float(os.environ.get("COOKGAME_MIX_MOVE_DISTANCE_M", "0.2"))
DEFAULT_CALIBRATION_PIXELS = int(os.environ.get("COOKGAME_MIX_CALIBRATION_PIXELS", "120"))
DEFAULT_MIX_LAPS = int(os.environ.get("COOKGAME_MIX_LAPS", "2"))

