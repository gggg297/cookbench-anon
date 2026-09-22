"""
auto_flipping configuration (EPM-adapted).
"""

from __future__ import annotations

import os

from epm.cerebellum.skills._shared_paths import realtime_products_json, userdata_root, window_title

USERDATA_PATH: str = str(userdata_root())
REALTIME_PRODUCTS_JSON: str = str(realtime_products_json())
GAME_WINDOW_TITLE: str = window_title("CookingSimulator")

DEFAULT_PLACE_RADIUS_M: float = float(os.environ.get("COOKGAME_FLIP_PLACE_RADIUS_M", "0.012"))
DEFAULT_MAX_ATTEMPTS: int = int(os.environ.get("COOKGAME_FLIP_MAX_ATTEMPTS", "3"))
POST_FLIP_EXIT_DELAY_S: float = float(os.environ.get("COOKGAME_POST_FLIP_EXIT_DELAY_S", "1.0"))

