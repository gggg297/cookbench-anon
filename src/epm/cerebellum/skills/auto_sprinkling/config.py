"""
auto_sprinkling configuration (EPM-adapted).
"""

from __future__ import annotations

import os

from epm.cerebellum.skills._shared_paths import realtime_products_json, userdata_root, window_title

USERDATA_PATH: str = str(userdata_root())
REALTIME_PRODUCTS_JSON: str = str(realtime_products_json())
GAME_WINDOW_TITLE: str = window_title("CookingSimulator")

DEFAULT_SPRINKLE_NUM: int = int(os.environ.get("COOKGAME_SPRINKLE_NUM", "3"))
ACTION_DELAY_SECONDS: float = float(os.environ.get("COOKGAME_SPRINKLE_ACTION_DELAY_S", "0.08"))
POST_CLICK_SLEEP_SECONDS: float = float(os.environ.get("COOKGAME_SPRINKLE_POST_CLICK_SLEEP_S", "0.8"))

GRAMS_TRACKING_ENABLED: bool = os.environ.get("COOKGAME_SPRINKLE_GRAMS_TRACKING", "1").strip().lower() in ("1", "true", "yes")
GRAMS_TRACKING_MODE: str = os.environ.get("COOKGAME_SPRINKLE_GRAMS_TRACKING_MODE", "interaction").strip().lower()

