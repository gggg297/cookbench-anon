"""
auto_cutting configuration (EPM-adapted).
"""

from __future__ import annotations

from epm.cerebellum.skills._shared_paths import realtime_products_json, userdata_root, window_title

USERDATA_PATH: str = str(userdata_root())
REALTIME_PRODUCTS_JSON: str = str(realtime_products_json())
GAME_WINDOW_TITLE: str = window_title("CookingSimulator")

DEFAULT_CUT_COUNT: int = 3
DEFAULT_CUT_INTERVAL: float = 0.25
