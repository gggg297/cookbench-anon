"""
auto_navigation package configuration (EPM-adapted).
"""

from __future__ import annotations

from epm.cerebellum.skills._shared_paths import userdata_root

DATA_ROOT: str = str(userdata_root())
