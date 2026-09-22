from __future__ import annotations

"""
NOTE: This module is intentionally not named `types.py` at import time when running as a script.
It previously shadowed the Python stdlib `types` module if executed via a file path under this folder.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PlanRef:
    high_level_id: str
    atomic_step_id: str


@dataclass(frozen=True)
class LoopDetection:
    loop_detected: bool
    consecutive_failures: int
    last_error_types: list[str]


@dataclass(frozen=True)
class StepRecord:
    """
    The canonical step record for EPM.

    This structure matches the JSONL schema described in `epm/memory/long_horizon_history.txt`.
    """

    step_id: int
    time: str
    observation_summary: str
    action_or_skill: str
    params: Dict[str, Any]
    result_summary: str
    errors: str
    loop_detection: LoopDetection
    screenshot_path: Optional[str] = None
    duration_s: Optional[float] = None
    episode_elapsed_s: Optional[float] = None

    # A-scheme alignment (optional)
    plan_ref: Optional[PlanRef] = None

    # Optional plan-vs-executed payloads (keep minimal; can be filled by Monitor)
    planned: Optional[Dict[str, Any]] = None
    executed: Optional[Dict[str, Any]] = None
    diff: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class Observation:
    time: str
    frame_id: str
    screenshot_path: Optional[str]
    state: Dict[str, Any]
    objects: list[Dict[str, Any]]
    events: list[Dict[str, Any]]


@dataclass(frozen=True)
class Decision:
    action_or_skill: str
    params: Dict[str, Any]
    plan_ref: Optional[PlanRef] = None
