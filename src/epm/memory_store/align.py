from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional


def diff_planned_executed(planned: Dict[str, Any], executed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a small diff payload for plan vs executed.

    This is deliberately simple and stable; richer diffs can live in higher-level Monitor code.
    """
    if planned == executed:
        return {"type": "same", "details": ""}
    if planned.get("action_or_skill") != executed.get("action_or_skill"):
        return {
            "type": "action_changed",
            "details": {"planned": planned.get("action_or_skill"), "executed": executed.get("action_or_skill")},
        }
    if planned.get("params") != executed.get("params"):
        return {"type": "param_changed", "details": {"planned": planned.get("params"), "executed": executed.get("params")}}
    return {"type": "replanned", "details": ""}

