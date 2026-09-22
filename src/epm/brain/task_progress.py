from __future__ import annotations

"""
Task-progress writer for long-horizon objective semantic state summaries.

EPM task_progress is not an atomic action queue. It is a higher-level semantic
memory that summarizes recipe constraints, current subgoal status,
commitments, resource bindings, blockers, timing checkpoints, and observed
execution notes. It should record state, not planning advice.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from epm.kb.recipes import Dish


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _split_recipe_to_sentences(recipe_text: str) -> List[str]:
    text = (recipe_text or "").replace("\n", " ").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(".") if p.strip()]
    return parts[:50]


def _base_recipe_payload(dish: Dish) -> dict[str, Any]:
    goal = (dish.recipe_text or "").replace("\n", " ").strip()
    if len(goal) > 1600:
        goal = goal[:1600] + "..."
    steps = _split_recipe_to_sentences(dish.recipe_text)
    return {
        "recipe_id": str(dish.id),
        "recipe_name": str(dish.dish_name),
        "raw_text": goal,
        "steps": steps,
    }


def write_task_progress_epm(*, path: str | Path, dish: Dish) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "version": 3,
        "update_policy": "semantic_overwrite",
        "last_updated": _utc_now_iso(),
        "recipe": _base_recipe_payload(dish),
        "goal_state": {
            "current_high_level_id": "H1",
            "current_high_level_goal": "Follow recipe",
            "current_subgoal": "",
            "status": "todo",
            "done_criteria": [],
            "current_activity": "",
        },
        "semantic_progress": {
            "completed_tasks": [],
            "recent_high_level_actions": [],
        },
        "commitments": [],
        "resource_bindings": {
            "held_item": "",
            "active_container": "",
            "active_workspace": "",
            "timers": [],
        },
        "blocking_conditions": [],
        "temporal_checkpoints": {
            "force_submit_remaining_steps": None,
            "timer_checkpoints": [],
        },
        "strategy_notes": {
        },
        "evidence": {
            "recipe_constraints": [],
            "last_feedback_summary": "",
            "recent_steps_summary": "",
            "agent_state_summary": "",
        },
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_task_progress_pe(*, path: str | Path, dish: Dish) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "version": 1,
        "update_policy": "rolling_plan",
        "last_updated": _utc_now_iso(),
        "task": {
            "task_id": "",
            "recipe_id": str(dish.id),
            "recipe_name": str(dish.dish_name),
            "goal_description": (dish.recipe_text or "").replace("\n", " ").strip(),
        },
        "planning": {
            "plan_version": 1,
            "plan_status": "idle",
            "step_cursor": 0,
            "high_level_id": "H1",
        },
        "plan_steps": [],
        "off_plan_events": [],
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
