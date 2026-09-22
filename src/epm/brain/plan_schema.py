from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    type: str  # "action" | "skill"
    name: str
    args: Dict[str, Any]
    expectation: str = ""


@dataclass(frozen=True)
class PlanResponse:
    high_level_id: str
    goal: str
    explanation: Optional[str]
    thoughts: str
    action_list: List[PlanStep]


_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)

def _iter_json_object_candidates(text: str) -> list[str]:
    """
    Return candidate JSON-object substrings using a balanced-brace scan.

    This is more robust than "first '{' .. last '}'" when the model outputs
    extra braces in thoughts, tool specs, etc.
    """
    candidates: list[str] = []
    s = text
    n = len(s)
    i = 0
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        start = i
        depth = 0
        in_str = False
        escape = False
        j = i
        while j < n:
            ch = s[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(s[start : j + 1])
                        break
            j += 1
        i = start + 1
    return candidates


def extract_json_object(text: str, *, strip_think_tags: bool = True) -> Dict[str, Any]:
    """
    Best-effort extraction for LLM output:
    - strips ```json fences
    - optionally strips <think>...</think> (keeps only the final answer portion)
    - expects a single JSON object
    """
    cleaned = _CODE_FENCE_RE.sub("", text.strip()).strip()
    if strip_think_tags:
        # Qwen "thinking" models often emit a <think>...</think> block before the final answer.
        # For parsing we keep only the content AFTER the closing tag to avoid JSON-like braces inside CoT.
        lower = cleaned.lower()
        end_tag = "</think>"
        idx = lower.rfind(end_tag)
        if idx != -1:
            cleaned = cleaned[idx + len(end_tag) :].strip()
    if not cleaned:
        raise ValueError("empty_response")

    # First try: direct parse (fast path)
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Second try: scan for balanced JSON object candidates.
    #
    # Important: if the model output is truncated, the *outer* plan object may be unbalanced,
    # but inner dicts (e.g. a single step) are still balanced. In that case, naive "first dict"
    # selection leads to confusing errors like missing_high_level_id. We therefore score and
    # prefer candidates that look like a plan: has high_level_id + action_list.
    candidates = _iter_json_object_candidates(cleaned)
    best: tuple[int, int, Dict[str, Any]] | None = None  # (score, length, obj)
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        score = 0
        if "high_level_id" in obj:
            score += 100
        if isinstance(obj.get("action_list"), list):
            score += 80
        if "goal" in obj:
            score += 20
        if "thoughts" in obj:
            score += 5
        tup = (score, len(cand), obj)
        if best is None or tup[:2] > best[:2]:
            best = tup
    if best is not None:
        return best[2]

    raise ValueError("no_json_object_found")


def _as_str(v: Any, *, allow_none: bool = False) -> Optional[str]:
    if v is None and allow_none:
        return None
    if isinstance(v, str):
        return v
    if v is None:
        return ""
    return str(v)


def _as_dict(v: Any) -> Dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _normalize_step_id(high_level_id: str, idx: int, provided: Optional[str]) -> str:
    if isinstance(provided, str) and provided.strip():
        candidate = provided.strip()
        # Models sometimes emit placeholder ids like "Hk.A1". Enforce consistency with
        # the current high_level_id so downstream alignment (task_progress) works.
        if candidate.startswith(f"{high_level_id}."):
            return candidate
    return f"{high_level_id}.A{idx}"


def parse_plan_response(obj: Dict[str, Any]) -> PlanResponse:
    high_level_id = _as_str(obj.get("high_level_id")) or ""
    if not high_level_id:
        raise ValueError("missing_high_level_id")

    goal = _as_str(obj.get("goal")) or ""
    explanation = _as_str(obj.get("explanation"), allow_none=True)
    thoughts = _as_str(obj.get("thoughts")) or ""

    raw_actions = obj.get("action_list")
    if not isinstance(raw_actions, list):
        raise ValueError("missing_action_list")

    steps: List[PlanStep] = []
    for i, item in enumerate(raw_actions, start=1):
        if not isinstance(item, dict):
            continue
        step_id = _normalize_step_id(high_level_id, i, _as_str(item.get("step_id"), allow_none=True))
        step_type = (_as_str(item.get("type")) or "action").strip().lower()
        if step_type not in ("action", "skill"):
            step_type = "action"
        name = (_as_str(item.get("name")) or "").strip()
        args = _as_dict(item.get("args"))
        expectation = (_as_str(item.get("expectation")) or "").strip()
        if not name:
            continue
        steps.append(PlanStep(step_id=step_id, type=step_type, name=name, args=args, expectation=expectation))

    if not steps:
        raise ValueError("empty_action_list")
    return PlanResponse(
        high_level_id=high_level_id,
        goal=goal,
        explanation=explanation,
        thoughts=thoughts,
        action_list=steps,
    )


def validate_plan_against_allowlist(
    plan: PlanResponse,
    *,
    allowed_actions: Sequence[str],
    allowed_skills: Sequence[str],
) -> Tuple[bool, str]:
    action_set = set(allowed_actions)
    skill_set = set(allowed_skills)
    for step in plan.action_list:
        if step.type == "action":
            if step.name not in action_set:
                return False, f"unknown_action:{step.name}"
        if step.type == "skill":
            if step.name not in skill_set:
                return False, f"unknown_skill:{step.name}"
    return True, ""
