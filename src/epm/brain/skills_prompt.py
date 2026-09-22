from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SkillCard:
    name: str
    description: str
    allowed_actions: List[str]


_ACTION_SPEC_LINE = re.compile(r"^\s*-\s*(?P<name>[a-zA-Z0-9_]+)\s*(?P<sig>\(.*\))")


def _parse_action_specs(text: str) -> Dict[str, str]:
    """
    Parse `action_specs.txt` into {action_name: "(sig)"}.
    """
    out: Dict[str, str] = {}
    for line in (text or "").splitlines():
        m = _ACTION_SPEC_LINE.match(line.strip())
        if not m:
            continue
        out[m.group("name")] = m.group("sig")
    return out


def _parse_skills_catalog(text: str) -> List[SkillCard]:
    """
    Parse `skills_catalog.json` (written by the agent) into cards.
    """
    if not text.strip():
        return []
    obj = json.loads(text)
    skills = obj.get("skills") if isinstance(obj, dict) else None
    if not isinstance(skills, list):
        return []
    cards: List[SkillCard] = []
    for s in skills:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        desc = s.get("description", "")
        allowed = s.get("allowed_actions", [])
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(desc, str):
            desc = str(desc)
        if not isinstance(allowed, list):
            allowed = []
        allowed_names = [str(a).strip() for a in allowed if isinstance(a, str) and a.strip()]
        if not allowed_names:
            continue
        cards.append(SkillCard(name=name.strip(), description=desc.strip(), allowed_actions=allowed_names))
    return cards


def list_skill_card_names(*, skills_catalog_json: str) -> List[str]:
    """
    Return all SkillCard names in catalog order.

    This is useful when the caller wants to inject the full interface reference
    (no routing/selection).
    """
    return [c.name for c in _parse_skills_catalog(skills_catalog_json)]


def _parse_parameter_heuristics(text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Return (by_action, by_skill) maps from parameter_heuristics.json.
    """
    if not text.strip():
        return {}, {}
    obj = json.loads(text)
    if not isinstance(obj, dict):
        return {}, {}
    by_action = obj.get("by_action") if isinstance(obj.get("by_action"), dict) else {}
    by_skill = obj.get("by_skill") if isinstance(obj.get("by_skill"), dict) else {}
    return by_action, by_skill


def select_skill_cards(
    *,
    skills_catalog_json: str,
    high_level_goal: str,
    recipe_text: str,
    feedback: str,
    max_cards: int = 0,
) -> List[str]:
    """
    Rule-based router: select a small subset of skills to inject into the planner prompt.

    The selection is deliberately simple so it is stable and easy to ablate.
    """
    cards = _parse_skills_catalog(skills_catalog_json)
    if not cards:
        return []

    goal = (high_level_goal or "").lower()
    recipe = (recipe_text or "").lower()
    fb = (feedback or "").lower()
    text = " ".join([goal, recipe, fb])

    def _want_any(words: List[str]) -> bool:
        return any(w in text for w in words)

    selected: List[str] = []

    def _add_by_name_pred(pred) -> None:
        for c in cards:
            if pred(c):
                selected.append(c.name)

    # Always include navigation if present.
    for c in cards:
        if "navigate" in c.name.lower():
            selected.append(c.name)
            break

    # Pouring / liquids (high priority for cooking tasks)
    if _want_any(
        [
            "pour",
            "poured",
            "ml",
            "oil",
            "wine",
            "juice",
            "water",
            "vinegar",
            "soy",
            "liquid",
        ]
    ):
        _add_by_name_pred(lambda c: c.name.lower().startswith("liquid-pourout") or ("pour" in c.name.lower()))

    # Sprinkling / spices (high priority for cooking tasks)
    if _want_any(
        [
            "sprinkle",
            "season",
            "salt",
            "pepper",
            "oregano",
            "thyme",
            "spice",
            "spices",
        ]
    ):
        _add_by_name_pred(lambda c: c.name.lower().startswith("spices-sprinkle") or ("sprinkle" in c.name.lower()))

    # Manipulation
    if _want_any(["pick", "take", "grab", "put", "place", "bowl", "plate", "pan", "pot", "container"]):
        for c in cards:
            n = c.name.lower()
            # Avoid pulling in throw-away / drop-release categories unless explicitly needed.
            if n.startswith("pickupobject(") or n.startswith("putobject("):
                selected.append(c.name)

    # Doors
    if _want_any(["fridge", "refrigerator", "oven", "microwave", "door", "open", "close"]):
        for c in cards:
            n = c.name.lower()
            if "openobject" in n or "closeobject" in n:
                selected.append(c.name)

    # Heating / toggles
    if _want_any(["heat", "cook", "boil", "fry", "turn on", "turn off", "switch", "toggle"]):
        for c in cards:
            n = c.name.lower()
            if "toggleon" in n or "toggleoff" in n:
                selected.append(c.name)

    # Cutting
    if _want_any(["cut", "slice", "chop", "dice", "cutting"]):
        for c in cards:
            if "sliceobject" in c.name.lower() or "cut" in c.name.lower():
                selected.append(c.name)

    # Mixing / blending
    if _want_any(["mix", "mixture", "blend", "blender", "stir", "whisk", "puree"]):
        _add_by_name_pred(lambda c: "blender" in c.name.lower() or "mixing" in c.name.lower())

    # Common tool interactions (helpful when the agent is stuck in a mode)
    if _want_any(["ladle", "scoop"]):
        _add_by_name_pred(lambda c: "ladle" in c.name.lower())
    if _want_any(["tongs", "flip", "turn over"]):
        _add_by_name_pred(lambda c: "tongs" in c.name.lower() or "flip" in c.name.lower())
    if _want_any(["faucet", "tap", "sink", "water"]):
        _add_by_name_pred(lambda c: "faucet" in c.name.lower())

    # Debug: if model invented actions, force more interface context
    if "unknown_action" in fb or "invalid_plan" in fb:
        for c in cards:
            if c.name not in selected:
                selected.append(c.name)

    # Deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for name in selected:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        if int(max_cards) > 0 and len(out) >= int(max_cards):
            break
    return out


def render_skill_cards_for_prompt(
    *,
    skills_catalog_json: str,
    action_specs_text: str,
    parameter_heuristics_json: str,
    selected_skill_names: List[str],
) -> str:
    """
    Render selected skill cards in an Anthropic-skills-inspired style.
    """
    cards = _parse_skills_catalog(skills_catalog_json)
    by_name = {c.name: c for c in cards}
    specs = _parse_action_specs(action_specs_text)
    by_action, by_skill = _parse_parameter_heuristics(parameter_heuristics_json)

    lines: List[str] = []
    lines.append("Skill Cards (use these like tools; ONLY call actions listed below):")

    for name in selected_skill_names:
        c = by_name.get(name)
        if c is None:
            continue
        lines.append("")
        lines.append(f"[SKILL] {c.name}")
        if c.description:
            lines.append(f"Description: {c.description}")
        skill_hint = by_skill.get(c.name) if isinstance(by_skill, dict) else None
        if isinstance(skill_hint, dict) and (skill_hint.get("defaults") or skill_hint.get("ranges") or skill_hint.get("notes")):
            lines.append("Skill hints: " + json.dumps(skill_hint, ensure_ascii=False))
        lines.append("Allowed actions (use exact name + exact kwargs):")
        for a in c.allowed_actions:
            sig = specs.get(a, "(...)")
            hint = by_action.get(a) if isinstance(by_action, dict) else None
            if isinstance(hint, dict) and (hint.get("defaults") or hint.get("ranges") or hint.get("notes")):
                lines.append(f"- {a}{sig}  # hints={json.dumps(hint, ensure_ascii=False)}")
            else:
                lines.append(f"- {a}{sig}")

    return "\n".join(lines).strip() + "\n"
