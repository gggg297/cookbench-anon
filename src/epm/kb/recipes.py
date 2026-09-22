from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Dish:
    id: int
    dish_name: str
    ingredients: List[str]
    recipe_text: str
    raw: Dict[str, Any]


def _stringify_recipe_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return "\n".join(parts)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, child in value.items():
            child_text = _stringify_recipe_value(child)
            if child_text:
                parts.append(f"{str(key).strip()}:\n{child_text}")
        return "\n\n".join(parts).strip()
    return str(value).strip()


def _render_recipe_text(*, dish_name: str, ingredients: List[str], recipe_obj: Dict[str, Any]) -> str:
    parts: list[str] = []
    if dish_name:
        parts.append(f"Dish: {dish_name}")
    clean_ingredients = [str(x).strip() for x in ingredients if str(x).strip()]
    if clean_ingredients:
        parts.append("Ingredients:")
        parts.extend(f"- {name}" for name in clean_ingredients)
    if recipe_obj:
        if parts:
            parts.append("")
        parts.append("Recipe sections:")
        for section_name, section_value in recipe_obj.items():
            text = _stringify_recipe_value(section_value)
            if not text:
                continue
            name = str(section_name).strip()
            if name:
                parts.append(f"[{name}]")
            parts.append(text)
            parts.append("")
    return "\n".join(parts).strip()


def load_dishes(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {p}, got {type(data).__name__}")
    return data


def get_dish_by_id(path: str | Path, dish_id: int) -> Dish:
    dishes = load_dishes(path)
    matched = next((d for d in dishes if isinstance(d, dict) and d.get("id") == dish_id), None)
    if not matched:
        raise KeyError(f"Dish id={dish_id} not found in {path}")

    dish_name = str(matched.get("dish_name", "")).strip() or f"dish_{dish_id}"
    ingredients = matched.get("ingredients") if isinstance(matched.get("ingredients"), list) else []

    recipe_obj = matched.get("recipe") if isinstance(matched.get("recipe"), dict) else {}
    recipe_text = _render_recipe_text(
        dish_name=dish_name,
        ingredients=[str(x) for x in ingredients],
        recipe_obj=recipe_obj,
    )

    return Dish(
        id=int(matched["id"]),
        dish_name=dish_name,
        ingredients=[str(x) for x in ingredients],
        recipe_text=recipe_text,
        raw=matched,
    )
