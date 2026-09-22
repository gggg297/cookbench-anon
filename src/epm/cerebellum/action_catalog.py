from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class CatalogAction:
    name: str
    note: str
    signature: str
    returns: str


@dataclass(frozen=True)
class CatalogCategory:
    name: str
    actions: List[CatalogAction]


_CATEGORY_RE = re.compile(r"^\s*\[(?P<name>[^\]]+)\]\s*(?:\(|（)?")
_ACTION_RE = re.compile(
    r"^\s*-\s*(?P<name>[^|\s]+)\s*\|\s*备注=(?P<note>[^|]*)\|\s*输入=\((?P<input>[^)]*)\)\s*\|\s*输出=(?P<out>.*)\s*$"
)


class ActionCatalog:
    def __init__(self, categories: List[CatalogCategory]) -> None:
        self.categories = categories

    @staticmethod
    def from_text(path: str | Path) -> "ActionCatalog":
        p = Path(path)
        lines = p.read_text(encoding="utf-8").splitlines()

        categories: List[CatalogCategory] = []
        current_name: Optional[str] = None
        current_actions: List[CatalogAction] = []

        def _flush() -> None:
            nonlocal current_name, current_actions
            if current_name is None:
                return
            categories.append(CatalogCategory(name=current_name, actions=list(current_actions)))
            current_name = None
            current_actions = []

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            cm = _CATEGORY_RE.match(line)
            if cm:
                _flush()
                current_name = cm.group("name").strip()
                continue

            am = _ACTION_RE.match(line)
            if am and current_name is not None:
                current_actions.append(
                    CatalogAction(
                        name=am.group("name").strip(),
                        note=am.group("note").strip(),
                        signature=f"({am.group('input').strip()})",
                        returns=am.group("out").strip(),
                    )
                )
                continue

        _flush()
        return ActionCatalog(categories=categories)

    def list_category_names(self) -> List[str]:
        return [c.name for c in self.categories]

    def select(self, *, category_queries: Iterable[str]) -> "ActionCatalog":
        queries = [q.strip() for q in category_queries if str(q).strip()]
        if not queries:
            return ActionCatalog(categories=[])

        selected: List[CatalogCategory] = []
        for cat in self.categories:
            for q in queries:
                if q == cat.name or q in cat.name:
                    selected.append(cat)
                    break
        return ActionCatalog(categories=selected)

    def to_prompt_text(self, *, max_actions_per_category: Optional[int] = None) -> str:
        lines: List[str] = []
        for cat in self.categories:
            lines.append(f"[{cat.name}]")
            actions = cat.actions
            if max_actions_per_category is not None:
                limit = int(max_actions_per_category)
                if limit > 0:
                    actions = actions[:limit]
            for a in actions:
                lines.append(f"- {a.name} {a.signature}  # {a.note}".rstrip())
            lines.append("")
        return "\n".join(lines).strip() + ("\n" if lines else "")
