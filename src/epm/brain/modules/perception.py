from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from epm.brain.interfaces import Percept, PerceptionModule
from epm.core.epm_types import Observation
from epm.cerebellum.skills.auto_perception.skill import PerceptionArgs, list_visible_items


@dataclass(frozen=True)
class OraclePerceptionConfig:
    realtime_products_path: Path
    max_items: int = 20
    only_on_screen: bool = True


class OraclePerception(PerceptionModule):
    """
    "透视观察"：从 realtime_products.json 直接读可见物品，作为结构化文本摘要。
    """

    def __init__(self, cfg: OraclePerceptionConfig) -> None:
        self.cfg = cfg

    def run(self, *, observation: Observation) -> Percept:
        try:
            items = list_visible_items(
                PerceptionArgs(
                    realtime_products_path=self.cfg.realtime_products_path,
                    only_on_screen=bool(self.cfg.only_on_screen),
                    max_items=int(self.cfg.max_items),
                )
            )
        except Exception:
            # Perception should not crash the control loop; degrade to empty list.
            items = []
        names = [x.get("name") for x in items if isinstance(x, dict) and x.get("name")]
        max_items = int(self.cfg.max_items)
        if max_items <= 0:
            show_names = names
        else:
            show_names = names[:max_items]
        text = "Visible items (oracle): " + ", ".join([str(n) for n in show_names])
        return Percept(text=text, slots={"visible_items": items})


@dataclass(frozen=True)
class NoPerception(PerceptionModule):
    def run(self, *, observation: Observation) -> Percept:
        return Percept(text="", slots={})
