from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class InteractionSnapshot:
    """
    Parsed view of CS_CamDump Alt+J interaction info.

    The mod writes key-value pairs to `realtime_interaction_info.txt`, e.g.:
      HasTarget: True
      ItemName: Lemon
      Action: PickUp
      Weight: 123 g
      PourAmount: 49 ml
      ContainerName: Paella Pan
    """

    has_target: bool
    item_name: str
    action: str
    weight: str
    pour_amount: str
    overflow_amount: str
    container_name: str
    container_contents: str
    timestamp: str


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _parse_kv(content: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def read_interaction_snapshot(*, userdata_root: Path, max_age_s: float = 2.0) -> Optional[InteractionSnapshot]:
    path = userdata_root / "realtime_interaction_info.txt"
    if not path.exists():
        return None
    age = time.time() - _mtime(path)
    if age < 0 or age > float(max_age_s):
        # The file may exist but be stale (Alt+J not active, or no new interaction).
        return None

    try:
        content = path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        return None

    data = _parse_kv(content)
    if not data:
        return None

    return InteractionSnapshot(
        has_target=(data.get("HasTarget", "False").strip().lower() == "true"),
        item_name=str(data.get("ItemName", "") or "").strip(),
        action=str(data.get("Action", "") or "").strip(),
        weight=str(data.get("Weight", "") or "").strip(),
        pour_amount=str(data.get("PourAmount", "") or "").strip(),
        overflow_amount=str(data.get("OverflowAmount", "") or "").strip(),
        container_name=str(data.get("ContainerName", "") or "").strip(),
        container_contents=str(data.get("ContainerContents", "") or "").strip(),
        timestamp=str(data.get("Timestamp", "") or "").strip(),
    )


def render_interaction_snapshot(snapshot: InteractionSnapshot) -> str:
    """
    Render a prompt-friendly explanation.

    Field semantics (from CS_CamDump Alt+J writer):
    - HasTarget: whether an interactable target (or pouring) is detected
    - ItemName/Action: what's currently pointed at / interaction UI indicates
    - Weight: UI-reported weight (often grams) when applicable
    - PourAmount/OverflowAmount: UI-reported pouring/overflow totals when pouring is active
    - ContainerName/ContainerContents: when the interaction involves a container
    """

    def _clip(s: str, n: int) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[: max(0, n - 3)] + "..."

    lines: list[str] = []
    lines.append("Alt+J interaction snapshot (what the crosshair/interaction UI reports):")
    lines.append(f"- Target detected: {'yes' if snapshot.has_target else 'no'}")
    if snapshot.item_name:
        lines.append(f"- Target item: {_clip(snapshot.item_name, 120)}")
    if snapshot.action:
        lines.append(f"- Suggested action: {_clip(snapshot.action, 120)}")
    if snapshot.weight:
        lines.append(f"- Weight (if applicable): {_clip(snapshot.weight, 80)}")
    if snapshot.pour_amount or snapshot.overflow_amount:
        if snapshot.pour_amount:
            lines.append(f"- Pour amount (total): {_clip(snapshot.pour_amount, 80)}")
        if snapshot.overflow_amount:
            lines.append(f"- Overflow amount: {_clip(snapshot.overflow_amount, 80)}")
    if snapshot.container_name:
        lines.append(f"- Container: {_clip(snapshot.container_name, 120)}")
    if snapshot.container_contents:
        lines.append(f"- Container contents: {_clip(snapshot.container_contents, 220)}")
        if snapshot.pour_amount:
            lines.append("- Note: `Container contents` may be incomplete while pouring; point at the container when NOT pouring to get the full list.")
    if snapshot.timestamp:
        lines.append(f"- Timestamp: {_clip(snapshot.timestamp, 64)}")
    return "\n".join(lines).strip()
