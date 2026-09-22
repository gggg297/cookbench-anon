from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from epm.cerebellum.cookbench_api import ACTION_DISPATCHER, resolve_action_callable


@dataclass(frozen=True)
class ActionSpec:
    name: str
    signature: str
    doc: str


def _first_line(doc: Optional[str]) -> str:
    if not doc:
        return ""
    return doc.strip().splitlines()[0].strip()


def _format_signature(fn: Callable[..., object]) -> str:
    try:
        sig = str(inspect.signature(fn))
    except Exception:
        sig = "(...)"
    return sig


def collect_action_specs(*, action_names: Optional[List[str]] = None) -> List[ActionSpec]:
    """
    Collect Python-level action signatures from the semantic action dispatcher.

    This is intended to be injected into the planner prompt so the model uses
    correct action names AND correct kwargs.
    """

    names = action_names or sorted(ACTION_DISPATCHER.keys())
    specs: List[ActionSpec] = []
    for name in names:
        fn = resolve_action_callable(name)
        if fn is None:
            continue
        specs.append(ActionSpec(name=name, signature=_format_signature(fn), doc=_first_line(getattr(fn, "__doc__", ""))))
    return specs


def to_prompt_text(*, action_names: Optional[List[str]] = None) -> str:
    specs = collect_action_specs(action_names=action_names)
    lines: List[str] = []
    lines.append("# Action Specs (authoritative)")
    lines.append("# Use ONLY these action names and ONLY these parameters.")
    lines.append("")
    for s in specs:
        trailer = f"  # {s.doc}" if s.doc else ""
        lines.append(f"- {s.name}{s.signature}{trailer}".rstrip())
    return "\n".join(lines).strip() + "\n"
