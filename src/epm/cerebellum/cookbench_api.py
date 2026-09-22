from __future__ import annotations

"""
`cookbench_api` is the Cerebellum facade: the single entry-point for action execution.

In this repo it currently runs in "local" mode (keyboard/mouse driven semantic actions),
but the interface is designed so a real CookBench backend can be swapped in later.
"""

import json
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol, get_args, get_origin

from epm.cerebellum import local_actions
from epm.cerebellum.local_actions import ACTION_DISPATCHER


RAW_INPUT_ACTION_NAMES = frozenset(
    {
        "press_keyboard",
        "hold_keyboard",
        "leave_keyboard",
        "click_mouse",
        "hold_mouse",
        "leave_mouse",
        "move_related_mouse",
        "scroll_down_the_wheel",
        "scroll_up_the_wheel",
    }
)


def resolve_action_callable(name: str):
    """Resolve a semantic action, or a raw input primitive explicitly allowlisted by name."""
    fn = ACTION_DISPATCHER.get(name)
    if fn is not None:
        return fn
    if name in RAW_INPUT_ACTION_NAMES:
        candidate = getattr(local_actions, name, None)
        return candidate if callable(candidate) else None
    return None


@dataclass(frozen=True)
class ActionResult:
    success: bool
    raw: Dict[str, Any]
    error: str = ""


class ActionAPI(Protocol):
    def list_actions(self) -> list[str]: ...
    def call_action(self, name: str, **params: Any) -> ActionResult: ...


@dataclass(frozen=True)
class LocalActionAPI(ActionAPI):
    """
    Action API implemented via `local_actions.ACTION_DISPATCHER`.

    `action_aliases.json` allows renaming high-level action names without changing code.
    """

    aliases_path: Path = Path(__file__).with_name("action_aliases.json")
    raw_input_only: bool = False

    @staticmethod
    def _coerce_param_value(value: Any, parameter: inspect.Parameter) -> Any:
        if not isinstance(value, str):
            return value

        raw = value.strip()
        if not raw:
            return value

        annotation = parameter.annotation
        if annotation is inspect._empty and parameter.default is not inspect._empty:
            annotation = type(parameter.default)

        origin = get_origin(annotation)
        if origin is not None:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                annotation = args[0]

        try:
            if annotation is bool:
                lowered = raw.lower()
                if lowered in {"true", "1", "yes", "y", "on"}:
                    return True
                if lowered in {"false", "0", "no", "n", "off"}:
                    return False
                return value
            if annotation is int:
                return int(float(raw))
            if annotation is float:
                return float(raw)
        except Exception:
            return value

        if parameter.default is not inspect._empty:
            default = parameter.default
            try:
                if isinstance(default, bool):
                    lowered = raw.lower()
                    if lowered in {"true", "1", "yes", "y", "on"}:
                        return True
                    if lowered in {"false", "0", "no", "n", "off"}:
                        return False
                if isinstance(default, int) and not isinstance(default, bool):
                    return int(float(raw))
                if isinstance(default, float):
                    return float(raw)
            except Exception:
                return value

        return value

    def list_actions(self) -> list[str]:
        if self.raw_input_only:
            return sorted(name for name in RAW_INPUT_ACTION_NAMES if resolve_action_callable(name) is not None)
        # Some high-level GUI flows are treated as "skills" (composite behaviors) for GPT tooling.
        # Keep them callable via ACTION_DISPATCHER / manual debugging, but hide them from the
        # action allowlist so they appear as `skill__*` tools instead of `action__*`.
        skill_like_actions = {
            # Treat composite behaviors as skills in the tool manifest.
            "auto_cut",
            "auto_filp",
            "auto_flip",
            "auto_mix",
            "auto_navigation",
            "auto_pour",
            "auto_sprinkle",
            "gui_buy_new_item",
            "gui_order_dish_via_computer",
            "gui_submit_dish_via_checkout_stand",
        }
        return sorted([k for k in ACTION_DISPATCHER.keys() if k not in skill_like_actions])

    def call_action(self, name: str, **params: Any) -> ActionResult:
        resolved = name
        if self.aliases_path.exists():
            try:
                aliases = json.loads(self.aliases_path.read_text(encoding="utf-8"))
                if isinstance(aliases, dict) and isinstance(aliases.get(name), str):
                    resolved = str(aliases[name])
            except Exception:
                resolved = name

        fn = resolve_action_callable(resolved)
        if fn is None:
            return ActionResult(False, raw={}, error=f"unknown_action:{name}")
        try:
            # Be forgiving about extra args coming from LLM plans:
            # drop unexpected kwargs unless the action explicitly accepts **kwargs.
            filtered = dict(params)
            try:
                sig = inspect.signature(fn)
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                if not accepts_var_kw:
                    allowed = {k for k, p in sig.parameters.items() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
                    filtered = {k: v for k, v in filtered.items() if k in allowed}
                filtered = {
                    k: self._coerce_param_value(v, sig.parameters[k])
                    if k in sig.parameters else v
                    for k, v in filtered.items()
                }
            except Exception:
                filtered = dict(params)

            out = fn(**filtered)

            # Convention: if an action returns a dict with a boolean `success` field,
            # use that as the executor success flag (so higher-level wrappers can
            # report structured failures without raising exceptions).
            if isinstance(out, dict) and isinstance(out.get("success"), bool):
                ok = bool(out.get("success"))
                err = ""
                if not ok:
                    err = str(out.get("error") or "action_returned_success_false")
                return ActionResult(ok, raw={"return": out, "resolved_action": resolved, "filtered_params": filtered}, error=err)

            return ActionResult(True, raw={"return": out, "resolved_action": resolved, "filtered_params": filtered})
        except Exception as e:
            return ActionResult(False, raw={}, error=str(e))


# Backward-compat / narrative name used by the paper/doc.
CookBenchActionAPI = LocalActionAPI
