from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from dataclasses import fields as dataclass_fields

from epm.brain.tool_schema import (
    dataclass_to_json_schema,
    signature_to_json_schema,
    _parse_args_descriptions,
    _sanitize_doc_text,
)
from epm.cerebellum.cookbench_api import resolve_action_callable
from epm.cerebellum.skills import registry as skill_registry


@dataclass(frozen=True)
class ToolDef:
    name: str
    kind: str  # "action" | "skill"
    target: str  # underlying action/skill name
    description: str
    parameters: Dict[str, Any]


_LOCAL_ACTIONS_TAG_CACHE: dict[str, str] | None = None
_GUI_ACTION_FLOW_CACHE: dict[str, str] | None = None


def _load_local_actions_tags(*, max_lines: int = 855) -> dict[str, str]:
    """
    Best-effort parse of section/comment tags in `epm.cerebellum.local_actions`.

    The repo has a human-authored comment structure in the first part of local_actions.py
    (e.g. lines containing "→ 3.12.5 冰箱控制"). Users want these tags reflected in tool
    descriptions so GPT sees the intended grouping/context.
    """
    global _LOCAL_ACTIONS_TAG_CACHE
    if _LOCAL_ACTIONS_TAG_CACHE is not None:
        return dict(_LOCAL_ACTIONS_TAG_CACHE)

    tags: dict[str, str] = {}
    try:
        import re
        from pathlib import Path

        import epm.cerebellum.local_actions as la  # type: ignore

        path = Path(getattr(la, "__file__", "")).resolve()
        if not path.exists():
            _LOCAL_ACTIONS_TAG_CACHE = {}
            return {}

        current_tag = ""
        # Example header lines:
        #   #           → 3.12.5 冰箱控制
        header_re = re.compile(r"→\s*(?P<tag>[0-9.]+\s+[^\r\n#]+)")
        def_re = re.compile(r"^\s*def\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")

        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for i, raw in enumerate(lines[: int(max_lines)]):
            line = raw.strip("\ufeff")
            hm = header_re.search(line)
            if hm:
                current_tag = hm.group("tag").strip()
                continue
            dm = def_re.match(line)
            if dm and current_tag:
                name = dm.group("name").strip()
                tags.setdefault(name, current_tag)
        _LOCAL_ACTIONS_TAG_CACHE = dict(tags)
        return dict(tags)
    except Exception:
        _LOCAL_ACTIONS_TAG_CACHE = {}
        return {}


def _strip_doc_sections(doc: str) -> str:
    """
    Clean action docstrings for tool `description`.

    User preference:
    - Do NOT include "Args:", "Returns:", "Flow:" or "Manual test:" blocks in tool descriptions.
      (Args are used separately to populate JSON Schema parameter descriptions.)
    """
    doc = (doc or "").strip()
    if not doc:
        return ""

    out_lines: list[str] = []
    lines = doc.splitlines()
    skip = False
    skip_mode = ""

    def _is_section_header(s: str) -> bool:
        return s in {"Args:", "Returns:"} or s.startswith(("Args:", "Returns:", "Flow:", "Manual test:", "GPT tool guidance:"))

    for raw in lines:
        s = raw.rstrip()
        st = s.strip()

        # Begin skipping blocks.
        if st == "Args:":
            skip = True
            skip_mode = "args"
            continue
        if st == "Returns:":
            skip = True
            skip_mode = "returns"
            continue
        if st == "Flow:":
            skip = True
            skip_mode = "flow"
            continue
        if st.startswith("Manual test:"):
            skip = True
            skip_mode = "manual"
            continue

        # End skipping when we hit another header (but keep that header).
        if skip and _is_section_header(st) and not (st == "Args:" or st == "Returns:" or st == "Flow:" or st.startswith("Manual test:")):
            skip = False
            skip_mode = ""

        if skip:
            # In Manual test blocks, also skip subsequent indented command lines.
            continue

        out_lines.append(s)

    # Remove accidental empty leading/trailing lines and compress excessive blank lines.
    cleaned: list[str] = []
    blank_streak = 0
    for line in out_lines:
        if not line.strip():
            blank_streak += 1
            if blank_streak <= 1:
                cleaned.append("")
            continue
        blank_streak = 0
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _decorate_action_description(*, action_name: str, base_desc: str) -> str:
    desc = (base_desc or "").strip()

    tags = _load_local_actions_tags()
    tag = tags.get(str(action_name), "").strip()
    if tag:
        desc = f"[{tag}] {desc}" if desc else f"[{tag}]"

    return desc


def _load_gui_action_flows() -> dict[str, str]:
    """
    Best-effort parse of GUI flow notes from:
      epm.cerebellum.gui_actions/GUI_ACTIONS_QUICKSTART.txt

    We treat lines like:
      python ... --action <name> ...
      (Flow: ...)
    as additional action documentation, appended to tool descriptions.
    """
    global _GUI_ACTION_FLOW_CACHE
    if _GUI_ACTION_FLOW_CACHE is not None:
        return dict(_GUI_ACTION_FLOW_CACHE)

    flows: dict[str, str] = {}
    try:
        import re
        from pathlib import Path

        import epm.cerebellum.gui_actions as ga  # type: ignore

        path = (Path(getattr(ga, "__file__", "")).resolve().parent / "GUI_ACTIONS_QUICKSTART.txt").resolve()
        if not path.exists():
            _GUI_ACTION_FLOW_CACHE = {}
            return {}

        action_re = re.compile(r"--action\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
        flow_re = re.compile(r"\(Flow:\s*(?P<flow>.*?)(?:\)\s*)?$")

        last_action = ""
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for raw in lines:
            line = raw.strip("\ufeff").strip()
            if not line:
                continue
            am = action_re.search(line)
            if am:
                last_action = am.group("name").strip()
                continue
            fm = flow_re.search(line)
            if fm and last_action:
                flows.setdefault(last_action, fm.group("flow").strip())
                last_action = ""

        _GUI_ACTION_FLOW_CACHE = dict(flows)
        return dict(flows)
    except Exception:
        _GUI_ACTION_FLOW_CACHE = {}
        return {}


def build_tool_manifest(*, allowed_actions: Sequence[str], allowed_skills: Sequence[str]) -> List[ToolDef]:
    """
    Build a tool manifest from the currently-allowed actions/skills.

    Tool names are prefixed to avoid collisions and make routing unambiguous:
    - action__<action_name>
    - skill__<skill_name>
    """
    tools: List[ToolDef] = []

    for a in allowed_actions:
        fn = resolve_action_callable(a)
        if fn is None:
            continue
        desc = _strip_doc_sections(getattr(fn, "__doc__", "") or "") or f"Execute action {a}."
        desc = _decorate_action_description(action_name=str(a), base_desc=desc)
        desc = _sanitize_doc_text(desc)
        tools.append(
            ToolDef(
                name=f"action__{a}",
                kind="action",
                target=a,
                description=desc,
                parameters=signature_to_json_schema(fn),
            )
        )

    for s in allowed_skills:
        # Prefer local_actions docstring (if present) for skill description and arg help.
        la_doc = ""
        try:
            import epm.cerebellum.local_actions as la  # type: ignore

            # Map skill names to local_actions entrypoints when names differ.
            name_map = {
                "auto_filp": "auto_flip",
                "auto_cutting": "auto_cut",
                "auto_flipping": "auto_flip",
                "auto_pouring": "auto_pour",
                "auto_sprinkling": "auto_sprinkle",
            }
            la_name = name_map.get(s, s)
            la_fn = getattr(la, la_name, None)
            if callable(la_fn):
                la_doc = getattr(la_fn, "__doc__", "") or ""
        except Exception:
            la_doc = ""

        args_cls = skill_registry.SKILL_ARG_CLASSES.get(s)
        drop = set(skill_registry.SKILL_DROP_FIELDS.get(s, set()))
        if args_cls is not None:
            if la_doc:
                allowed_names = {f.name for f in dataclass_fields(args_cls)}
                arg_desc = _parse_args_descriptions(la_doc, allowed_names=allowed_names)
            else:
                arg_desc = {}
            params = dataclass_to_json_schema(args_cls, drop_fields=drop, arg_desc=arg_desc)
        else:
            # Fallback: no args
            params = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
        desc = _strip_doc_sections(la_doc) if la_doc else ""
        if not desc:
            desc = (skill_registry.SKILL_DESCRIPTIONS.get(s) or "").strip() or f"Execute skill {s}."
        desc = _sanitize_doc_text(desc)
        tools.append(
            ToolDef(
                name=f"skill__{s}",
                kind="skill",
                target=s,
                description=desc,
                parameters=params,
            )
        )

    return tools


def to_openai_tools(tools: Sequence[ToolDef]) -> List[Dict[str, Any]]:
    """
    Convert ToolDef list to OpenAI-compatible `tools=[{type:'function', function:{...}}]`.
    """
    out: List[Dict[str, Any]] = []
    for t in tools:
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
        )
    return out
