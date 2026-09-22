from __future__ import annotations

import inspect
import re
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, Dict, Literal, Optional, get_args, get_origin


def _json_type_for(py_type: Any) -> str:
    origin = get_origin(py_type)
    if origin is Literal:
        # Literal["left","right"] -> string
        return "string"
    if py_type in (int,):
        return "integer"
    if py_type in (float,):
        return "number"
    if py_type in (bool,):
        return "boolean"
    return "string"


def _enum_for(py_type: Any) -> list[Any] | None:
    origin = get_origin(py_type)
    if origin is Literal:
        vals = list(get_args(py_type))
        return vals if vals else None
    return None


def _parse_args_descriptions(doc: str, allowed_names: set[str] | None = None) -> dict[str, str]:
    """
    Parse a simple docstring "Args:" section into {param_name: description}.

    Supported format (common in this repo):

      Args:
        foo: explanation...
        bar: explanation...

    Parsing is best-effort; unknown formats are ignored.
    """
    doc = (doc or "").strip()
    if not doc:
        return {}

    lines = doc.splitlines()
    start = None
    for i, raw in enumerate(lines):
        if raw.strip() == "Args:":
            start = i + 1
            break
    if start is None:
        return {}

    out: dict[str, str] = {}
    # Stop when we hit another top-level section header.
    stop_re = re.compile(r"^\s*(Returns|Return|Raises|Examples|Manual test|GPT tool guidance|Flow)\s*:")
    item_re = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*(?P<desc>.+?)\s*$")

    current_name: str | None = None
    current_desc: list[str] = []

    for raw in lines[start:]:
        if stop_re.match(raw.strip()):
            break

        m = item_re.match(raw)
        if m:
            name = m.group("name").strip()
            if allowed_names is not None and name not in allowed_names:
                # Treat non-parameter headers (e.g., "Note:") as continuation if applicable.
                if current_name and raw.startswith((" ", "\t")) and raw.strip():
                    current_desc.append(raw.strip())
                continue
            # flush previous
            if current_name and current_desc:
                out.setdefault(current_name, " ".join(current_desc).strip())
            current_name = name
            current_desc = [m.group("desc").strip()]
            continue

        # Continuation line: indented text after an arg line.
        if current_name and raw.startswith((" ", "\t")) and raw.strip():
            current_desc.append(raw.strip())
            continue

    if current_name and current_desc:
        out.setdefault(current_name, " ".join(current_desc).strip())
    return out


def _sanitize_doc_text(text: str) -> str:
    # Avoid raw double quotes inside JSON descriptions.
    # json.dumps would escape them, but we prefer single quotes for readability.
    return (text or "").replace('"', "'").strip()


def signature_to_json_schema(fn) -> Dict[str, Any]:
    """
    Convert a Python callable signature into a minimal JSON Schema object for OpenAI tools.

    Notes:
    - We only support keyword params (EPM actions are kwargs-only usage).
    - Types are best-effort; unknown types default to string.
    """
    sig = inspect.signature(fn)
    props: Dict[str, Any] = {}
    required: list[str] = []
    allowed = {k for k, p in sig.parameters.items() if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)}
    arg_desc = _parse_args_descriptions(getattr(fn, "__doc__", "") or "", allowed_names=allowed)
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
            continue
        ann = p.annotation if p.annotation is not inspect._empty else str
        jtype = _json_type_for(ann)
        schema: Dict[str, Any] = {"type": jtype}
        enum = _enum_for(ann)
        if enum is not None:
            schema["enum"] = enum
        if name in arg_desc:
            schema["description"] = _sanitize_doc_text(str(arg_desc[name]))
        if p.default is not inspect._empty:
            schema["default"] = p.default
        else:
            required.append(name)
        props[name] = schema
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def dataclass_to_json_schema(
    cls: type,
    *,
    drop_fields: Optional[set[str]] = None,
    arg_desc: Optional[dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Convert a dataclass args object into a minimal JSON Schema.
    """
    if not is_dataclass(cls):
        raise TypeError("not_a_dataclass")
    drop_fields = drop_fields or set()
    arg_desc = arg_desc or {}
    props: Dict[str, Any] = {}
    required: list[str] = []
    for f in fields(cls):
        if f.name in drop_fields:
            continue
        ann = f.type or str
        schema: Dict[str, Any] = {"type": _json_type_for(ann)}
        enum = _enum_for(ann)
        if enum is not None:
            schema["enum"] = enum
        if f.name in arg_desc:
            schema["description"] = _sanitize_doc_text(str(arg_desc[f.name]))
        if f.default is not MISSING:
            schema["default"] = f.default
        elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore[attr-defined]
            # Can't serialize factory; treat as optional.
            pass
        else:
            required.append(f.name)
        props[f.name] = schema
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}
