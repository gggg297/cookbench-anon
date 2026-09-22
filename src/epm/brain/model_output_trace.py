from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Sequence


def split_think_and_final(text: str) -> tuple[str, str]:
    raw = str(text or "")
    low = raw.lower()
    start = low.find("<think>")
    end = low.rfind("</think>")
    if start != -1 and end != -1 and start < end:
        think = raw[start + len("<think>") : end].strip()
        final = raw[end + len("</think>") :].strip()
        return think, final
    return "", raw.strip()


_VISIBLE_REASONING_KEYS: tuple[str, ...] = (
    "reasoning_content",
    "reasoning",
    "thinking",
    "analysis",
)


def _text_from_reasoning_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_text_from_reasoning_value(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if not isinstance(value, dict):
        return ""
    for key in ("text", "content", "summary", "thinking", "reasoning_content", "analysis"):
        text = _text_from_reasoning_value(value.get(key))
        if text:
            return text
    return ""


def _collect_reasoning_fields(value: Any, *, path: str, found: list[tuple[str, str]]) -> None:
    """Collect only provider-exposed reasoning fields from an API response payload."""
    if isinstance(value, dict):
        block_type = str(value.get("type", "") or "").strip().lower()
        if block_type in {"thinking", "reasoning"}:
            text = _text_from_reasoning_value(value)
            if text:
                found.append((f"{path}.type={block_type}", text))
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if str(key).strip().lower() in _VISIBLE_REASONING_KEYS:
                text = _text_from_reasoning_value(child)
                if text:
                    found.append((child_path, text))
                continue
            _collect_reasoning_fields(child, path=child_path, found=found)
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _collect_reasoning_fields(child, path=f"{path}[{index}]", found=found)


def extract_visible_reasoning(*, response_payload: Any = None, response_text: str = "") -> tuple[str, tuple[str, ...]]:
    """Return provider-exposed reasoning, then fall back to an explicit think tag.

    This function intentionally does not infer a hidden chain of thought. It only
    records fields that the provider returned in the API response.
    """
    found: list[tuple[str, str]] = []
    _collect_reasoning_fields(response_payload, path="response_payload", found=found)
    if found:
        unique: list[tuple[str, str]] = []
        seen_text: set[str] = set()
        for source, text in found:
            if text not in seen_text:
                unique.append((source, text))
                seen_text.add(text)
        return "\n\n".join(text for _, text in unique), tuple(source for source, _ in unique)

    thinking, _ = split_think_and_final(response_text)
    if thinking:
        return thinking, ("response_text.<think>",)
    return "", ()


def trace_dirs_for_raw_dir(raw_dir: Path) -> tuple[Path, Path]:
    name = str(raw_dir.name or "").strip()
    if name.endswith("_raw"):
        prefix = name[: -len("_raw")]
    else:
        prefix = name
    return raw_dir.with_name(f"{prefix}_thinking"), raw_dir.with_name(f"{prefix}_final")


def write_trace_files(*, thinking_dir: Path, final_dir: Path, filename: str, text: str) -> tuple[str, str]:
    think_text, final_text = split_think_and_final(text)
    thinking_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    (thinking_dir / filename).write_text(think_text, encoding="utf-8")
    (final_dir / filename).write_text((final_text or str(text or "").strip()), encoding="utf-8")
    return think_text, final_text


def write_trace_files_for_raw_dir(*, raw_dir: Path, filename: str, text: str) -> tuple[str, str]:
    thinking_dir, final_dir = trace_dirs_for_raw_dir(raw_dir)
    return write_trace_files(thinking_dir=thinking_dir, final_dir=final_dir, filename=filename, text=text)


def _slugify_trace_part(value: str, *, default: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return default
    text = re.sub(r"[^a-z0-9._-]+", "_", raw).strip("._-")
    return text or default


def default_model_trace_dir(*, request_metrics_path: Path | str | None) -> Path | None:
    text = str(request_metrics_path or "").strip()
    if not text:
        return None
    return Path(text).resolve().parent / "model_call_traces"


def write_model_call_trace(
    *,
    trace_dir: Path | str | None,
    call_name: str,
    prompt_text: str,
    response_text: str = "",
    response_payload: Any = None,
    error_text: str = "",
    provider: str = "",
    model: str = "",
    attempt: int | None = None,
    screenshot_paths: Sequence[str | Path] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    root = Path(trace_dir).resolve() if str(trace_dir or "").strip() else None
    if root is None:
        return {}
    root.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = str(time.time_ns() % 1_000_000_000).zfill(9)
    base = f"{stamp}_{suffix}_{_slugify_trace_part(call_name, default='model_call')}"
    if attempt is not None:
        base += f"_attempt_{int(attempt)}"

    prompt_path = root / f"{base}.prompt.txt"
    response_path = root / f"{base}.response.txt"
    thinking_path = root / f"{base}.thinking.txt"
    meta_path = root / f"{base}.meta.json"

    prompt_raw = str(prompt_text or "")
    response_raw = str(response_text or "")
    error_raw = str(error_text or "")
    visible_reasoning, reasoning_sources = extract_visible_reasoning(
        response_payload=response_payload,
        response_text=response_raw,
    )

    prompt_path.write_text(prompt_raw, encoding="utf-8")
    response_path.write_text(response_raw, encoding="utf-8")
    thinking_path.write_text(visible_reasoning, encoding="utf-8")

    meta: dict[str, Any] = {
        "call_name": str(call_name or ""),
        "provider": str(provider or ""),
        "model": str(model or ""),
        "attempt": (int(attempt) if attempt is not None else None),
        "has_response_payload": response_payload is not None,
        "prompt_chars": len(prompt_raw),
        "response_chars": len(response_raw),
        "visible_reasoning_chars": len(visible_reasoning),
        "visible_reasoning_sources": list(reasoning_sources),
        "error_chars": len(error_raw),
        "error_text": error_raw,
        "screenshot_paths": [str(p) for p in (screenshot_paths or []) if str(p or "").strip()],
        "paths": {
            "prompt": str(prompt_path),
            "response": str(response_path),
            "thinking": str(thinking_path),
            "meta": str(meta_path),
        },
    }
    if response_payload is not None:
        meta["response_payload"] = response_payload
    if isinstance(extra_meta, dict) and extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "prompt": str(prompt_path),
        "response": str(response_path),
        "thinking": str(thinking_path),
        "meta": str(meta_path),
    }
