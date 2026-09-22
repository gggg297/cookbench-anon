from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional

from epm.brain.openai_compat import DEFAULT_CHAT_COMPLETIONS_PATH, join_openai_compatible_url, normalize_openai_compatible_path


DEFAULT_ANTHROPIC_MESSAGES_PATH = "/v1/messages"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

OPENAI_PROVIDER_TYPES = {"openai", "openai_compatible"}
ANTHROPIC_PROVIDER_TYPES = {"anthropic_messages"}


def normalize_provider_type(value: str) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return "openai_compatible"
    aliases = {
        "openai": "openai",
        "openai_compatible": "openai_compatible",
        "chat_completions": "openai_compatible",
        "anthropic_messages": "anthropic_messages",
        "anthropic": "anthropic_messages",
        "messages": "anthropic_messages",
    }
    return aliases.get(text, text)


def provider_api_mode(provider: str) -> str:
    normalized = normalize_provider_type(provider)
    if normalized in OPENAI_PROVIDER_TYPES:
        return "openai_compatible"
    if normalized in ANTHROPIC_PROVIDER_TYPES:
        return "anthropic_messages"
    return normalized


def provider_request_url(cfg: Any) -> str:
    provider = normalize_provider_type(str(getattr(cfg, "provider", "") or ""))
    request_path = str(getattr(cfg, "request_path", "") or "").strip()
    base_url = str(getattr(cfg, "base_url", "") or "")
    if provider in OPENAI_PROVIDER_TYPES:
        path = normalize_openai_compatible_path(
            request_path or str(getattr(cfg, "chat_completions_path", "") or ""),
            default=DEFAULT_CHAT_COMPLETIONS_PATH,
        )
        return join_openai_compatible_url(base_url, path)
    if provider in ANTHROPIC_PROVIDER_TYPES:
        path = normalize_openai_compatible_path(
            request_path or str(getattr(cfg, "messages_path", "") or ""),
            default=DEFAULT_ANTHROPIC_MESSAGES_PATH,
        )
        return join_openai_compatible_url(base_url, path)
    raise ValueError(f"unsupported_provider_type:{provider}")


def provider_request_headers(cfg: Any, api_key: str) -> dict[str, str]:
    provider = normalize_provider_type(str(getattr(cfg, "provider", "") or ""))
    headers: dict[str, str] = {"Content-Type": "application/json"}

    header_name = str(getattr(cfg, "auth_header_name", "") or "").strip()
    header_prefix = str(getattr(cfg, "auth_header_prefix", "") or "")
    if not header_name:
        header_name = "Authorization" if provider in OPENAI_PROVIDER_TYPES else "x-api-key"
    if not header_prefix and not str(getattr(cfg, "auth_header_prefix", "") or "").strip():
        header_prefix = "Bearer " if provider in OPENAI_PROVIDER_TYPES and header_name.lower() == "authorization" else ""
    if api_key:
        headers[header_name] = f"{header_prefix}{api_key}"

    extra_headers = getattr(cfg, "extra_headers", None)
    if isinstance(extra_headers, dict):
        for key, value in extra_headers.items():
            key_text = str(key or "").strip()
            value_text = str(value or "").strip()
            if key_text and value_text:
                headers[key_text] = value_text

    if provider in ANTHROPIC_PROVIDER_TYPES:
        lower_keys = {key.lower() for key in headers}
        if "anthropic-version" not in lower_keys:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_VERSION
    return headers


def _parse_data_url_image(url: str) -> dict[str, Any]:
    text = str(url or "").strip()
    if not text.startswith("data:") or "," not in text:
        raise ValueError("unsupported_image_url_format")
    header, data = text.split(",", 1)
    mime = "image/jpeg"
    if ";" in header:
        mime = header[5:].split(";", 1)[0] or mime
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime,
            "data": data,
        },
    }


def _openai_content_to_anthropic(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content or " "}]
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content or " ")}]
    out: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "") or "").strip().lower()
        if block_type == "text":
            out.append({"type": "text", "text": str(block.get("text", "") or " ")})
            continue
        if block_type == "image_url":
            image_url = block.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else ""
            out.append(_parse_data_url_image(str(url or "")))
    if not out:
        out.append({"type": "text", "text": " "})
    return out


def openai_messages_to_anthropic(messages: Iterable[dict[str, Any]]) -> tuple[Optional[str], list[dict[str, Any]]]:
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user") or "user").strip().lower()
        content = message.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            else:
                blocks = _openai_content_to_anthropic(content)
                joined = "\n".join(str(block.get("text", "") or "") for block in blocks if block.get("type") == "text").strip()
                if joined:
                    system_parts.append(joined)
            continue
        anthropic_role = "assistant" if role == "assistant" else "user"
        anthropic_messages.append(
            {
                "role": anthropic_role,
                "content": _openai_content_to_anthropic(content),
            }
        )
    if not anthropic_messages:
        anthropic_messages = [{"role": "user", "content": [{"type": "text", "text": " "}]}]
    system_text = "\n\n".join(part.strip() for part in system_parts if str(part or "").strip()).strip()
    return (system_text or None), anthropic_messages


def openai_tools_to_anthropic(tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if str(tool.get("type", "") or "").strip().lower() != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name", "") or "").strip()
        if not name:
            continue
        out.append(
            {
                "name": name,
                "description": str(function.get("description", "") or ""),
                "input_schema": function.get("parameters") if isinstance(function.get("parameters"), dict) else {"type": "object", "properties": {}},
            }
        )
    return out


def anthropic_response_to_openai(response_json: Any) -> dict[str, Any]:
    if not isinstance(response_json, dict):
        raise ValueError("non_dict_response")
    content_blocks = response_json.get("content")
    if not isinstance(content_blocks, list):
        raise ValueError("missing_content_blocks")

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "") or "").strip().lower()
        if block_type == "text":
            text_parts.append(str(block.get("text", "") or ""))
            continue
        if block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id", "") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "") or ""),
                        "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                    },
                }
            )

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(part for part in text_parts if str(part or "").strip()).strip(),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message}],
        "usage": response_json.get("usage", {}),
        "_provider_raw": response_json,
    }
