from __future__ import annotations

"""
Small OpenAI-compatible chat client used by baseline pipelines (Reflexion/EPM/CaP).

We intentionally keep this dependency-light (urllib) and compatible with the existing
`OpenAIPlanner` request format so the same endpoints can be reused.
"""

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, Sequence
from PIL import Image

from epm.brain.openai_compat import (
    DEFAULT_CHAT_COMPLETIONS_PATH,
    build_openai_token_limit_payload,
    join_openai_compatible_url,
    normalize_openai_compatible_path,
)
from epm.brain.provider_compat import (
    ANTHROPIC_PROVIDER_TYPES,
    OPENAI_PROVIDER_TYPES,
    anthropic_response_to_openai,
    openai_messages_to_anthropic,
    provider_api_mode,
    provider_request_headers,
    provider_request_url,
)
from epm.brain.model_output_trace import default_model_trace_dir, write_model_call_trace
from epm.brain.prompt_cache import (
    PromptCachePlan,
    build_extra_body,
    prepare_prompt_cache,
    request_cache_metrics,
    should_split_messages,
    should_use_qwen_cache_control,
    usage_cache_metrics,
)
from epm.core.api_key_pool_source import collect_api_key_candidates
from epm.core.http_403_pause import attach_network_context, classify_network_exception
from epm.core.balance_status import balance_check_enabled, maybe_refresh_balance_status, provider_root


class _ChatSettings(Protocol):
    provider: str
    base_url: str
    request_path: str
    model: str
    messages_path: str
    api_key_env: str
    api_key_pool_env: str
    api_key: str
    auth_header_name: str
    auth_header_prefix: str
    extra_headers: Optional[Dict[str, str]]
    timeout_s: float
    max_retries: int
    temperature: float
    max_tokens: int
    use_vision: bool
    image_max_side: int
    image_format: str
    jpeg_quality: int
    prompt_cache_enabled: bool
    prompt_cache_force: bool
    prompt_cache_min_chars: int
    prompt_cache_ttl_s: int
    prompt_cache_dir: str
    gemini_cache_api_base_url: str
    request_metrics_path: Optional[Path]


def _mask_secret(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:2]}***{text[-4:]}"


def _candidate_api_keys(cfg: _ChatSettings) -> list[str]:
    keys, _ = collect_api_key_candidates(
        explicit_key=str(getattr(cfg, "api_key", "") or ""),
        api_key_env=str(getattr(cfg, "api_key_env", "") or ""),
        api_key_pool_env=str(getattr(cfg, "api_key_pool_env", "") or ""),
        base_url=str(getattr(cfg, "base_url", "") or ""),
    )
    return keys


def _read_api_key(cfg: _ChatSettings) -> str:
    api_key_env = str(getattr(cfg, "api_key_env", "") or "").strip()
    api_key_pool_env = str(getattr(cfg, "api_key_pool_env", "") or "").strip()
    env_key = str(os.environ.get(api_key_env, "") or "").strip() if api_key_env else ""
    candidates, _ = collect_api_key_candidates(
        explicit_key=str(getattr(cfg, "api_key", "") or ""),
        api_key_env=api_key_env,
        api_key_pool_env=api_key_pool_env,
        base_url=str(getattr(cfg, "base_url", "") or ""),
    )
    if not candidates:
        return env_key
    chosen = env_key if (env_key and env_key in candidates) else candidates[0]
    if api_key_env and chosen:
        os.environ[api_key_env] = chosen
    if api_key_env:
        basis = "||".join(
            [
                api_key_env,
                api_key_pool_env,
                str(getattr(cfg, "base_url", "") or "").strip(),
                str(getattr(cfg, "model", "") or "").strip(),
                str(getattr(cfg, "provider", "") or "").strip(),
            ]
        )
        digest = hashlib.md5(basis.encode("utf-8")).hexdigest()[:12]
        try:
            chosen_index = candidates.index(chosen)
        except Exception:
            chosen_index = 0
        os.environ[f"EPM_API_KEY_ACTIVE_{digest}"] = f"{chosen_index + 1}/{len(candidates)}"
    return chosen


def _api_key_debug_info(cfg: _ChatSettings, api_key: str) -> dict[str, Any]:
    candidates = _candidate_api_keys(cfg)
    slot = ""
    if api_key and api_key in candidates:
        slot = f"{candidates.index(api_key) + 1}/{len(candidates)}"
    elif len(candidates) == 1 and api_key:
        slot = "1/1"
    return {
        "api_key_env": str(getattr(cfg, "api_key_env", "") or ""),
        "api_key_pool_env": str(getattr(cfg, "api_key_pool_env", "") or ""),
        "api_key_masked": _mask_secret(api_key),
        "api_key_slot": slot,
    }


def _log_api_key_use(*, cfg: _ChatSettings, api_key: str, call: str) -> dict[str, Any]:
    info = _api_key_debug_info(cfg, api_key)
    logging.info(
        "[EPM] api_request channel=chat_client call=%s provider=%s model=%s key=%s slot=%s env=%s pool_env=%s",
        call,
        str(getattr(cfg, "provider", "") or ""),
        str(getattr(cfg, "model", "") or ""),
        info["api_key_masked"],
        info["api_key_slot"],
        info["api_key_env"],
        info["api_key_pool_env"],
    )
    return info


def _maybe_log_balance(*, cfg: _ChatSettings, api_key: str, call: str) -> dict[str, Any]:
    if not balance_check_enabled(str(getattr(cfg, "base_url", "") or "")):
        return {}
    memory_dir = str(os.environ.get("EPM_RUN_MEMORY_DIR", "") or "").strip()
    if not memory_dir:
        return {}
    try:
        episode_step = int(str(os.environ.get("EPM_EPISODE_STEP", "") or "0").strip())
    except Exception:
        episode_step = 0
    try:
        cadence_steps = int(str(os.environ.get("EPM_BALANCE_CADENCE_STEPS", "") or "10").strip())
    except Exception:
        cadence_steps = 10
    info, refreshed = maybe_refresh_balance_status(
        base_url=str(getattr(cfg, "base_url", "") or ""),
        api_key=api_key,
        memory_dir=memory_dir,
        episode_step=episode_step,
        cadence_steps=cadence_steps,
        timeout_s=min(8.0, max(1.0, float(getattr(cfg, "timeout_s", 60.0) or 60.0))),
        channel="chat_client",
        call=call,
    )
    if refreshed:
        if info.get("error"):
            logging.warning("[EPM] provider_balance channel=chat_client call=%s error=%s", call, info.get("error"))
        else:
            logging.info(
                "[EPM] provider_balance channel=chat_client call=%s step=%s balance_usd=%s used_usd=%s total_usd=%s key=%s",
                call,
                info.get("last_checked_step"),
                info.get("balance_usd"),
                info.get("used_usd"),
                info.get("total_usd"),
                _mask_secret(api_key),
            )
    return {
        "provider_balance_usd": info.get("balance_usd"),
        "provider_used_usd": info.get("used_usd"),
        "provider_total_usd": info.get("total_usd"),
        "provider_balance_step": info.get("last_checked_step"),
        "provider_balance_error": info.get("error", ""),
    }


def _encode_image_data_url(
    path: str | Path,
    *,
    max_side: int = 1024,
    image_format: str = "jpeg",
    jpeg_quality: int = 85,
) -> str:
    fmt = (image_format or "jpeg").strip().lower()
    if fmt not in ("jpeg", "jpg", "png"):
        fmt = "jpeg"

    img = Image.open(Path(path))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    if fmt in ("jpeg", "jpg") and img.mode == "RGBA":
        img = img.convert("RGB")

    if max_side and max(img.size) > int(max_side):
        img.thumbnail((int(max_side), int(max_side)), resample=Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    if fmt in ("jpeg", "jpg"):
        img.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _normalize_screenshot_paths(
    screenshot_path: str | Path | None,
    screenshot_paths: Optional[Sequence[str | Path]],
) -> list[str]:
    ordered: list[str] = []
    if screenshot_paths:
        for item in screenshot_paths:
            text = str(item or "").strip()
            if text:
                ordered.append(text)
    if screenshot_path:
        text = str(screenshot_path).strip()
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def _make_messages(
    *,
    prompt_text: str,
    screenshot_path: str | None,
    screenshot_paths: Optional[Sequence[str | Path]],
    use_vision: bool,
    image_max_side: int,
    image_format: str,
    jpeg_quality: int,
    cache_plan: PromptCachePlan | None = None,
) -> list[dict[str, Any]]:
    plan = cache_plan or PromptCachePlan()
    image_paths = _normalize_screenshot_paths(screenshot_path, screenshot_paths)
    if should_split_messages(plan):
        messages: list[dict[str, Any]] = []
        if plan.mode != "gemini_explicit" and plan.parts.system_text.strip():
            messages.append({"role": "system", "content": plan.parts.system_text})
        content: list[dict[str, Any]] = []
        if should_use_qwen_cache_control(plan) and plan.parts.stable_user_text.strip():
            content.append(
                {
                    "type": "text",
                    "text": plan.parts.stable_user_text,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        dynamic_text = (
            plan.parts.dynamic_user_text
            if plan.mode == "gemini_explicit"
            else (plan.parts.dynamic_user_text or plan.parts.user_text)
        )
        if dynamic_text.strip() or not image_paths or not use_vision:
            content.append({"type": "text", "text": dynamic_text or " "})
        if use_vision and image_paths:
            for path in image_paths:
                image_url = _encode_image_data_url(
                    path,
                    max_side=int(image_max_side),
                    image_format=str(image_format or "jpeg"),
                    jpeg_quality=int(jpeg_quality),
                )
                content.append({"type": "image_url", "image_url": {"url": image_url}})
        if not content:
            content.append({"type": "text", "text": " "})
        messages.append({"role": "user", "content": content if (use_vision and image_paths) or should_use_qwen_cache_control(plan) else dynamic_text})
        return messages
    if not use_vision or not image_paths:
        return [{"role": "user", "content": prompt_text}]
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for path in image_paths:
        image_url = _encode_image_data_url(
            path,
            max_side=int(image_max_side),
            image_format=str(image_format or "jpeg"),
            jpeg_quality=int(jpeg_quality),
        )
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return [{"role": "user", "content": content}]


def _extract_openai_message(out: Any) -> dict[str, Any]:
    if not isinstance(out, dict):
        raise ValueError("non_dict_response")
    choices = out.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("no_choices_in_response")
    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise ValueError("missing_message")
    return msg


def _normalize_timeout(timeout_s: float) -> float | None:
    """
    Normalize timeout semantics to match planner.py:
    - timeout_s <= 0 => no timeout (None)
    - otherwise use the positive float value
    """
    try:
        t = float(timeout_s)
    except Exception:
        return 60.0
    if t <= 0:
        return None
    return t


def _append_jsonl(path: Optional[Path], row: Dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _trim_text(text: Any, *, max_chars: int = 2000) -> str:
    s = str(text or "")
    if len(s) <= int(max_chars):
        return s
    return s[: max(0, int(max_chars) - 14)] + "...<truncated>"


def _headers_preview(headers: Any, *, max_items: int = 12, max_value_chars: int = 240) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        items = list(headers.items())
    except Exception:
        return out
    for idx, (k, v) in enumerate(items):
        if idx >= int(max_items):
            break
        out[str(k)] = _trim_text(v, max_chars=max_value_chars)
    return out


def _exception_chain(exc: BaseException, *, max_depth: int = 6) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and len(chain) < int(max_depth):
        ident = id(cur)
        if ident in seen:
            break
        seen.add(ident)
        chain.append(cur)
        nxt = cur.__cause__ if cur.__cause__ is not None else cur.__context__
        cur = nxt if isinstance(nxt, BaseException) else None
    return chain


def _request_error_metrics(exc: BaseException) -> dict[str, Any]:
    chain = _exception_chain(exc)
    root = chain[-1] if chain else exc
    out: dict[str, Any] = {
        "error": repr(exc),
        "error_type": type(exc).__name__,
        "error_str": str(exc),
        "error_args": [repr(x) for x in getattr(exc, "args", ())],
        "error_chain": [
            {
                "type": type(it).__name__,
                "repr": repr(it),
                "str": str(it),
            }
            for it in chain
        ],
        "root_error_type": type(root).__name__,
        "root_error_repr": repr(root),
        "root_error_str": str(root),
    }

    error_kind = "exception"
    if isinstance(exc, urllib.error.HTTPError):
        error_kind = "http"
        out["http_status"] = int(exc.code)
        out["http_reason"] = str(exc.reason)
        try:
            setattr(exc, "_epm_http_reason", out["http_reason"])
        except Exception:
            pass
        out["http_headers"] = _headers_preview(exc.headers)
        try:
            body = exc.read()
        except Exception as body_err:
            out["http_body_read_error"] = repr(body_err)
            body = b""
        if isinstance(body, bytes):
            out["http_body_preview"] = _trim_text(body.decode("utf-8", errors="replace"))
        elif body:
            out["http_body_preview"] = _trim_text(body)
        if out.get("http_body_preview"):
            try:
                setattr(exc, "_epm_http_body_preview", str(out["http_body_preview"]))
            except Exception:
                pass
    elif isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        out["url_error_reason_type"] = type(reason).__name__ if reason is not None else ""
        out["url_error_reason_repr"] = repr(reason)
        out["url_error_reason_str"] = str(reason) if reason is not None else ""
        if isinstance(reason, ssl.SSLError):
            error_kind = "ssl"
        elif isinstance(reason, (TimeoutError, socket.timeout)):
            error_kind = "timeout"
        else:
            error_kind = "url"
    elif isinstance(exc, ssl.SSLError):
        error_kind = "ssl"
    elif isinstance(exc, (TimeoutError, socket.timeout)):
        error_kind = "timeout"

    if error_kind == "exception":
        if isinstance(root, ssl.SSLError):
            error_kind = "ssl"
        elif isinstance(root, (TimeoutError, socket.timeout)):
            error_kind = "timeout"
    out["error_kind"] = error_kind
    return out


def _estimate_text_tokens(text: str) -> int:
    raw = str(text or "").encode("utf-8")
    if not raw:
        return 0
    return max(1, int(math.ceil(len(raw) / 4.0)))


def _extract_openai_usage(out: Any) -> dict[str, Any]:
    if not isinstance(out, dict):
        return {}
    usage = out.get("usage")
    return usage if isinstance(usage, dict) else {}


def _usage_metrics(out: Any) -> dict[str, Any]:
    usage = _extract_openai_usage(out)
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    metrics: dict[str, Any] = {}
    if prompt_tokens is not None:
        metrics["input_tokens"] = prompt_tokens
    if completion_tokens is not None:
        metrics["output_tokens"] = completion_tokens
    if total_tokens is not None:
        metrics["total_tokens"] = total_tokens
    if usage:
        metrics["usage"] = usage
    metrics.update(usage_cache_metrics(out))
    return metrics


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    raw: Dict[str, Any]


def chat_complete_text(
    *,
    cfg: _ChatSettings,
    prompt: str,
    screenshot_path: Optional[str] = None,
    screenshot_paths: Optional[Sequence[str | Path]] = None,
    force_use_vision: Optional[bool] = None,
    max_tokens: Optional[int] = None,
) -> ChatCompletion:
    """
    Best-effort text completion against an OpenAI-compatible chat completions endpoint.

    Returns:
      ChatCompletion(content=..., raw=...) where content is `choices[0].message.content`.
    """

    provider = (getattr(cfg, "provider", "") or "").strip().lower().replace("-", "_")
    if provider not in OPENAI_PROVIDER_TYPES and provider not in ANTHROPIC_PROVIDER_TYPES:
        raise ValueError(f"unsupported_provider_for_chat_client:{provider}")
    if not getattr(cfg, "base_url", ""):
        raise ValueError("missing_base_url")
    if not getattr(cfg, "model", ""):
        raise ValueError("missing_model")

    url = provider_request_url(cfg)
    api_key = _read_api_key(cfg)
    api_key_info = _log_api_key_use(cfg=cfg, api_key=api_key, call="chat_complete_text")
    headers = provider_request_headers(cfg, api_key)

    use_vision = bool(getattr(cfg, "use_vision", False)) if force_use_vision is None else bool(force_use_vision)
    request_metrics_path = getattr(cfg, "request_metrics_path", None)
    cache_plan = prepare_prompt_cache(cfg=cfg, prompt_text=prompt, request_metrics_path=request_metrics_path)
    normalized_screenshot_paths = _normalize_screenshot_paths(screenshot_path, screenshot_paths)
    openai_messages = _make_messages(
        prompt_text=prompt,
        screenshot_path=screenshot_path,
        screenshot_paths=screenshot_paths,
        use_vision=use_vision,
        image_max_side=int(getattr(cfg, "image_max_side", 768) or 768),
        image_format=str(getattr(cfg, "image_format", "jpeg") or "jpeg"),
        jpeg_quality=int(getattr(cfg, "jpeg_quality", 70) or 70),
        cache_plan=cache_plan,
    )
    payload: Dict[str, Any]
    if provider in OPENAI_PROVIDER_TYPES:
        payload = {
            "model": getattr(cfg, "model"),
            "messages": openai_messages,
            "temperature": float(getattr(cfg, "temperature", 0.0)),
        }
        payload.update(
            build_openai_token_limit_payload(
                model=str(getattr(cfg, "model")),
                max_tokens=int(getattr(cfg, "max_tokens", 800) if max_tokens is None else max_tokens),
            )
        )
    else:
        system_text, anthropic_messages = openai_messages_to_anthropic(openai_messages)
        payload = {
            "model": getattr(cfg, "model"),
            "messages": anthropic_messages,
            "temperature": float(getattr(cfg, "temperature", 0.0)),
            "max_tokens": int(getattr(cfg, "max_tokens", 800) if max_tokens is None else max_tokens),
        }
        if system_text:
            payload["system"] = system_text
    extra_body = build_extra_body(cache_plan)
    if extra_body:
        payload["extra_body"] = extra_body

    timeout_s = _normalize_timeout(getattr(cfg, "timeout_s", 60.0))
    max_retries = int(getattr(cfg, "max_retries", 2) or 2)
    payload_bytes = 0
    try:
        payload_bytes = len(json.dumps(payload).encode("utf-8"))
    except Exception:
        payload_bytes = 0
    last_err: Exception | None = None
    trace_dir = default_model_trace_dir(request_metrics_path=request_metrics_path)
    for attempt in range(max_retries + 1):
        start_ts = time.time()
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            out = json.loads(raw)
            if provider in ANTHROPIC_PROVIDER_TYPES:
                out = anthropic_response_to_openai(out)
            msg = _extract_openai_message(out)
            content = msg.get("content")
            if not isinstance(content, str):
                raise ValueError("missing_message_content")
            try:
                write_model_call_trace(
                    trace_dir=trace_dir,
                    call_name="chat_complete_text",
                    prompt_text=prompt,
                    response_text=content,
                    response_payload=out,
                    provider=str(getattr(cfg, "provider", "") or ""),
                    model=str(getattr(cfg, "model", "") or ""),
                    attempt=int(attempt),
                    screenshot_paths=normalized_screenshot_paths,
                    extra_meta={
                        "use_vision": bool(use_vision),
                        "request_url": str(url or ""),
                    },
                )
            except Exception:
                pass
            balance_info = _maybe_log_balance(cfg=cfg, api_key=api_key, call="chat_complete_text")
            _append_jsonl(
                request_metrics_path,
                {
                    "ts": start_ts,
                    "planner_mode": "chat_client",
                    "api_mode": provider_api_mode(provider),
                    "provider": str(getattr(cfg, "provider", "") or ""),
                    "model": str(getattr(cfg, "model", "") or ""),
                    "request_url": str(url or ""),
                    "call": "chat_complete_text",
                    **api_key_info,
                    "attempt": int(attempt),
                    "max_retries": int(max_retries),
                    "timeout_cfg_s": float(getattr(cfg, "timeout_s", 60.0)),
                    "timeout_effective_s": timeout_s,
                    "prompt_chars": len(prompt or ""),
                    "prompt_bytes": len((prompt or "").encode("utf-8")),
                    "input_tokens_est": _estimate_text_tokens(prompt or ""),
                    "payload_bytes": int(payload_bytes),
                    "use_vision": bool(use_vision),
                    "has_screenshot": bool(normalized_screenshot_paths),
                    "screenshot_count": len(normalized_screenshot_paths),
                    **request_cache_metrics(cache_plan),
                    **_usage_metrics(out),
                    **balance_info,
                    "success": True,
                    "latency_s": time.time() - start_ts,
                },
            )
            return ChatCompletion(content=content, raw=(out if isinstance(out, dict) else {}))
        except Exception as e:
            last_err = e
            try:
                write_model_call_trace(
                    trace_dir=trace_dir,
                    call_name="chat_complete_text",
                    prompt_text=prompt,
                    response_text="",
                    response_payload=None,
                    error_text=repr(e),
                    provider=str(getattr(cfg, "provider", "") or ""),
                    model=str(getattr(cfg, "model", "") or ""),
                    attempt=int(attempt),
                    screenshot_paths=normalized_screenshot_paths,
                    extra_meta={
                        "use_vision": bool(use_vision),
                        "request_url": str(url or ""),
                    },
                )
            except Exception:
                pass
            balance_info = _maybe_log_balance(cfg=cfg, api_key=api_key, call="chat_complete_text")
            _append_jsonl(
                request_metrics_path,
                {
                    "ts": start_ts,
                    "planner_mode": "chat_client",
                    "api_mode": provider_api_mode(provider),
                    "provider": str(getattr(cfg, "provider", "") or ""),
                    "model": str(getattr(cfg, "model", "") or ""),
                    "request_url": str(url or ""),
                    "call": "chat_complete_text",
                    **api_key_info,
                    "attempt": int(attempt),
                    "max_retries": int(max_retries),
                    "timeout_cfg_s": float(getattr(cfg, "timeout_s", 60.0)),
                    "timeout_effective_s": timeout_s,
                    "prompt_chars": len(prompt or ""),
                    "prompt_bytes": len((prompt or "").encode("utf-8")),
                    "input_tokens_est": _estimate_text_tokens(prompt or ""),
                    "payload_bytes": int(payload_bytes),
                    "use_vision": bool(use_vision),
                    "has_screenshot": bool(normalized_screenshot_paths),
                    "screenshot_count": len(normalized_screenshot_paths),
                    **request_cache_metrics(cache_plan),
                    **balance_info,
                    "success": False,
                    "latency_s": time.time() - start_ts,
                    **_request_error_metrics(e),
                },
            )
            if attempt == 0 and cache_plan.requested and cache_plan.reason:
                logging.warning(
                    "[EPM] chat_prompt_cache status=%s reason=%s",
                    cache_plan.status,
                    cache_plan.reason,
                )
            network_signal = classify_network_exception(e, source="chat_client.chat_complete_text")
            if network_signal is not None:
                attach_network_context(
                    network_signal,
                    api_key_env=getattr(cfg, "api_key_env", ""),
                    api_key_pool_env=getattr(cfg, "api_key_pool_env", ""),
                    api_key_masked=str(api_key_info.get("api_key_masked", "") or ""),
                    api_key_slot=str(api_key_info.get("api_key_slot", "") or ""),
                    base_url=getattr(cfg, "base_url", ""),
                    model=getattr(cfg, "model", ""),
                    provider=getattr(cfg, "provider", ""),
                )
                raise network_signal from e
            # Backoff (avoid thundering herd on overloaded local endpoints)
            time.sleep(0.2 * (attempt + 1))
            continue
    if last_err is not None:
        network_signal = classify_network_exception(last_err, source="chat_client.chat_complete_text.final")
        if network_signal is not None:
            attach_network_context(
                network_signal,
                api_key_env=getattr(cfg, "api_key_env", ""),
                api_key_pool_env=getattr(cfg, "api_key_pool_env", ""),
                api_key_masked=str(api_key_info.get("api_key_masked", "") or ""),
                api_key_slot=str(api_key_info.get("api_key_slot", "") or ""),
                base_url=getattr(cfg, "base_url", ""),
                model=getattr(cfg, "model", ""),
                provider=getattr(cfg, "provider", ""),
            )
            raise network_signal from last_err
        raise RuntimeError(f"chat_complete_failed:{last_err!r}") from last_err
    raise RuntimeError("chat_complete_failed:unknown_request_failure")
