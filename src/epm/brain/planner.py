from __future__ import annotations

"""
Planner module (per-high-level rolling horizon).

This file intentionally contains both:
- ScriptedPlanner (debug / offline)
- OpenAIPlanner (OpenAI-compatible HTTP endpoint)

So the folder stays small and the responsibilities are obvious.
"""

import base64
import hashlib
import io
import json
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
from typing import Any, Dict, List, Optional, Protocol
from loguru import logger as logging
from PIL import Image

from epm.brain.interfaces import PlannerContext, PlannerModule
from epm.brain.model_output_trace import (
    default_model_trace_dir,
    extract_visible_reasoning,
    split_think_and_final,
    write_model_call_trace,
)
from epm.brain.plan_schema import PlanResponse, PlanStep, extract_json_object, parse_plan_response
from epm.brain.prompt_cache import (
    PromptCachePlan,
    build_extra_body,
    prepare_prompt_cache,
    request_cache_metrics,
    should_split_messages,
    should_use_qwen_cache_control,
    usage_cache_metrics,
)
from epm.brain.http_qwen3vl import Qwen3VlHttpClient
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
    openai_tools_to_anthropic,
    provider_api_mode,
    provider_request_headers,
    provider_request_url,
)
from epm.brain.tools_manifest import build_tool_manifest, to_openai_tools
from epm.core.api_key_pool_source import collect_api_key_candidates
from epm.core.http_403_pause import attach_network_context, classify_network_exception
from epm.core.settings import LlmSettings
from epm.core.settings import VlmSettings
from epm.core.balance_status import balance_check_enabled, maybe_refresh_balance_status, provider_root


class Planner(Protocol):
    def plan(self, *, prompt: str, screenshot_path: str | None = None) -> PlanResponse: ...


def _display_path(path: str | Path) -> str:
    text = str(path or "").strip()
    if not text:
        return ""
    try:
        root = Path(__file__).resolve().parents[3]
        return os.path.relpath(text, start=str(root))
    except Exception:
        return text


def _raise_wrapped_request_failure(*, last_err: Exception | None, label: str, source: str) -> None:
    if last_err is None:
        raise RuntimeError(f"{label}:unknown_request_failure")
    network_signal = classify_network_exception(last_err, source=source)
    if network_signal is not None:
        raise network_signal from last_err
    raise RuntimeError(f"{label}:{last_err}") from last_err


def _mask_secret(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:2]}***{text[-4:]}"


# ----------------------------
# Scripted planner (debug)
# ----------------------------
@dataclass(frozen=True)
class ScriptedPlannerConfig:
    plan_path: Path


class ScriptedPlanner(PlannerModule):
    """
    Reads a plan JSON from disk.

    This lets you validate the full plan-execute loop before integrating an LLM/VLM.
    """

    def __init__(self, cfg: ScriptedPlannerConfig) -> None:
        self.cfg = cfg

    def plan(self, *, context: PlannerContext, prompt: str) -> PlanResponse:
        if not self.cfg.plan_path.exists():
            plan = parse_plan_response(
                {
                    "high_level_id": "H1",
                    "goal": "noop",
                    "explanation": None,
                    "thoughts": "no scripted plan provided; noop",
                    "action_list": [
                        {"step_id": "H1.A1", "type": "action", "name": "noop", "args": {}, "expectation": ""}
                    ],
                }
            )
            return _coerce_single_retry_step(plan, context=context)
        raw_text = self.cfg.plan_path.read_text(encoding="utf-8")
        return _parse_plan_with_instance_selection_fallback(context=context, text=raw_text)


# ----------------------------
# OpenAI-compatible endpoint planner
# ----------------------------
def _read_api_key(cfg: Any) -> str:
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


def _candidate_api_keys(cfg: Any) -> list[str]:
    keys, _ = collect_api_key_candidates(
        explicit_key=str(getattr(cfg, "api_key", "") or ""),
        api_key_env=str(getattr(cfg, "api_key_env", "") or ""),
        api_key_pool_env=str(getattr(cfg, "api_key_pool_env", "") or ""),
        base_url=str(getattr(cfg, "base_url", "") or ""),
    )
    return keys


def _api_key_debug_info(cfg: Any, api_key: str) -> dict[str, Any]:
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


def _log_api_key_use(*, cfg: Any, api_key: str, call: str, channel: str) -> dict[str, Any]:
    info = _api_key_debug_info(cfg, api_key)
    logging.info(
        f"[EPM] api_request channel={channel} call={call} provider={str(getattr(cfg, 'provider', '') or '')} "
        f"model={str(getattr(cfg, 'model', '') or '')} key={info['api_key_masked']} slot={info['api_key_slot']} "
        f"env={info['api_key_env']} pool_env={info['api_key_pool_env']}"
    )
    return info


def _maybe_log_balance(*, cfg: Any, api_key: str, call: str, channel: str) -> dict[str, Any]:
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
        channel=channel,
        call=call,
    )
    if refreshed:
        if info.get("error"):
            logging.warning(f"[EPM] provider_balance channel={channel} call={call} error={info.get('error')}")
        else:
            logging.info(
                f"[EPM] provider_balance channel={channel} call={call} step={info.get('last_checked_step')} "
                f"balance_usd={info.get('balance_usd')} used_usd={info.get('used_usd')} "
                f"total_usd={info.get('total_usd')} key={_mask_secret(api_key)}"
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
) -> tuple[str, Dict[str, Any]]:
    fmt = (image_format or "jpeg").strip().lower()
    if fmt not in ("jpeg", "jpg", "png"):
        fmt = "jpeg"

    src_path = Path(path)
    raw = src_path.read_bytes()
    img = Image.open(src_path)
    orig_size = tuple(int(x) for x in img.size)
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
        data_url = f"data:image/jpeg;base64,{b64}"
        encoded_fmt = "jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{b64}"
        encoded_fmt = "png"
    encoded_bytes = len(buf.getvalue())
    meta = {
        "image_source_bytes": len(raw),
        "image_encoded_bytes": encoded_bytes,
        "image_data_url_bytes": len(data_url.encode("utf-8")),
        "image_format": encoded_fmt,
        "image_max_side": int(max_side),
        "image_jpeg_quality": int(jpeg_quality) if encoded_fmt == "jpeg" else None,
        "image_orig_size": {"width": int(orig_size[0]), "height": int(orig_size[1])},
        "image_final_size": {"width": int(img.size[0]), "height": int(img.size[1])},
        "image_compression_ratio": (
            float(encoded_bytes) / float(len(raw)) if len(raw) > 0 else None
        ),
    }
    return data_url, meta


def _make_messages(
    *,
    prompt_text: str,
    screenshot_path: str | None,
    use_vision: bool,
    image_max_side: int,
    image_format: str,
    jpeg_quality: int,
    cache_plan: PromptCachePlan | None = None,
) -> tuple[list[dict[str, Any]], Dict[str, Any]]:
    plan = cache_plan or PromptCachePlan()
    if should_split_messages(plan):
        messages: list[dict[str, Any]] = []
        if plan.mode != "gemini_explicit" and plan.parts.system_text.strip():
            messages.append({"role": "system", "content": plan.parts.system_text})
        image_meta: Dict[str, Any] = {}
        content_blocks: list[dict[str, Any]] = []
        if should_use_qwen_cache_control(plan) and plan.parts.stable_user_text.strip():
            content_blocks.append(
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
        if dynamic_text.strip() or not screenshot_path or not use_vision:
            content_blocks.append({"type": "text", "text": dynamic_text or " "})
        if use_vision and screenshot_path:
            image_url, image_meta = _encode_image_data_url(
                screenshot_path,
                max_side=int(image_max_side),
                image_format=str(image_format or "jpeg"),
                jpeg_quality=int(jpeg_quality),
            )
            content_blocks.append({"type": "image_url", "image_url": {"url": image_url}})
        if not content_blocks:
            content_blocks.append({"type": "text", "text": " "})
        if use_vision and screenshot_path:
            messages.append({"role": "user", "content": content_blocks})
            return messages, image_meta
        text_blocks = [b for b in content_blocks if b.get("type") == "text"]
        if len(text_blocks) == 1 and not should_use_qwen_cache_control(plan):
            messages.append({"role": "user", "content": str(text_blocks[0].get("text") or "")})
        else:
            messages.append({"role": "user", "content": content_blocks})
        return messages, image_meta
    if not use_vision or not screenshot_path:
        return [{"role": "user", "content": prompt_text}], {}
    image_url, image_meta = _encode_image_data_url(
        screenshot_path,
        max_side=int(image_max_side),
        image_format=str(image_format or "jpeg"),
        jpeg_quality=int(jpeg_quality),
    )
    content = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    return [{"role": "user", "content": content}], image_meta


def _tools_prompt_suffix() -> str:
    # Keep this short; PromptBuilder already provides most structure.
    return (
        "\n\n"
        "You may use tool calls (function calling) to choose the next steps.\n"
        "Prefer tool calls over plain text. If you use tool calls, do NOT include a JSON plan in message content.\n"
    )


def _is_instance_disambiguation_feedback(feedback: str) -> bool:
    return "blocking=instance_disambiguation" in str(feedback or "").strip().lower()


def _normalize_timeout(timeout_s: float) -> Optional[float]:
    """
    urllib timeout behavior:
    - None: no timeout (block until response / socket break)
    - >0: bounded timeout in seconds
    """
    try:
        t = float(timeout_s)
    except Exception:
        return None
    return None if t <= 0 else t


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
        if isinstance(reason, (ssl.SSLError,)):
            error_kind = "ssl"
        elif isinstance(reason, (TimeoutError, socket.timeout)):
            error_kind = "timeout"
        else:
            error_kind = "url"
    elif isinstance(exc, (ssl.SSLError,)):
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
    # Rough cross-model estimate for dashboard/debugging when provider usage is absent.
    return max(1, int(math.ceil(len(raw) / 4.0)))


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


def _split_think_and_final(text: str) -> tuple[str, str]:
    raw = str(text or "")
    if not raw.strip():
        return "", ""
    low = raw.lower()
    start = low.find("<think>")
    end = low.rfind("</think>")
    if start != -1 and end != -1 and end > start:
        think = raw[start + len("<think>") : end].strip()
        final = raw[end + len("</think>") :].strip()
        return think, final
    return "", raw.strip()


def _planner_attempt_kind(*, attempt: int) -> str:
    return "initial" if int(attempt) == 0 else "format_repair"


def _coerce_single_retry_step(plan: PlanResponse, *, context: PlannerContext) -> PlanResponse:
    if not _is_instance_disambiguation_feedback(context.feedback):
        return plan
    if len(plan.action_list) <= 1:
        return plan
    first = plan.action_list[0]
    return PlanResponse(
        high_level_id=context.high_level_id,
        goal=plan.goal,
        explanation=plan.explanation,
        thoughts=plan.thoughts,
        action_list=[
            PlanStep(
                step_id=f"{context.high_level_id}.A1",
                type=first.type,
                name=first.name,
                args=first.args,
                expectation=first.expectation,
            )
        ],
    )


def _safe_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def _feedback_line_value(feedback: str, key: str) -> str:
    prefix = f"{key}="
    for raw in str(feedback or "").splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return ""


def _extract_failed_step_type_and_name(feedback: str) -> tuple[str, str]:
    # line shape: "type=skill name=auto_navigation"
    m = re.search(r"(?:^|\n)type=(action|skill)\s+name=([^\n]+)", str(feedback or ""))
    if not m:
        return "", ""
    return str(m.group(1)).strip().lower(), str(m.group(2)).strip()


def _build_retry_plan_from_selection(
    *,
    context: PlannerContext,
    selected_name: str,
    selected_instance_id: int,
) -> PlanResponse:
    step_type, step_name = _extract_failed_step_type_and_name(context.feedback)
    if step_type not in ("action", "skill") or not step_name:
        raise ValueError("instance_selection_missing_failed_step")

    args_raw = _feedback_line_value(context.feedback, "args")
    args: Dict[str, Any] = {}
    if args_raw:
        try:
            parsed = json.loads(args_raw)
            if isinstance(parsed, dict):
                args = dict(parsed)
        except Exception:
            args = {}

    instance_arg = _feedback_line_value(context.feedback, "instance_meta.instance_arg")
    name_arg = _feedback_line_value(context.feedback, "instance_meta.name_arg")
    if not instance_arg:
        # Fallback: choose the first *_instance_id arg if metadata is missing.
        for k in args.keys():
            if isinstance(k, str) and k.endswith("_instance_id"):
                instance_arg = k
                break
    if not instance_arg:
        raise ValueError("instance_selection_missing_instance_arg")

    args[instance_arg] = int(selected_instance_id)
    if name_arg:
        args[name_arg] = str(selected_name)

    expectation = f"Retry {step_name} with {instance_arg}={int(selected_instance_id)} for {selected_name!r}."
    step = PlanStep(
        step_id=f"{context.high_level_id}.A1",
        type=step_type,
        name=step_name,
        args=args,
        expectation=expectation,
    )
    return PlanResponse(
        high_level_id=context.high_level_id,
        goal=context.high_level_goal or "Retry failed step with selected instance_id",
        explanation="instance_id_selection",
        thoughts="Selected one candidate and retry the failed step.",
        action_list=[step],
    )


def _try_parse_instance_selection_plan(*, context: PlannerContext, text: str) -> Optional[PlanResponse]:
    if not _is_instance_disambiguation_feedback(context.feedback):
        return None
    obj = extract_json_object(text, strip_think_tags=True)
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name") or "").strip()
    iid = _safe_int(obj.get("instance_id"))
    if not name or iid is None:
        return None
    return _build_retry_plan_from_selection(context=context, selected_name=name, selected_instance_id=int(iid))


def _parse_plan_with_instance_selection_fallback(*, context: PlannerContext, text: str) -> PlanResponse:
    if _is_instance_disambiguation_feedback(context.feedback):
        try:
            p = _try_parse_instance_selection_plan(context=context, text=text)
            if p is not None:
                return p
        except Exception:
            pass
    obj = extract_json_object(text, strip_think_tags=True)
    plan = parse_plan_response(obj)
    return _coerce_single_retry_step(plan, context=context)


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


def _extract_openai_finish_reason(out: Any) -> str:
    if not isinstance(out, dict):
        return ""
    choices = out.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    fr = c0.get("finish_reason")
    return str(fr) if isinstance(fr, str) else ""


def _extract_openai_usage(out: Any) -> dict[str, Any]:
    if not isinstance(out, dict):
        return {}
    usage = out.get("usage")
    return usage if isinstance(usage, dict) else {}


def _warn_if_model_output_truncated(out: Any, *, tag: str) -> None:
    fr = _extract_openai_finish_reason(out)
    if fr != "length":
        return
    usage = _extract_openai_usage(out)
    logging.warning(
        f"[EPM] {tag}_finish_reason=length (model output truncated). usage={usage if usage else '{}'}"
    )


def _plan_len_ok(n: int, plan_min_steps: int, plan_max_steps: int) -> bool:
    min_steps = int(plan_min_steps)
    max_steps = int(plan_max_steps)
    if max_steps < 0:
        return n >= min_steps
    return min_steps <= n <= max_steps


def _plan_len_expected(plan_min_steps: int, plan_max_steps: int) -> str:
    min_steps = int(plan_min_steps)
    max_steps = int(plan_max_steps)
    if max_steps < 0:
        return f">={min_steps}"
    return f"{min_steps}..{max_steps}"



def _plan_len_validate(n: int, plan_min_steps: int, plan_max_steps: int, *, allow_under_min: bool, context: str) -> None:
    if _plan_len_ok(n, plan_min_steps, plan_max_steps):
        return
    min_steps = int(plan_min_steps)
    max_steps = int(plan_max_steps)
    if allow_under_min and n >= 1:
        if max_steps >= 0 and n > max_steps:
            expect = _plan_len_expected(plan_min_steps, plan_max_steps)
            raise ValueError(f"{context}_length_out_of_range:{n} expected={expect}")
        expect = _plan_len_expected(plan_min_steps, plan_max_steps)
        logging.warning(
            f"[EPM] {context}_length_below_min:{n} min={min_steps} expected={expect}; allowing incremental plan."
        )
        return
    expect = _plan_len_expected(plan_min_steps, plan_max_steps)
    raise ValueError(f"{context}_length_out_of_range:{n} expected={expect}")


def _tool_calls_to_plan(
    *,
    high_level_id: str,
    high_level_goal: str,
    msg: dict[str, Any],
    plan_min_steps: int,
    plan_max_steps: int,
    allow_under_min: bool = True,
) -> PlanResponse:
    tool_calls = msg.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise ValueError("no_tool_calls")

    steps: list[PlanStep] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        name = fn.get("name")
        args_s = fn.get("arguments")
        if not isinstance(name, str) or not name:
            continue
        if not isinstance(args_s, str) or not args_s.strip():
            args_s = "{}"
        try:
            args = json.loads(args_s)
        except Exception:
            args = {}
        if not isinstance(args, dict):
            args = {}

        if name.startswith("action__"):
            step_type = "action"
            step_name = name[len("action__") :]
        elif name.startswith("skill__"):
            step_type = "skill"
            step_name = name[len("skill__") :]
        else:
            continue

        steps.append(
            PlanStep(
                step_id="",  # filled later
                type=step_type,
                name=step_name,
                args=args,
                expectation="",
            )
        )

    _plan_len_validate(
        len(steps),
        plan_min_steps,
        plan_max_steps,
        allow_under_min=bool(allow_under_min),
        context="tool_calls",
    )

    coerced: list[PlanStep] = []
    for i, s in enumerate(steps, start=1):
        coerced.append(
            PlanStep(
                step_id=f"{high_level_id}.A{i}",
                type=s.type,
                name=s.name,
                args=s.args,
                expectation=s.expectation,
            )
        )

    content = msg.get("content")
    thoughts = content.strip() if isinstance(content, str) else ""
    return PlanResponse(
        high_level_id=high_level_id,
        goal=high_level_goal or "Follow recipe",
        explanation=None,
        thoughts=thoughts,
        action_list=coerced,
    )


@dataclass(frozen=True)
class OpenAIPlannerConfig:
    llm: LlmSettings
    allowed_actions: List[str]
    allowed_skills: List[str]


class OpenAIPlanner(PlannerModule):
    def __init__(self, cfg: OpenAIPlannerConfig) -> None:
        self.cfg = cfg
        if self.cfg.llm.provider not in OPENAI_PROVIDER_TYPES and self.cfg.llm.provider not in ANTHROPIC_PROVIDER_TYPES:
            raise ValueError(f"unsupported_llm_provider:{self.cfg.llm.provider}")
        if not self.cfg.llm.model:
            raise ValueError("llm.model is required for planner_mode=llm")
        if not self.cfg.llm.base_url:
            raise ValueError("llm.base_url is required for planner_mode=llm")
        self._tool_calling_enabled = bool(getattr(self.cfg.llm, "use_tools", False))

    def set_tool_calling_enabled(self, enabled: bool) -> None:
        self._tool_calling_enabled = bool(enabled)

    def probe_tool_calling_support(self) -> Optional[bool]:
        if self.cfg.llm.provider not in OPENAI_PROVIDER_TYPES and self.cfg.llm.provider not in ANTHROPIC_PROVIDER_TYPES:
            return False
        prompt = "Tool-calling probe: call any one tool from the provided tool list once."
        try:
            out = self._chat_raw(prompt=prompt, screenshot_path=None, use_tools=True)
        except Exception:
            return None
        try:
            msg = _extract_openai_message(out)
        except Exception:
            return False
        tool_calls = msg.get("tool_calls")
        return bool(isinstance(tool_calls, list) and tool_calls)

    def plan(self, *, context: PlannerContext, prompt: str) -> PlanResponse:
        instance_id_selection_mode = _is_instance_disambiguation_feedback(context.feedback)
        use_tools = bool(self._tool_calling_enabled) and (not instance_id_selection_mode)
        if use_tools:
            out = self._chat_raw(prompt=prompt, screenshot_path=context.observation.screenshot_path, use_tools=True)
            msg = _extract_openai_message(out)
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                plan = _tool_calls_to_plan(
                    high_level_id=context.high_level_id,
                    high_level_goal=context.high_level_goal,
                    msg=msg,
                    plan_min_steps=1,
                    plan_max_steps=32,
                )
                visible_reasoning, _ = extract_visible_reasoning(response_payload=out)
                if visible_reasoning and not plan.thoughts:
                    return PlanResponse(
                        high_level_id=plan.high_level_id,
                        goal=plan.goal,
                        explanation=plan.explanation,
                        thoughts=visible_reasoning,
                        action_list=plan.action_list,
                    )
                return plan
            # Fallback to legacy JSON plan in content.
            content = msg.get("content")
            if not isinstance(content, str):
                raise ValueError("missing_message_content")
            return _parse_plan_with_instance_selection_fallback(context=context, text=content)

        content = self._chat(prompt=prompt, screenshot_path=context.observation.screenshot_path)
        return _parse_plan_with_instance_selection_fallback(context=context, text=content)

    def _chat(self, *, prompt: str, screenshot_path: str | None) -> str:
        out = self._chat_raw(prompt=prompt, screenshot_path=screenshot_path, use_tools=False)
        msg = _extract_openai_message(out)
        content = msg.get("content")
        if not isinstance(content, str):
            raise ValueError("missing_message_content")
        return content

    def _chat_raw(self, *, prompt: str, screenshot_path: str | None, use_tools: bool) -> Dict[str, Any]:
        provider = str(getattr(self.cfg.llm, "provider", "") or "").strip().lower().replace("-", "_")
        url = provider_request_url(self.cfg.llm)
        api_key = _read_api_key(self.cfg.llm)
        _log_api_key_use(cfg=self.cfg.llm, api_key=api_key, call="_chat_raw", channel="llm")
        headers = provider_request_headers(self.cfg.llm, api_key)

        prompt_text = prompt + (_tools_prompt_suffix() if use_tools else "")
        cache_plan = prepare_prompt_cache(cfg=self.cfg.llm, prompt_text=prompt_text)
        openai_messages = _make_messages(
            prompt_text=prompt_text,
            screenshot_path=screenshot_path,
            use_vision=bool(self.cfg.llm.use_vision),
            image_max_side=int(getattr(self.cfg.llm, "image_max_side", 768) or 768),
            image_format=str(getattr(self.cfg.llm, "image_format", "jpeg") or "jpeg"),
            jpeg_quality=int(getattr(self.cfg.llm, "jpeg_quality", 70) or 70),
            cache_plan=cache_plan,
        )[0]
        payload: Dict[str, Any]
        if provider in OPENAI_PROVIDER_TYPES:
            payload = {
                "model": self.cfg.llm.model,
                "messages": openai_messages,
                "temperature": float(self.cfg.llm.temperature),
            }
            payload.update(build_openai_token_limit_payload(model=self.cfg.llm.model, max_tokens=int(self.cfg.llm.max_tokens)))
        else:
            system_text, anthropic_messages = openai_messages_to_anthropic(openai_messages)
            payload = {
                "model": self.cfg.llm.model,
                "messages": anthropic_messages,
                "temperature": float(self.cfg.llm.temperature),
                "max_tokens": int(self.cfg.llm.max_tokens),
            }
            if system_text:
                payload["system"] = system_text
        extra_body = build_extra_body(cache_plan)
        if extra_body:
            payload["extra_body"] = extra_body

        if use_tools:
            manifest = build_tool_manifest(allowed_actions=self.cfg.allowed_actions, allowed_skills=self.cfg.allowed_skills)
            openai_tools = to_openai_tools(manifest)
            if provider in OPENAI_PROVIDER_TYPES:
                payload["tools"] = openai_tools
                payload["tool_choice"] = "auto"
            else:
                payload["tools"] = openai_tools_to_anthropic(openai_tools)
                payload["tool_choice"] = {"type": "auto"}

        data = json.dumps(payload).encode("utf-8")
        trace_dir = default_model_trace_dir(request_metrics_path=getattr(self.cfg.llm, "request_metrics_path", None))
        last_err: Optional[Exception] = None
        for attempt in range(int(self.cfg.llm.max_retries) + 1):
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=_normalize_timeout(float(self.cfg.llm.timeout_s))) as resp:
                    body = resp.read().decode("utf-8")
                out = json.loads(body)
                if provider in ANTHROPIC_PROVIDER_TYPES:
                    out = anthropic_response_to_openai(out)
                if not isinstance(out, dict):
                    raise ValueError("non_dict_response")
                try:
                    msg = _extract_openai_message(out)
                    content = msg.get("content")
                    response_text = content if isinstance(content, str) else json.dumps(msg, ensure_ascii=False, indent=2)
                    write_model_call_trace(
                        trace_dir=trace_dir,
                        call_name=("openai_planner_chat_raw_tools" if use_tools else "openai_planner_chat_raw"),
                        prompt_text=prompt_text,
                        response_text=response_text,
                        response_payload=out,
                        provider=str(getattr(self.cfg.llm, "provider", "") or ""),
                        model=str(getattr(self.cfg.llm, "model", "") or ""),
                        attempt=int(attempt),
                        screenshot_paths=([screenshot_path] if screenshot_path else []),
                        extra_meta={"use_tools": bool(use_tools), "request_url": str(url or "")},
                    )
                except Exception:
                    pass
                _warn_if_model_output_truncated(out, tag="llm")
                _maybe_log_balance(cfg=self.cfg.llm, api_key=api_key, call="_chat_raw", channel="llm")
                return out
            except Exception as e:
                last_err = e
                try:
                    write_model_call_trace(
                        trace_dir=trace_dir,
                        call_name=("openai_planner_chat_raw_tools" if use_tools else "openai_planner_chat_raw"),
                        prompt_text=prompt_text,
                        response_text="",
                        error_text=repr(e),
                        provider=str(getattr(self.cfg.llm, "provider", "") or ""),
                        model=str(getattr(self.cfg.llm, "model", "") or ""),
                        attempt=int(attempt),
                        screenshot_paths=([screenshot_path] if screenshot_path else []),
                        extra_meta={"use_tools": bool(use_tools), "request_url": str(url or "")},
                    )
                except Exception:
                    pass
                _maybe_log_balance(cfg=self.cfg.llm, api_key=api_key, call="_chat_raw", channel="llm")
                if attempt == 0 and cache_plan.requested and cache_plan.reason:
                    logging.warning(
                        "[EPM] llm_prompt_cache status=%s reason=%s",
                        cache_plan.status,
                        cache_plan.reason,
                    )
                network_signal = classify_network_exception(e, source="planner.openai_llm._chat_raw")
                if network_signal is not None:
                    attach_network_context(
                        network_signal,
                        api_key_env=getattr(self.cfg.llm, "api_key_env", ""),
                        api_key_pool_env=getattr(self.cfg.llm, "api_key_pool_env", ""),
                        api_key_masked=str(_api_key_debug_info(self.cfg.llm, api_key).get("api_key_masked", "") or ""),
                        api_key_slot=str(_api_key_debug_info(self.cfg.llm, api_key).get("api_key_slot", "") or ""),
                        base_url=getattr(self.cfg.llm, "base_url", ""),
                        model=getattr(self.cfg.llm, "model", ""),
                        provider=getattr(self.cfg.llm, "provider", ""),
                        channel="llm",
                    )
                    raise network_signal from e
                time.sleep(0.5 * (attempt + 1))
        _raise_wrapped_request_failure(
            last_err=last_err,
            label="llm_request_failed",
            source="planner.openai_llm._chat_raw.final",
        )


@dataclass(frozen=True)
class Qwen3VlHttpPlannerConfig:
    llm: LlmSettings


class Qwen3VlHttpPlanner(PlannerModule):
    """
    Planner backed by the custom Qwen3-VL HTTP server (/v1/chat -> {"text": "..."}).
    """

    def __init__(self, cfg: Qwen3VlHttpPlannerConfig) -> None:
        self.cfg = cfg
        if self.cfg.llm.provider != "qwen3vl_http":
            raise ValueError(f"unsupported_llm_provider:{self.cfg.llm.provider}")
        if not self.cfg.llm.base_url:
            raise ValueError("llm.base_url is required for qwen3vl_http")
        self.client = Qwen3VlHttpClient(
            base_url=self.cfg.llm.base_url,
            connect_timeout_s=5.0,
            read_timeout_s=_normalize_timeout(float(self.cfg.llm.timeout_s)),
        )

    def plan(self, *, context: PlannerContext, prompt: str) -> PlanResponse:
        try:
            text = self.client.chat(
                prompt=prompt,
                image_path=(context.observation.screenshot_path if self.cfg.llm.use_vision else None),
                max_new_tokens=int(self.cfg.llm.max_tokens),
            )
            try:
                write_model_call_trace(
                    trace_dir=default_model_trace_dir(request_metrics_path=getattr(self.cfg.llm, "request_metrics_path", None)),
                    call_name="qwen3vl_http_planner_plan",
                    prompt_text=prompt,
                    response_text=text,
                    provider=str(getattr(self.cfg.llm, "provider", "") or ""),
                    model=str(getattr(self.cfg.llm, "model", "") or ""),
                    screenshot_paths=[context.observation.screenshot_path] if (self.cfg.llm.use_vision and context.observation.screenshot_path) else [],
                    extra_meta={"planner_kind": "llm_qwen3vl_http"},
                )
            except Exception:
                pass
        except Exception as e:
            try:
                write_model_call_trace(
                    trace_dir=default_model_trace_dir(request_metrics_path=getattr(self.cfg.llm, "request_metrics_path", None)),
                    call_name="qwen3vl_http_planner_plan",
                    prompt_text=prompt,
                    response_text="",
                    error_text=repr(e),
                    provider=str(getattr(self.cfg.llm, "provider", "") or ""),
                    model=str(getattr(self.cfg.llm, "model", "") or ""),
                    screenshot_paths=[context.observation.screenshot_path] if (self.cfg.llm.use_vision and context.observation.screenshot_path) else [],
                    extra_meta={"planner_kind": "llm_qwen3vl_http"},
                )
            except Exception:
                pass
            raise
        return _parse_plan_with_instance_selection_fallback(context=context, text=text)


@dataclass(frozen=True)
class VlmPlannerConfig:
    """
    Planner backed by a multimodal model config (VLM settings).

    This is used for the "monolithic VLM does planning" variant:
      screenshot + text prompt -> plan JSON

    It intentionally reuses the VLM config block so users don't have to duplicate
    provider/base_url/model settings into `llm` when they only have one MLLM.
    """

    vlm: VlmSettings
    allowed_actions: List[str]
    allowed_skills: List[str]
    save_last_raw_path: Optional[Path] = None
    save_per_step_dir: Optional[Path] = None
    request_metrics_path: Optional[Path] = None
    plan_min_steps: int = 3
    plan_max_steps: int = 8
    allow_incremental_plan: bool = True


class VlmPlanner(PlannerModule):
    def __init__(self, cfg: VlmPlannerConfig) -> None:
        self.cfg = cfg
        if self.cfg.vlm.provider not in OPENAI_PROVIDER_TYPES and self.cfg.vlm.provider not in ANTHROPIC_PROVIDER_TYPES and self.cfg.vlm.provider != "qwen3vl_http":
            raise ValueError(f"unsupported_vlm_provider_for_planner:{self.cfg.vlm.provider}")
        if self.cfg.vlm.provider != "qwen3vl_http" and not self.cfg.vlm.model:
            raise ValueError("vlm.model is required for planner_mode=vlm (openai-compatible)")
        if not self.cfg.vlm.base_url:
            raise ValueError("vlm.base_url is required for planner_mode=vlm")

        self._qwen_client: Optional[Qwen3VlHttpClient] = None
        if self.cfg.vlm.provider == "qwen3vl_http":
            self._qwen_client = Qwen3VlHttpClient(
                base_url=self.cfg.vlm.base_url,
                connect_timeout_s=5.0,
                read_timeout_s=_normalize_timeout(float(self.cfg.vlm.timeout_s)),
            )
        self._openai_tools: Optional[List[Dict[str, Any]]] = None
        if self.cfg.vlm.provider in OPENAI_PROVIDER_TYPES or self.cfg.vlm.provider in ANTHROPIC_PROVIDER_TYPES:
            manifest = build_tool_manifest(allowed_actions=self.cfg.allowed_actions, allowed_skills=self.cfg.allowed_skills)
            self._openai_tools = to_openai_tools(manifest)
        self._tool_calling_enabled = bool(getattr(self.cfg.vlm, "use_tools", False)) and self._openai_tools is not None

    def set_tool_calling_enabled(self, enabled: bool) -> None:
        if not enabled:
            self._tool_calling_enabled = False
            return
        self._tool_calling_enabled = self._openai_tools is not None

    def probe_tool_calling_support(self) -> Optional[bool]:
        if self.cfg.vlm.provider not in OPENAI_PROVIDER_TYPES and self.cfg.vlm.provider not in ANTHROPIC_PROVIDER_TYPES:
            return False
        if self._openai_tools is None:
            return False
        prompt = "Tool-calling probe: call any one tool from the provided tool list once."
        try:
            out = self._chat_raw(prompt=prompt, screenshot_path=None)
        except Exception:
            return None
        try:
            msg = _extract_openai_message(out)
        except Exception:
            return False
        tool_calls = msg.get("tool_calls")
        return bool(isinstance(tool_calls, list) and tool_calls)

    def plan(self, *, context: PlannerContext, prompt: str) -> PlanResponse:
        screenshot_path = context.observation.screenshot_path if bool(self.cfg.vlm.use_vision) else None
        instance_id_selection_mode = _is_instance_disambiguation_feedback(context.feedback)
        use_tools = bool(self._tool_calling_enabled) and self._openai_tools is not None and (not instance_id_selection_mode)
        if instance_id_selection_mode:
            effective_plan_min = 1
            effective_plan_max = 1
        else:
            effective_plan_min = int(self.cfg.plan_min_steps)
            effective_plan_max = int(self.cfg.plan_max_steps)

        last_err: Optional[Exception] = None
        last_content: str = ""
        for attempt in range(int(self.cfg.vlm.max_retries) + 1):
            if attempt == 0:
                cur_prompt = prompt
            else:
                cur_prompt = self._repair_prompt(
                    previous_output=last_content,
                    high_level_id=context.high_level_id,
                    plan_min_steps=effective_plan_min,
                    plan_max_steps=effective_plan_max,
                    instance_id_selection_mode=instance_id_selection_mode,
                )

            if use_tools:
                out = self._chat_raw(prompt=cur_prompt, screenshot_path=screenshot_path)
                fr = _extract_openai_finish_reason(out)
                usage = _extract_openai_usage(out)
                # Write meta for offline debugging.
                try:
                    if self.cfg.save_last_raw_path is not None:
                        meta_path = self.cfg.save_last_raw_path.with_name("vlm_planner_last_meta.json")
                        meta = {
                            "finish_reason": fr,
                            "usage": usage,
                            "model": out.get("model") if isinstance(out, dict) else None,
                        }
                        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                except Exception:
                    pass
                try:
                    msg = _extract_openai_message(out)
                except Exception as e:
                    last_err = e
                    time.sleep(0.4 * (attempt + 1))
                    continue
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    plan = _tool_calls_to_plan(
                        high_level_id=context.high_level_id,
                        high_level_goal=context.high_level_goal,
                        msg=msg,
                        # Tool calling is naturally incremental: the model may emit 1 tool call at a time.
                        # Respect configured plan length to keep PE/EPM aligned.
                        plan_min_steps=effective_plan_min,
                        plan_max_steps=effective_plan_max,
                        allow_under_min=bool(self.cfg.allow_incremental_plan),
                    )
                    return plan
                content = msg.get("content")
                last_content = content if isinstance(content, str) else ""
            else:
                last_content = self._chat(prompt=cur_prompt, screenshot_path=screenshot_path)

            if self.cfg.save_last_raw_path is not None:
                try:
                    self.cfg.save_last_raw_path.parent.mkdir(parents=True, exist_ok=True)
                    self.cfg.save_last_raw_path.write_text(last_content, encoding="utf-8")
                    think_text, final_text = split_think_and_final(last_content)
                    self.cfg.save_last_raw_path.with_name("vlm_planner_last_thinking.txt").write_text(
                        think_text,
                        encoding="utf-8",
                    )
                    self.cfg.save_last_raw_path.with_name("vlm_planner_last_final.txt").write_text(
                        (final_text or last_content).strip(),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
            think_text, final_text = _split_think_and_final(last_content)
            if self.cfg.save_per_step_dir is not None:
                try:
                    self.cfg.save_per_step_dir.mkdir(parents=True, exist_ok=True)
                    fid = str(getattr(context.observation, "frame_id", "") or "").strip()
                    step_tag = f"step_{int(fid):06d}" if fid.isdigit() else (fid or "step_unknown")
                    raw_path = self.cfg.save_per_step_dir / f"{step_tag}.txt"
                    raw_attempt_path = self.cfg.save_per_step_dir / f"{step_tag}.attempt_{int(attempt)}.txt"
                    raw_path.write_text(last_content, encoding="utf-8")
                    raw_attempt_path.write_text(last_content, encoding="utf-8")
                    think_dir = self.cfg.save_per_step_dir.with_name("vlm_planner_thinking")
                    final_dir = self.cfg.save_per_step_dir.with_name("vlm_planner_final")
                    think_dir.mkdir(parents=True, exist_ok=True)
                    final_dir.mkdir(parents=True, exist_ok=True)
                    think_path = think_dir / f"{step_tag}.txt"
                    final_path = final_dir / f"{step_tag}.txt"
                    think_attempt_path = think_dir / f"{step_tag}.attempt_{int(attempt)}.txt"
                    final_attempt_path = final_dir / f"{step_tag}.attempt_{int(attempt)}.txt"
                    think_path.write_text(think_text, encoding="utf-8")
                    final_path.write_text((final_text or last_content).strip(), encoding="utf-8")
                    think_attempt_path.write_text(think_text, encoding="utf-8")
                    final_attempt_path.write_text((final_text or last_content).strip(), encoding="utf-8")
                    think_preview = (think_text or "").replace("\n", " ").strip()
                    final_preview = ((final_text or last_content).strip()).replace("\n", " ").strip()
                    if len(think_preview) > 220:
                        think_preview = think_preview[:220] + "..."
                    if len(final_preview) > 220:
                        final_preview = final_preview[:220] + "..."
                    logging.info(
                        f"[EPM] planner_output step={step_tag} planner_attempt={int(attempt)} kind={_planner_attempt_kind(attempt=attempt)} think_chars={len(think_text)} final_chars={len((final_text or last_content).strip())}"
                    )
                    logging.info(
                        f"[EPM] planner_output_paths raw='{_display_path(raw_path)}' raw_attempt='{_display_path(raw_attempt_path)}' thinking='{_display_path(think_path)}' thinking_attempt='{_display_path(think_attempt_path)}' final='{_display_path(final_path)}' final_attempt='{_display_path(final_attempt_path)}'"
                    )
                    if think_text:
                        logging.info(f"[EPM] planner_output_preview think={think_preview!r} final={final_preview!r}")
                    else:
                        logging.info(
                            f"[EPM] planner_output_preview think='<empty: no explicit <think>...</think> block returned>' final={final_preview!r}"
                        )
                except Exception:
                    pass
            try:
                plan = _parse_plan_with_instance_selection_fallback(context=context, text=last_content)
                _plan_len_validate(
                    len(plan.action_list),
                    effective_plan_min,
                    effective_plan_max,
                    allow_under_min=bool(self.cfg.allow_incremental_plan),
                    context="action_list",
                )
                if plan.high_level_id != context.high_level_id:
                    coerced_steps: list[PlanStep] = []
                    for i, s in enumerate(plan.action_list, start=1):
                        coerced_steps.append(
                            PlanStep(
                                step_id=f"{context.high_level_id}.A{i}",
                                type=s.type,
                                name=s.name,
                                args=s.args,
                                expectation=s.expectation,
                            )
                        )
                    plan = PlanResponse(
                        high_level_id=context.high_level_id,
                        goal=plan.goal,
                        explanation=plan.explanation,
                        thoughts=plan.thoughts,
                        action_list=coerced_steps,
                    )
                return plan
            except Exception as e:
                last_err = e
                time.sleep(0.4 * (attempt + 1))

        hint = ""
        if self.cfg.save_last_raw_path is not None:
            hint = f" (see {self.cfg.save_last_raw_path})"
        raise RuntimeError(f"vlm_planner_invalid_json_output last_err={last_err!r}{hint}")

    def _chat(self, *, prompt: str, screenshot_path: str | None) -> str:
        if self.cfg.vlm.provider == "qwen3vl_http":
            assert self._qwen_client is not None
            start_ts = time.time()
            try:
                text = self._qwen_client.chat(
                    prompt=prompt,
                    image_path=screenshot_path,
                    max_new_tokens=int(self.cfg.vlm.max_tokens),
                )
                _append_jsonl(
                    self.cfg.request_metrics_path,
                    {
                        "ts": start_ts,
                        "planner_mode": "vlm",
                        "api_mode": "qwen3vl_http",
                        "provider": str(self.cfg.vlm.provider or ""),
                        "model": str(self.cfg.vlm.model or ""),
                        "request_url": str(self.cfg.vlm.base_url or ""),
                        "call": "_chat",
                        "attempt": 0,
                        "max_retries": int(self.cfg.vlm.max_retries),
                        "timeout_cfg_s": float(self.cfg.vlm.timeout_s),
                        "timeout_effective_s": _normalize_timeout(float(self.cfg.vlm.timeout_s)),
                        "prompt_chars": len(prompt or ""),
                        "prompt_bytes": len((prompt or "").encode("utf-8")),
                        "input_tokens_est": _estimate_text_tokens(prompt or ""),
                        "has_screenshot": bool(screenshot_path),
                        "success": True,
                        "latency_s": time.time() - start_ts,
                    },
                )
                return text
            except Exception as e:
                _append_jsonl(
                    self.cfg.request_metrics_path,
                    {
                        "ts": start_ts,
                        "planner_mode": "vlm",
                        "api_mode": "qwen3vl_http",
                        "provider": str(self.cfg.vlm.provider or ""),
                        "model": str(self.cfg.vlm.model or ""),
                        "request_url": str(self.cfg.vlm.base_url or ""),
                        "call": "_chat",
                        "attempt": 0,
                        "max_retries": int(self.cfg.vlm.max_retries),
                        "timeout_cfg_s": float(self.cfg.vlm.timeout_s),
                        "timeout_effective_s": _normalize_timeout(float(self.cfg.vlm.timeout_s)),
                        "prompt_chars": len(prompt or ""),
                        "prompt_bytes": len((prompt or "").encode("utf-8")),
                        "input_tokens_est": _estimate_text_tokens(prompt or ""),
                        "has_screenshot": bool(screenshot_path),
                        "success": False,
                        "latency_s": time.time() - start_ts,
                        **_request_error_metrics(e),
                    },
                )
                network_signal = classify_network_exception(e, source="planner._chat.qwen3vl_http")
                if network_signal is not None:
                    attach_network_context(
                        network_signal,
                        api_key_env=getattr(self.cfg.vlm, "api_key_env", ""),
                        api_key_pool_env=getattr(self.cfg.vlm, "api_key_pool_env", ""),
                        api_key_masked=str(api_key_info.get("api_key_masked", "") or ""),
                        api_key_slot=str(api_key_info.get("api_key_slot", "") or ""),
                        base_url=getattr(self.cfg.vlm, "base_url", ""),
                        model=getattr(self.cfg.vlm, "model", ""),
                        provider=getattr(self.cfg.vlm, "provider", ""),
                        channel="vlm",
                    )
                    raise network_signal from e
                raise

        provider = str(getattr(self.cfg.vlm, "provider", "") or "").strip().lower().replace("-", "_")
        url = provider_request_url(self.cfg.vlm)
        api_key = _read_api_key(self.cfg.vlm)
        api_key_info = _log_api_key_use(cfg=self.cfg.vlm, api_key=api_key, call="_chat", channel="vlm")
        headers = provider_request_headers(self.cfg.vlm, api_key)

        cache_plan = prepare_prompt_cache(
            cfg=self.cfg.vlm,
            prompt_text=prompt,
            request_metrics_path=self.cfg.request_metrics_path,
        )
        messages, image_meta = _make_messages(
            prompt_text=prompt,
            screenshot_path=screenshot_path,
            use_vision=bool(self.cfg.vlm.use_vision),
            image_max_side=int(getattr(self.cfg.vlm, "image_max_side", 768) or 768),
            image_format=str(getattr(self.cfg.vlm, "image_format", "jpeg") or "jpeg"),
            jpeg_quality=int(getattr(self.cfg.vlm, "jpeg_quality", 70) or 70),
            cache_plan=cache_plan,
        )
        payload: Dict[str, Any]
        if provider in OPENAI_PROVIDER_TYPES:
            payload = {
                "model": self.cfg.vlm.model,
                "messages": messages,
                "temperature": float(self.cfg.vlm.temperature),
            }
            payload.update(build_openai_token_limit_payload(model=self.cfg.vlm.model, max_tokens=int(self.cfg.vlm.max_tokens)))
        else:
            system_text, anthropic_messages = openai_messages_to_anthropic(messages)
            payload = {
                "model": self.cfg.vlm.model,
                "messages": anthropic_messages,
                "temperature": float(self.cfg.vlm.temperature),
                "max_tokens": int(self.cfg.vlm.max_tokens),
            }
            if system_text:
                payload["system"] = system_text
        extra_body = build_extra_body(cache_plan)
        if extra_body:
            payload["extra_body"] = extra_body
        data = json.dumps(payload).encode("utf-8")
        if image_meta:
            logging.info(
                "[EPM] image_compression "
                f"src_bytes={image_meta.get('image_source_bytes')} "
                f"encoded_bytes={image_meta.get('image_encoded_bytes')} "
                f"data_url_bytes={image_meta.get('image_data_url_bytes')} "
                f"fmt={image_meta.get('image_format')} "
                f"orig={image_meta.get('image_orig_size')} "
                f"final={image_meta.get('image_final_size')} "
                f"ratio={image_meta.get('image_compression_ratio')}"
            )

        last_err: Optional[Exception] = None
        for attempt in range(int(self.cfg.vlm.max_retries) + 1):
            start_ts = time.time()
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=_normalize_timeout(float(self.cfg.vlm.timeout_s))) as resp:
                    body = resp.read().decode("utf-8")
                out = json.loads(body)
                if provider in ANTHROPIC_PROVIDER_TYPES:
                    out = anthropic_response_to_openai(out)
                _warn_if_model_output_truncated(out, tag="vlm")
                choices = out.get("choices") if isinstance(out, dict) else None
                if not isinstance(choices, list) or not choices:
                    raise ValueError("no_choices_in_response")
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, str):
                    raise ValueError("missing_message_content")
                balance_info = _maybe_log_balance(cfg=self.cfg.vlm, api_key=api_key, call="_chat", channel="vlm")
                _append_jsonl(
                    self.cfg.request_metrics_path,
                    {
                        "ts": start_ts,
                        "planner_mode": "vlm",
                        "api_mode": provider_api_mode(provider),
                        "provider": str(self.cfg.vlm.provider or ""),
                        "model": str(self.cfg.vlm.model or ""),
                        "request_url": str(url or ""),
                        "call": "_chat",
                        **api_key_info,
                        "attempt": int(attempt),
                        "max_retries": int(self.cfg.vlm.max_retries),
                        "timeout_cfg_s": float(self.cfg.vlm.timeout_s),
                        "timeout_effective_s": _normalize_timeout(float(self.cfg.vlm.timeout_s)),
                        "prompt_chars": len(prompt or ""),
                        "prompt_bytes": len((prompt or "").encode("utf-8")),
                        "input_tokens_est": _estimate_text_tokens(prompt or ""),
                        "payload_bytes": len(data),
                        "has_screenshot": bool(screenshot_path),
                        **image_meta,
                        **request_cache_metrics(cache_plan),
                        **_usage_metrics(out),
                        **balance_info,
                        "success": True,
                        "latency_s": time.time() - start_ts,
                    },
                )
                return content
            except Exception as e:
                last_err = e
                balance_info = _maybe_log_balance(cfg=self.cfg.vlm, api_key=api_key, call="_chat", channel="vlm")
                _append_jsonl(
                    self.cfg.request_metrics_path,
                    {
                        "ts": start_ts,
                        "planner_mode": "vlm",
                        "api_mode": provider_api_mode(provider),
                        "provider": str(self.cfg.vlm.provider or ""),
                        "model": str(self.cfg.vlm.model or ""),
                        "request_url": str(url or ""),
                        "call": "_chat",
                        **api_key_info,
                        "attempt": int(attempt),
                        "max_retries": int(self.cfg.vlm.max_retries),
                        "timeout_cfg_s": float(self.cfg.vlm.timeout_s),
                        "timeout_effective_s": _normalize_timeout(float(self.cfg.vlm.timeout_s)),
                        "prompt_chars": len(prompt or ""),
                        "prompt_bytes": len((prompt or "").encode("utf-8")),
                        "input_tokens_est": _estimate_text_tokens(prompt or ""),
                        "payload_bytes": len(data),
                        "has_screenshot": bool(screenshot_path),
                        **image_meta,
                        **request_cache_metrics(cache_plan),
                        **balance_info,
                        "success": False,
                        "latency_s": time.time() - start_ts,
                        **_request_error_metrics(e),
                    },
                )
                network_signal = classify_network_exception(e, source="planner._chat")
                if network_signal is not None:
                    attach_network_context(
                        network_signal,
                        api_key_env=getattr(self.cfg.vlm, "api_key_env", ""),
                        api_key_pool_env=getattr(self.cfg.vlm, "api_key_pool_env", ""),
                        api_key_masked=str(api_key_info.get("api_key_masked", "") or ""),
                        api_key_slot=str(api_key_info.get("api_key_slot", "") or ""),
                        base_url=getattr(self.cfg.vlm, "base_url", ""),
                        model=getattr(self.cfg.vlm, "model", ""),
                        provider=getattr(self.cfg.vlm, "provider", ""),
                        channel="vlm",
                    )
                    raise network_signal from e
                if attempt == 0 and cache_plan.requested and cache_plan.reason:
                    logging.warning(
                        f"[EPM] vlm_prompt_cache status={cache_plan.status} reason={cache_plan.reason}"
                    )
                time.sleep(0.5 * (attempt + 1))
        _raise_wrapped_request_failure(
            last_err=last_err,
            label="vlm_planner_request_failed",
            source="planner._chat.final",
        )

    def _chat_raw(self, *, prompt: str, screenshot_path: str | None) -> Dict[str, Any]:
        if self.cfg.vlm.provider == "qwen3vl_http":
            # Normalize to an OpenAI-like shape for shared parsing.
            assert self._qwen_client is not None
            try:
                text = self._qwen_client.chat(
                    prompt=prompt,
                    image_path=screenshot_path,
                    max_new_tokens=int(self.cfg.vlm.max_tokens),
                )
                try:
                    write_model_call_trace(
                        trace_dir=default_model_trace_dir(request_metrics_path=self.cfg.request_metrics_path),
                        call_name="vlm_planner_qwen3vl_http_chat_raw",
                        prompt_text=prompt,
                        response_text=text,
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        screenshot_paths=([screenshot_path] if screenshot_path else []),
                        extra_meta={"planner_kind": "vlm_qwen3vl_http"},
                    )
                except Exception:
                    pass
            except Exception as e:
                try:
                    write_model_call_trace(
                        trace_dir=default_model_trace_dir(request_metrics_path=self.cfg.request_metrics_path),
                        call_name="vlm_planner_qwen3vl_http_chat_raw",
                        prompt_text=prompt,
                        response_text="",
                        error_text=repr(e),
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        screenshot_paths=([screenshot_path] if screenshot_path else []),
                        extra_meta={"planner_kind": "vlm_qwen3vl_http"},
                    )
                except Exception:
                    pass
                raise
            return {"choices": [{"message": {"role": "assistant", "content": text}}]}

        provider = str(getattr(self.cfg.vlm, "provider", "") or "").strip().lower().replace("-", "_")
        url = provider_request_url(self.cfg.vlm)
        api_key = _read_api_key(self.cfg.vlm)
        api_key_info = _log_api_key_use(cfg=self.cfg.vlm, api_key=api_key, call="_chat_raw", channel="vlm")
        headers = provider_request_headers(self.cfg.vlm, api_key)

        prompt_text = prompt + _tools_prompt_suffix()
        cache_plan = prepare_prompt_cache(
            cfg=self.cfg.vlm,
            prompt_text=prompt_text,
            request_metrics_path=self.cfg.request_metrics_path,
        )
        messages, image_meta = _make_messages(
            prompt_text=prompt_text,
            screenshot_path=screenshot_path,
            use_vision=bool(self.cfg.vlm.use_vision),
            image_max_side=int(getattr(self.cfg.vlm, "image_max_side", 768) or 768),
            image_format=str(getattr(self.cfg.vlm, "image_format", "jpeg") or "jpeg"),
            jpeg_quality=int(getattr(self.cfg.vlm, "jpeg_quality", 70) or 70),
            cache_plan=cache_plan,
        )
        payload: Dict[str, Any]
        if provider in OPENAI_PROVIDER_TYPES:
            payload = {
                "model": self.cfg.vlm.model,
                "messages": messages,
                "temperature": float(self.cfg.vlm.temperature),
                "tools": list(self._openai_tools or []),
                "tool_choice": "auto",
            }
            payload.update(build_openai_token_limit_payload(model=self.cfg.vlm.model, max_tokens=int(self.cfg.vlm.max_tokens)))
        else:
            system_text, anthropic_messages = openai_messages_to_anthropic(messages)
            payload = {
                "model": self.cfg.vlm.model,
                "messages": anthropic_messages,
                "temperature": float(self.cfg.vlm.temperature),
                "max_tokens": int(self.cfg.vlm.max_tokens),
                "tools": openai_tools_to_anthropic(list(self._openai_tools or [])),
                "tool_choice": {"type": "auto"},
            }
            if system_text:
                payload["system"] = system_text
        extra_body = build_extra_body(cache_plan)
        if extra_body:
            payload["extra_body"] = extra_body
        data = json.dumps(payload).encode("utf-8")
        trace_dir = default_model_trace_dir(request_metrics_path=self.cfg.request_metrics_path)
        if image_meta:
            logging.info(
                "[EPM] image_compression "
                f"src_bytes={image_meta.get('image_source_bytes')} "
                f"encoded_bytes={image_meta.get('image_encoded_bytes')} "
                f"data_url_bytes={image_meta.get('image_data_url_bytes')} "
                f"fmt={image_meta.get('image_format')} "
                f"orig={image_meta.get('image_orig_size')} "
                f"final={image_meta.get('image_final_size')} "
                f"ratio={image_meta.get('image_compression_ratio')}"
            )

        last_err: Optional[Exception] = None
        for attempt in range(int(self.cfg.vlm.max_retries) + 1):
            start_ts = time.time()
            try:
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=_normalize_timeout(float(self.cfg.vlm.timeout_s))) as resp:
                    body = resp.read().decode("utf-8")
                out = json.loads(body)
                if provider in ANTHROPIC_PROVIDER_TYPES:
                    out = anthropic_response_to_openai(out)
                if not isinstance(out, dict):
                    raise ValueError("non_dict_response")
                try:
                    msg = _extract_openai_message(out)
                    content = msg.get("content")
                    response_text = content if isinstance(content, str) else json.dumps(msg, ensure_ascii=False, indent=2)
                    write_model_call_trace(
                        trace_dir=trace_dir,
                        call_name="vlm_planner_chat_raw",
                        prompt_text=prompt_text,
                        response_text=response_text,
                        response_payload=out,
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        attempt=int(attempt),
                        screenshot_paths=([screenshot_path] if screenshot_path else []),
                        extra_meta={"request_url": str(url or ""), "planner_kind": "vlm_openai_compatible"},
                    )
                except Exception:
                    pass
                _warn_if_model_output_truncated(out, tag="vlm")
                balance_info = _maybe_log_balance(cfg=self.cfg.vlm, api_key=api_key, call="_chat_raw", channel="vlm")
                _append_jsonl(
                    self.cfg.request_metrics_path,
                    {
                        "ts": start_ts,
                        "planner_mode": "vlm",
                        "api_mode": provider_api_mode(provider),
                        "provider": str(self.cfg.vlm.provider or ""),
                        "model": str(self.cfg.vlm.model or ""),
                        "request_url": str(url or ""),
                        "call": "_chat_raw",
                        **api_key_info,
                        "attempt": int(attempt),
                        "max_retries": int(self.cfg.vlm.max_retries),
                        "timeout_cfg_s": float(self.cfg.vlm.timeout_s),
                        "timeout_effective_s": _normalize_timeout(float(self.cfg.vlm.timeout_s)),
                        "prompt_chars": len(prompt_text or ""),
                        "prompt_bytes": len((prompt_text or "").encode("utf-8")),
                        "input_tokens_est": _estimate_text_tokens(prompt_text or ""),
                        "payload_bytes": len(data),
                        "has_screenshot": bool(screenshot_path),
                        **image_meta,
                        **request_cache_metrics(cache_plan),
                        **_usage_metrics(out),
                        **balance_info,
                        "success": True,
                        "latency_s": time.time() - start_ts,
                    },
                )
                return out
            except Exception as e:
                last_err = e
                try:
                    write_model_call_trace(
                        trace_dir=trace_dir,
                        call_name="vlm_planner_chat_raw",
                        prompt_text=prompt_text,
                        response_text="",
                        error_text=repr(e),
                        provider=str(getattr(self.cfg.vlm, "provider", "") or ""),
                        model=str(getattr(self.cfg.vlm, "model", "") or ""),
                        attempt=int(attempt),
                        screenshot_paths=([screenshot_path] if screenshot_path else []),
                        extra_meta={"request_url": str(url or ""), "planner_kind": "vlm_openai_compatible"},
                    )
                except Exception:
                    pass
                balance_info = _maybe_log_balance(cfg=self.cfg.vlm, api_key=api_key, call="_chat_raw", channel="vlm")
                _append_jsonl(
                    self.cfg.request_metrics_path,
                    {
                        "ts": start_ts,
                        "planner_mode": "vlm",
                        "api_mode": provider_api_mode(provider),
                        "provider": str(self.cfg.vlm.provider or ""),
                        "model": str(self.cfg.vlm.model or ""),
                        "request_url": str(url or ""),
                        "call": "_chat_raw",
                        **api_key_info,
                        "attempt": int(attempt),
                        "max_retries": int(self.cfg.vlm.max_retries),
                        "timeout_cfg_s": float(self.cfg.vlm.timeout_s),
                        "timeout_effective_s": _normalize_timeout(float(self.cfg.vlm.timeout_s)),
                        "prompt_chars": len(prompt_text or ""),
                        "prompt_bytes": len((prompt_text or "").encode("utf-8")),
                        "input_tokens_est": _estimate_text_tokens(prompt_text or ""),
                        "payload_bytes": len(data),
                        "has_screenshot": bool(screenshot_path),
                        **image_meta,
                        **request_cache_metrics(cache_plan),
                        **balance_info,
                        "success": False,
                        "latency_s": time.time() - start_ts,
                        **_request_error_metrics(e),
                    },
                )
                network_signal = classify_network_exception(e, source="planner._chat_raw")
                if network_signal is not None:
                    attach_network_context(
                        network_signal,
                        api_key_env=getattr(self.cfg.vlm, "api_key_env", ""),
                        api_key_pool_env=getattr(self.cfg.vlm, "api_key_pool_env", ""),
                        base_url=getattr(self.cfg.vlm, "base_url", ""),
                        model=getattr(self.cfg.vlm, "model", ""),
                        provider=getattr(self.cfg.vlm, "provider", ""),
                        channel="vlm",
                    )
                    raise network_signal from e
                if attempt == 0 and cache_plan.requested and cache_plan.reason:
                    logging.warning(
                        f"[EPM] vlm_prompt_cache status={cache_plan.status} reason={cache_plan.reason}"
                    )
                time.sleep(0.5 * (attempt + 1))
        _raise_wrapped_request_failure(
            last_err=last_err,
            label="vlm_planner_request_failed",
            source="planner._chat_raw.final",
        )

    @staticmethod
    def _repair_prompt(
        *,
        previous_output: str,
        high_level_id: str,
        plan_min_steps: int,
        plan_max_steps: int,
        instance_id_selection_mode: bool = False,
    ) -> str:
        prev = (previous_output or "").strip()
        if instance_id_selection_mode:
            return (
                "Your previous response did not follow the required response format.\n"
                "Rewrite your answer to be a SINGLE JSON object and NOTHING else.\n"
                "- No markdown fences.\n"
                "- No commentary.\n"
                "- No extra keys.\n"
                "- No <think>...</think> blocks. Output ONLY the final JSON.\n"
                "- Use valid JSON with double quotes, no trailing commas.\n"
                "- Output ONLY two keys: name, instance_id.\n"
                "\n"
                "Return this exact JSON schema:\n"
                '{\n  "name": "candidate name",\n  "instance_id": 123\n}\n'
                "\n"
                "Previous response (invalid):\n"
                f"{prev}\n"
            )
        return (
            "Your previous response did not follow the required response format.\n"
            "Rewrite your answer to be a SINGLE JSON object and NOTHING else.\n"
            "- No markdown fences.\n"
            "- No commentary.\n"
            "- No extra keys.\n"
            "- No <think>...</think> blocks. Output ONLY the final JSON.\n"
            "- Use valid JSON with double quotes, no trailing commas.\n"
            + (
                f"- action_list length MUST be >= {int(plan_min_steps)} (no maximum).\n"
                if int(plan_max_steps) < 0
                else f"- action_list length MUST be within [{int(plan_min_steps)}, {int(plan_max_steps)}]. Do NOT output more than {int(plan_max_steps)} steps.\n"
            )
            +
            "- Keep it compact: thoughts <= 1 sentence; each expectation <= 1 sentence.\n"
            "- Use ONLY the exact action/skill names from the provided schemas. Do NOT use pseudo verbs like goto/pick_up/pour.\n"
            "\n"
            "Return this exact JSON schema:\n"
            "{\n"
            f'  "high_level_id": "{str(high_level_id)}",\n'
            '  "goal": "one sentence goal",\n'
            '  "explanation": null | "why last attempt failed",\n'
            '  "thoughts": "brief planning rationale",\n'
            '  "action_list": [\n'
            f'    {{"step_id":"{str(high_level_id)}.A1","type":"action","name":"EXACT_ACTION_NAME_FROM_SCHEMA","args":{{"required_arg_name":"required_arg_value"}},"expectation":"a concrete observable outcome"}}\n'
            "  ]\n"
            "}\n"
            "\n"
            "Previous response (invalid):\n"
            f"{prev}\n"
        )
