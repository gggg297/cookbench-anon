from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol


_DYNAMIC_USER_MARKERS = (
    "\nObservation:\n",
    "\nFeedback from last execution:\n",
    "\nMust-fix precondition blocker:\n",
    "\nMust-respect visual warning:\n",
    "\nAuto query for disambiguation:\n",
    "\nCross-step query memory:\n",
    "\nSTM window snapshot:\n",
    "\nPlanner-facing visual summary:\n",
    "\nAgent state snapshot:\n",
    "\nCountdown status:\n",
    "\nTask progress hard constraints:\n",
)


class _CacheSettings(Protocol):
    provider: str
    base_url: str
    model: str
    api_key_env: str
    api_key: str
    prompt_cache_enabled: bool
    prompt_cache_force: bool
    prompt_cache_min_chars: int
    prompt_cache_ttl_s: int
    prompt_cache_dir: str
    gemini_cache_api_base_url: str


@dataclass(frozen=True)
class PromptCacheParts:
    prompt_text: str
    system_text: str
    user_text: str
    stable_user_text: str
    dynamic_user_text: str

    @property
    def prefix_text(self) -> str:
        pieces = []
        if self.system_text.strip():
            pieces.append(self.system_text.strip())
        if self.stable_user_text.strip():
            pieces.append(self.stable_user_text.strip())
        return "\n\n".join(pieces).strip()

    @property
    def prefix_chars(self) -> int:
        return len(self.prefix_text)


@dataclass(frozen=True)
class PromptCachePlan:
    family: str = ""
    mode: str = "disabled"
    requested: bool = False
    enabled: bool = False
    status: str = "disabled"
    reason: str = ""
    parts: PromptCacheParts = field(
        default_factory=lambda: PromptCacheParts(
            prompt_text="",
            system_text="",
            user_text="",
            stable_user_text="",
            dynamic_user_text="",
        )
    )
    cache_name: str = ""


def read_api_key(cfg: _CacheSettings) -> str:
    env_name = getattr(cfg, "api_key_env", "") or ""
    return (getattr(cfg, "api_key", "") or "").strip() or (os.environ.get(env_name) or "").strip()


def detect_cache_family(model: str) -> str:
    low = str(model or "").strip().lower()
    if "gemini" in low:
        return "gemini"
    if "qwen" in low:
        return "qwen"
    return ""


def _is_official_gemini_api_url(url: str) -> bool:
    raw = str(url or "").strip()
    if not raw:
        return False
    parsed = urllib.parse.urlparse(raw)
    host = (parsed.netloc or parsed.path or "").strip().lower()
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    return host == "generativelanguage.googleapis.com"


def split_prompt_for_cache(prompt_text: str) -> PromptCacheParts:
    raw = str(prompt_text or "")
    system_text = ""
    user_text = raw
    if raw.startswith("SYSTEM:\n"):
        body = raw[len("SYSTEM:\n") :]
        sep = "\nUSER:\n"
        idx = body.find(sep)
        if idx >= 0:
            system_text = body[:idx]
            user_text = body[idx + len(sep) :]
    stable_user = user_text
    dynamic_user = ""
    earliest_idx: Optional[int] = None
    for marker in _DYNAMIC_USER_MARKERS:
        idx = user_text.find(marker)
        if idx >= 0 and (earliest_idx is None or idx < earliest_idx):
            earliest_idx = idx
    if earliest_idx is not None:
        stable_user = user_text[:earliest_idx]
        dynamic_user = user_text[earliest_idx:]
    return PromptCacheParts(
        prompt_text=raw,
        system_text=system_text,
        user_text=user_text,
        stable_user_text=stable_user,
        dynamic_user_text=dynamic_user,
    )


def request_cache_metrics(plan: PromptCachePlan) -> dict[str, Any]:
    return {
        "cache_family": plan.family or "",
        "cache_mode": plan.mode,
        "cache_requested": bool(plan.requested),
        "cache_enabled": bool(plan.enabled),
        "cache_status": plan.status,
        "cache_reason": plan.reason,
        "cache_prefix_chars": int(plan.parts.prefix_chars),
        "cache_name": plan.cache_name or "",
    }


def usage_cache_metrics(out: Any) -> dict[str, Any]:
    if not isinstance(out, dict):
        return {
            "cache_hit": False,
            "cache_hit_tokens": 0,
            "cache_creation_tokens": 0,
        }
    usage = out.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    prompt_details = usage.get("prompt_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    input_details = usage.get("input_tokens_details")
    input_details = input_details if isinstance(input_details, dict) else {}
    usage_md = out.get("usage_metadata")
    usage_md = usage_md if isinstance(usage_md, dict) else {}
    cached_tokens = int(
        prompt_details.get("cached_tokens")
        or input_details.get("cached_tokens")
        or usage_md.get("cached_content_token_count")
        or 0
    )
    creation_tokens = int(
        prompt_details.get("cache_creation_input_tokens")
        or usage_md.get("cache_creation_input_token_count")
        or 0
    )
    return {
        "cache_hit": cached_tokens > 0,
        "cache_hit_tokens": cached_tokens,
        "cache_creation_tokens": creation_tokens,
    }


def build_extra_body(plan: PromptCachePlan) -> dict[str, Any]:
    if plan.mode == "gemini_explicit" and plan.cache_name:
        return {"google": {"cached_content": plan.cache_name}}
    return {}


def should_split_messages(plan: PromptCachePlan) -> bool:
    return plan.mode in ("gemini_explicit", "qwen_explicit")


def should_use_qwen_cache_control(plan: PromptCachePlan) -> bool:
    return plan.mode == "qwen_explicit"


def prepare_prompt_cache(
    *,
    cfg: _CacheSettings,
    prompt_text: str,
    request_metrics_path: Path | None = None,
) -> PromptCachePlan:
    parts = split_prompt_for_cache(prompt_text)
    family = detect_cache_family(getattr(cfg, "model", ""))
    enabled = bool(getattr(cfg, "prompt_cache_enabled", True))
    if not enabled or not family:
        return PromptCachePlan(
            family=family,
            requested=False,
            enabled=False,
            status="disabled",
            reason=("unsupported_model_family" if not family else "prompt_cache_disabled"),
            parts=parts,
        )
    prefix_chars = parts.prefix_chars
    min_chars = int(getattr(cfg, "prompt_cache_min_chars", 4096) or 4096)
    if prefix_chars < min_chars:
        return PromptCachePlan(
            family=family,
            requested=False,
            enabled=False,
            status="skipped",
            reason=f"prefix_too_short:{prefix_chars}<{min_chars}",
            parts=parts,
        )
    if not parts.stable_user_text.strip():
        return PromptCachePlan(
            family=family,
            requested=False,
            enabled=False,
            status="skipped",
            reason="missing_stable_user_prefix",
            parts=parts,
        )
    if family == "qwen":
        return PromptCachePlan(
            family=family,
            mode="qwen_explicit",
            requested=True,
            enabled=True,
            status="prepared",
            parts=parts,
        )
    if family == "gemini":
        cache_name, status, reason = _ensure_gemini_cache_name(
            cfg=cfg,
            parts=parts,
            request_metrics_path=request_metrics_path,
        )
        return PromptCachePlan(
            family=family,
            mode=("gemini_explicit" if cache_name else "disabled"),
            requested=True,
            enabled=bool(cache_name),
            status=status,
            reason=reason,
            parts=parts,
            cache_name=cache_name,
        )
    return PromptCachePlan(
        family=family,
        requested=False,
        enabled=False,
        status="disabled",
        reason="unsupported_family",
        parts=parts,
    )


def _infer_gemini_cache_api_base_url(cfg: _CacheSettings) -> str:
    override = str(getattr(cfg, "gemini_cache_api_base_url", "") or "").strip().rstrip("/")
    if override and _is_official_gemini_api_url(override):
        return override
    base_url = str(getattr(cfg, "base_url", "") or "").strip().rstrip("/")
    if _is_official_gemini_api_url(base_url):
        return "https://generativelanguage.googleapis.com"
    return ""


def _ensure_gemini_cache_name(
    *,
    cfg: _CacheSettings,
    parts: PromptCacheParts,
    request_metrics_path: Path | None,
) -> tuple[str, str, str]:
    api_base = _infer_gemini_cache_api_base_url(cfg)
    if not api_base:
        return "", "unsupported_endpoint", "gemini_explicit_cache_requires_official_google_endpoint"
    api_key = read_api_key(cfg)
    if not api_key:
        return "", "disabled", "missing_api_key"
    cache_dir = _cache_dir(cfg=cfg, request_metrics_path=request_metrics_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ttl_s = max(60, int(getattr(cfg, "prompt_cache_ttl_s", 3600) or 3600))
    digest = hashlib.sha256(
        json.dumps(
            {
                "model": str(getattr(cfg, "model", "") or ""),
                "system": parts.system_text,
                "stable_user": parts.stable_user_text,
                "ttl_s": ttl_s,
                "api_base": api_base,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    meta_path = cache_dir / f"gemini_cache_{digest}.json"
    now_s = time.time()
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cache_name = str(meta.get("cache_name") or "").strip()
            expire_at_s = float(meta.get("expire_at_s") or 0.0)
            if cache_name and expire_at_s > now_s + 5.0:
                return cache_name, "reused", ""
        except Exception:
            pass
    if not api_base:
        return "", "unsupported_endpoint", "missing_gemini_cache_api_base_url"
    payload: dict[str, Any] = {
        "model": f"models/{str(getattr(cfg, 'model', '') or '').strip()}",
        "contents": [
            {
                "role": "user",
                "parts": [{"text": parts.stable_user_text}],
            }
        ],
        "ttl": f"{ttl_s}s",
    }
    if parts.system_text.strip():
        payload["systemInstruction"] = {
            "role": "system",
            "parts": [{"text": parts.system_text}],
        }
    try:
        url = f"{api_base}/v1beta/cachedContents?key={urllib.parse.quote(api_key)}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        out = json.loads(body)
        cache_name = str(out.get("name") or "").strip()
        if not cache_name:
            return "", "create_failed", "missing_cache_name_in_response"
        expire_at_s = now_s + ttl_s
        meta_path.write_text(
            json.dumps(
                {
                    "cache_name": cache_name,
                    "expire_at_s": expire_at_s,
                    "model": str(getattr(cfg, "model", "") or ""),
                    "prefix_chars": parts.prefix_chars,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return cache_name, "created", ""
    except Exception as exc:
        logging.warning("[EPM] gemini_cache_create_failed: %r", exc)
        return "", "create_failed", repr(exc)


def _cache_dir(*, cfg: _CacheSettings, request_metrics_path: Path | None) -> Path:
    override = str(getattr(cfg, "prompt_cache_dir", "") or "").strip()
    if override:
        return Path(override)
    if request_metrics_path is not None:
        return Path(request_metrics_path).resolve().parent / "prompt_caches"
    return Path.cwd() / "prompt_caches"
