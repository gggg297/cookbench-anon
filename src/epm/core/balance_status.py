from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_BALANCE_CACHE: dict[str, dict[str, Any]] = {}


def provider_root(base_url: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base


def usage_url(base_url: str) -> str:
    root = provider_root(base_url)
    return root.rstrip("/") + "/api/usage/token/"


def balance_check_enabled(base_url: str) -> bool:
    """Whether prepaid-balance polling applies to this endpoint.

    Balance polling is opt-in: set ``EPM_BALANCE_CHECK_HOSTS`` to a comma-separated
    list of endpoint hosts (e.g. ``api.example.com``) whose ``/api/usage/token/``
    endpoint should be queried. Endpoints that do not implement that route simply
    report an error and are otherwise ignored.
    """
    host = provider_root(base_url).lower()
    if not host:
        return False
    configured = os.environ.get("EPM_BALANCE_CHECK_HOSTS", "")
    hosts = [item.strip().lower() for item in configured.split(",") if item.strip()]
    if not hosts:
        return False
    return any(item in host for item in hosts)


def query_token_usage(*, base_url: str, api_key: str, timeout_s: float = 8.0) -> tuple[dict[str, Any] | None, str]:
    key = str(api_key or "").strip()
    if not key:
        return None, "missing_api_key"
    url = usage_url(base_url)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return None, f"http_error:{e.code}:{body[:300]}"
    except Exception as e:
        return None, f"request_failed:{e!r}"

    try:
        obj = json.loads(raw)
    except Exception as e:
        return None, f"json_parse_failed:{e!r}"
    if not isinstance(obj, dict):
        return None, "non_dict_response"

    data = obj.get("data")
    if not isinstance(data, dict):
        return None, f"missing_data:{str(obj)[:300]}"

    out = {
        "url": url,
        "message": str(obj.get("message") or ""),
        "name": str(data.get("name") or ""),
        "object": str(data.get("object") or ""),
        "balance_usd": _as_float(data.get("balance_usd")),
        "used_usd": _as_float(data.get("used_usd")),
        "total_usd": _as_float(data.get("total_usd")),
        "total_available": _as_int(data.get("total_available")),
        "total_granted": _as_int(data.get("total_granted")),
        "total_used": _as_int(data.get("total_used")),
        "expires_at": _as_int(data.get("expires_at")),
        "model_limits_enabled": bool(data.get("model_limits_enabled", False)),
        "unlimited_quota": bool(data.get("unlimited_quota", False)),
        "raw": obj,
    }
    return out, ""


def get_cached_balance(memory_dir: str) -> dict[str, Any]:
    key = str(memory_dir or "").strip()
    if not key:
        return {}
    payload = _BALANCE_CACHE.get(key)
    return dict(payload) if isinstance(payload, dict) else {}


def maybe_refresh_balance_status(
    *,
    base_url: str,
    api_key: str,
    memory_dir: str | Path,
    episode_step: int,
    cadence_steps: int = 10,
    timeout_s: float = 8.0,
    channel: str = "",
    call: str = "",
) -> tuple[dict[str, Any], bool]:
    memory_key = str(memory_dir or "").strip()
    current_step = max(0, int(episode_step))
    cadence = max(1, int(cadence_steps))
    existing = get_cached_balance(memory_key)
    try:
        last_checked_step = int(existing.get("last_checked_step"))
    except Exception:
        last_checked_step = -1
    if last_checked_step >= 0 and current_step - last_checked_step < cadence:
        cached = dict(existing)
        cached["cached"] = True
        cached["refreshed"] = False
        return cached, False

    info, err = query_token_usage(base_url=base_url, api_key=api_key, timeout_s=timeout_s)
    payload = dict(existing) if isinstance(existing, dict) else {}
    payload.update(
        {
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "last_checked_step": int(current_step),
            "cadence_steps": int(cadence),
            "channel": str(channel or ""),
            "call": str(call or ""),
            "provider_root": provider_root(base_url),
            "api_key_masked": _mask_secret(api_key),
            "error": str(err or ""),
            "cached": False,
            "refreshed": True,
        }
    )
    if info is not None:
        payload.update(
            {
                "name": info.get("name"),
                "object": info.get("object"),
                "balance_usd": info.get("balance_usd"),
                "used_usd": info.get("used_usd"),
                "total_usd": info.get("total_usd"),
                "total_available": info.get("total_available"),
                "total_granted": info.get("total_granted"),
                "total_used": info.get("total_used"),
                "expires_at": info.get("expires_at"),
                "unlimited_quota": info.get("unlimited_quota"),
                "url": info.get("url"),
                "message": info.get("message"),
            }
        )
    if memory_key:
        _BALANCE_CACHE[memory_key] = dict(payload)
    return payload, True


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _mask_secret(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:2]}***{text[-4:]}"
