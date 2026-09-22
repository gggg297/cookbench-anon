from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epm.core.balance_status import balance_check_enabled, query_token_usage, provider_root


_BALANCE_CACHE_TTL_S = 30.0
_BALANCE_CACHE: dict[tuple[str, str], tuple[float, dict[str, Any] | None, str]] = {}


@dataclass(frozen=True)
class ApiKeyCandidateSelection:
    keys: list[str]
    sources: list[str]
    local_matched: bool = False
    local_path: str = ""
    diagnostics: list[str] | None = None


def _configs_root() -> Path:
    return Path(__file__).resolve().parents[3] / "configs"


def local_api_keys_example_path() -> Path:
    return _configs_root() / "api_keys.local.example.json"


def local_api_keys_path() -> Path:
    return _configs_root() / "api_keys.local.json"


def parse_api_key_pool(raw: str) -> list[str]:
    text = str(raw or "")
    if not text.strip():
        return []
    parts = re.split(r"[;,\r\n]+", text)
    return _dedupe([str(part or "").strip() for part in parts if str(part or "").strip()])


def collect_api_key_candidates(
    *,
    explicit_key: str,
    api_key_env: str,
    api_key_pool_env: str,
    base_url: str = "",
) -> tuple[list[str], list[str]]:
    selection = collect_api_key_candidates_detailed(
        explicit_key=explicit_key,
        api_key_env=api_key_env,
        api_key_pool_env=api_key_pool_env,
        base_url=base_url,
    )
    return selection.keys, selection.sources


def collect_api_key_candidates_detailed(
    *,
    explicit_key: str,
    api_key_env: str,
    api_key_pool_env: str,
    base_url: str = "",
) -> ApiKeyCandidateSelection:
    local_selection = _load_local_api_key_candidates(
        api_key_env=api_key_env,
        api_key_pool_env=api_key_pool_env,
        base_url=base_url,
    )
    if local_selection.local_matched:
        if local_selection.keys:
            return local_selection
        diag = " | ".join(local_selection.diagnostics or [])
        raise RuntimeError(
            "local_api_keys_no_usable_key:"
            f"path={local_selection.local_path or local_api_keys_path()} "
            f"env={api_key_env or '-'} pool_env={api_key_pool_env or '-'} "
            f"provider={provider_root(base_url) or base_url or '-'} "
            f"detail={diag or 'all matching local groups were empty or below threshold'}"
        )

    items: list[tuple[str, str]] = []
    if str(explicit_key or "").strip():
        items.append((str(explicit_key).strip(), "config.api_key"))
    if str(api_key_pool_env or "").strip():
        for idx, key in enumerate(parse_api_key_pool(os.environ.get(api_key_pool_env, "")), start=1):
            items.append((key, f"pool:{api_key_pool_env}[{idx}]"))
    if str(api_key_env or "").strip():
        env_key = str(os.environ.get(api_key_env, "") or "").strip()
        if env_key:
            items.append((env_key, f"env:{api_key_env}"))

    out_keys: list[str] = []
    out_sources: list[str] = []
    seen: set[str] = set()
    for key, source in items:
        if key in seen:
            continue
        seen.add(key)
        out_keys.append(key)
        out_sources.append(source)
    return ApiKeyCandidateSelection(keys=out_keys, sources=out_sources, local_path=str(local_api_keys_path()))


def _load_local_api_key_candidates(
    *,
    api_key_env: str,
    api_key_pool_env: str,
    base_url: str,
) -> ApiKeyCandidateSelection:
    path = local_api_keys_path()
    if not path.exists():
        return ApiKeyCandidateSelection(keys=[], sources=[], local_matched=False, local_path=str(path))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"local_api_keys_parse_failed:path={path}:error={e!r}") from e
    if not isinstance(raw, dict):
        raise RuntimeError(f"local_api_keys_invalid_root:path={path}:expected_object")

    entries = _normalize_local_entries(raw)
    if not entries:
        return ApiKeyCandidateSelection(keys=[], sources=[], local_matched=False, local_path=str(path))

    matched_entries: list[dict[str, Any]] = []
    all_keys: list[str] = []
    all_sources: list[str] = []
    diagnostics: list[str] = []
    seen: set[str] = set()
    for idx, entry in enumerate(entries, start=1):
        if not _entry_matches(
            entry,
            api_key_env=api_key_env,
            api_key_pool_env=api_key_pool_env,
            base_url=base_url,
        ):
            continue
        matched_entries.append(entry)
        keys, sources, entry_diags = _resolve_entry_keys(
            entry,
            entry_index=idx,
            base_url=base_url,
        )
        diagnostics.extend(entry_diags)
        for key, source in zip(keys, sources):
            if key in seen:
                continue
            seen.add(key)
            all_keys.append(key)
            all_sources.append(source)

    return ApiKeyCandidateSelection(
        keys=all_keys,
        sources=all_sources,
        local_matched=bool(matched_entries),
        local_path=str(path),
        diagnostics=diagnostics,
    )


def _normalize_local_entries(raw: dict[str, Any]) -> list[dict[str, Any]]:
    entries_raw = raw.get("api_key_priority")
    if isinstance(entries_raw, list):
        return [entry for entry in entries_raw if isinstance(entry, dict)]

    # Backward compatibility for the old mapping-style file.
    pools = raw.get("api_key_pools")
    if not isinstance(pools, dict):
        return []

    entries: list[dict[str, Any]] = []
    for name, entry in pools.items():
        label = str(name or "").strip()
        if isinstance(entry, dict):
            keys_raw = entry.get("keys")
            keys = [str(x or "").strip() for x in list(keys_raw or []) if str(x or "").strip()]
            entries.append(
                {
                    "name": str(entry.get("label") or label or "local_pool"),
                    "api_key_env": label,
                    "api_key_pool_env": label,
                    "provider_root_contains": str(entry.get("provider_root_contains") or "").strip(),
                    "min_balance_usd": entry.get("min_balance_usd"),
                    "keys": keys,
                }
            )
            continue
        if isinstance(entry, list):
            keys = [str(x or "").strip() for x in entry if str(x or "").strip()]
            entries.append(
                {
                    "name": label or "local_pool",
                    "api_key_env": label,
                    "api_key_pool_env": label,
                    "provider_root_contains": "",
                    "min_balance_usd": None,
                    "keys": keys,
                }
            )
    return entries


def _entry_matches(
    entry: dict[str, Any],
    *,
    api_key_env: str,
    api_key_pool_env: str,
    base_url: str,
) -> bool:
    provider_root_value = provider_root(base_url).lower()
    provider_filters = _as_str_list(entry.get("provider_root_contains"))
    env_filters = [
        str(entry.get("api_key_env") or "").strip(),
        str(entry.get("api_key_pool_env") or "").strip(),
    ]
    requested_envs = {str(api_key_env or "").strip(), str(api_key_pool_env or "").strip()}
    requested_envs.discard("")

    provider_match = bool(provider_filters) and any(text.lower() in provider_root_value for text in provider_filters)
    env_match = any(text and text in requested_envs for text in env_filters)
    return provider_match or env_match


def _resolve_entry_keys(
    entry: dict[str, Any],
    *,
    entry_index: int,
    base_url: str,
) -> tuple[list[str], list[str], list[str]]:
    keys = _dedupe([str(x or "").strip() for x in list(entry.get("keys") or []) if str(x or "").strip()])
    name = str(entry.get("name") or entry.get("label") or f"group_{entry_index}").strip() or f"group_{entry_index}"
    if not keys:
        return [], [], [f"{name}:empty_keys"]

    balances_enabled = balance_check_enabled(base_url)
    min_balance_usd = _as_float(entry.get("min_balance_usd"))
    should_balance_check = balances_enabled or (min_balance_usd is not None)
    threshold = min_balance_usd

    out_keys: list[str] = []
    out_sources: list[str] = []
    diagnostics: list[str] = []

    for key_index, key in enumerate(keys, start=1):
        source = f"local:{name}[{key_index}]"
        if not should_balance_check:
            out_keys.append(key)
            out_sources.append(source)
            continue

        if not balances_enabled:
            diagnostics.append(f"{name}[{key_index}]:unsupported_balance_probe")
            continue

        info, err = _query_cached_balance(base_url=base_url, api_key=key)
        if err:
            diagnostics.append(f"{name}[{key_index}]:balance_probe_failed:{err}")
            if _should_accept_key_when_balance_probe_failed(err):
                out_keys.append(key)
                out_sources.append(f"{source}|balance_probe_unverified={err}")
            continue
        balance = _as_float((info or {}).get("balance_usd"))
        if balance is None:
            diagnostics.append(f"{name}[{key_index}]:balance_missing")
            continue
        if threshold is not None and balance < float(threshold):
            diagnostics.append(f"{name}[{key_index}]:balance_below_threshold:{balance:.6f}<{float(threshold):.6f}")
            continue
        out_keys.append(key)
        out_sources.append(f"{source}|balance={balance:.6f}")

    if not out_keys and not diagnostics:
        diagnostics.append(f"{name}:no_usable_keys")
    return out_keys, out_sources, diagnostics


def _query_cached_balance(*, base_url: str, api_key: str) -> tuple[dict[str, Any] | None, str]:
    root = provider_root(base_url)
    cache_key = (root, str(api_key or "").strip())
    now = time.time()
    cached = _BALANCE_CACHE.get(cache_key)
    if cached is not None and (now - cached[0]) < _BALANCE_CACHE_TTL_S:
        return cached[1], cached[2]
    info, err = query_token_usage(base_url=base_url, api_key=api_key, timeout_s=8.0)
    _BALANCE_CACHE[cache_key] = (now, info, str(err or ""))
    return info, str(err or "")


def _should_accept_key_when_balance_probe_failed(err: str) -> bool:
    text = str(err or "").strip().lower()
    if not text:
        return False
    accepted_markers = (
        "http_error:429",
        "timeout",
        "ssl",
        "connection",
        "remote_disconnected",
        "unexpected_eof",
    )
    return any(marker in text for marker in accepted_markers)


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    return []


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out
