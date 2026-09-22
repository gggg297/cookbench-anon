from __future__ import annotations

import argparse
import faulthandler
import io
import json
import os
import re
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from loguru import logger as logging

SCRIPT_PATH = Path(__file__).resolve()
EPM_ROOT = SCRIPT_PATH.parents[1]
SRC_DIR = EPM_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from epm.core.agent import AgentConfig, EpmAgent  # noqa: E402
from epm.core.api_key_pool_source import collect_api_key_candidates, parse_api_key_pool  # noqa: E402
from epm.core.http_403_pause import Http403PauseRequired, NetworkAbortRequired, NetworkPauseRequired  # noqa: E402
from epm.core.notifier import maybe_send_email_notification  # noqa: E402
from epm.core.prompt_ablation import (  # noqa: E402
    RAW_INPUT_ACTIONS,
    audit_prompt_trace_directory,
    resolve_ablation_reductions,
    restrict_to_actions_only,
    restrict_to_raw_input_actions,
)
from epm.core.settings import load_settings  # noqa: E402
from epm.core.balance_status import balance_check_enabled, get_cached_balance, maybe_refresh_balance_status  # noqa: E402
from epm.cerebellum.game_hotkeys import bootstrap_all  # noqa: E402
from epm.cerebellum.raw_input_controller import StopRequested, RawInputController  # noqa: E402


try:
    # Emit Python stack traces for fatal native crashes (for example Win32 access
    # violations inside ctypes / pywin32 / cv2 / mss paths). Dashboard already
    # captures stderr, so this gives us a best-effort traceback instead of a silent
    # large Windows exit code only.
    faulthandler.enable(file=sys.stderr, all_threads=True)
except Exception:
    pass


def _network_probable_causes(err: BaseException) -> str:
    kind = str(getattr(err, "kind", "") or "").strip().lower()
    http_status = getattr(err, "http_status", None)
    if http_status == 401:
        return "invalid_api_key_or_bearer_token/key_revoked_or_expired/wrong_provider_channel_or_endpoint_auth"
    if kind == "ssl":
        return (
            "network_unstable/tls_handshake_or_stream_interrupted/"
            "server_closed_connection_early/proxy_or_gateway_reset"
        )
    if kind == "timeout":
        return "api_slow/network_latency/provider_overloaded/request_timeout"
    if kind in {"connection_reset", "connection_aborted", "remote_disconnected", "connection_error"}:
        return "network_unstable/server_reset_connection/proxy_interrupted/upstream_disconnected"
    if kind == "http_403" or http_status == 403:
        return "api_key_invalid_or_expired/quota_or_account_restricted/provider_rejected_request"
    if http_status == 429:
        return "rate_limit/quota_exceeded/provider_throttling"
    if isinstance(http_status, int) and http_status >= 500:
        return "provider_server_error/upstream_gateway_error/temporary_overload"
    return "network_path_unstable/provider_temporary_failure/request_chain_interrupted"


def _network_diagnosis(err: BaseException) -> str:
    kind = str(getattr(err, "kind", "") or "").strip().lower()
    http_status = getattr(err, "http_status", None)
    if http_status == 401:
        return "认证失败：当前 Bearer API key 无效、已过期、被撤销，或与当前 provider/base_url 不匹配。"
    if kind == "http_403" or http_status == 403:
        return "请求被 provider 拒绝：当前 key 常见原因是额度不足、账户受限、预扣费失败，或该渠道暂时不允许继续调用。"
    if http_status == 429:
        return "接口被限流：provider 认为当前请求频率过高，或该账户/模型通道触发了速率限制。"
    if kind == "ssl":
        return "TLS/SSL 连接异常：握手过程中断、证书链异常，或中间代理/网关提前关闭了加密连接。"
    if kind == "timeout":
        return "请求超时：provider 返回太慢，或网络链路抖动导致在超时时间内没收到响应。"
    if kind in {"connection_reset", "connection_aborted", "remote_disconnected", "connection_error"}:
        return "连接被对端或中间网关直接断开：请求已发出，但 provider/代理在完整返回前关闭了连接。"
    if kind == "dns":
        return "DNS 解析失败：当前机器无法把 provider 域名解析成 IP。"
    if isinstance(http_status, int) and http_status >= 500:
        return "Provider 服务器内部异常：上游服务、网关或模型后端暂时不可用。"
    return "网络链路或 provider 临时异常：请求没有正常走完整个响应链路。"


def _network_common_triggers(err: BaseException) -> str:
    kind = str(getattr(err, "kind", "") or "").strip().lower()
    http_status = getattr(err, "http_status", None)
    if http_status == 401:
        return "key 填错、key 失效、key 不属于这个第三方渠道、base_url 配错到了别的供应商。"
    if kind == "http_403" or http_status == 403:
        return "余额不足、预扣费失败、账户风控、模型权限未开通、provider 把该 key 暂时封禁。"
    if http_status == 429:
        return "同一 key 请求过密、短时间并发太高、provider 对余额查询或模型接口单独限流。"
    if kind == "ssl":
        return "本地网络不稳、代理转发异常、上游网关 TLS 关闭连接、企业/校园网络中间人代理。"
    if kind == "timeout":
        return "模型响应太慢、请求体太大、provider 高峰拥塞、网络丢包。"
    if kind in {"connection_reset", "connection_aborted", "remote_disconnected", "connection_error"}:
        return "provider 网关重置连接、代理链路中断、远端服务重启、长连接被中间层回收。"
    if kind == "dns":
        return "本机 DNS 配置异常、代理没接管 DNS、当前网络环境无法访问该域名。"
    if isinstance(http_status, int) and http_status >= 500:
        return "provider 后端服务异常、模型服务重启、网关/负载均衡临时故障。"
    return "provider 临时抖动、代理链路异常、网络环境不稳定。"


def _network_suggested_action(err: BaseException) -> str:
    kind = str(getattr(err, "kind", "") or "").strip().lower()
    http_status = getattr(err, "http_status", None)
    if http_status == 401:
        return "检查 api_keys.local.json 中这把 key 是否正确、是否属于当前渠道；若有多把 key，直接换下一把。"
    if kind == "http_403" or http_status == 403:
        return "优先切换到下一把 key；若所有 key 都 403，再检查余额、账户权限和 provider 返回体中的 quota/permission 提示。"
    if http_status == 429:
        return "降低请求频率或稍后重试；如果是余额接口 429，不代表模型 key 本身无效。"
    if kind == "ssl":
        return "稍后重试；若频繁出现，检查代理/VPN、系统时间、证书链和网络环境。"
    if kind == "timeout":
        return "稍后重试；必要时减小 prompt/图像体积，或提高 timeout。"
    if kind in {"connection_reset", "connection_aborted", "remote_disconnected", "connection_error"}:
        return "这通常不是动作逻辑错误；优先视为网络/网关抖动，稍后重试并观察是否持续发生。"
    if kind == "dns":
        return "先确认当前机器能否访问 provider 域名，再检查 DNS、代理和防火墙。"
    if isinstance(http_status, int) and http_status >= 500:
        return "等待 provider 恢复后再试；这类错误通常与本地动作或 prompt 无关。"
    return "先看 http_status/kind/http_body_preview，再判断是 key、限流还是网络链路问题。"


def _network_error_detail(
    err: BaseException,
    *,
    run_name: str = "",
    dish_id: int | None = None,
    episode_step: int | None = None,
    last_completed_step_id: int | None = None,
    last_physical_step_id: int | None = None,
    resume_step_id: int | None = None,
) -> str:
    def _clean(value: Any) -> str:
        return str(value or "").replace("\r", " ").replace("\n", " ").strip()

    parts: list[str] = []
    if run_name:
        parts.append(f"run={run_name}")
    if isinstance(dish_id, int):
        parts.append(f"dish={dish_id}")
    if isinstance(episode_step, int):
        parts.append(f"episode_step={episode_step}")
    if isinstance(last_completed_step_id, int):
        parts.append(f"last_completed_step_id={last_completed_step_id}")
    if isinstance(last_physical_step_id, int):
        parts.append(f"last_physical_step_id={last_physical_step_id}")
    if isinstance(resume_step_id, int):
        parts.append(f"resume_step_id={resume_step_id}")
    source = _clean(getattr(err, "source", ""))
    if source:
        parts.append(f"source={source}")
    kind = _clean(getattr(err, "kind", ""))
    if kind:
        parts.append(f"kind={kind}")
    http_status = getattr(err, "http_status", None)
    if isinstance(http_status, int):
        parts.append(f"http_status={http_status}")
    retry_after_s = getattr(err, "retry_after_s", None)
    if isinstance(retry_after_s, int):
        parts.append(f"retry_after_s={retry_after_s}")
    http_reason = _clean(getattr(err, "http_reason", ""))
    if http_reason:
        parts.append(f"http_reason={http_reason}")
    http_body_preview = _clean(getattr(err, "http_body_preview", ""))
    if http_body_preview:
        parts.append(f"http_body_preview={http_body_preview}")
    api_key_masked = _clean(getattr(err, "api_key_masked", ""))
    if api_key_masked:
        parts.append(f"api_key_masked={api_key_masked}")
    api_key_slot = _clean(getattr(err, "api_key_slot", ""))
    if api_key_slot:
        parts.append(f"api_key_slot={api_key_slot}")
    probable_causes = _clean(_network_probable_causes(err))
    if probable_causes:
        parts.append(f"probable_causes={probable_causes}")
    diagnosis = _clean(_network_diagnosis(err))
    if diagnosis:
        parts.append(f"diagnosis={diagnosis}")
    common_triggers = _clean(_network_common_triggers(err))
    if common_triggers:
        parts.append(f"common_triggers={common_triggers}")
    suggested_action = _clean(_network_suggested_action(err))
    if suggested_action:
        parts.append(f"suggested_action={suggested_action}")
    message = _clean(err)
    if message:
        parts.append(f"message={message}")
    return " | ".join(parts)


def _configure_logging(*, log_mode: str, use_color: bool, default_log_level: str = "INFO") -> None:
    utf8_console = False
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        enc = (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "")
        utf8_console = enc == "utf8"
    except Exception:
        utf8_console = False

    console_encoding = getattr(sys.stdout, "encoding", None) or "utf-8"

    def _to_console_safe(text: str) -> str:
        s = str(text or "")
        try:
            return s.encode(console_encoding, errors="backslashreplace").decode(console_encoding, errors="ignore")
        except Exception:
            # Fallback should never fail and keeps diagnostics readable.
            return s.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="ignore")

    def _patch(record: dict) -> None:
        tag = (os.environ.get("EPM_LOG_TAG") or "").strip()
        record.setdefault("extra", {})
        record["extra"]["tag"] = tag or "EPM"
        msg = str(record.get("message") or "")
        if msg.startswith("[EPM] "):
            msg = msg[6:]
        elif msg == "[EPM]":
            msg = ""
        # Keep full Unicode when console is UTF-8; otherwise downgrade safely to avoid handler crash.
        record["message"] = msg if utf8_console else _to_console_safe(msg)

    _mode = str(log_mode or "").strip().lower()
    del _mode
    level = str(os.environ.get("EPM_LOG_LEVEL") or default_log_level or "INFO").strip().upper() or "INFO"

    logging.remove()
    logging.configure(patcher=_patch)
    logging.add(
        sys.stdout,
        level=level,
        colorize=bool(use_color) and (not bool(os.environ.get("NO_COLOR"))),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level}</level> | "
            "<magenta>[{extra[tag]}]</magenta> "
            "<level>{message}</level>"
        ),
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    last_err: Exception | None = None
    for sleep_s in (0.0, 0.02, 0.05, 0.1, 0.2):
        if sleep_s:
            time.sleep(sleep_s)
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_err = e
        except OSError as e:
            last_err = e
    try:
        path.write_text(content, encoding="utf-8")
    except Exception:
        _ = last_err
        return


def _append_startup_error_event(
    *,
    memory_dir: Path,
    phase: str,
    error: str,
    result: str = "failure",
    extra: dict[str, Any] | None = None,
) -> None:
    raw_error = str(error or "").strip()
    if not raw_error:
        return
    path = Path(memory_dir) / "error_events.log"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        one_line_error = re.sub(r"\s+", " ", raw_error).strip()
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "episode_step": 0,
            "logical_step": 0,
            "phase": str(phase or "").strip() or "startup",
            "result": str(result or "").strip() or "failure",
            "step_id": "-",
            "type": "startup",
            "name": str((extra or {}).get("name") or "").strip() or "restart_env",
            "args": dict(extra or {}),
            "error": one_line_error,
        }
        line = (
            f"{payload['ts']} | step=0 | logical=0 | phase={payload['phase']} | result={payload['result']} "
            f"| step_id=- | type=startup | name={payload['name']} | error={payload['error']}"
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        return


def _mask_secret(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:2]}***{text[-4:]}"


@dataclass
class _ApiKeyRecord:
    key: str
    sources: list[str] = field(default_factory=list)
    selected_count: int = 0
    http_403_count: int = 0
    last_error: str = ""
    last_selected_at: str = ""
    last_http_403_at: str = ""
    disabled_until_ts: float = 0.0


@dataclass
class _ApiKeyPool:
    api_key_env: str
    api_key_pool_env: str
    base_url: str
    model: str
    provider: str
    channels: set[str] = field(default_factory=set)
    records: list[_ApiKeyRecord] = field(default_factory=list)
    current_index: int = -1

    def display_name(self) -> str:
        base = self.base_url or self.provider or self.api_key_env or "unknown_provider"
        model = self.model or ""
        return f"{base}|{model}" if model else base


class _ApiKeyPoolManager:
    def __init__(self, *, memory_dir: Path, quarantine_s: int) -> None:
        self.memory_dir = memory_dir
        self.quarantine_s = max(0, int(quarantine_s))
        self._pools: dict[str, _ApiKeyPool] = {}

    def _pool_key(self, *, api_key_env: str, api_key_pool_env: str, base_url: str, model: str, provider: str) -> str:
        return "||".join(
            [
                str(api_key_env or "").strip(),
                str(api_key_pool_env or "").strip(),
                str(base_url or "").strip(),
                str(model or "").strip(),
                str(provider or "").strip(),
            ]
        )

    def _collect_records(self, *, cfg: Any, api_key_env: str, api_key_pool_env: str) -> list[_ApiKeyRecord]:
        ordered: list[_ApiKeyRecord] = []
        by_key: dict[str, _ApiKeyRecord] = {}

        def _add(raw_key: str, source: str) -> None:
            key = str(raw_key or "").strip()
            label = str(source or "").strip()
            if not key:
                return
            record = by_key.get(key)
            if record is None:
                record = _ApiKeyRecord(key=key)
                by_key[key] = record
                ordered.append(record)
            if label and label not in record.sources:
                record.sources.append(label)

        keys, sources = collect_api_key_candidates(
            explicit_key=str(getattr(cfg, "api_key", "") or ""),
            api_key_env=api_key_env,
            api_key_pool_env=api_key_pool_env,
            base_url=str(getattr(cfg, "base_url", "") or ""),
        )
        for key, source in zip(keys, sources):
            _add(key, source)
        return ordered

    @staticmethod
    def _record_sources_text(record: _ApiKeyRecord) -> str:
        labels = [str(x or "").strip() for x in list(getattr(record, "sources", [])) if str(x or "").strip()]
        return "|".join(labels) if labels else "unknown"

    @staticmethod
    def _find_key_index(records: list[_ApiKeyRecord], key: str) -> int:
        active = str(key or "").strip()
        if not active:
            return -1
        for idx, record in enumerate(records):
            if record.key == active:
                return idx
        return -1

    def register_cfg(self, *, cfg: Any, channel: str) -> None:
        api_key_env = str(getattr(cfg, "api_key_env", "") or "").strip()
        api_key_pool_env = str(getattr(cfg, "api_key_pool_env", "") or "").strip()
        if not api_key_env:
            return
        records = self._collect_records(cfg=cfg, api_key_env=api_key_env, api_key_pool_env=api_key_pool_env)
        if not records:
            return
        pool_id = self._pool_key(
            api_key_env=api_key_env,
            api_key_pool_env=api_key_pool_env,
            base_url=str(getattr(cfg, "base_url", "") or ""),
            model=str(getattr(cfg, "model", "") or ""),
            provider=str(getattr(cfg, "provider", "") or ""),
        )
        pool = self._pools.get(pool_id)
        if pool is None:
            pool = _ApiKeyPool(
                api_key_env=api_key_env,
                api_key_pool_env=api_key_pool_env,
                base_url=str(getattr(cfg, "base_url", "") or ""),
                model=str(getattr(cfg, "model", "") or ""),
                provider=str(getattr(cfg, "provider", "") or ""),
                records=records,
            )
            self._pools[pool_id] = pool
        pool.channels.add(str(channel or "").strip() or "unknown")

    def has_pools(self) -> bool:
        return bool(self._pools)

    def initialize(self, *, log: Any) -> None:
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        for pool in self._pools.values():
            parsed_pool_keys = parse_api_key_pool(os.environ.get(pool.api_key_pool_env, ""))
            env_key_present = bool(str(os.environ.get(pool.api_key_env, "") or "").strip())
            active = str(os.environ.get(pool.api_key_env, "") or "").strip()
            chosen_index = self._find_key_index(pool.records, active)
            explicit_key = ""
            if pool.records:
                for record in pool.records:
                    if "config.api_key" in record.sources:
                        explicit_key = record.key
                        break
            if chosen_index < 0 and explicit_key:
                chosen_index = self._find_key_index(pool.records, explicit_key)
            if chosen_index < 0:
                chosen_index = self._pick_available_index(pool, now_ts=time.time(), exclude_index=None)
            if chosen_index < 0:
                continue
            self._activate(pool, chosen_index, now_iso=now_iso, reason="init")
            active_record = pool.records[chosen_index]
            log.info(
                f"[EPM] api_key_pool_init env={pool.api_key_env} pool_env={pool.api_key_pool_env} "
                f"provider={pool.display_name()} active={_mask_secret(active_record.key)} "
                f"active_index={chosen_index + 1}/{len(pool.records)} sources={self._record_sources_text(active_record)} "
                f"pool_env_count={len(parsed_pool_keys)} env_key_present={env_key_present}"
            )
            if pool.api_key_pool_env and len(parsed_pool_keys) == 0:
                log.warning(
                    f"[EPM] api_key_pool_env_empty env={pool.api_key_env} pool_env={pool.api_key_pool_env} "
                    "No keys were parsed from the pool env in this process. "
                    "Check whether the env var is visible in the current shell and whether keys are separated by ';' or ','."
                )
        self._persist()

    def maybe_rotate_after_http_403(
        self,
        *,
        err: NetworkPauseRequired,
        log: Any,
    ) -> tuple[bool, str]:
        pool = self._find_pool_for_error(err)
        if pool is None:
            return False, "no_matching_api_key_pool"
        now_ts = time.time()
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        current_index = self._current_index(pool)
        if current_index < 0:
            current_index = self._pick_available_index(pool, now_ts=now_ts, exclude_index=None)
            if current_index < 0:
                detail = self._build_pool_exhausted_detail(pool)
                self._persist()
                return False, detail
            self._activate(pool, current_index, now_iso=now_iso, reason="late_init")
        current = pool.records[current_index]
        current.http_403_count += 1
        current.last_http_403_at = now_iso
        current.last_error = self._format_error_detail(err)
        if self.quarantine_s > 0:
            current.disabled_until_ts = now_ts + float(self.quarantine_s)
        next_index = self._pick_available_index(pool, now_ts=now_ts, exclude_index=current_index)
        if next_index < 0:
            detail = self._build_pool_exhausted_detail(pool)
            self._persist()
            return False, detail
        self._activate(pool, next_index, now_iso=now_iso, reason="http_403_rotate")
        next_record = pool.records[next_index]
        detail = (
            f"provider={pool.display_name()} env={pool.api_key_env} "
            f"invalidated_index={current_index + 1}/{len(pool.records)} invalidated={_mask_secret(current.key)} "
            f"replacement_index={next_index + 1}/{len(pool.records)} replacement={_mask_secret(next_record.key)} "
            f"replacement_sources={self._record_sources_text(next_record)} "
            f"quarantine_s={self.quarantine_s} http_403_count={current.http_403_count}"
        )
        log.warning(f"[EPM] api_key_rotated {detail}")
        self._persist()
        return True, detail

    def maybe_rotate_after_http_401(
        self,
        *,
        err: NetworkAbortRequired,
        log: Any,
    ) -> tuple[bool, str]:
        pool = self._find_pool_for_error(err)
        if pool is None:
            return False, "no_matching_api_key_pool"
        now_ts = time.time()
        now_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        current_index = self._current_index(pool)
        if current_index < 0:
            current_index = self._pick_available_index(pool, now_ts=now_ts, exclude_index=None)
            if current_index < 0:
                detail = self._build_pool_exhausted_detail(pool)
                self._persist()
                return False, detail
            self._activate(pool, current_index, now_iso=now_iso, reason="late_init")
        current = pool.records[current_index]
        current.last_error = self._format_error_detail(err)
        current.disabled_until_ts = now_ts + float(10 * 365 * 24 * 3600)
        next_index = self._pick_available_index(pool, now_ts=now_ts, exclude_index=current_index)
        if next_index < 0:
            detail = self._build_pool_exhausted_detail(pool)
            self._persist()
            return False, detail
        self._activate(pool, next_index, now_iso=now_iso, reason="http_401_rotate")
        next_record = pool.records[next_index]
        detail = (
            f"provider={pool.display_name()} env={pool.api_key_env} "
            f"invalidated_index={current_index + 1}/{len(pool.records)} invalidated={_mask_secret(current.key)} "
            f"replacement_index={next_index + 1}/{len(pool.records)} replacement={_mask_secret(next_record.key)} "
            f"replacement_sources={self._record_sources_text(next_record)} "
            f"reason=invalid_api_key_or_bearer_token"
        )
        log.warning(f"[EPM] api_key_rotated_invalid {detail}")
        self._persist()
        return True, detail

    def _format_error_detail(self, err: BaseException) -> str:
        parts = [f"source={getattr(err, 'source', '')}", f"kind={getattr(err, 'kind', '')}"]
        reason = str(getattr(err, "http_reason", "") or "").strip()
        if reason:
            parts.append(f"reason={reason}")
        preview = str(getattr(err, "http_body_preview", "") or "").strip()
        if preview:
            parts.append(f"body={preview}")
        return "; ".join(parts)

    def _current_index(self, pool: _ApiKeyPool) -> int:
        if 0 <= int(pool.current_index) < len(pool.records):
            return int(pool.current_index)
        active = str(os.environ.get(pool.api_key_env, "") or "").strip()
        for idx, record in enumerate(pool.records):
            if record.key == active:
                pool.current_index = idx
                return idx
        return -1

    def _pick_available_index(self, pool: _ApiKeyPool, *, now_ts: float, exclude_index: int | None) -> int:
        for idx, record in enumerate(pool.records):
            if exclude_index is not None and idx == exclude_index:
                continue
            if float(record.disabled_until_ts or 0.0) > now_ts:
                continue
            return idx
        return -1

    def _activate(self, pool: _ApiKeyPool, index: int, *, now_iso: str, reason: str) -> None:
        pool.current_index = int(index)
        record = pool.records[index]
        record.selected_count += 1
        record.last_selected_at = now_iso
        os.environ[pool.api_key_env] = record.key
        record.last_error = record.last_error or f"activated:{reason}"

    def _find_pool_for_error(self, err: BaseException) -> _ApiKeyPool | None:
        api_key_env = str(getattr(err, "api_key_env", "") or "").strip()
        api_key_pool_env = str(getattr(err, "api_key_pool_env", "") or "").strip()
        base_url = str(getattr(err, "request_base_url", "") or "").strip()
        model = str(getattr(err, "request_model", "") or "").strip()
        provider = str(getattr(err, "request_provider", "") or "").strip()
        pool_id = self._pool_key(
            api_key_env=api_key_env,
            api_key_pool_env=api_key_pool_env,
            base_url=base_url,
            model=model,
            provider=provider,
        )
        if pool_id in self._pools:
            return self._pools[pool_id]
        if api_key_env:
            matches = [pool for pool in self._pools.values() if pool.api_key_env == api_key_env]
            if len(matches) == 1:
                return matches[0]
        return None

    def _build_pool_exhausted_detail(self, pool: _ApiKeyPool) -> str:
        parts: list[str] = []
        now_ts = time.time()
        for record in pool.records:
            disabled_for_s = max(0, int(record.disabled_until_ts - now_ts))
            parts.append(
                f"{_mask_secret(record.key)}"
                f"(index={pool.records.index(record) + 1}/{len(pool.records)},sources={self._record_sources_text(record)},selected={record.selected_count},http_403={record.http_403_count},cooldown_s={disabled_for_s},last_error={record.last_error})"
            )
        return (
            f"api_key_pool_exhausted provider={pool.display_name()} env={pool.api_key_env} "
            f"pool_env={pool.api_key_pool_env} keys=[{'; '.join(parts)}]"
        )

    def _persist(self) -> None:
        payload = {
            "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "pools": [],
        }
        now_ts = time.time()
        for pool in self._pools.values():
            payload["pools"].append(
                {
                    "api_key_env": pool.api_key_env,
                    "api_key_pool_env": pool.api_key_pool_env,
                    "base_url": pool.base_url,
                    "model": pool.model,
                    "provider": pool.provider,
                    "channels": sorted(pool.channels),
                    "active_key": _mask_secret(pool.records[pool.current_index].key) if 0 <= pool.current_index < len(pool.records) else "",
                    "active_index": (int(pool.current_index) + 1) if 0 <= pool.current_index < len(pool.records) else 0,
                    "pool_size": len(pool.records),
                    "keys": [
                        {
                            "key": _mask_secret(record.key),
                            "sources": list(record.sources),
                            "selected_count": int(record.selected_count),
                            "http_403_count": int(record.http_403_count),
                            "last_selected_at": record.last_selected_at,
                            "last_http_403_at": record.last_http_403_at,
                            "last_error": record.last_error,
                            "cooldown_s": max(0, int(record.disabled_until_ts - now_ts)),
                            "disabled": bool(record.disabled_until_ts > now_ts),
                        }
                        for record in pool.records
                    ],
                }
            )
        _atomic_write_text(self.memory_dir / "api_key_pool_status.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _coerce_json_value(raw: str) -> Any:
    text = raw.strip()
    if text == "":
        return ""
    try:
        return json.loads(text)
    except Exception:
        return text


def _set_nested_value(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in dotted_key.split(".") if p != ""]
    if not parts:
        raise ValueError("empty override key")
    cur: dict[str, Any] = cfg
    for i, part in enumerate(parts):
        is_last = i == (len(parts) - 1)
        if is_last:
            cur[part] = value
            return
        nxt = cur.get(part)
        if nxt is None:
            nxt = {}
            cur[part] = nxt
        if not isinstance(nxt, dict):
            raise ValueError(f"override path conflict at '{part}' (not an object)")
        cur = nxt


def _apply_config_overrides(cfg_path: Path, overrides: list[str]) -> None:
    if not overrides:
        return
    raw = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict):
        raise ValueError("config JSON must be an object at top-level")
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"invalid --set format (expect key=value): {item!r}")
        key, value_raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --set key: {item!r}")
        value = _coerce_json_value(value_raw)
        _set_nested_value(raw, key, value)
    _atomic_write_text(cfg_path, json.dumps(raw, ensure_ascii=False, indent=2))


def _collect_dot_overrides(unknown: list[str]) -> list[str]:
    overrides: list[str] = []
    i = 0
    while i < len(unknown):
        token = unknown[i]
        if token in ("--restartenv", "--restart-env", "--restart_env"):
            i += 1
            continue
        if not token.startswith("--"):
            raise ValueError(f"unexpected argument: {token!r}")
        key = token[2:]
        if not key:
            raise ValueError("invalid override flag")
        if "=" in key:
            key, value = key.split("=", 1)
            if "." not in key:
                raise ValueError(f"override key must be dot path: {key!r}")
            overrides.append(f"{key}={value}")
            i += 1
            continue
        if "." not in key:
            raise ValueError(f"override key must be dot path: {key!r}")
        if i + 1 >= len(unknown):
            raise ValueError(f"missing value for override: {key!r}")
        value = unknown[i + 1]
        overrides.append(f"{key}={value}")
        i += 2
    return overrides


def _normalize_run_name_part(value: str, *, fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    for ch in ('/', '\\', ':', '*', '?', '"', "<", ">", "|"):
        text = text.replace(ch, "_")
    return text.replace(" ", "_")


def _default_run_name(*, pipeline: str, model: str, prompt_ablation_profile: str, dish_id: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pipeline_part = _normalize_run_name_part(str(pipeline), fallback="pipeline")
    model_part = _normalize_run_name_part(str(model), fallback="model")
    profile_part = _normalize_run_name_part(str(prompt_ablation_profile), fallback="full")
    dish_part = _normalize_run_name_part(str(dish_id), fallback="dish")
    return f"{ts}-{pipeline_part}-{model_part}-{profile_part}-{dish_part}"


def _planner_model_tag(settings: Any) -> str:
    try:
        mode = str(getattr(settings.brain, "planner_mode", "") or "").strip().lower()
    except Exception:
        mode = ""
    if mode == "vlm":
        model = str(getattr(getattr(settings, "vlm", None), "model", "") or "").strip()
    else:
        model = str(getattr(getattr(settings, "llm", None), "model", "") or "").strip()
    return _normalize_run_name_part(model, fallback="model") if model else "model"


def _read_order_state(memory_dir: Path) -> dict:
    path = memory_dir / "order_state.json"
    if not path.exists():
        return {
            "ordered_dish_ids": [],
            "ordered_dish_names": [],
            "last_ordered_dish_id": None,
            "last_ordered_dish_name": "",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "ordered_dish_ids": [],
            "ordered_dish_names": [],
            "last_ordered_dish_id": None,
            "last_ordered_dish_name": "",
        }
    if not isinstance(data, dict):
        return {
            "ordered_dish_ids": [],
            "ordered_dish_names": [],
            "last_ordered_dish_id": None,
            "last_ordered_dish_name": "",
        }
    return data


def _write_order_state(memory_dir: Path, state: dict) -> None:
    _atomic_write_text(memory_dir / "order_state.json", json.dumps(state, ensure_ascii=False, indent=2) + "\n")


def _ensure_dish_ordered(*, memory_dir: Path, dish_id: int, dish_name: str, log: Any, max_attempts: int = 3) -> bool:
    state = _read_order_state(memory_dir)
    ordered_ids = state.get("ordered_dish_ids")
    if not isinstance(ordered_ids, list):
        ordered_ids = []
    if int(dish_id) in {int(x) for x in ordered_ids if isinstance(x, (int, str))}:
        return True

    from epm.cerebellum.local_actions import gui_order_dish_via_computer

    def _extract_order_debug(res_obj: Any) -> str:
        if not isinstance(res_obj, dict):
            return ""
        # Prefer structured field from order flow.
        order_res = res_obj.get("order_result")
        if isinstance(order_res, dict):
            decorations = order_res.get("decorations")
            if isinstance(decorations, dict):
                dbg = decorations.get("debug_screenshot")
                if isinstance(dbg, str) and dbg.strip():
                    return dbg.strip()
            raw_err = order_res.get("error")
            if isinstance(raw_err, str):
                marker = "debug="
                if marker in raw_err:
                    return raw_err.split(marker, 1)[1].strip().split()[0].strip("'\"")
        # Fallback: parse top-level error.
        top_err = res_obj.get("error")
        if isinstance(top_err, str):
            marker = "debug="
            if marker in top_err:
                return top_err.split(marker, 1)[1].strip().split()[0].strip("'\"")
        return ""

    def _extract_order_nested_error(res_obj: Any) -> str:
        if not isinstance(res_obj, dict):
            return ""
        order_res = res_obj.get("order_result")
        if isinstance(order_res, dict):
            raw_err = str(order_res.get("error") or "").strip()
            if raw_err:
                return raw_err
        return ""

    def _format_order_nav_debug(res_obj: Any) -> list[str]:
        if not isinstance(res_obj, dict):
            return []
        nav_debug = res_obj.get("nav_debug")
        if not isinstance(nav_debug, dict):
            return []

        lines: list[str] = []
        summary_parts: list[str] = []

        nav_error = res_obj.get("nav_error")
        if isinstance(nav_error, str) and nav_error.strip():
            summary_parts.append(f"nav_error={nav_error.strip()}")

        preflight_parts: list[str] = []
        for key in ("f12_ok", "f11_ok"):
            if key in res_obj:
                preflight_parts.append(f"{key}={res_obj.get(key)}")
        if preflight_parts:
            summary_parts.append("preflight[" + " ".join(preflight_parts) + "]")
        if "nav_bypassed_by_open_ui" in res_obj:
            summary_parts.append(f"nav_bypassed_by_open_ui={res_obj.get('nav_bypassed_by_open_ui')}")

        for key in (
            "resolved_target",
            "resolved_target_instance_id",
            "navigation_failure_summary",
            "blocked_by_mode",
        ):
            value = nav_debug.get(key)
            if value not in (None, "", [], {}):
                summary_parts.append(f"{key}={value}")

        if summary_parts:
            lines.append("[EPM] order_nav_debug " + " | ".join(summary_parts))

        stdout_tail = nav_debug.get("navigator_stdout_tail")
        if isinstance(stdout_tail, str) and stdout_tail.strip():
            compact_tail = " | ".join(part.strip() for part in stdout_tail.splitlines() if part.strip())
            if len(compact_tail) > 1200:
                compact_tail = compact_tail[-1200:]
            lines.append(f"[EPM] order_nav_stdout_tail={compact_tail}")

        computer_ui = res_obj.get("computer_ui")
        if isinstance(computer_ui, dict) and computer_ui:
            lines.append(f"[EPM] order_computer_ui={computer_ui}")

        for key in ("navigation_failure_signals", "docking_debug", "precise_adjust_debug", "posture_decision", "hint"):
            value = nav_debug.get(key)
            if value not in (None, "", [], {}):
                lines.append(f"[EPM] order_nav_{key}={value}")

        return lines

    for attempt in range(1, int(max_attempts) + 1):
        res = gui_order_dish_via_computer(dish_name=str(dish_name))
        if isinstance(res, dict) and bool(res.get("success")):
            ordered_ids = [int(x) for x in ordered_ids if isinstance(x, (int, str))]
            if int(dish_id) not in ordered_ids:
                ordered_ids.append(int(dish_id))
            ordered_names = state.get("ordered_dish_names")
            if not isinstance(ordered_names, list):
                ordered_names = []
            if dish_name and dish_name not in ordered_names:
                ordered_names.append(str(dish_name))
            state.update(
                {
                    "ordered_dish_ids": ordered_ids,
                    "ordered_dish_names": ordered_names,
                    "last_ordered_dish_id": int(dish_id),
                    "last_ordered_dish_name": str(dish_name),
                }
            )
            _write_order_state(memory_dir, state)
            return True
        log.warning(
            f"[EPM] order_failed attempt={attempt} dish_id={dish_id} dish_name={dish_name!r} "
            f"error={(res.get('error') if isinstance(res, dict) else getattr(res, 'error', ''))}"
        )
        nested_err = _extract_order_nested_error(res)
        if nested_err:
            log.warning(f"[EPM] order_gui_error={nested_err}")
        dbg = _extract_order_debug(res)
        if dbg:
            log.warning(f"[EPM] order_debug_screenshot={dbg}")
        for line in _format_order_nav_debug(res):
            log.warning(line)
        time.sleep(0.5)

    final_error = f"order_failed_exhausted_attempts dish_id={dish_id} dish_name={dish_name!r}"
    try:
        if isinstance(res, dict):
            top_err = str(res.get("error") or "").strip()
            nested_err = _extract_order_nested_error(res)
            detail_bits = []
            if top_err:
                detail_bits.append(f"top_error={top_err}")
            if nested_err:
                detail_bits.append(f"gui_error={nested_err}")
            dbg = _extract_order_debug(res)
            if dbg:
                detail_bits.append(f"debug_screenshot={dbg}")
            if detail_bits:
                final_error += " | " + " | ".join(detail_bits)
    except Exception:
        pass
    _append_startup_error_event(
        memory_dir=memory_dir,
        phase="order_gate",
        error=final_error,
        result="failure",
        extra={
            "name": "order_dish_via_computer",
            "dish_id": int(dish_id),
            "dish_name": str(dish_name),
        },
    )
    return False


def _find_latest_run_dir(runs_root: Path) -> Path | None:
    if not runs_root.exists():
        return None
    candidates = [
        p
        for p in runs_root.iterdir()
        if p.is_dir() and re.match(r"^\d{8}_\d{6}_\d{6}[-_].+", p.name)
    ]
    if not candidates:
        candidates = [p for p in runs_root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_resume_state(memory_dir: Path) -> dict:
    path = memory_dir / "resume_state.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _read_last_step_id_from_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if end <= 0:
                return None
            buf = b""
            pos = end
            chunk = 4096
            while pos > 0 and b"\n" not in buf:
                read_size = min(chunk, pos)
                pos -= read_size
                f.seek(pos)
                buf = f.read(read_size) + buf
            for line in reversed(buf.splitlines()):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s.decode("utf-8", errors="replace"))
                except Exception:
                    continue
                step = obj.get("step_id")
                if isinstance(step, int):
                    return step
                if isinstance(step, str) and step.isdigit():
                    return int(step)
                return None
    except Exception:
        return None
    return None


def _resolve_resume_step_id(memory_dir: Path) -> tuple[int, str]:
    state = _read_resume_state(memory_dir)
    status = str(state.get("status") or "").strip().lower()
    reason = str(state.get("reason") or "").strip().lower()
    if status in {"waiting_network", "waiting_http_403", "fatal"} and (
        "network_" in reason or "http_403" in reason or "http_429" in reason
    ):
        for key, source in (
            ("last_completed_step_id", "resume_state.last_completed_step_id"),
            ("last_physical_step_id", "resume_state.last_physical_step_id"),
            ("last_network_resume_step_id", "resume_state.last_network_resume_step_id"),
            ("last_http_403_resume_step_id", "resume_state.http_403"),
        ):
            step = state.get(key)
            if isinstance(step, int) and step > 0:
                return step, source
            if isinstance(step, str) and step.isdigit() and int(step) > 0:
                return int(step), source
    step = state.get("last_step_id")
    if isinstance(step, int):
        return step, "resume_state"
    if isinstance(step, str) and step.isdigit():
        return int(step), "resume_state"
    main = _read_last_step_id_from_jsonl(memory_dir / "long_horizon_history.txt")
    if isinstance(main, int):
        return main, "long_horizon_history"
    best = None
    for path in memory_dir.glob("long_horizon_history.txt.fallback.*"):
        step_id = _read_last_step_id_from_jsonl(path)
        if isinstance(step_id, int):
            if best is None or step_id > best:
                best = step_id
    if isinstance(best, int):
        return best, "long_horizon_fallback"
    return 0, "none"


def _is_network_interrupted_resume_state(state: dict[str, Any]) -> bool:
    status = str(state.get("status") or "").strip().lower()
    reason = str(state.get("reason") or "").strip().lower()
    if status in {"waiting_network", "waiting_http_403"}:
        return True
    if status == "fatal" and (
        "network_" in reason
        or "http_403" in reason
        or "http_429" in reason
        or "timeout" in reason
        or "connection_" in reason
        or "remote_disconnected" in reason
    ):
        return True
    return False


def _backup_before_rewrite(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + f".resume_cleanup_backup.{stamp}")
    try:
        backup.write_text(path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    except Exception:
        return


def _resume_rewind_backup_dir(*, memory_dir: Path, resume_step_id: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return memory_dir / "resume_rewind_backups" / f"{stamp}_to_step_{int(resume_step_id):06d}"


def _backup_file_to_dir(path: Path, *, backup_root: Path, rel: str) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        dst = backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        return True
    except Exception:
        return False


def _backup_dir_to_dir(path: Path, *, backup_root: Path, rel: str) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    try:
        dst = backup_root / rel
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(path, dst)
        return True
    except Exception:
        return False


def _write_removed_lines_backup(*, backup_root: Path | None, rel: str, lines: list[str]) -> None:
    if backup_root is None or not lines:
        return
    try:
        dst = backup_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(dst, ("\n".join(lines) + "\n") if lines else "")
    except Exception:
        return


def _reset_json_file(path: Path, *, payload: dict[str, Any] | None = None) -> bool:
    try:
        _atomic_write_text(path, json.dumps(payload or {}, ensure_ascii=False, indent=2) + "\n")
        return True
    except Exception:
        return False


def _reset_text_file(path: Path, *, text: str = "") -> bool:
    try:
        _atomic_write_text(path, (str(text) if text else ""))
        return True
    except Exception:
        return False


def _backup_and_unlink(path: Path, *, backup_root: Path | None = None, backup_rel: str = "") -> bool:
    if not path.exists():
        return False
    try:
        if backup_root is not None and backup_rel:
            if path.is_dir():
                _backup_dir_to_dir(path, backup_root=backup_root, rel=f"full_before/{backup_rel}")
            else:
                _backup_file_to_dir(path, backup_root=backup_root, rel=f"full_before/{backup_rel}")
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink()
        return True
    except Exception:
        return False


def _trim_jsonl_by_step_field(
    path: Path,
    *,
    field_names: tuple[str, ...],
    max_step_id: int,
    backup_root: Path | None = None,
    backup_rel: str = "",
) -> bool:
    if not path.exists():
        return False
    kept: list[str] = []
    removed: list[str] = []
    changed = False
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            kept.append(line)
            continue
        if not isinstance(obj, dict):
            kept.append(line)
            continue
        step_val = None
        for field in field_names:
            raw = obj.get(field)
            if isinstance(raw, int):
                step_val = raw
                break
            if isinstance(raw, str) and raw.isdigit():
                step_val = int(raw)
                break
        if isinstance(step_val, int) and step_val > int(max_step_id):
            changed = True
            removed.append(line)
            continue
        kept.append(line)
    if changed:
        if backup_root is not None and backup_rel:
            _backup_file_to_dir(path, backup_root=backup_root, rel=f"full_before/{backup_rel}")
            _write_removed_lines_backup(backup_root=backup_root, rel=f"trimmed_away/{backup_rel}", lines=removed)
        _backup_before_rewrite(path)
        _atomic_write_text(path, ("\n".join(kept) + "\n") if kept else "")
    return changed


def _trim_step_trace(path: Path, *, max_step_id: int, backup_root: Path | None = None, backup_rel: str = "") -> bool:
    if not path.exists():
        return False
    kept: list[str] = []
    removed: list[str] = []
    changed = False
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return False
    for line in lines:
        m = re.search(r"\bstep=(\d+)\b", line)
        if m is not None and int(m.group(1)) > int(max_step_id):
            changed = True
            removed.append(line)
            continue
        kept.append(line)
    if changed:
        if backup_root is not None and backup_rel:
            _backup_file_to_dir(path, backup_root=backup_root, rel=f"full_before/{backup_rel}")
            _write_removed_lines_backup(backup_root=backup_root, rel=f"trimmed_away/{backup_rel}", lines=removed)
        _backup_before_rewrite(path)
        _atomic_write_text(path, ("\n".join(kept) + "\n") if kept else "")
    return changed


def _trim_numeric_step_named_files(
    dir_path: Path,
    *,
    max_step_id: int,
    backup_root: Path | None = None,
    backup_rel: str = "",
) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    removed = 0
    for path in dir_path.iterdir():
        if not path.is_file():
            continue
        m = re.search(r"step_(\d+)", path.name)
        if m is None:
            continue
        if int(m.group(1)) <= int(max_step_id):
            continue
        try:
            if backup_root is not None and backup_rel:
                _backup_file_to_dir(path, backup_root=backup_root, rel=f"trimmed_away/{backup_rel}/{path.name}")
            path.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def _trim_step_screenshots(
    dir_path: Path,
    *,
    max_step_id: int,
    backup_root: Path | None = None,
    backup_rel: str = "",
) -> int:
    if not dir_path.exists() or not dir_path.is_dir():
        return 0
    removed = 0
    for path in dir_path.iterdir():
        if not path.is_file():
            continue
        m = re.match(r"step_(\d+)(?:_post)?\.(png|jpg|jpeg)$", path.name, flags=re.IGNORECASE)
        if m is None:
            continue
        if int(m.group(1)) <= int(max_step_id):
            continue
        try:
            if backup_root is not None and backup_rel:
                _backup_file_to_dir(path, backup_root=backup_root, rel=f"trimmed_away/{backup_rel}/{path.name}")
            path.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def _is_non_physical_trace_step(*, step_type: str, step_name: str) -> bool:
    return (str(step_type or "").strip().lower(), str(step_name or "").strip().lower()) in {
        ("skill", "query_scene_objects"),
        ("skill", "auto_perception"),
        ("skill", "list_supported_items"),
    }


def _collect_post_resume_physical_actions(*, memory_dir: Path, resume_step_id: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    trace_path = memory_dir / "step_trace.log"
    if trace_path.exists():
        try:
            for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _TRACE_RESULT_RE.search(line)
                if not m or str(m.group("kind") or "") != "result":
                    continue
                step_no = int(m.group("step"))
                if step_no <= int(resume_step_id):
                    continue
                step_type = str(m.group("type") or "")
                step_name = str(m.group("name") or "")
                if _is_non_physical_trace_step(step_type=step_type, step_name=step_name):
                    continue
                out.append(
                    {
                        "step": step_no,
                        "step_id": str(m.group("step_id") or ""),
                        "type": step_type,
                        "name": step_name,
                        "result": str(m.group("result") or ""),
                        "error": _strip_repr_quotes(str(m.group("error") or "")),
                    }
                )
        except Exception:
            return out
    return out


def _confirm_manual_rewind_with_physical_actions(
    *,
    memory_dir: Path,
    resume_step_id: int,
    log: Any,
    force_yes: bool = False,
) -> bool:
    actions = _collect_post_resume_physical_actions(memory_dir=memory_dir, resume_step_id=resume_step_id)
    if not actions:
        return True
    sample = ", ".join(
        f"step={int(item['step'])}:{item['type']}:{item['name']}"
        for item in actions[:6]
    )
    warn = (
        f"[EPM] rewind_warning target_step={int(resume_step_id)} "
        f"physical_actions_after_target={len(actions)} samples=[{sample}] "
        f"note=rewind_will_trim_logs_and_reset_run_memory_but_cannot_auto_restore_physical_world_state"
    )
    log.warning(warn)
    if force_yes:
        log.warning("[EPM] rewind_warning user_confirmation=auto_yes")
        return True
    if not sys.stdin or not sys.stdin.isatty():
        log.error("[EPM] rewind_confirmation_required but stdin is not interactive; re-run with --yes-rewind to continue.")
        return False
    try:
        print("\n[EPM] Warning: there are physical actions after the requested rewind step.")
        print(f"[EPM] target_step={int(resume_step_id)} physical_actions_after_target={len(actions)}")
        if sample:
            print(f"[EPM] samples: {sample}")
        print("[EPM] Logs/memory will be rewound, but the game world itself cannot be auto-restored.")
        answer = input("[EPM] Continue rewind? [yes/no]: ").strip().lower()
    except EOFError:
        answer = ""
    return answer in {"y", "yes"}


def _cleanup_resume_artifacts(
    *,
    memory_dir: Path,
    screenshot_dir: Path,
    resume_step_id: int,
    log: Any,
    force: bool = False,
    reason_tag: str = "",
) -> bool:
    state = _read_resume_state(memory_dir)
    if not force and not _is_network_interrupted_resume_state(state):
        return False
    try:
        last_step_id = int(state.get("last_step_id") or 0)
    except Exception:
        last_step_id = 0
    if int(resume_step_id) < 0 or last_step_id <= int(resume_step_id):
        return False

    cleanup_reason = str(reason_tag or "").strip() or (
        "resume_rewind_after_network_interrupt" if _is_network_interrupted_resume_state(state) else "resume_rewind_manual"
    )
    backup_root = _resume_rewind_backup_dir(memory_dir=memory_dir, resume_step_id=resume_step_id)
    backup_root.mkdir(parents=True, exist_ok=True)

    changed_items: list[str] = []
    manifest: dict[str, Any] = {
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "resume_step_id": int(resume_step_id),
        "previous_last_step_id": int(last_step_id),
        "reason": cleanup_reason,
        "note": "memory/log state rewound; physical game world is not auto-restored",
        "changed_items": changed_items,
    }

    def _mark(item: str) -> None:
        if item not in changed_items:
            changed_items.append(item)

    def _trim_step_dir(dir_name: str) -> None:
        removed = _trim_numeric_step_named_files(
            memory_dir / dir_name,
            max_step_id=resume_step_id,
            backup_root=backup_root,
            backup_rel=f"memory/{dir_name}",
        )
        if removed > 0:
            _mark(f"{dir_name}(-{removed})")

    def _reset_json_state(name: str, payload: dict[str, Any] | None = None) -> None:
        path = memory_dir / name
        if path.exists():
            _backup_file_to_dir(path, backup_root=backup_root, rel=f"full_before/memory/{name}")
        if _reset_json_file(path, payload=payload):
            _mark(name)

    def _reset_text_state(name: str, text: str = "") -> None:
        path = memory_dir / name
        if path.exists():
            _backup_file_to_dir(path, backup_root=backup_root, rel=f"full_before/memory/{name}")
        if _reset_text_file(path, text=text):
            _mark(name)

    def _remove_state_path(name: str) -> None:
        if _backup_and_unlink(memory_dir / name, backup_root=backup_root, backup_rel=f"memory/{name}"):
            _mark(name)

    if _trim_jsonl_by_step_field(
        memory_dir / "long_horizon_history.txt",
        field_names=("step_id",),
        max_step_id=resume_step_id,
        backup_root=backup_root,
        backup_rel="memory/long_horizon_history.txt",
    ):
        _mark("long_horizon_history.txt")
    for path in sorted(memory_dir.glob("long_horizon_history.txt.fallback.*")):
        if _trim_jsonl_by_step_field(
            path,
            field_names=("step_id",),
            max_step_id=resume_step_id,
            backup_root=backup_root,
            backup_rel=f"memory/{path.name}",
        ):
            _mark(path.name)
    if _trim_step_trace(
        memory_dir / "step_trace.log",
        max_step_id=resume_step_id,
        backup_root=backup_root,
        backup_rel="memory/step_trace.log",
    ):
        _mark("step_trace.log")
    if _trim_jsonl_by_step_field(
        memory_dir / "planner_thoughts.jsonl",
        field_names=("episode_step",),
        max_step_id=resume_step_id,
        backup_root=backup_root,
        backup_rel="memory/planner_thoughts.jsonl",
    ):
        _mark("planner_thoughts.jsonl")

    _trim_step_dir("task_progress_updates")
    _trim_step_dir("precondition_checks")
    _trim_step_dir("prompts")
    _trim_step_dir("reflexion_final")
    _trim_step_dir("reflexion_raw")
    _trim_step_dir("reflexion_thinking")
    _trim_step_dir("vlm_planner_final")
    _trim_step_dir("vlm_planner_raw")
    _trim_step_dir("vlm_planner_thinking")

    removed_screenshots = _trim_step_screenshots(
        screenshot_dir,
        max_step_id=resume_step_id,
        backup_root=backup_root,
        backup_rel="screenshots",
    )
    if removed_screenshots > 0:
        _mark(f"screenshots(-{removed_screenshots})")

    status_path = memory_dir / "current_step_status.json"
    try:
        if status_path.exists():
            payload = json.loads(status_path.read_text(encoding="utf-8", errors="replace"))
            if isinstance(payload, dict):
                ep_step = payload.get("episode_step")
                ep_step_int = int(ep_step) if isinstance(ep_step, (int, str)) and str(ep_step).isdigit() else 0
                if ep_step_int > int(resume_step_id):
                    _backup_file_to_dir(status_path, backup_root=backup_root, rel="full_before/memory/current_step_status.json")
                    payload["episode_step"] = int(resume_step_id)
                    payload["phase"] = "resume_rewind_cleaned"
                    payload["result"] = ""
                    payload["error"] = ""
                    payload["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
                    _backup_before_rewrite(status_path)
                    _atomic_write_text(status_path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                    _mark("current_step_status.json")
    except Exception:
        pass

    _reset_json_state(
        "pe_active_plan.json",
        payload={
            "cursor_index": 0,
            "plan": [],
            "episode_step": int(resume_step_id),
            "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    _remove_state_path("agent_state.json")
    _remove_state_path("task_progress_epm.txt")
    _remove_state_path("reflexion_memory.txt")
    _remove_state_path("reflexion_progress_memory.txt")
    _remove_state_path("last_failure_raw.json")
    _remove_state_path("pe_plan_history.jsonl")
    _reset_json_state("cap_runtime_state.json", payload={})
    _reset_json_state("put_place_occupancy.json", payload={})
    _reset_text_state("latest_precondition_feedback.txt", text="")
    _reset_text_state("latest_task_progress_feedback.txt", text="")
    _reset_text_state("latest_visual_anomaly_feedback.txt", text="")
    _reset_text_state("stm_window.txt", text="")

    _backup_file_to_dir(memory_dir / "resume_state.json", backup_root=backup_root, rel="full_before/memory/resume_state.json")
    state["last_step_id"] = int(resume_step_id)
    for key in ("last_completed_step_id", "last_physical_step_id"):
        value = state.get(key)
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if isinstance(value, int):
            if value > int(resume_step_id):
                state[key] = int(resume_step_id)
                state[key.replace("_id", "_type")] = ""
                state[key.replace("_id", "_name")] = ""
        else:
            state.pop(key, None)
    prev_network_step = state.get("last_network_resume_step_id")
    if isinstance(prev_network_step, str) and prev_network_step.isdigit():
        prev_network_step = int(prev_network_step)
    if isinstance(prev_network_step, int) and prev_network_step > int(resume_step_id):
        state["last_network_resume_step_id"] = int(resume_step_id)
        state.pop("last_network_waited_s", None)
    prev_http403_step = state.get("last_http_403_resume_step_id")
    if isinstance(prev_http403_step, str) and prev_http403_step.isdigit():
        prev_http403_step = int(prev_http403_step)
    if isinstance(prev_http403_step, int) and prev_http403_step > int(resume_step_id):
        state["last_http_403_resume_step_id"] = int(resume_step_id)
        state.pop("last_http_403_waited_s", None)
    state["status"] = "resume_rewind_cleaned"
    state["rewind_backup_dir"] = str(backup_root)
    state["reason"] = f"{cleanup_reason}:trimmed_from={last_step_id}:to={int(resume_step_id)}"
    state["updated_at"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    _atomic_write_text(memory_dir / "resume_state.json", json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    _mark("resume_state.json")

    manifest["final_resume_state"] = {
        "last_step_id": state.get("last_step_id"),
        "last_completed_step_id": state.get("last_completed_step_id"),
        "last_physical_step_id": state.get("last_physical_step_id"),
        "last_network_resume_step_id": state.get("last_network_resume_step_id"),
        "last_http_403_resume_step_id": state.get("last_http_403_resume_step_id"),
        "status": state.get("status"),
        "reason": state.get("reason"),
    }
    _atomic_write_text(backup_root / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(
        backup_root / "README.txt",
        (
            "resume rewind backup\n"
            f"target_step={int(resume_step_id)}\n"
            f"previous_last_step={int(last_step_id)}\n"
            f"reason={cleanup_reason}\n"
            "note=memory/log state was rewound; physical world was not auto-restored\n"
        ),
    )
    log.warning(
        f"[EPM] resume_cleanup=applied trimmed_from={last_step_id} to={int(resume_step_id)} "
        f"reason={cleanup_reason} backup_dir={backup_root} files={changed_items}"
    )
    return True


def _write_resume_state(
    *,
    memory_dir: Path,
    run_name: str,
    dish_id: int,
    last_step_id: int,
    status: str,
    reason: str = "",
    last_time_iso: str = "",
    agent: EpmAgent | None = None,
    last_network_resume_step_id: int | None = None,
    last_network_waited_s: int | None = None,
    last_http_403_resume_step_id: int | None = None,
    last_http_403_waited_s: int | None = None,
) -> None:
    prev = _read_resume_state(memory_dir)
    payload = {
        "run_name": str(run_name),
        "dish_id": int(dish_id),
        "last_step_id": int(last_step_id),
        "status": str(status),
        "reason": str(reason),
        "last_time": str(last_time_iso or ""),
        "updated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    last_completed_step_id = getattr(agent, "last_completed_step_id", None)
    last_completed_step_type = getattr(agent, "last_completed_step_type", None)
    last_completed_step_name = getattr(agent, "last_completed_step_name", None)
    last_physical_step_id = getattr(agent, "last_physical_step_id", None)
    last_physical_step_type = getattr(agent, "last_physical_step_type", None)
    last_physical_step_name = getattr(agent, "last_physical_step_name", None)
    if not isinstance(last_completed_step_id, int) or last_completed_step_id < 0:
        prev_val = prev.get("last_completed_step_id")
        if isinstance(prev_val, int):
            last_completed_step_id = prev_val
            last_completed_step_type = prev.get("last_completed_step_type")
            last_completed_step_name = prev.get("last_completed_step_name")
    if not isinstance(last_physical_step_id, int) or last_physical_step_id < 0:
        prev_val = prev.get("last_physical_step_id")
        if isinstance(prev_val, int):
            last_physical_step_id = prev_val
            last_physical_step_type = prev.get("last_physical_step_type")
            last_physical_step_name = prev.get("last_physical_step_name")
    if isinstance(last_completed_step_id, int):
        payload["last_completed_step_id"] = int(last_completed_step_id)
        payload["last_completed_step_type"] = str(last_completed_step_type or "")
        payload["last_completed_step_name"] = str(last_completed_step_name or "")
    if isinstance(last_physical_step_id, int):
        payload["last_physical_step_id"] = int(last_physical_step_id)
        payload["last_physical_step_type"] = str(last_physical_step_type or "")
        payload["last_physical_step_name"] = str(last_physical_step_name or "")
    if not isinstance(last_network_resume_step_id, int):
        prev_network_resume = prev.get("last_network_resume_step_id")
        if isinstance(prev_network_resume, int):
            last_network_resume_step_id = prev_network_resume
    if isinstance(last_network_resume_step_id, int):
        payload["last_network_resume_step_id"] = int(last_network_resume_step_id)
    if not isinstance(last_network_waited_s, int):
        prev_waited = prev.get("last_network_waited_s")
        if isinstance(prev_waited, int):
            last_network_waited_s = prev_waited
    if isinstance(last_network_waited_s, int):
        payload["last_network_waited_s"] = int(last_network_waited_s)
    if isinstance(last_http_403_resume_step_id, int):
        payload["last_http_403_resume_step_id"] = int(last_http_403_resume_step_id)
    elif isinstance(payload.get("last_network_resume_step_id"), int):
        payload["last_http_403_resume_step_id"] = int(payload["last_network_resume_step_id"])
    if isinstance(last_http_403_waited_s, int):
        payload["last_http_403_waited_s"] = int(last_http_403_waited_s)
    elif isinstance(payload.get("last_network_waited_s"), int):
        payload["last_http_403_waited_s"] = int(payload["last_network_waited_s"])
    _atomic_write_text(memory_dir / "resume_state.json", json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _handle_network_pause(
    *,
    log: Any,
    memory_dir: Path,
    run_name: str,
    dish_id: int,
    agent: EpmAgent,
    err: NetworkPauseRequired,
    accumulated_wait_s: int,
    wait_s_default: int,
    max_wait_s: int,
    retry_attempt_limit: int,
    api_key_pool_manager: _ApiKeyPoolManager | None = None,
    api_key_rotation_enabled: bool = True,
    api_key_rotation_after_http_403_wait_s: int = 60,
) -> tuple[bool, int]:
    resume_step_id = int(
        getattr(agent, "last_completed_step_id", 0)
        or getattr(agent, "last_physical_step_id", 0)
        or getattr(agent, "step_id", 0)
    )
    total_wait_s = int(accumulated_wait_s)
    detail = _network_error_detail(
        err,
        run_name=run_name,
        dish_id=dish_id,
        episode_step=int(getattr(agent, "step_id", 0)),
        last_completed_step_id=int(getattr(agent, "last_completed_step_id", 0)),
        last_physical_step_id=int(getattr(agent, "last_physical_step_id", 0)),
        resume_step_id=resume_step_id,
    )
    if (
        api_key_rotation_enabled
        and api_key_pool_manager is not None
        and int(getattr(err, "http_status", 0) or 0) == 403
    ):
        rotated, rotate_detail = api_key_pool_manager.maybe_rotate_after_http_403(err=err, log=log)
        if rotated:
            log.warning(f"[EPM] network_pause_http_403_rotated {rotate_detail}")
            return True, total_wait_s
        if rotate_detail:
            detail = f"{detail} | {rotate_detail}"
    wait_s = int(getattr(err, "retry_after_s", 0) or 0)
    if wait_s <= 0:
        wait_s = max(1, int(wait_s_default))
    next_attempt = max(0, int(total_wait_s) // max(1, int(wait_s_default))) + 1
    next_total_wait_s = int(total_wait_s) + int(wait_s)
    if next_attempt <= int(retry_attempt_limit) and next_total_wait_s <= int(max_wait_s):
        log.warning(
            f"[EPM] network_pause_wait attempt={next_attempt}/{int(retry_attempt_limit)} "
            f"wait_s={wait_s} accumulated_wait_s={next_total_wait_s}/{int(max_wait_s)} detail={detail}"
        )
        resumed_early = _poll_wait_for_network_recovery(
            err=err,
            wait_budget_s=wait_s,
            log=log,
            label="network_pause",
        )
        if resumed_early:
            log.info(
                f"[EPM] network_pause_resume_early attempt={next_attempt}/{int(retry_attempt_limit)} "
                f"wait_budget_s={wait_s}"
            )
        return True, next_total_wait_s

    reason = f"network_abort_after_retries:{detail}"
    log.error(
        f"[EPM] network_pause_retry_exhausted attempts={next_attempt - 1}/{int(retry_attempt_limit)} "
        f"accumulated_wait_s={int(total_wait_s)}/{int(max_wait_s)} detail={detail}"
    )
    _write_resume_state(
        memory_dir=memory_dir,
        run_name=run_name,
        dish_id=dish_id,
        last_step_id=resume_step_id,
        agent=agent,
        status="fatal",
        reason=reason,
        last_network_resume_step_id=resume_step_id,
        last_network_waited_s=total_wait_s,
        last_http_403_resume_step_id=(resume_step_id if isinstance(err, Http403PauseRequired) else None),
        last_http_403_waited_s=(total_wait_s if isinstance(err, Http403PauseRequired) else None),
    )
    return False, total_wait_s


def _sync_memory_templates(*, memory_dir: Path, epm_root: Path, pipeline: str) -> dict[str, bool]:
    templates = ["reference_plan.txt"]
    if str(pipeline or "").strip().lower() == "reflexion":
        templates.append("reflexion_memory.txt")
    results: dict[str, bool] = {}
    for name in templates:
        src = epm_root / "memory" / name
        dst = memory_dir / name
        if not src.exists():
            results[name] = False
            continue
        try:
            text = src.read_text(encoding="utf-8-sig", errors="replace")
        except Exception:
            results[name] = False
            continue
        _atomic_write_text(dst, text)
        results[name] = True
    return results


def _probe_endpoint_reachable(*, base_url: str, timeout_s: float = 3.0) -> tuple[bool, str]:
    url = str(base_url or "").strip()
    if not url:
        return False, "missing_base_url"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            return True, f"http_{status}"
    except urllib.error.HTTPError as e:
        # Any HTTP response means the endpoint is reachable again at the network layer.
        return True, f"http_{int(getattr(e, 'code', 0) or 0)}"
    except Exception as e:
        return False, f"{type(e).__name__}:{e}"


def _poll_wait_for_network_recovery(
    *,
    err: BaseException,
    wait_budget_s: int,
    log: Any,
    label: str,
    poll_interval_s: float = 1.0,
) -> bool:
    base_url = str(getattr(err, "request_base_url", "") or "").strip()
    kind = str(getattr(err, "kind", "") or "").strip().lower()
    if not base_url or kind in {"http_403", "http_429"}:
        time.sleep(float(max(1, int(wait_budget_s))))
        return False

    deadline = time.time() + float(max(1, int(wait_budget_s)))
    attempt = 0
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        attempt += 1
        ok, probe_detail = _probe_endpoint_reachable(
            base_url=base_url,
            timeout_s=min(3.0, max(0.5, remaining)),
        )
        if ok:
            log.info(
                f"[EPM] {label}_probe_recovered base_url={base_url} "
                f"probe_attempt={attempt} detail={probe_detail}"
            )
            return True
        sleep_s = min(float(poll_interval_s), max(0.0, deadline - time.time()))
        if sleep_s > 0:
            time.sleep(sleep_s)
    return False


def _reset_run_memory(*, memory_dir: Path, epm_root: Path, pipeline: str) -> None:
    keep = {"reference_plan.txt"}
    if str(pipeline or "").strip().lower() == "reflexion":
        keep.add("reflexion_memory.txt")
    if memory_dir.exists():
        for p in memory_dir.iterdir():
            if p.name in keep:
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
            except Exception:
                pass
    _sync_memory_templates(memory_dir=memory_dir, epm_root=epm_root, pipeline=pipeline)


def _reset_agent_state(path: Path) -> None:
    try:
        template_path = EPM_ROOT / "memory" / "agent_state.template.json"
        if template_path.exists():
            text = template_path.read_text(encoding="utf-8-sig", errors="replace").strip()
            _atomic_write_text(path, (text + "\n") if text else "{}\n")
        else:
            _atomic_write_text(path, "{}\n")
    except Exception:
        pass


_TRACE_RESULT_RE = re.compile(
    r"step=(?P<step>\d+)\s+"
    r"kind=(?P<kind>\w+)\s+"
    r"step_id=(?P<step_id>\S+)\s+"
    r"type=(?P<type>\S+)\s+"
    r"name=(?P<name>\S+)\s+"
    r"args=(?P<args>\{.*?\})\s+"
    r"result='(?P<result>[^']*)'\s+"
    r"error=(?P<error>.*)$"
)


def _strip_repr_quotes(text: str) -> str:
    s = (text or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _summarize_failures(*, memory_dir: Path, log: Any) -> None:
    trace_path = memory_dir / "step_trace.log"
    history_path = memory_dir / "long_horizon_history.txt"
    total_results = 0
    failures: list[dict[str, Any]] = []

    if trace_path.exists():
        try:
            for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _TRACE_RESULT_RE.search(line)
                if not m:
                    continue
                if str(m.group("kind")) != "result":
                    continue
                total_results += 1
                result = str(m.group("result") or "").strip().lower()
                if result == "success":
                    continue
                err = _strip_repr_quotes(str(m.group("error") or ""))
                failures.append(
                    {
                        "step": int(m.group("step")),
                        "step_id": str(m.group("step_id") or ""),
                        "name": str(m.group("name") or ""),
                        "error": err,
                    }
                )
        except Exception:
            pass

    # Fallback: summarize from long-horizon history if step trace is unavailable.
    if total_results == 0 and history_path.exists():
        try:
            for line in history_path.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    continue
                total_results += 1
                result = str(obj.get("result_summary", "")).strip().lower()
                if result == "success":
                    continue
                failures.append(
                    {
                        "step": int(obj.get("step_id") or 0),
                        "step_id": str(((obj.get("plan_ref") or {}).get("atomic_step_id") or "")),
                        "name": str(obj.get("action_or_skill") or ""),
                        "error": str(obj.get("errors") or ""),
                    }
                )
        except Exception:
            pass

    if total_results <= 0:
        log.info("[EPM] failure_summary=unavailable (no step records)")
        return

    fail_count = len(failures)
    success_count = max(0, total_results - fail_count)
    if fail_count <= 0:
        log.info(f"[EPM] failure_summary total={total_results} success={success_count} failure=0")
        return

    log.warning(f"[EPM] failure_summary total={total_results} success={success_count} failure={fail_count}")
    for item in failures[-5:]:
        log.warning(
            f"[EPM] failure step={item['step']} step_id={item['step_id']} action={item['name']} error={item['error']!r}"
        )


def _collect_failure_summary(memory_dir: Path) -> dict[str, Any]:
    trace_path = memory_dir / "step_trace.log"
    history_path = memory_dir / "long_horizon_history.txt"
    total_results = 0
    failures: list[dict[str, Any]] = []

    if trace_path.exists():
        try:
            for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _TRACE_RESULT_RE.search(line)
                if not m:
                    continue
                if str(m.group("kind")) != "result":
                    continue
                total_results += 1
                result = str(m.group("result") or "").strip().lower()
                if result == "success":
                    continue
                failures.append(
                    {
                        "step": int(m.group("step")),
                        "step_id": str(m.group("step_id") or ""),
                        "name": str(m.group("name") or ""),
                        "error": _strip_repr_quotes(str(m.group("error") or "")),
                    }
                )
        except Exception:
            pass

    if total_results == 0 and history_path.exists():
        try:
            for line in history_path.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if not isinstance(obj, dict):
                    continue
                total_results += 1
                result = str(obj.get("result_summary", "")).strip().lower()
                if result == "success":
                    continue
                failures.append(
                    {
                        "step": int(obj.get("step_id") or 0),
                        "step_id": str(((obj.get("plan_ref") or {}).get("atomic_step_id") or "")),
                        "name": str(obj.get("action_or_skill") or ""),
                        "error": str(obj.get("errors") or ""),
                    }
                )
        except Exception:
            pass

    fail_count = len(failures)
    success_count = max(0, total_results - fail_count)
    return {
        "total_results": int(total_results),
        "success_count": int(success_count),
        "fail_count": int(fail_count),
        "recent_failures": failures[-5:],
    }


def _latest_feedback_file(run_root: Path) -> Path | None:
    matches = sorted(run_root.glob("recipe_feedback_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _read_json_with_fallback(path: Path) -> dict[str, Any]:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            raw = path.read_text(encoding=enc, errors="ignore")
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise last_err
    return {}


def _extract_feedback_summary(feedback_obj: dict[str, Any]) -> tuple[str, list[str]]:
    if not isinstance(feedback_obj, dict):
        return "", []
    complaints = feedback_obj.get("complaints")
    if not isinstance(complaints, dict):
        return "", []
    lines: list[str] = []
    missing_items: list[str] = []
    for section_name in ("flavors", "technique", "temperature"):
        items = complaints.get(section_name)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            product = str(item.get("LocalizedProductName") or "").strip()
            message = str(item.get("Message") or "").strip()
            reason = str(item.get("Reason") or "").strip()
            text = " / ".join([x for x in (product, message or reason) if x])
            if text:
                lines.append(text)
            if message.lower() == "not enough" and product:
                missing_items.append(product)
    unwanted = complaints.get("unwantedProducts")
    if isinstance(unwanted, list):
        for item in unwanted:
            if not isinstance(item, dict):
                continue
            product = str(item.get("LocalizedProductName") or "").strip()
            if product:
                lines.append(f"unwanted / {product}")
    return " | ".join(lines[:12]), list(dict.fromkeys(missing_items))


def _collect_submit_feedback_summary(
    *,
    run_root: Path,
    memory_dir: Path,
    recipes_path: Path,
    dish_id: int,
) -> dict[str, Any]:
    records = _load_history_records(memory_dir)
    submit_action_success = False
    submit_action_step_id = 0
    submit_action_error = ""
    for row in records:
        action_name = str(row.get("action_or_skill") or "").strip()
        if action_name != "gui_submit_dish_via_checkout_stand":
            continue
        result_summary = str(row.get("result_summary") or "").strip().lower()
        if result_summary == "success":
            submit_action_success = True
            try:
                submit_action_step_id = int(row.get("step_id") or 0)
            except Exception:
                submit_action_step_id = 0
        elif not submit_action_error:
            submit_action_error = str(row.get("errors") or "").strip()

    expected_dish_name = ""
    try:
        from epm.kb.recipes import get_dish_by_id

        expected_dish_name = str(get_dish_by_id(str(recipes_path), int(dish_id)).dish_name or "").strip()
    except Exception:
        expected_dish_name = ""

    def _norm(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip()).lower()

    feedback_file = _latest_feedback_file(run_root)
    feedback_obj: dict[str, Any] = {}
    feedback_error = ""
    feedback_verified = False
    feedback_dish_name = ""
    taste_score: float | None = None
    feedback_summary = ""
    missing_items: list[str] = []
    if feedback_file is not None:
        try:
            feedback_obj = _read_json_with_fallback(feedback_file)
            feedback_dish_name = str(feedback_obj.get("dishName") or feedback_obj.get("dish_name") or "").strip()
            if not expected_dish_name or _norm(feedback_dish_name) == _norm(expected_dish_name):
                feedback_verified = True
            else:
                feedback_error = f"feedback_dish_mismatch:expected={expected_dish_name!r}:got={feedback_dish_name!r}"
            raw_score = feedback_obj.get("taste_score")
            if raw_score not in ("", None):
                try:
                    taste_score = float(raw_score)
                except Exception:
                    taste_score = None
            feedback_summary, missing_items = _extract_feedback_summary(feedback_obj)
        except Exception as e:
            feedback_error = f"feedback_read_error:{e!r}"
    else:
        feedback_error = "feedback_file_missing"

    final_submit_success = bool(feedback_verified or submit_action_success)
    return {
        "expected_dish_name": expected_dish_name,
        "submit_action_success": bool(submit_action_success),
        "submit_action_step_id": int(submit_action_step_id),
        "submit_action_error": submit_action_error,
        "feedback_file": str(feedback_file) if feedback_file else "",
        "feedback_file_found": bool(feedback_file),
        "feedback_verified": bool(feedback_verified),
        "feedback_error": feedback_error,
        "feedback_dish_name": feedback_dish_name,
        "taste_score": taste_score,
        "feedback_summary": feedback_summary,
        "missing_items": missing_items,
        "final_submit_success": bool(final_submit_success),
    }


def _load_history_records(memory_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidates = [memory_dir / "long_horizon_history.txt"]
    candidates.extend(sorted(memory_dir.glob("long_horizon_history.txt.fallback.*")))
    for path in candidates:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                s = line.strip()
                if not s:
                    continue
                obj = json.loads(s)
                if isinstance(obj, dict):
                    records.append(obj)
        except Exception:
            continue

    dedup: dict[int, dict[str, Any]] = {}
    for rec in records:
        try:
            step_id = int(rec.get("step_id") or 0)
        except Exception:
            continue
        dedup[step_id] = rec
    return [dedup[k] for k in sorted(dedup.keys())]


def _format_minutes_text(seconds: float) -> str:
    minutes = max(0.0, float(seconds or 0.0)) / 60.0
    return f"{minutes:.2f} min"


def _write_run_time_summary(
    *,
    run_root: Path,
    memory_dir: Path,
    run_name: str,
    dish_id: int,
    log: Any,
) -> None:
    records = _load_history_records(memory_dir)
    resume_state = _read_resume_state(memory_dir)
    status = str(resume_state.get("status") or "").strip() or "unknown"
    reason = str(resume_state.get("reason") or "").strip()

    executed_steps = len(records)
    planning_s = 0.0
    execution_s = 0.0
    summed_total_s = 0.0
    episode_elapsed_s = 0.0

    for rec in records:
        diff = rec.get("diff") if isinstance(rec.get("diff"), dict) else {}
        timing = diff.get("timing_s") if isinstance(diff.get("timing_s"), dict) else {}
        try:
            planning_s += float(timing.get("planning") or 0.0)
        except Exception:
            pass
        try:
            execution_s += float(timing.get("execution") or 0.0)
        except Exception:
            pass
        try:
            summed_total_s += float(rec.get("duration_s") or 0.0)
        except Exception:
            pass
        try:
            episode_elapsed_s = max(episode_elapsed_s, float(rec.get("episode_elapsed_s") or 0.0))
        except Exception:
            pass

    total_s = max(episode_elapsed_s, summed_total_s)
    avg_total_s = (total_s / executed_steps) if executed_steps > 0 else 0.0

    lines = [
        "Run Time Summary",
        f"run_name: {run_name}",
        f"dish_id: {int(dish_id)}",
        f"status: {status}",
        f"reason: {reason}",
        f"executed_steps: {executed_steps}",
        f"planning_time: {_format_minutes_text(planning_s)}",
        f"execution_time: {_format_minutes_text(execution_s)}",
        f"total_time: {_format_minutes_text(total_s)}",
        f"average_time_per_step: {_format_minutes_text(avg_total_s)}",
    ]
    if executed_steps > 0:
        lines.append(f"average_planning_time_per_step: {_format_minutes_text(planning_s / executed_steps)}")
        lines.append(f"average_execution_time_per_step: {_format_minutes_text(execution_s / executed_steps)}")

    out_path = run_root / "run_time_summary.txt"
    _atomic_write_text(out_path, "\n".join(lines) + "\n")
    log.info(f"[EPM] run_time_summary={out_path}")


def _read_last_jsonl_obj(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            buf = b""
            while pos > 0:
                step = min(4096, pos)
                pos -= step
                f.seek(pos)
                buf = f.read(step) + buf
                lines = buf.splitlines()
                while lines:
                    raw = lines.pop()
                    s = raw.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s.decode("utf-8", errors="replace"))
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        return obj
    except Exception:
        return {}
    return {}


def _load_provider_balance_summary(memory_dir: Path) -> dict[str, Any]:
    raw = get_cached_balance(str(memory_dir))
    if isinstance(raw, dict) and raw:
        return raw
    metrics = _read_last_jsonl_obj(memory_dir / "planner_request_metrics.jsonl")
    if not isinstance(metrics, dict):
        return {}
    if (
        metrics.get("provider_balance_usd") is None
        and metrics.get("provider_used_usd") is None
        and metrics.get("provider_total_usd") is None
        and not str(metrics.get("provider_balance_error") or "").strip()
    ):
        return {}
    return {
        "updated_at": metrics.get("ts"),
        "last_checked_step": metrics.get("provider_balance_step"),
        "balance_usd": metrics.get("provider_balance_usd"),
        "used_usd": metrics.get("provider_used_usd"),
        "total_usd": metrics.get("provider_total_usd"),
        "error": metrics.get("provider_balance_error", ""),
        "api_key_masked": metrics.get("api_key_masked", ""),
    }


def _maybe_send_low_balance_alert(
    *,
    runtime: Any,
    memory_dir: Path,
    run_name: str,
    dish_id: int,
    current_step_id: int,
    log: Any,
) -> bool:
    balance = _load_provider_balance_summary(memory_dir)
    if not balance:
        return False
    err = str(balance.get("error") or "").strip()
    if err:
        return False
    try:
        remaining = float(balance.get("balance_usd"))
    except Exception:
        return False
    threshold = float(getattr(runtime, "notify_email_low_balance_usd_threshold", 10.0) or 10.0)
    if remaining >= threshold:
        return False
    if str(os.environ.get("EPM_LOW_BALANCE_ALERT_SENT", "") or "").strip() == "1":
        return False
    subject = (
        f"[EPM][LOW_BALANCE] run={run_name} dish={dish_id} "
        f"balance=${remaining:.3f} step={int(current_step_id)}"
    )
    body = "\n".join(
        [
            "EPM Low Balance Warning",
            "",
            f"run_name: {run_name}",
            f"dish_id: {int(dish_id)}",
            f"episode_step: {int(current_step_id)}",
            f"balance_usd: {balance.get('balance_usd')}",
            f"used_usd: {balance.get('used_usd')}",
            f"total_usd: {balance.get('total_usd')}",
            f"threshold_usd: {threshold}",
            f"updated_at: {balance.get('updated_at', '')}",
            f"key: {balance.get('api_key_masked', '')}",
            f"planner_metrics: {memory_dir / 'planner_request_metrics.jsonl'}",
        ]
    )
    log.error(
        f"[EPM] LOW_BALANCE_WARNING balance_usd={remaining:.3f} "
        f"threshold_usd={threshold:.3f} step={int(current_step_id)} run={run_name} dish={dish_id}"
    )
    sent = maybe_send_email_notification(runtime=runtime, subject=subject, body=body, log=log)
    if not sent:
        return False
    os.environ["EPM_LOW_BALANCE_ALERT_SENT"] = "1"
    return True


def _is_timeout_network_issue(err: BaseException) -> bool:
    return str(getattr(err, "kind", "") or "").strip().lower() == "timeout"


def _timeout_budget_s_for_error(*, settings: Any, err: BaseException) -> int:
    channel = str(getattr(err, "request_channel", "") or "").strip().lower()
    try:
        if channel == "llm":
            return max(1, int(float(getattr(settings.llm, "timeout_s", 600) or 600)))
        return max(1, int(float(getattr(settings.vlm, "timeout_s", 600) or 600)))
    except Exception:
        return 600


def _log_startup_provider_balance_once(
    *,
    settings: Any,
    memory_dir: Path,
    log: Any,
) -> None:
    seen: set[tuple[str, str, str]] = set()
    for channel, cfg in (("llm", getattr(settings, "llm", None)), ("vlm", getattr(settings, "vlm", None))):
        if cfg is None:
            continue
        base_url = str(getattr(cfg, "base_url", "") or "").strip()
        if not balance_check_enabled(base_url):
            continue
        api_key_env = str(getattr(cfg, "api_key_env", "") or "").strip()
        api_key = str(os.environ.get(api_key_env, "") or "").strip() if api_key_env else ""
        if not api_key:
            api_key = str(getattr(cfg, "api_key", "") or "").strip()
        if not api_key:
            continue
        dedupe_key = (channel, provider_root, api_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        info, _ = maybe_refresh_balance_status(
            base_url=base_url,
            api_key=api_key,
            memory_dir=str(memory_dir),
            episode_step=0,
            cadence_steps=10**9,
            timeout_s=min(8.0, max(1.0, float(getattr(cfg, "timeout_s", 60.0) or 60.0))),
            channel=channel,
            call="run_start",
        )
        err = str(info.get("error") or "").strip()
        if err:
            log.warning(
                f"[EPM] provider_balance_startup_skipped channel={channel} env={api_key_env or '-'} "
                f"key={_mask_secret(api_key)} error={err}"
            )
            continue
        log.info(
            f"[EPM] provider_balance_startup channel={channel} env={api_key_env or '-'} "
            f"key={_mask_secret(api_key)} balance_usd={info.get('balance_usd')} "
            f"used_usd={info.get('used_usd')} total_usd={info.get('total_usd')}"
        )


def _build_email_summary(
    *,
    run_root: Path,
    memory_dir: Path,
    run_name: str,
    dish_id: int,
    recipes_path: Path,
    return_code: int,
) -> tuple[str, str]:
    resume = _read_resume_state(memory_dir)
    status = str(resume.get("status") or "finished").strip() or "finished"
    reason = str(resume.get("reason") or "").strip()
    manual_intervention_required = "manual_intervention_required" in reason.lower()
    status_tag = "OK" if status in {"done", "finished"} and int(return_code) == 0 else "ATTN"
    if manual_intervention_required:
        status_tag = "MANUAL"

    records = _load_history_records(memory_dir)
    executed_steps = len(records)
    planning_s = 0.0
    execution_s = 0.0
    summed_total_s = 0.0
    episode_elapsed_s = 0.0
    for rec in records:
        diff = rec.get("diff") if isinstance(rec.get("diff"), dict) else {}
        timing = diff.get("timing_s") if isinstance(diff.get("timing_s"), dict) else {}
        try:
            planning_s += float(timing.get("planning") or 0.0)
        except Exception:
            pass
        try:
            execution_s += float(timing.get("execution") or 0.0)
        except Exception:
            pass
        try:
            summed_total_s += float(rec.get("duration_s") or 0.0)
        except Exception:
            pass
        try:
            episode_elapsed_s = max(episode_elapsed_s, float(rec.get("episode_elapsed_s") or 0.0))
        except Exception:
            pass
    total_s = max(episode_elapsed_s, summed_total_s)

    failure = _collect_failure_summary(memory_dir)
    submit_feedback = _collect_submit_feedback_summary(
        run_root=run_root,
        memory_dir=memory_dir,
        recipes_path=recipes_path,
        dish_id=dish_id,
    )
    balance = _load_provider_balance_summary(memory_dir)
    score = submit_feedback.get("taste_score", None)
    subject_suffix = ""
    if score is not None:
        try:
            subject_suffix = f" score={float(score):.3f}"
        except Exception:
            subject_suffix = f" score={score}"
    manual_suffix = " manual_intervention_required" if manual_intervention_required else ""
    subject = f"[EPM][{status_tag}] run={run_name} dish={dish_id} status={status}{manual_suffix}{subject_suffix}"
    try:
        agent_state = json.loads((memory_dir / "agent_state.json").read_text(encoding="utf-8-sig", errors="replace"))
        if not isinstance(agent_state, dict):
            agent_state = {}
    except Exception:
        agent_state = {}

    lines: list[str] = [
        "EPM Run Notification",
        "",
        "[Run]",
        f"run_name: {run_name}",
        f"dish_id: {int(dish_id)}",
        f"status: {status}",
        f"reason: {reason or '(empty)'}",
        f"manual_intervention_required: {manual_intervention_required}",
        f"return_code: {int(return_code)}",
        f"updated_at: {resume.get('updated_at', '')}",
        "",
        "[Progress]",
        f"last_step_id: {resume.get('last_step_id', '')}",
        f"last_completed_step_id: {resume.get('last_completed_step_id', '')}",
        f"last_completed_step: {resume.get('last_completed_step_type', '')}:{resume.get('last_completed_step_name', '')}",
        f"last_physical_step_id: {resume.get('last_physical_step_id', '')}",
        f"last_physical_step: {resume.get('last_physical_step_type', '')}:{resume.get('last_physical_step_name', '')}",
        f"executed_steps: {executed_steps}",
        "",
        "[Timing]",
        f"planning_time: {_format_minutes_text(planning_s)}",
        f"execution_time: {_format_minutes_text(execution_s)}",
        f"total_time: {_format_minutes_text(total_s)}",
            "",
            "[Failures]",
            f"result_total: {failure.get('total_results', 0)}",
            f"result_success: {failure.get('success_count', 0)}",
            f"result_failure: {failure.get('fail_count', 0)}",
    ]

    recent_failures = failure.get("recent_failures") or []
    if recent_failures:
        lines.append("recent_failures:")
        for item in recent_failures:
            lines.append(
                f"  - step={item.get('step')} step_id={item.get('step_id')} "
                f"name={item.get('name')} error={item.get('error')}"
            )
    else:
        lines.append("recent_failures: (none)")

    lines.extend(
        [
            "",
            "[Feedback]",
            f"expected_dish_name: {submit_feedback.get('expected_dish_name', '')}",
            f"submit_action_success: {submit_feedback.get('submit_action_success', False)}",
            f"submit_action_step_id: {submit_feedback.get('submit_action_step_id', 0)}",
            f"submit_action_error: {submit_feedback.get('submit_action_error', '') or '(none)'}",
            f"feedback_file_found: {submit_feedback.get('feedback_file_found', False)}",
            f"feedback_verified: {submit_feedback.get('feedback_verified', False)}",
            f"final_submit_success: {submit_feedback.get('final_submit_success', False)}",
            f"feedback_dish_name: {submit_feedback.get('feedback_dish_name', '')}",
            f"taste_score: {submit_feedback.get('taste_score', None)}",
            f"feedback_summary: {submit_feedback.get('feedback_summary', '') or '(none)'}",
            f"missing_items: {', '.join(submit_feedback.get('missing_items', [])) or '(none)'}",
            f"feedback_error: {submit_feedback.get('feedback_error', '') or '(none)'}",
            "",
            "[Agent State]",
            f"mode: {agent_state.get('mode', '')}",
            f"is_held: {agent_state.get('is_held', '')}",
            f"held_item: {agent_state.get('held_item', '')}",
            f"active_container: {agent_state.get('active_container', '')}",
            f"force_submit_active: {agent_state.get('force_submit_active', '')}",
            f"force_submit_stage: {agent_state.get('force_submit_stage', '')}",
            f"force_submit_remaining_steps: {agent_state.get('force_submit_remaining_steps', '')}",
            "",
            "[Network]",
            f"last_network_resume_step_id: {resume.get('last_network_resume_step_id', '')}",
            f"last_network_waited_s: {resume.get('last_network_waited_s', '')}",
            f"last_http_403_resume_step_id: {resume.get('last_http_403_resume_step_id', '')}",
            f"last_http_403_waited_s: {resume.get('last_http_403_waited_s', '')}",
            "",
            "[Balance]",
            f"updated_at: {balance.get('updated_at', '')}",
            f"last_checked_step: {balance.get('last_checked_step', '')}",
            f"provider_root: {balance.get('provider_root', '')}",
            f"key: {balance.get('api_key_masked', '')}",
            f"balance_usd: {balance.get('balance_usd', '')}",
            f"used_usd: {balance.get('used_usd', '')}",
            f"total_usd: {balance.get('total_usd', '')}",
            f"error: {balance.get('error', '') or '(none)'}",
            "",
            "[Paths]",
            f"memory_dir: {memory_dir}",
            f"run_root: {run_root}",
            f"feedback_file: {submit_feedback.get('feedback_file', '')}",
        ]
    )
    return subject, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("epm/epm_config.json"),
        help="Path to epm_config.json (or a directory containing it).",
    )
    parser.add_argument("--dish-id", type=int, default=None)
    parser.add_argument(
        "--ablation-reduce",
        default="",
        metavar="NO_COMPONENT[,NO_COMPONENT...]",
        help=(
            "Comma-separated reductions, e.g. no_body,no_strategy. "
            "Allowed: no_body, no_perception_sup, no_strategy, "
            "no_history, no_feedback, no_rgb, no_skill, no_action."
        ),
    )
    parser.add_argument("--steps", type=int, default=None, help="Debug mode: run a fixed number of steps.")
    parser.add_argument("--max-steps", type=int, default=None, help="Safety limit for until-done mode.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Update config JSON before run: key=value (dot path). Writes back to the config file.",
    )
    parser.add_argument(
        "--stm-window-size",
        "--stm_window_size",
        "--window-size",
        "--window_size",
        dest="stm_window_size",
        type=int,
        default=None,
        help="Override runtime.stm_window_size from config.",
    )
    parser.add_argument(
        "--restart-env",
        "--restart_env",
        "--restartenv",
        dest="restart_env",
        action="store_true",
        help="Restart the environment before any actions.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional run folder name (default: YYYYMMDD_HHMMSS-pipeline-dish_id).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing run folder (use --run-name or the latest run under epm/runs).",
    )
    parser.add_argument(
        "--resume-step-id",
        type=int,
        default=None,
        help="When used with --resume, rewind/restore to this exact step id before continuing.",
    )
    parser.add_argument(
        "--rollback-steps",
        type=int,
        default=0,
        help="When used with --resume, rewind this many steps from the detected resume point before continuing.",
    )
    parser.add_argument(
        "--yes-rewind",
        action="store_true",
        help="Auto-confirm manual rewind even if there are physical actions after the target step.",
    )
    parser.add_argument(
        "--refresh-memory",
        action="store_true",
        help="Overwrite run memory templates with repo-level epm/memory/* (reference_plan/reflexion_memory).",
    )
    parser.add_argument(
        "--log-mode",
        type=str,
        default="",
        help="Log mode override: 'verbose' or 'minimal' (default: from runtime.log_mode).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in console logs (or set NO_COLOR=1).",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        cleaned: list[str] = []
        for tok in unknown:
            if tok == "--restartenv":
                args.restart_env = True
                continue
            cleaned.append(tok)
        unknown = cleaned

    cfg_path = Path(args.config).resolve()
    if args.resume_step_id is not None and int(args.resume_step_id) < 0:
        parser.error("--resume-step-id must be >= 0.")
    if int(args.rollback_steps or 0) < 0:
        parser.error("--rollback-steps must be >= 0.")
    if (args.resume_step_id is not None or int(args.rollback_steps or 0) > 0) and not bool(args.resume):
        parser.error("--resume-step-id / --rollback-steps require --resume.")
    if cfg_path.is_dir():
        cfg_path = (cfg_path / "epm_config.json").resolve()
    try:
        overrides = list(args.set or [])
        overrides.extend(_collect_dot_overrides(list(unknown)))
        _apply_config_overrides(cfg_path, overrides)
    except Exception as e:
        parser.error(str(e))
    os.environ["EPM_CONFIG"] = str(cfg_path)
    settings = load_settings(cfg_path)
    prompt_ablation = resolve_ablation_reductions(args.ablation_reduce)
    prompt_ablation_profile = prompt_ablation.profile
    dish_id = settings.dish_id if args.dish_id is None else int(args.dish_id)
    pipeline_tag = _normalize_run_name_part(str(settings.brain.pipeline), fallback="pipeline")
    prompt_ablation_label = "-".join(prompt_ablation.reductions) or "full"
    prompt_ablation_tag = _normalize_run_name_part(prompt_ablation_label, fallback="full")
    dish_tag = _normalize_run_name_part(str(dish_id), fallback="dish")
    os.environ["EPM_LOG_TAG"] = f"{pipeline_tag}-{prompt_ablation_tag}-{dish_tag}"
    log_mode = (args.log_mode or "").strip().lower() or (getattr(settings.runtime, "log_mode", "") or "verbose").strip().lower()
    if log_mode not in ("verbose", "minimal"):
        log_mode = "verbose"
    use_color = bool(getattr(settings.runtime, "log_color", True)) and (not bool(args.no_color))
    _configure_logging(
        log_mode=log_mode,
        use_color=use_color,
        default_log_level=str(getattr(settings.runtime, "log_level", "INFO") or "INFO"),
    )
    log = logging.bind(component="EPM")
    disabled_groups_tag = ",".join(sorted(prompt_ablation.disabled_groups)) or "none"
    log.info(
        f"[EPM] ablation_reduce={list(prompt_ablation.reductions)} disabled_groups={disabled_groups_tag}"
    )

    resume = bool(args.resume)
    runs_root = (EPM_ROOT / "runs").resolve()
    if resume:
        if (args.run_name or "").strip():
            run_name = (args.run_name or "").strip()
            run_root = (runs_root / run_name).resolve()
            if not run_root.exists():
                log.error(f"[EPM] resume=true error=run_not_found run_root={run_root}")
                return 2
        else:
            latest = _find_latest_run_dir(runs_root)
            if latest is None:
                log.error(f"[EPM] resume=true error=no_runs_found runs_root={runs_root}")
                return 2
            run_root = latest.resolve()
            run_name = run_root.name
    else:
        run_name = (args.run_name or "").strip()
        if not run_name:
            run_name = _default_run_name(
                pipeline=settings.brain.pipeline,
                model=_planner_model_tag(settings),
                prompt_ablation_profile=prompt_ablation_label,
                dish_id=dish_id,
            )
        run_root = (runs_root / run_name).resolve()
    screenshot_dir = run_root / "screenshots"
    memory_dir = run_root / "memory"
    if resume:
        if not memory_dir.exists():
            log.error(f"[EPM] resume=true error=memory_dir_not_found memory_dir={memory_dir}")
            return 2
        screenshot_dir.mkdir(parents=True, exist_ok=True)
    else:
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        memory_dir.mkdir(parents=True, exist_ok=True)

    (memory_dir / "effective_prompt_ablation.json").write_text(
        json.dumps(
            {
                "profile": prompt_ablation_profile,
                "disabled_groups": sorted(prompt_ablation.disabled_groups),
                "ablation_reduce": list(prompt_ablation.reductions),
                "action_skill_interface": (
                    "semantic actions and skills disabled; raw keyboard/mouse actions only"
                    if restrict_to_raw_input_actions(
                        prompt_ablation_profile, disabled_groups=prompt_ablation.disabled_groups
                    )
                    else "high-level skills disabled; semantic action APIs only"
                    if restrict_to_actions_only(
                        prompt_ablation_profile, disabled_groups=prompt_ablation.disabled_groups
                    )
                    else "always_enabled"
                ),
                "raw_input_actions": (
                    list(RAW_INPUT_ACTIONS)
                    if restrict_to_raw_input_actions(
                        prompt_ablation_profile, disabled_groups=prompt_ablation.disabled_groups
                    )
                    else []
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if bool(args.refresh_memory):
        results = _sync_memory_templates(
            memory_dir=memory_dir,
            epm_root=EPM_ROOT,
            pipeline=settings.brain.pipeline,
        )
        ok = [k for k, v in results.items() if v]
        skipped = [k for k, v in results.items() if not v]
        log.info(f"[EPM] refresh_memory=ok files={ok}")
        if skipped:
            log.info(f"[EPM] refresh_memory=skipped files={skipped}")

    api_key_pool_manager = _ApiKeyPoolManager(
        memory_dir=memory_dir,
        quarantine_s=int(getattr(settings.runtime, "api_key_rotation_quarantine_s", 1800) or 1800),
    )
    api_key_pool_manager.register_cfg(cfg=settings.llm, channel="llm")
    api_key_pool_manager.register_cfg(cfg=settings.vlm, channel="vlm")
    if api_key_pool_manager.has_pools():
        api_key_pool_manager.initialize(log=log)
        log.info(f"[EPM] api_key_pool_status_path={memory_dir / 'api_key_pool_status.json'}")
    try:
        max_steps_for_balance = int(args.steps) if args.steps is not None else int(getattr(settings.runtime, "max_steps", 300) or 300)
    except Exception:
        max_steps_for_balance = 300
    os.environ["EPM_BALANCE_CADENCE_STEPS"] = str(max(10**6, max_steps_for_balance + 1))
    _log_startup_provider_balance_once(settings=settings, memory_dir=memory_dir, log=log)

    restart_env = bool(getattr(settings.runtime, "restart_env", False)) or bool(args.restart_env)
    if restart_env:
        try:
            from epm.cerebellum.gui_actions.restart_env_gui import restart_environment

            log.info("[EPM] restart_env=begin")
            res = restart_environment(window_title=settings.capture.window_title)
            if isinstance(res, dict) and not bool(res.get("success", False)):
                err = str(res.get("error") or "restart_environment_returned_unsuccessful").strip()
                log.error(f"[EPM] restart_env=unsuccessful result={res}")
                _append_startup_error_event(
                    memory_dir=memory_dir,
                    phase="startup_restart_env",
                    error=err,
                    result="failure",
                    extra={"name": "restart_env", "raw_result": res},
                )
            else:
                log.info(f"[EPM] restart_env=done result={res}")
        except Exception as e:
            log.error(f"[EPM] restart_env=failed error={e!r}")
            _append_startup_error_event(
                memory_dir=memory_dir,
                phase="startup_restart_env",
                error=repr(e),
                result="failure",
                extra={"name": "restart_env"},
            )

    resume_step_id, resume_source = _resolve_resume_step_id(memory_dir) if resume else (0, "none")
    if resume:
        resume_state = _read_resume_state(memory_dir)
        resume_dish = resume_state.get("dish_id")
        if args.dish_id is None and isinstance(resume_dish, int):
            dish_id = int(resume_dish)
            dish_tag = _normalize_run_name_part(str(dish_id), fallback="dish")
            os.environ["EPM_LOG_TAG"] = f"{pipeline_tag}-{prompt_ablation_tag}-{dish_tag}"
            log.info(f"[EPM] resume=true using_dish_id_from_resume_state dish_id={dish_id}")
        elif isinstance(resume_dish, int) and int(resume_dish) != int(dish_id):
            log.error(f"[EPM] resume=true error=dish_id_mismatch resume_dish_id={resume_dish} dish_id={dish_id}")
            return 2
        detected_resume_step_id = int(resume_step_id)
        if args.resume_step_id is not None:
            cli_resume_step_id = int(args.resume_step_id)
            if cli_resume_step_id > detected_resume_step_id:
                log.error(
                    f"[EPM] resume=true error=resume_step_id_out_of_range "
                    f"requested={cli_resume_step_id} detected={detected_resume_step_id}"
                )
                return 2
            resume_step_id = cli_resume_step_id
            resume_source = f"cli.resume_step_id({cli_resume_step_id})"
        elif int(args.rollback_steps or 0) > 0:
            rollback_steps = int(args.rollback_steps)
            resume_step_id = max(0, detected_resume_step_id - rollback_steps)
            resume_source = f"{resume_source}-rollback({rollback_steps})"
        manual_rewind = int(resume_step_id) < int(detected_resume_step_id)
        if manual_rewind and not _confirm_manual_rewind_with_physical_actions(
            memory_dir=memory_dir,
            resume_step_id=int(resume_step_id),
            log=log,
            force_yes=bool(args.yes_rewind),
        ):
            log.error(
                f"[EPM] resume_rewind=cancelled target_step={int(resume_step_id)} "
                f"detected_resume_step_id={int(detected_resume_step_id)}"
            )
            return 2
        _cleanup_resume_artifacts(
            memory_dir=memory_dir,
            screenshot_dir=screenshot_dir,
            resume_step_id=int(resume_step_id),
            log=log,
            force=bool(manual_rewind),
            reason_tag=("resume_rewind_manual" if manual_rewind else ""),
        )

    def _finalize_run(return_code: int) -> int:
        try:
            audit_summary = audit_prompt_trace_directory(
                memory_dir=memory_dir,
                profile=prompt_ablation_profile,
                disabled_groups=prompt_ablation.disabled_groups,
            )
            if not bool(audit_summary.get("ok", False)):
                log.error(
                    "[EPM] prompt_ablation_trace_audit=failed "
                    f"violations={audit_summary.get('violation_count', 0)}"
                )
                return_code = max(int(return_code), 3)
            else:
                log.info(f"[EPM] prompt_ablation_trace_audit=passed traces={audit_summary.get('trace_count', 0)}")
        except Exception as e:
            log.warning(f"[EPM] prompt_ablation_trace_audit=error err={e!r}")
        try:
            _summarize_failures(memory_dir=memory_dir, log=log)
        except Exception:
            pass
        try:
            _write_run_time_summary(
                run_root=run_root,
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                log=log,
            )
        except Exception:
            pass
        try:
            subject, body = _build_email_summary(
                run_root=run_root,
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                recipes_path=settings.paths.recipes_path,
                return_code=int(return_code),
            )
            maybe_send_email_notification(runtime=settings.runtime, subject=subject, body=body, log=log)
        except Exception as e:
            log.warning(f"[EPM] email_notify skipped due_to_exception err={e!r}")
        return int(return_code)

    def _handle_timeout_retry_or_abort(err: BaseException, *, timeout_retry_count: int) -> tuple[str, int]:
        detail = _network_error_detail(
            err,
            run_name=run_name,
            dish_id=dish_id,
            episode_step=int(getattr(agent, "step_id", 0)),
            last_completed_step_id=int(getattr(agent, "last_completed_step_id", 0)),
            last_physical_step_id=int(getattr(agent, "last_physical_step_id", 0)),
        )
        resume_step_id = int(
            getattr(agent, "last_completed_step_id", 0)
            or getattr(agent, "last_physical_step_id", 0)
            or getattr(agent, "step_id", 0)
        )
        timeout_budget_s = _timeout_budget_s_for_error(settings=settings, err=err)
        next_count = int(timeout_retry_count) + 1
        channel = str(getattr(err, "request_channel", "") or "").strip() or "planner"
        source = str(getattr(err, "source", "") or "").strip() or "unknown"
        timeout_retry_wait_s = max(1, int(getattr(settings.runtime, "network_wait_s", 15) or 15))
        timeout_retry_limit = max(1, int(getattr(settings.runtime, "network_retry_attempts", 10) or 10))
        if next_count <= timeout_retry_limit:
            subject = (
                f"[EPM][TIMEOUT_WARN] run={run_name} dish={dish_id} "
                f"step={int(getattr(agent, 'step_id', 0))} channel={channel}"
            )
            body = "\n".join(
                [
                    "EPM Planner Timeout Warning",
                    "",
                    "A planner request timed out after the configured timeout budget.",
                    "The run will retry exactly once.",
                    "",
                    f"run_name: {run_name}",
                    f"dish_id: {int(dish_id)}",
                    f"episode_step: {int(getattr(agent, 'step_id', 0))}",
                    f"resume_step_id: {resume_step_id}",
                    f"channel: {channel}",
                    f"source: {source}",
                    f"timeout_budget_s: {timeout_budget_s}",
                    f"retry_wait_s: {timeout_retry_wait_s}",
                    f"retry_attempt: {next_count}/{timeout_retry_limit}",
                    f"detail: {detail}",
                    f"memory_dir: {memory_dir}",
                ]
            )
            log.warning(
                f"[EPM] timeout_retry_wait timeout_budget_s={timeout_budget_s} "
                f"channel={channel} wait_s={timeout_retry_wait_s} "
                f"attempt={next_count}/{timeout_retry_limit} detail={detail}"
            )
            maybe_send_email_notification(runtime=settings.runtime, subject=subject, body=body, log=log)
            resumed_early = _poll_wait_for_network_recovery(
                err=err,
                wait_budget_s=timeout_retry_wait_s,
                log=log,
                label="timeout_retry",
            )
            if resumed_early:
                log.info(
                    f"[EPM] timeout_retry_resume_early channel={channel} "
                    f"attempt={next_count}/{timeout_retry_limit} wait_budget_s={timeout_retry_wait_s}"
                )
            return "retry", next_count

        subject = (
            f"[EPM][TIMEOUT_FATAL] run={run_name} dish={dish_id} "
            f"step={int(getattr(agent, 'step_id', 0))} channel={channel}"
        )
        body = "\n".join(
            [
                "EPM Planner Timeout Fatal",
                "",
                "The planner request timed out twice in a row.",
                "The run is being aborted.",
                "",
                f"run_name: {run_name}",
                f"dish_id: {int(dish_id)}",
                f"episode_step: {int(getattr(agent, 'step_id', 0))}",
                f"resume_step_id: {resume_step_id}",
                f"channel: {channel}",
                f"source: {source}",
                f"timeout_budget_s: {timeout_budget_s}",
                f"retry_wait_s: {timeout_retry_wait_s}",
                f"retry_attempt: {next_count}/{timeout_retry_limit}",
                f"detail: {detail}",
                f"memory_dir: {memory_dir}",
            ]
        )
        log.error(
            f"[EPM] timeout_retry_exhausted timeout_budget_s={timeout_budget_s} "
            f"channel={channel} attempts={timeout_retry_limit} "
            f"detail={detail}"
        )
        maybe_send_email_notification(runtime=settings.runtime, subject=subject, body=body, log=log)
        _write_resume_state(
            memory_dir=memory_dir,
            run_name=run_name,
            dish_id=dish_id,
            last_step_id=resume_step_id,
            agent=agent,
            status="fatal",
            reason=f"network_timeout_after_retry:{detail}",
            last_network_resume_step_id=resume_step_id,
            last_network_waited_s=int(timeout_retry_wait_s) * int(timeout_retry_limit),
        )
        return "fatal", next_count

    log.info(
        f"[EPM] config={args.config} dish_id={dish_id} steps={args.steps} run_name={run_name} "
        f"log_mode={log_mode} resume={resume} resume_step_id={int(resume_step_id)} "
        f"resume_step_source={resume_source} rollback_steps={int(args.rollback_steps or 0)}"
    )
    log.info(f"[EPM] run_root={run_root}")
    if resume and resume_step_id > 0:
        log.info(f"[EPM] resume=true last_step_id={resume_step_id} source={resume_source}")

    # Ensure global stop hotkey thread is up early (before any actions run).
    try:
        _ = RawInputController()
    except Exception:
        pass

    agent_state_path = (memory_dir / "agent_state.json").resolve()
    if not resume:
        _reset_run_memory(
            memory_dir=memory_dir,
            epm_root=EPM_ROOT,
            pipeline=settings.brain.pipeline,
        )
        _reset_agent_state(agent_state_path)
    os.environ["EPM_AGENT_STATE_PATH"] = str(agent_state_path)
    os.environ["EPM_RUN_ROOT"] = str(run_root)
    os.environ["EPM_RUN_MEMORY_DIR"] = str(memory_dir)

    agent = EpmAgent(
        AgentConfig(
            memory_dir=memory_dir,
            realtime_products_path=settings.paths.realtime_products_path,
            camera_info_path=settings.paths.camera_info_path,
            agent_state_path=agent_state_path,
            recipes_path=settings.paths.recipes_path,
            dish_id=dish_id,
            screenshot_dir=screenshot_dir,
            window_title=settings.capture.window_title,
            activate_window_each_step=settings.capture.activate_window_each_step,
            screenshot_format=str(getattr(settings.capture, "screenshot_format", "jpeg") or "jpeg"),
            screenshot_jpeg_quality=int(getattr(settings.capture, "screenshot_jpeg_quality", 70) or 70),
            stm_window_size=(
                int(args.stm_window_size)
                if args.stm_window_size is not None
                else settings.runtime.stm_window_size
            ),
            action_catalog_path=settings.brain.action_catalog_path,
            exposed_action_categories=settings.brain.exposed_action_categories,
            max_actions_per_category=settings.brain.max_actions_per_category,
            inject_memory=settings.brain.inject_memory,
            inject_tool_schemas=settings.brain.inject_tool_schemas,
            scripted_plan_path=settings.brain.scripted_plan_path,
            verbose=(bool(settings.runtime.verbose) and (log_mode != "minimal")),
            planner_mode=settings.brain.planner_mode,
            llm=settings.llm,
            vlm=settings.vlm,
            pipeline=settings.brain.pipeline,
            perception_mode=settings.brain.perception_mode,
            max_visible_items=settings.brain.max_visible_items,
            plan_min_steps=settings.brain.plan_min_steps,
            plan_max_steps=settings.brain.plan_max_steps,
            allow_incremental_plan=settings.brain.allow_incremental_plan,
            task_progress_maintenance=settings.brain.task_progress_maintenance,
            step_success_judge=settings.brain.step_success_judge,
            subgoal_done_judge=settings.brain.subgoal_done_judge,
            visual_anomaly_observer=settings.brain.visual_anomaly_observer,
            precondition_checker=settings.brain.precondition_checker,
            reflexion=settings.brain.reflexion,
            prompt_policy=settings.brain.prompt_policy,
            prompt_ablation_profile=prompt_ablation_profile,
            prompt_disabled_groups=prompt_ablation.disabled_groups,
            prompt_layout_path=settings.brain.prompt_layout_path,
            log_mode=log_mode,
            log_color=use_color,
            resume=resume,
            force_submit_step_threshold=settings.runtime.force_submit_step_threshold,
            force_submit_active=settings.runtime.force_submit_active,
            force_submit_within_steps=settings.runtime.force_submit_within_steps,
        )
    )
    if resume and resume_step_id > 0:
        agent.step_id = int(resume_step_id)
        resume_state = _read_resume_state(memory_dir)
        for key in ("last_completed_step_id", "last_physical_step_id"):
            value = resume_state.get(key)
            if isinstance(value, int):
                setattr(agent, key, int(value))
        for key in ("last_completed_step_type", "last_completed_step_name", "last_physical_step_type", "last_physical_step_name"):
            value = resume_state.get(key)
            if isinstance(value, str):
                setattr(agent, key, str(value))
        try:
            seed = getattr(agent, "step_id", 0)
            set_counter = getattr(getattr(agent, "pipeline", None), "set_atomic_counter", None)
            if callable(set_counter):
                set_counter(int(seed))
        except Exception:
            pass

    network_pause_enabled = bool(getattr(settings.runtime, "network_pause_enabled", True))
    network_wait_s = max(1, int(getattr(settings.runtime, "network_wait_s", 15) or 15))
    network_max_wait_s = max(network_wait_s, int(getattr(settings.runtime, "network_max_wait_s", 150) or 150))
    network_retry_attempts = max(1, int(getattr(settings.runtime, "network_retry_attempts", 10) or 10))
    api_key_rotation_enabled = bool(getattr(settings.runtime, "api_key_rotation_enabled", True))
    api_key_rotation_after_http_403_wait_s = max(
        1, int(getattr(settings.runtime, "api_key_rotation_after_http_403_wait_s", 60) or 60)
    )
    network_accumulated_wait_s = 0

    # Bootstrap common CS_CamDump hotkeys (best-effort).
    # - F12: realtime_products.json scan
    # - F11: realtime_radar_scan.txt
    # - Alt+J: realtime_interaction_info.txt / UDP
    try:
        from epm.vision.screen_capture import activate_window
        from epm.cerebellum.local_actions import io_controller

        bootstrap_stdout = None
        if log_mode == "minimal":
            bootstrap_stdout = io.StringIO()
        if bootstrap_stdout is not None:
            with redirect_stdout(bootstrap_stdout):
                status = bootstrap_all(
                    realtime_products_path=settings.paths.realtime_products_path,
                    window_title=settings.capture.window_title,
                    userdata_root=settings.paths.realtime_products_path.parent,
                    activate_window=activate_window,
                    io_controller=io_controller,
                )
        else:
            status = bootstrap_all(
                realtime_products_path=settings.paths.realtime_products_path,
                window_title=settings.capture.window_title,
                userdata_root=settings.paths.realtime_products_path.parent,
                activate_window=activate_window,
                io_controller=io_controller,
            )
        log.info(f"[EPM] hotkeys_bootstrap={status}")
        if bootstrap_stdout is not None:
            txt = bootstrap_stdout.getvalue()
            if txt.strip():
                (memory_dir / "bootstrap_stdout.log").write_text(txt, encoding="utf-8")
    except Exception as e:
        log.info(f"[EPM] hotkeys_bootstrap=failed error={e!r}")

    # Hard gate: ensure dish order is placed before any planning/execution.
    # Relaxed for resume: if we already executed steps, do not force re-order.
    if resume and agent.step_id > 0:
        log.info(f"[EPM] order_gate=skip resume_step_id={agent.step_id}")
    else:
        log.info(f"[EPM] order_gate=run resume={resume} resume_step_id={agent.step_id}")
        try:
            if not _ensure_dish_ordered(memory_dir=memory_dir, dish_id=dish_id, dish_name=agent.dish.dish_name, log=log):
                log.error("[EPM] order_failed exhausted attempts; aborting before planning")
                _write_resume_state(
                    memory_dir=memory_dir,
                    run_name=run_name,
                    dish_id=dish_id,
                    last_step_id=agent.step_id,
                    agent=agent,
                    status="fatal",
                    reason="order_failed",
                )
                return _finalize_run(2)
        except StopRequested:
            log.warning("[EPM] stopped=true reason=global_stop_hotkey")
            _write_resume_state(
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                last_step_id=agent.step_id,
                agent=agent,
                status="stopped",
                reason="global_stop_hotkey",
            )
            return _finalize_run(0)

    if args.steps is not None:
        ended_status = ""
        requested_steps = int(args.steps)
        completed_steps = 0
        timeout_retry_count = 0
        while completed_steps < requested_steps:
            os.environ["EPM_EPISODE_STEP"] = str(int(agent.step_id))
            try:
                obs = agent.run_one_step()
                network_accumulated_wait_s = 0
                timeout_retry_count = 0
            except NetworkPauseRequired as e:
                if _is_timeout_network_issue(e):
                    action, timeout_retry_count = _handle_timeout_retry_or_abort(
                        e,
                        timeout_retry_count=timeout_retry_count,
                    )
                    if action == "retry":
                        continue
                    return _finalize_run(2)
                if not network_pause_enabled:
                    detail = _network_error_detail(
                        e,
                        run_name=run_name,
                        dish_id=dish_id,
                        episode_step=int(getattr(agent, "step_id", 0)),
                        last_completed_step_id=int(getattr(agent, "last_completed_step_id", 0)),
                        last_physical_step_id=int(getattr(agent, "last_physical_step_id", 0)),
                    )
                    log.error(f"[EPM] network_pause_disabled {detail}")
                    _write_resume_state(
                        memory_dir=memory_dir,
                        run_name=run_name,
                        dish_id=dish_id,
                        last_step_id=agent.step_id,
                        agent=agent,
                        status="fatal",
                        reason=f"network_pause_disabled:{detail}",
                        last_network_resume_step_id=(
                            int(getattr(agent, "last_completed_step_id", 0))
                            or int(getattr(agent, "last_physical_step_id", 0))
                            or int(agent.step_id)
                        ),
                        last_http_403_resume_step_id=agent.step_id,
                    )
                    return _finalize_run(2)
                should_continue, network_accumulated_wait_s = _handle_network_pause(
                    log=log,
                    memory_dir=memory_dir,
                    run_name=run_name,
                    dish_id=dish_id,
                    agent=agent,
                    err=e,
                    accumulated_wait_s=network_accumulated_wait_s,
                    wait_s_default=network_wait_s,
                    max_wait_s=network_max_wait_s,
                    retry_attempt_limit=network_retry_attempts,
                    api_key_pool_manager=api_key_pool_manager,
                    api_key_rotation_enabled=api_key_rotation_enabled,
                    api_key_rotation_after_http_403_wait_s=api_key_rotation_after_http_403_wait_s,
                )
                if not should_continue:
                    return _finalize_run(2)
                continue
            except NetworkAbortRequired as e:
                if _is_timeout_network_issue(e):
                    action, timeout_retry_count = _handle_timeout_retry_or_abort(
                        e,
                        timeout_retry_count=timeout_retry_count,
                    )
                    if action == "retry":
                        continue
                    return _finalize_run(2)
                detail = _network_error_detail(
                    e,
                    run_name=run_name,
                    dish_id=dish_id,
                    episode_step=int(getattr(agent, "step_id", 0)),
                    last_completed_step_id=int(getattr(agent, "last_completed_step_id", 0)),
                    last_physical_step_id=int(getattr(agent, "last_physical_step_id", 0)),
                )
                if (
                    api_key_rotation_enabled
                    and api_key_pool_manager is not None
                    and int(getattr(e, "http_status", 0) or 0) == 401
                ):
                    rotated, rotate_detail = api_key_pool_manager.maybe_rotate_after_http_401(err=e, log=log)
                    if rotated:
                        log.warning(f"[EPM] network_abort_http_401_rotated {rotate_detail}")
                        continue
                    if rotate_detail:
                        detail = f"{detail} | {rotate_detail}"
                log.error(f"[EPM] network_abort {detail}")
                _write_resume_state(
                    memory_dir=memory_dir,
                    run_name=run_name,
                    dish_id=dish_id,
                    last_step_id=(
                        int(getattr(agent, "last_physical_step_id", 0))
                        or int(getattr(agent, "last_completed_step_id", 0))
                        or int(agent.step_id)
                    ),
                    agent=agent,
                    status="fatal",
                    reason=f"network_abort:{detail}",
                    last_network_resume_step_id=(
                        int(getattr(agent, "last_completed_step_id", 0))
                        or int(getattr(agent, "last_physical_step_id", 0))
                        or int(agent.step_id)
                    ),
                    last_network_waited_s=network_accumulated_wait_s,
                )
                return _finalize_run(2)
            except StopRequested:
                log.warning("[EPM] stopped=true reason=global_stop_hotkey")
                _write_resume_state(
                    memory_dir=memory_dir,
                    run_name=run_name,
                    dish_id=dish_id,
                    last_step_id=agent.step_id,
                    agent=agent,
                    status="stopped",
                    reason="global_stop_hotkey",
                )
                ended_status = "stopped"
                break
            except RuntimeError as e:
                if str(e).startswith(("fatal_episode_error:", "prompt_ablation_violation")):
                    log.error(f"[EPM] fatal=true error={e}")
                    _write_resume_state(
                        memory_dir=memory_dir,
                        run_name=run_name,
                        dish_id=dish_id,
                        last_step_id=agent.step_id,
                        agent=agent,
                        status="fatal",
                        reason=str(e),
                    )
                    return _finalize_run(2)
                raise
            except Exception as e:
                log.exception(f"[EPM] fatal=true unhandled_exception={type(e).__name__}:{e}")
                try:
                    _atomic_write_text(memory_dir / "fatal_traceback.txt", traceback.format_exc().rstrip() + "\n")
                except Exception as trace_err:
                    log.warning(f"[EPM] fatal_traceback_write_failed error={trace_err!r}")
                _write_resume_state(
                    memory_dir=memory_dir,
                    run_name=run_name,
                    dish_id=dish_id,
                    last_step_id=agent.step_id,
                    agent=agent,
                    status="fatal",
                    reason=f"unhandled_exception:{type(e).__name__}:{e}",
                )
                return _finalize_run(2)
            _write_resume_state(
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                last_step_id=agent.step_id,
                agent=agent,
                status="running",
                last_time_iso=getattr(obs, "time", "") or "",
            )
            _maybe_send_low_balance_alert(
                runtime=settings.runtime,
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                current_step_id=int(agent.step_id),
                log=log,
            )
            completed_steps += 1
        if not ended_status:
            _write_resume_state(
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                last_step_id=agent.step_id,
                agent=agent,
                status="steps_limit",
                reason="steps_completed",
            )
        return _finalize_run(0)

    # Default: run until done (or until max_steps).
    max_steps = int(args.max_steps) if args.max_steps is not None else int(settings.runtime.max_steps)
    completed_steps = 0
    last_obs: Any | None = None
    timeout_retry_count = 0
    while completed_steps < max_steps:
        os.environ["EPM_EPISODE_STEP"] = str(int(agent.step_id))
        try:
            obs = agent.run_one_step()
            last_obs = obs
            network_accumulated_wait_s = 0
            timeout_retry_count = 0
        except NetworkPauseRequired as e:
            if _is_timeout_network_issue(e):
                action, timeout_retry_count = _handle_timeout_retry_or_abort(
                    e,
                    timeout_retry_count=timeout_retry_count,
                )
                if action == "retry":
                    continue
                return _finalize_run(2)
            if not network_pause_enabled:
                detail = _network_error_detail(
                    e,
                    run_name=run_name,
                    dish_id=dish_id,
                    episode_step=int(getattr(agent, "step_id", 0)),
                    last_completed_step_id=int(getattr(agent, "last_completed_step_id", 0)),
                    last_physical_step_id=int(getattr(agent, "last_physical_step_id", 0)),
                )
                log.error(f"[EPM] network_pause_disabled {detail}")
                _write_resume_state(
                    memory_dir=memory_dir,
                    run_name=run_name,
                    dish_id=dish_id,
                    last_step_id=agent.step_id,
                    agent=agent,
                    status="fatal",
                    reason=f"network_pause_disabled:{detail}",
                    last_network_resume_step_id=(
                        int(getattr(agent, "last_completed_step_id", 0))
                        or int(getattr(agent, "last_physical_step_id", 0))
                        or int(agent.step_id)
                    ),
                    last_http_403_resume_step_id=agent.step_id,
                )
                return _finalize_run(2)
            should_continue, network_accumulated_wait_s = _handle_network_pause(
                log=log,
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                agent=agent,
                err=e,
                accumulated_wait_s=network_accumulated_wait_s,
                wait_s_default=network_wait_s,
                max_wait_s=network_max_wait_s,
                retry_attempt_limit=network_retry_attempts,
                api_key_pool_manager=api_key_pool_manager,
                api_key_rotation_enabled=api_key_rotation_enabled,
                api_key_rotation_after_http_403_wait_s=api_key_rotation_after_http_403_wait_s,
            )
            if not should_continue:
                return _finalize_run(2)
            continue
        except NetworkAbortRequired as e:
            if _is_timeout_network_issue(e):
                action, timeout_retry_count = _handle_timeout_retry_or_abort(
                    e,
                    timeout_retry_count=timeout_retry_count,
                )
                if action == "retry":
                    continue
                return _finalize_run(2)
            detail = _network_error_detail(
                e,
                run_name=run_name,
                dish_id=dish_id,
                episode_step=int(getattr(agent, "step_id", 0)),
                last_completed_step_id=int(getattr(agent, "last_completed_step_id", 0)),
                last_physical_step_id=int(getattr(agent, "last_physical_step_id", 0)),
            )
            if (
                api_key_rotation_enabled
                and api_key_pool_manager is not None
                and int(getattr(e, "http_status", 0) or 0) == 401
            ):
                rotated, rotate_detail = api_key_pool_manager.maybe_rotate_after_http_401(err=e, log=log)
                if rotated:
                    log.warning(f"[EPM] network_abort_http_401_rotated {rotate_detail}")
                    continue
                if rotate_detail:
                    detail = f"{detail} | {rotate_detail}"
            log.error(f"[EPM] network_abort {detail}")
            _write_resume_state(
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                last_step_id=(
                    int(getattr(agent, "last_completed_step_id", 0))
                    or int(getattr(agent, "last_physical_step_id", 0))
                    or int(agent.step_id)
                ),
                agent=agent,
                status="fatal",
                reason=f"network_abort:{detail}",
                last_network_resume_step_id=(
                    int(getattr(agent, "last_completed_step_id", 0))
                    or int(getattr(agent, "last_physical_step_id", 0))
                    or int(agent.step_id)
                ),
                last_network_waited_s=network_accumulated_wait_s,
            )
            return _finalize_run(2)
        except StopRequested:
            log.warning("[EPM] stopped=true reason=global_stop_hotkey")
            _write_resume_state(
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                last_step_id=agent.step_id,
                agent=agent,
                status="stopped",
                reason="global_stop_hotkey",
            )
            break
        except RuntimeError as e:
            if str(e).startswith(("fatal_episode_error:", "prompt_ablation_violation")):
                log.error(f"[EPM] fatal=true error={e}")
                _write_resume_state(
                    memory_dir=memory_dir,
                    run_name=run_name,
                    dish_id=dish_id,
                    last_step_id=agent.step_id,
                    agent=agent,
                    status="fatal",
                    reason=str(e),
                )
                return _finalize_run(2)
            raise
        except Exception as e:
            log.exception(f"[EPM] fatal=true unhandled_exception={type(e).__name__}:{e}")
            _write_resume_state(
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                last_step_id=agent.step_id,
                agent=agent,
                status="fatal",
                reason=f"unhandled_exception:{type(e).__name__}:{e}",
            )
            return _finalize_run(2)
        _write_resume_state(
            memory_dir=memory_dir,
            run_name=run_name,
            dish_id=dish_id,
            last_step_id=agent.step_id,
            agent=agent,
            status="running",
            last_time_iso=getattr(obs, "time", "") or "",
        )
        _maybe_send_low_balance_alert(
            runtime=settings.runtime,
            memory_dir=memory_dir,
            run_name=run_name,
            dish_id=dish_id,
            current_step_id=int(agent.step_id),
            log=log,
        )
        completed_steps += 1
        if agent.is_done(obs):
            done_reason = ""
            try:
                done_reason = str(agent.get_done_reason() or "").strip()
            except Exception:
                done_reason = ""
            log.info(f"[EPM] done=true reason={done_reason or 'done_state_detected'}")
            _write_resume_state(
                memory_dir=memory_dir,
                run_name=run_name,
                dish_id=dish_id,
                last_step_id=agent.step_id,
                agent=agent,
                status="done",
                reason=(done_reason or "done_state_detected"),
                last_time_iso=getattr(obs, "time", "") or "",
            )
            break
    else:
        log.warning(
            f"[EPM] stopped=true reason=max_steps_reached max_steps={max_steps} "
            f"completed_steps={completed_steps} last_step_id={agent.step_id}"
        )
        _write_resume_state(
            memory_dir=memory_dir,
            run_name=run_name,
            dish_id=dish_id,
            last_step_id=agent.step_id,
            agent=agent,
            status="max_steps",
            reason="max_steps_reached",
            last_time_iso=getattr(last_obs, "time", "") or "",
        )
        return _finalize_run(1)
    return _finalize_run(0)


if __name__ == "__main__":
    raise SystemExit(main())
