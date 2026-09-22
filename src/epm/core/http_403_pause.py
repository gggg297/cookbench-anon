from __future__ import annotations

import http.client
import json
import re
import socket
import ssl
import urllib.error
from email.utils import parsedate_to_datetime
from typing import Any, Optional

try:
    import requests
except Exception:  # pragma: no cover - requests is available in normal runtime
    requests = None  # type: ignore[assignment]


_RETRYABLE_HTTP_STATUS = {403, 408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524}


def _trim_text(text: Any, *, max_chars: int = 240) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s)
    if len(s) <= int(max_chars):
        return s
    return s[: max(0, int(max_chars) - 3)] + "..."


def _iter_exception_chain(exc: BaseException):
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        yield cur
        if isinstance(cur, urllib.error.URLError):
            reason = getattr(cur, "reason", None)
            if isinstance(reason, BaseException) and id(reason) not in seen:
                yield reason
                seen.add(id(reason))
        nxt = cur.__cause__ if cur.__cause__ is not None else cur.__context__
        cur = nxt if isinstance(nxt, BaseException) else None


def extract_http_status(exc: BaseException) -> int | None:
    for cur in _iter_exception_chain(exc):
        if isinstance(cur, urllib.error.HTTPError):
            try:
                return int(cur.code)
            except Exception:
                return None
        if requests is not None and isinstance(cur, requests.HTTPError):
            resp = getattr(cur, "response", None)
            try:
                status = getattr(resp, "status_code", None)
                return int(status) if status is not None else None
            except Exception:
                return None
    return None


def extract_http_reason(exc: BaseException) -> str:
    cached = getattr(exc, "_epm_http_reason", None)
    if isinstance(cached, str) and cached.strip():
        return cached.strip()
    for cur in _iter_exception_chain(exc):
        if isinstance(cur, urllib.error.HTTPError):
            try:
                return str(cur.reason or "").strip()
            except Exception:
                return ""
        if requests is not None and isinstance(cur, requests.HTTPError):
            try:
                resp = getattr(cur, "response", None)
                return str(getattr(resp, "reason", "") or "").strip()
            except Exception:
                return ""
    return ""


def extract_http_body_preview(exc: BaseException, *, max_chars: int = 240) -> str:
    cached = getattr(exc, "_epm_http_body_preview", None)
    if isinstance(cached, str) and cached.strip():
        return _trim_text(cached, max_chars=max_chars)
    for cur in _iter_exception_chain(exc):
        if isinstance(cur, urllib.error.HTTPError):
            try:
                body = cur.read()
            except Exception:
                body = b""
            if isinstance(body, bytes):
                text = body.decode("utf-8", errors="replace")
            else:
                text = str(body or "")
            text = _trim_text(text, max_chars=max_chars)
            if text:
                try:
                    setattr(exc, "_epm_http_body_preview", text)
                except Exception:
                    pass
            return text
        if requests is not None and isinstance(cur, requests.HTTPError):
            try:
                resp = getattr(cur, "response", None)
                text = _trim_text(getattr(resp, "text", "") or "", max_chars=max_chars)
            except Exception:
                text = ""
            if text:
                try:
                    setattr(exc, "_epm_http_body_preview", text)
                except Exception:
                    pass
            return text
    return ""


def _extract_http_error_hint(exc: BaseException) -> str:
    body = extract_http_body_preview(exc, max_chars=240)
    if not body:
        return ""
    try:
        obj = json.loads(body)
    except Exception:
        return body
    if isinstance(obj, dict):
        err = obj.get("error")
        if isinstance(err, dict):
            parts: list[str] = []
            msg = _trim_text(err.get("message"), max_chars=160)
            code = _trim_text(err.get("code"), max_chars=64)
            typ = _trim_text(err.get("type"), max_chars=64)
            if code:
                parts.append(f"code={code}")
            if typ:
                parts.append(f"type={typ}")
            if msg:
                parts.append(f"message={msg}")
            if parts:
                return "; ".join(parts)
    return body


def _format_network_exception_message(*, prefix: str, exc: BaseException) -> str:
    parts: list[str] = [prefix]
    status = extract_http_status(exc)
    if status is not None:
        parts.append(f"status={status}")
    reason = extract_http_reason(exc)
    if reason:
        parts.append(f"reason={reason}")
    retry_after_s = extract_retry_after_s(exc)
    if isinstance(retry_after_s, int) and retry_after_s > 0:
        parts.append(f"retry_after_s={retry_after_s}")
    hint = _extract_http_error_hint(exc)
    if hint:
        parts.append(f"detail={hint}")
    else:
        parts.append(f"error={type(exc).__name__}:{exc}")
    return " | ".join(parts)


def extract_retry_after_s(exc: BaseException) -> int | None:
    for cur in _iter_exception_chain(exc):
        if isinstance(cur, urllib.error.HTTPError):
            try:
                raw = cur.headers.get("Retry-After")
            except Exception:
                raw = None
            text = str(raw or "").strip()
            if not text:
                return None
            if text.isdigit():
                return max(1, int(text))
            try:
                dt = parsedate_to_datetime(text)
                if dt is None:
                    return None
                import datetime as _dt

                now = _dt.datetime.now(dt.tzinfo)
                return max(1, int((dt - now).total_seconds()))
            except Exception:
                return None
        if requests is not None and isinstance(cur, requests.HTTPError):
            try:
                resp = getattr(cur, "response", None)
                raw = resp.headers.get("Retry-After") if resp is not None else None
            except Exception:
                raw = None
            text = str(raw or "").strip()
            if not text:
                return None
            if text.isdigit():
                return max(1, int(text))
            try:
                dt = parsedate_to_datetime(text)
                if dt is None:
                    return None
                import datetime as _dt

                now = _dt.datetime.now(dt.tzinfo)
                return max(1, int((dt - now).total_seconds()))
            except Exception:
                return None
    return None


def is_http_403_error(exc: BaseException) -> bool:
    return extract_http_status(exc) == 403


def _network_error_kind(exc: BaseException) -> str | None:
    status = extract_http_status(exc)
    if status is not None:
        return f"http_{status}"
    for cur in _iter_exception_chain(exc):
        if isinstance(cur, urllib.error.URLError):
            reason = getattr(cur, "reason", None)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                return "timeout"
            if isinstance(reason, ssl.SSLError):
                return "ssl"
            if isinstance(reason, ConnectionResetError):
                return "connection_reset"
            if isinstance(reason, ConnectionRefusedError):
                return "connection_refused"
            if isinstance(reason, socket.gaierror):
                return "dns"
            return "url"
        if isinstance(cur, (TimeoutError, socket.timeout)):
            return "timeout"
        if isinstance(cur, ssl.SSLError):
            return "ssl"
        if isinstance(cur, ConnectionResetError):
            return "connection_reset"
        if isinstance(cur, ConnectionAbortedError):
            return "connection_aborted"
        if isinstance(cur, ConnectionRefusedError):
            return "connection_refused"
        if isinstance(cur, socket.gaierror):
            return "dns"
        if isinstance(cur, http.client.RemoteDisconnected):
            return "remote_disconnected"
        if requests is not None:
            if isinstance(cur, requests.exceptions.Timeout):
                return "timeout"
            if isinstance(cur, requests.exceptions.ConnectionError):
                return "connection_error"
            if isinstance(cur, requests.exceptions.SSLError):
                return "ssl"
            if isinstance(cur, requests.exceptions.RequestException):
                return "request"
    return None


def is_network_error(exc: BaseException) -> bool:
    return _network_error_kind(exc) is not None


def is_retryable_network_error(exc: BaseException) -> bool:
    status = extract_http_status(exc)
    if status is not None:
        return int(status) in _RETRYABLE_HTTP_STATUS
    kind = _network_error_kind(exc)
    return kind in {
        "timeout",
        "connection_reset",
        "connection_aborted",
        "remote_disconnected",
        "connection_error",
    }


class NetworkPauseRequired(RuntimeError):
    def __init__(
        self,
        *,
        source: str,
        message: str,
        kind: str = "",
        http_status: Optional[int] = None,
        retry_after_s: Optional[int] = None,
        http_reason: str = "",
        http_body_preview: str = "",
    ) -> None:
        self.source = str(source or "").strip() or "unknown"
        self.kind = str(kind or "").strip()
        self.http_status = int(http_status) if isinstance(http_status, int) else None
        self.retry_after_s = int(retry_after_s) if isinstance(retry_after_s, int) and retry_after_s > 0 else None
        self.http_reason = str(http_reason or "").strip()
        self.http_body_preview = _trim_text(http_body_preview, max_chars=240)
        super().__init__(str(message or "network_pause_required"))

    @classmethod
    def from_exception(cls, exc: BaseException, *, source: str) -> "NetworkPauseRequired":
        return cls(
            source=source,
            kind=str(_network_error_kind(exc) or ""),
            http_status=extract_http_status(exc),
            message=_format_network_exception_message(prefix="network_pause_required", exc=exc),
            retry_after_s=extract_retry_after_s(exc),
            http_reason=extract_http_reason(exc),
            http_body_preview=extract_http_body_preview(exc),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "http_status": self.http_status,
            "retry_after_s": self.retry_after_s,
            "http_reason": self.http_reason,
            "http_body_preview": self.http_body_preview,
            "message": str(self),
        }


class NetworkAbortRequired(RuntimeError):
    def __init__(
        self,
        *,
        source: str,
        message: str,
        kind: str = "",
        http_status: Optional[int] = None,
        http_reason: str = "",
        http_body_preview: str = "",
    ) -> None:
        self.source = str(source or "").strip() or "unknown"
        self.kind = str(kind or "").strip()
        self.http_status = int(http_status) if isinstance(http_status, int) else None
        self.http_reason = str(http_reason or "").strip()
        self.http_body_preview = _trim_text(http_body_preview, max_chars=240)
        super().__init__(str(message or "network_abort_required"))

    @classmethod
    def from_exception(cls, exc: BaseException, *, source: str) -> "NetworkAbortRequired":
        return cls(
            source=source,
            kind=str(_network_error_kind(exc) or ""),
            http_status=extract_http_status(exc),
            message=_format_network_exception_message(prefix="network_abort_required", exc=exc),
            http_reason=extract_http_reason(exc),
            http_body_preview=extract_http_body_preview(exc),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "kind": self.kind,
            "http_status": self.http_status,
            "http_reason": self.http_reason,
            "http_body_preview": self.http_body_preview,
            "message": str(self),
        }


class Http403PauseRequired(NetworkPauseRequired):
    @classmethod
    def from_exception(cls, exc: BaseException, *, source: str) -> "Http403PauseRequired":
        return cls(
            source=source,
            kind="http_403",
            http_status=403,
            message=_format_network_exception_message(prefix="http_403_pause_required", exc=exc),
            retry_after_s=extract_retry_after_s(exc),
            http_reason=extract_http_reason(exc),
            http_body_preview=extract_http_body_preview(exc),
        )


def classify_network_exception(exc: BaseException, *, source: str) -> RuntimeError | None:
    if isinstance(exc, (NetworkPauseRequired, NetworkAbortRequired)):
        return exc
    if not is_network_error(exc):
        return None
    if is_retryable_network_error(exc):
        if extract_http_status(exc) == 403:
            return Http403PauseRequired.from_exception(exc, source=source)
        return NetworkPauseRequired.from_exception(exc, source=source)
    return NetworkAbortRequired.from_exception(exc, source=source)


def attach_network_context(
    exc: RuntimeError | None,
    *,
    api_key_env: str = "",
    api_key_pool_env: str = "",
    api_key_masked: str = "",
    api_key_slot: str = "",
    base_url: str = "",
    model: str = "",
    provider: str = "",
    channel: str = "",
) -> RuntimeError | None:
    if exc is None:
        return None
    if api_key_env:
        setattr(exc, "api_key_env", str(api_key_env))
    if api_key_pool_env:
        setattr(exc, "api_key_pool_env", str(api_key_pool_env))
    if api_key_masked:
        setattr(exc, "api_key_masked", str(api_key_masked))
    if api_key_slot:
        setattr(exc, "api_key_slot", str(api_key_slot))
    if base_url:
        setattr(exc, "request_base_url", str(base_url))
    if model:
        setattr(exc, "request_model", str(model))
    if provider:
        setattr(exc, "request_provider", str(provider))
    if channel:
        setattr(exc, "request_channel", str(channel))
    return exc
