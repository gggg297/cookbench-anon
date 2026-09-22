from __future__ import annotations

import os
import smtplib
from email.utils import formataddr
from email.message import EmailMessage
from typing import Any


def _split_recipients(raw: str) -> list[str]:
    text = str(raw or "")
    if not text.strip():
        return []
    out: list[str] = []
    for part in text.replace(";", ",").split(","):
        addr = str(part or "").strip()
        if addr:
            out.append(addr)
    return out


def maybe_send_email_notification(*, runtime: Any, subject: str, body: str, log: Any | None = None) -> bool:
    if not bool(getattr(runtime, "notify_email_enabled", False)):
        return False
    host = str(getattr(runtime, "notify_email_smtp_host", "") or "").strip()
    port = int(getattr(runtime, "notify_email_smtp_port", 587) or 587)
    nickname = str(getattr(runtime, "notify_email_username", "") or "").strip()
    from_addr = str(getattr(runtime, "notify_email_from", "") or "").strip() or nickname
    password_env = str(getattr(runtime, "notify_email_password_env", "") or "").strip()
    recipients = _split_recipients(str(getattr(runtime, "notify_email_to", "") or ""))
    use_tls = bool(getattr(runtime, "notify_email_use_tls", True))
    password = str(os.environ.get(password_env, "") or "").strip() if password_env else ""
    login_username = from_addr

    if not host or not from_addr or not recipients:
        if log is not None:
            log.warning("[EPM] email_notify skipped missing_config")
        return False

    msg = EmailMessage()
    msg["Subject"] = str(subject or "EPM notification")
    msg["From"] = formataddr((nickname, from_addr)) if nickname else from_addr
    msg["To"] = ", ".join(recipients)
    msg.set_content(str(body or ""), subtype="plain", charset="utf-8")

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.ehlo()
            if use_tls:
                server.starttls()
                server.ehlo()
            if login_username:
                server.login(login_username, password)
            server.send_message(msg)
        if log is not None:
            log.info(f"[EPM] email_notify sent to={','.join(recipients)} subject={subject!r}")
        return True
    except Exception as e:
        if log is not None:
            log.warning(f"[EPM] email_notify failed err={e!r}")
        return False
