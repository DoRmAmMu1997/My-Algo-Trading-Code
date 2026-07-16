"""Shared fail-safe redaction for broker, OAuth, and notifier diagnostics."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "<redacted>"
_SENSITIVE_PARTS = (
    "token", "jkey", "password", "passwd", "mpin", "totp", "secret",
    "appkey", "app_key", "api_key", "apikey", "authorization", "cookie",
    "session", "pan", "dob", "birth", "client_id", "clientid",
)
_ASSIGNMENT_RE = re.compile(
    r"(?i)(token|jkey|password|passwd|mpin|totp|secret|app[_-]?key|api[_-]?key|"
    r"authorization|session|pan|dob)(\s*[\"']?\s*[:=]\s*[\"']?)([^\s,;&}\"']+)"
)
_TELEGRAM_URL_RE = re.compile(r"(?i)(api\.telegram\.org/bot)[^/\s]+")


def _sensitive_key(key: object) -> bool:
    normalized = re.sub(r"[^a-z0-9_]", "", str(key).lower())
    return any(part in normalized for part in _SENSITIVE_PARTS)


def redact_text(value: object, secrets: Iterable[str] = ()) -> str:
    """Redact credential assignments, Telegram URL tokens, and known secrets."""
    text = str(value)
    for secret in sorted({str(item) for item in secrets if str(item)}, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    text = _TELEGRAM_URL_RE.sub(r"\1<redacted>", text)
    return _ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", text)


def redact_payload(value: Any, secrets: Iterable[str] = ()) -> Any:
    """Recursively return a log-safe copy without mutating the broker response."""
    if isinstance(value, Mapping):
        return {
            key: REDACTED if _sensitive_key(key) else redact_payload(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(redact_payload(item, secrets) for item in value)
    if isinstance(value, list):
        return [redact_payload(item, secrets) for item in value]
    if isinstance(value, set):
        return {redact_payload(item, secrets) for item in value}
    if isinstance(value, str):
        return redact_text(value, secrets)
    return value


class RedactingFilter(logging.Filter):
    """Last-line log guard, including DEBUG records and exception strings."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(str(secret) for secret in secrets if str(secret))

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(record.msg, self._secrets)
        if record.args:
            record.args = redact_payload(record.args, self._secrets)
        if record.exc_info:
            # Format now, redact, and clear exc_info so logging cannot append the
            # original secret-bearing exception again after this filter returns.
            formatter = logging.Formatter()
            record.msg = f"{record.msg}\n{redact_text(formatter.formatException(record.exc_info), self._secrets)}"
            record.exc_info = None
            record.exc_text = None
        return True


def install_redaction_filter(logger: logging.Logger, secrets: Iterable[str] = ()) -> None:
    """Install one filter on a logger and all of its current handlers."""
    guard = RedactingFilter(secrets)
    logger.addFilter(guard)
    for handler in logger.handlers:
        handler.addFilter(guard)
