"""Shared fail-safe redaction for broker, OAuth, and notifier diagnostics.

Broker replies, login errors, and Telegram failures are exactly the payloads
an operator most wants to see in a log -- and exactly the payloads most likely
to carry a session token, password, TOTP, or personal identifier.  This module
gives every diagnostic call site one shared way to strip those values before
they can reach a log file or an alert.

Which helper to use when
------------------------
- ``redact_text``    : you have ONE string (an exception message, a response
  body, a URL) and want a safe copy of it.
- ``redact_payload`` : you have a STRUCTURE (the dict/list a broker returned)
  and want a safe deep copy -- sensitive KEYS are blanked wholesale, and every
  string VALUE inside is additionally run through ``redact_text``.
- ``RedactingFilter`` / ``install_redaction_filter`` : a last-line guard you
  can attach to a logger so even records someone forgets to redact are
  scrubbed on their way out (DEBUG lines and exception tracebacks included).

Redaction here deliberately prefers false positives (blanking something
harmless) over false negatives (leaking a credential): a blanked field costs a
little debugging convenience, a leaked token is a live-money security problem.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED = "<redacted>"

# Substrings that mark a dict KEY as sensitive.  Matching is by substring on a
# normalized (lowercased, punctuation-stripped) key, so "jKey", "JKey" and
# "j-key" all hit "jkey", and "accessToken"/"access_token" both hit "token".
# The cost of that looseness is occasional over-redaction (e.g. a key named
# "companyName" contains "pan") -- accepted on purpose, see the module note.
_SENSITIVE_PARTS = (
    "token", "jkey", "password", "passwd", "mpin", "totp", "secret",
    "appkey", "app_key", "api_key", "apikey", "authorization", "cookie",
    "session", "pan", "dob", "birth", "client_id", "clientid",
)

# Credential ASSIGNMENTS embedded in free text.  Matches shapes such as
#   token=abc123        password: "hunter2"      jKey='SESSIONKEY'
# i.e. a known credential word, then ':' or '=' (optionally quoted), then the
# value itself.  Group 1 keeps the name, group 2 keeps the separator, and the
# value (group 3) is replaced -- so the log still shows WHICH field was
# present, just never what it contained.
_ASSIGNMENT_RE = re.compile(
    r"(?i)(token|jkey|password|passwd|mpin|totp|secret|app[_-]?key|api[_-]?key|"
    r"authorization|session|pan|dob)(\s*[\"']?\s*[:=]\s*[\"']?)([^\s,;&}\"']+)"
)

# Telegram bakes the bot token INTO the request URL
# (https://api.telegram.org/bot<token>/sendMessage), so an HTTP exception's
# text leaks it verbatim.  Replace everything between "/bot" and the next "/".
_TELEGRAM_URL_RE = re.compile(r"(?i)(api\.telegram\.org/bot)[^/\s]+")


def _sensitive_key(key: object) -> bool:
    """Return whether a mapping key names credential/identity material."""
    normalized = re.sub(r"[^a-z0-9_]", "", str(key).lower())
    return any(part in normalized for part in _SENSITIVE_PARTS)


def redact_text(value: object, secrets: Iterable[str] = ()) -> str:
    """Return a log-safe string: known secrets, URL tokens, and assignments gone.

    ``secrets`` are the exact values THIS call site already holds (the
    password it sent, the token it used).  They are replaced first, longest
    value first, so a secret that happens to contain a shorter secret cannot
    leave a recognizable fragment behind.  The Telegram-URL and
    assignment-pattern passes then catch credentials the caller did not know
    were embedded in the text.
    """
    text = str(value)
    for secret in sorted({str(item) for item in secrets if str(item)}, key=len, reverse=True):
        text = text.replace(secret, REDACTED)
    text = _TELEGRAM_URL_RE.sub(r"\1<redacted>", text)
    return _ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}", text)


def environment_secrets(
    environ: Mapping[str, str],
    *,
    min_length: int = 8,
) -> tuple[str, ...]:
    """Collect the credential VALUES currently set in the environment.

    Intended for :func:`install_redaction_filter`, so the runner's log guard
    knows every real secret without a hand-maintained list that a newly added
    broker would silently fall off.  Selection is by KEY name using the same
    ``_SENSITIVE_PARTS`` rule the payload redactor uses, so `DHAN_ACCESS_TOKEN`,
    `TELEGRAM_BOT_TOKEN`, `SHOONYA_TOTP_SECRET` and friends are all picked up
    automatically.

    ``min_length`` exists to keep logs readable and is a genuine safety
    trade-off: a 4-digit MPIN would otherwise blank every matching digit
    sequence in the file, including strike prices and quantities, which makes
    the log useless and can itself hide a problem.  Short values are therefore
    left to the assignment-pattern pass in :func:`redact_text`, which still
    catches them when they appear as ``mpin=1234``.
    """
    secrets: set[str] = set()
    for key, value in environ.items():
        if not _sensitive_key(key):
            continue
        text = str(value).strip()
        if len(text) >= max(1, int(min_length)):
            secrets.add(text)
    return tuple(sorted(secrets, key=len, reverse=True))


def redact_payload(value: Any, secrets: Iterable[str] = ()) -> Any:
    """Recursively return a log-safe copy without mutating the broker response.

    Dictionaries get key-based blanking (a key like ``jKey`` loses its whole
    value regardless of what that value looks like); every nested string also
    goes through :func:`redact_text`.  Non-container, non-string leaves
    (numbers, booleans, ``None``) pass through untouched.  The original object
    is never modified -- the caller may still need the real values to trade.
    """
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
    if isinstance(value, BaseException):
        # An exception passed as a LAZY log argument (`log.warning("...: %s",
        # exc)`) is the most common way this codebase surfaces failures, and
        # its text is exactly where a credential-bearing URL shows up. Without
        # this branch the object passed through untouched and logging rendered
        # it with %s AFTER every filter had run -- leaking the secret the guard
        # exists to stop. Return the redacted STRING: the only consumer of a
        # redacted copy is a log line.
        return redact_text(value, secrets)
    return value


class RedactingFilter(logging.Filter):
    """Last-line log guard, including DEBUG records and exception strings.

    Attached to a logger (or handler), this rewrites each record in place
    BEFORE it is formatted: the message text, the lazy ``%s`` arguments, and
    -- most importantly -- any attached exception.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = tuple(str(secret) for secret in secrets if str(secret))

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(record.msg, self._secrets)
        if record.args:
            record.args = redact_payload(record.args, self._secrets)
        if record.exc_info:
            # A raw exception object can carry the offending request/response
            # in its args, and logging would normally format that traceback
            # AFTER all filters have run -- bypassing this guard entirely.
            # Format the traceback NOW, redact the resulting text, fold it
            # into the message, and clear exc_info/exc_text so the logging
            # machinery has nothing secret-bearing left to append.
            formatter = logging.Formatter()
            record.msg = f"{record.msg}\n{redact_text(formatter.formatException(record.exc_info), self._secrets)}"
            record.exc_info = None
            record.exc_text = None
        return True


def install_redaction_filter(logger: logging.Logger, secrets: Iterable[str] = ()) -> None:
    """Install one filter on a logger and all of its current handlers.

    The filter goes on the handlers as well because logger-level filters only
    see records CREATED through that logger -- records propagated up from
    child loggers reach the handlers directly.  Handlers added after this call
    are not covered; install the filter after logging setup is complete.
    """
    guard = RedactingFilter(secrets)
    logger.addFilter(guard)
    for handler in logger.handlers:
        handler.addFilter(guard)
