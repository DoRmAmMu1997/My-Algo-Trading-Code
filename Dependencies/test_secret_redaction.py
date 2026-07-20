"""Canary tests for MAT-108 credential-safe diagnostics."""

from __future__ import annotations

import io
import logging

from Dependencies.secret_redaction import (
    REDACTED,
    RedactingFilter,
    install_redaction_filter,
    redact_payload,
    redact_text,
)


def test_recursive_redaction_covers_broker_and_identity_fields():
    payload = {
        "stat": "Ok",
        "data": {"jKey": "CANARY-JKEY", "password": "CANARY-PASSWORD"},
        "authorization": "Bearer CANARY-AUTH",
        "profile": [{"pan": "ABCDE1234F", "dob": "1990-01-01"}],
    }
    safe = redact_payload(payload)
    rendered = repr(safe)
    assert "CANARY" not in rendered
    assert "ABCDE1234F" not in rendered
    assert rendered.count(REDACTED) >= 5


def test_telegram_exception_url_is_redacted():
    text = redact_text("POST https://api.telegram.org/bot123456:CANARY/sendMessage failed")
    assert "123456:CANARY" not in text
    assert "bot<redacted>/sendMessage" in text


def test_debug_and_exception_records_never_emit_canary_secrets():
    secret = "CANARY-SUPER-SECRET"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter([secret]))
    logger = logging.getLogger("mat108.canary")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    logger.debug("request=%s", {"jKey": secret, "password": secret})
    try:
        raise RuntimeError(f"authorization=Bearer-{secret}")
    except RuntimeError:
        logger.exception("broker request failed for token=%s", secret)

    output = stream.getvalue()
    assert secret not in output
    assert REDACTED in output


def test_redaction_reaches_secrets_inside_tuples_and_sets():
    """Broker payloads can nest sequences; every container type is walked."""
    payload = (
        "password=CANARY-TUPLE",
        {"CANARY-SET-SECRET", "harmless"},
        ["token=CANARY-LIST"],
    )
    safe = redact_payload(payload, ("CANARY-SET-SECRET",))
    rendered = repr(safe)
    assert "CANARY" not in rendered
    assert isinstance(safe, tuple)
    assert isinstance(safe[1], set)


def test_install_redaction_filter_covers_logger_and_existing_handlers():
    """One install call must guard both the logger and its current handlers."""
    secret = "CANARY-INSTALL-SECRET"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("mat110.install.canary")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    install_redaction_filter(logger, [secret])
    logger.debug("session token is %s somewhere in this line", secret)

    output = stream.getvalue()
    assert secret not in output
    assert REDACTED in output
    # The guard sits on the handler too, so records that reach the handler
    # directly (propagated from child loggers) are also scrubbed.
    assert any(isinstance(f, RedactingFilter) for f in handler.filters)
