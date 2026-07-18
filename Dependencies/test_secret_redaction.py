"""Canary tests for MAT-108 credential-safe diagnostics."""

from __future__ import annotations

import io
import logging

from Dependencies.secret_redaction import REDACTED, RedactingFilter, redact_payload, redact_text


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
