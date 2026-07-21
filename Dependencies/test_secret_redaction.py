"""Canary tests for MAT-108 credential-safe diagnostics."""

from __future__ import annotations

import io
import logging

from Dependencies.secret_redaction import (
    REDACTED,
    RedactingFilter,
    environment_secrets,
    install_redaction_filter,
    redact_payload,
    redact_text,
)


class TestEnvironmentSecrets:
    """The runner's log guard derives its secret list from the environment, so
    a newly added broker's credentials cannot be forgotten."""

    def test_collects_credential_values_by_key_name(self):
        environ = {
            "DHAN_ACCESS_TOKEN": "eyJhbGciOiJIUzUxMiJ9.CANARY-TOKEN",
            "TELEGRAM_BOT_TOKEN": "1234567890:CANARY-BOT-TOKEN",
            "SHOONYA_TOTP_SECRET": "CANARYTOTPSECRET",
            "FLATTRADE_API_SECRET": "CANARY-FLATTRADE-SECRET",
            # Not credential-shaped: must NOT be redacted out of logs.
            "RENKO_LOTS": "1",
            "MARKET_DATA_SOURCE": "WEBSOCKET",
            "LOG_LEVEL": "INFO",
        }
        secrets = environment_secrets(environ)
        assert "eyJhbGciOiJIUzUxMiJ9.CANARY-TOKEN" in secrets
        assert "1234567890:CANARY-BOT-TOKEN" in secrets
        assert "CANARYTOTPSECRET" in secrets
        assert "CANARY-FLATTRADE-SECRET" in secrets
        assert "WEBSOCKET" not in secrets
        assert "1" not in secrets

    def test_short_values_are_skipped_so_logs_stay_readable(self):
        """A 4-digit MPIN would otherwise blank every matching digit run in the
        log -- including strike prices and quantities."""
        environ = {"MPIN": "1234", "DHAN_API_SECRET": "long-enough-secret"}
        secrets = environment_secrets(environ)
        assert "1234" not in secrets
        assert "long-enough-secret" in secrets
        # ... but the assignment pattern still catches it in free text.
        assert "1234" not in redact_text("mpin=1234 submitted", secrets)

    def test_blank_and_whitespace_values_are_ignored(self):
        environ = {"DHAN_ACCESS_TOKEN": "", "DHAN_API_SECRET": "        "}
        assert environment_secrets(environ) == ()

    def test_longest_first_so_no_recognizable_fragment_survives(self):
        environ = {"A_TOKEN": "abcdefgh", "B_TOKEN": "abcdefghijkl"}
        assert environment_secrets(environ) == ("abcdefghijkl", "abcdefgh")

    def test_min_length_is_configurable_and_floored_at_one(self):
        environ = {"X_SECRET": "abc"}
        assert environment_secrets(environ, min_length=3) == ("abc",)
        assert environment_secrets(environ, min_length=0) == ("abc",)


def test_exception_objects_are_redacted_not_passed_through():
    """`log.warning("...: %s", exc)` is this codebase's standard failure shape.

    An exception is not a str/Mapping/list, so it used to fall through to the
    untouched-leaf return -- and logging then rendered it with %s AFTER every
    filter had run, leaking whatever its message carried.
    """
    secret = "CANARY-ACCESS-TOKEN"
    exc = OSError(f"connect failed: wss://host?token={secret}")

    safe = redact_payload((1.0, exc), (secret,))

    assert secret not in repr(safe)
    assert isinstance(safe[1], str), "an exception leaf must become a redacted string"
    assert REDACTED in safe[1]
    # Non-secret leaves are still passed through untouched.
    assert safe[0] == 1.0


def test_installed_filter_scrubs_the_dhan_websocket_token_from_an_exception():
    """Regression for the leak found reviewing the websocket producer.

    dhanhq's marketfeed builds `wss://api-feed.dhan.co?version=2&token=<TOKEN>
    &clientId=...`, so a connection error can carry the LIVE trading token in
    its message. The runner logs those exceptions on the reconnect path, and
    this log file is routinely shared when diagnosing a session.
    """
    token = "eyJhbGciOiJIUzUxMiJ9.CANARY-LIVE-TOKEN.signature"
    url = f"wss://api-feed.dhan.co?version=2&token={token}&clientId=1100000000&authType=2"

    stream = io.StringIO()
    logger = logging.getLogger("dhan_ws_leak_canary")
    logger.handlers.clear()
    logger.propagate = False
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    install_redaction_filter(logger, environment_secrets({"DHAN_ACCESS_TOKEN": token}))

    # 1. The shape the reconnect loop logs: exception rendered via %s.
    logger.warning("Websocket feed error (reconnect in %.0fs): %s", 1.0, OSError(url))
    # 2. The shape log.exception() produces: a full traceback.
    try:
        raise ConnectionError(f"failed to connect to {url}")
    except ConnectionError:
        logger.exception("Websocket supervisor cycle error")

    output = stream.getvalue()
    logger.handlers.clear()

    assert token not in output, "live access token leaked into the log"
    assert "CANARY-LIVE-TOKEN" not in output
    assert REDACTED in output
    # The operator can still see WHAT failed and where.
    assert "api-feed.dhan.co" in output
    assert "Websocket feed error" in output


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
