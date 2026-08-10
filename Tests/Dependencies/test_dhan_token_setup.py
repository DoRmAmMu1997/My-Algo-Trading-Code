"""Tests for the token-expiry advisory in the Dhan OAuth setup script.

The three weekday messages steer a live operator's next action, so each case
is locked in: mid-session expiry warns loudly, pre-open expiry gets the calmer
"session not covered" note, and post-close expiry is reported as the good
outcome.  Timestamps are built from the LOCAL clock (the same clock the helper
renders with), which keeps the tests deterministic on an IST box and on a UTC
CI runner alike.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta

from Dependencies.dhan_token_setup import _describe_token_expiry


def _token_with_exp(expires_at: datetime) -> str:
    """Build a JWT-shaped token whose middle segment carries ``exp``."""
    payload = (
        base64.urlsafe_b64encode(
            json.dumps({"exp": int(expires_at.timestamp())}).encode("utf-8")
        )
        .decode("ascii")
        .rstrip("=")
    )
    return f"header.{payload}.signature"


def _upcoming_weekday_at(hour: int, minute: int, *, weekday: int = 0) -> datetime:
    """Return the NEXT occurrence of ``weekday`` (0=Monday) at HH:MM local time."""
    now = datetime.now().astimezone()
    days_ahead = (weekday - now.weekday()) % 7 or 7
    return (now + timedelta(days=days_ahead)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )


def test_mid_session_weekday_expiry_warns_during_market_hours():
    expiry = _upcoming_weekday_at(11, 0)
    text = _describe_token_expiry(_token_with_exp(expiry))
    assert "DURING market hours" in text


def test_pre_open_weekday_expiry_notes_uncovered_session_not_market_hours():
    expiry = _upcoming_weekday_at(8, 0)
    text = _describe_token_expiry(_token_with_exp(expiry))
    assert "DURING market hours" not in text
    assert "BEFORE that day's 09:15 open" in text


def test_post_close_weekday_expiry_reports_the_session_as_covered():
    expiry = _upcoming_weekday_at(17, 0)
    text = _describe_token_expiry(_token_with_exp(expiry))
    assert "DURING market hours" not in text
    assert "covers that" in text


def test_weekend_expiry_carries_no_session_message():
    expiry = _upcoming_weekday_at(11, 0, weekday=6)  # next Sunday, mid-day
    text = _describe_token_expiry(_token_with_exp(expiry))
    assert text.startswith("Token expires ")
    assert "\n" not in text


def test_non_jwt_token_degrades_to_an_unknown_validity_note():
    assert "unknown" in _describe_token_expiry("not-a-jwt")
