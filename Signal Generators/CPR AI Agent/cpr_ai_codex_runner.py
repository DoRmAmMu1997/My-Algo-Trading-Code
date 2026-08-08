"""Prepare the deliberately narrow child process used for an optional Codex turn.

The configuration below is data, not a trading integration.  It gives a child
only the four frozen MCP reads and removes ambient credentials so an SDK issue
cannot turn into a shell, web, workspace, or order capability.

The subprocess import is intentional: callers and model input cannot choose
the fixed local interpreter or child script passed to it.
"""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cpr_ai_agent import CPRAgentRunResult, CPRToolCallRecord
from cpr_ai_codex_subprocess import build_isolated_thread_config

_SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "HOMEDRIVE",
        "HOMEPATH",
        "APPDATA",
        "LOCALAPPDATA",
        "CODEX_HOME",
    }
)
_SENSITIVE_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "API_KEY", "CREDENTIAL", "TRADING", "BROKER", "DHAN", "TELEGRAM")


def safe_subprocess_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy only minimal process plumbing and never infer a credential is safe.

    An allowlist is stronger than attempting to recognize every broker's next
    secret name.  It also removes model API keys: the Codex desktop/session
    authentication mechanism must not be exported into a trade process.
    """

    source = os.environ if source is None else source
    return {
        key: value
        for key, value in source.items()
        if key.upper() in _SAFE_ENVIRONMENT_KEYS and not any(marker in key.upper() for marker in _SENSITIVE_MARKERS)
    }


def build_codex_thread_config(snapshot_path: str, python_executable: str, _agent_directory: str) -> dict[str, Any]:
    """Compatibility wrapper around the child's single authoritative builder."""

    return build_isolated_thread_config(snapshot_path, python_executable)


def run_codex_turn(**kwargs: Any) -> CPRAgentRunResult:
    """Run the optional SDK adapter in a temp-only, sanitized child process.

    The parent writes only the already-frozen public context into an ephemeral
    snapshot file.  The child receives no current workspace, ambient secrets,
    or way to invoke a shell.  Its JSON result is still untrusted and undergoes
    the host's separate four-tool, schema, and policy validation.
    """

    context = kwargs.get("context")
    if not isinstance(context, Mapping):
        raise ValueError("Codex turn requires a frozen CPR context mapping.")
    script = Path(__file__).with_name("cpr_ai_codex_subprocess.py")
    with tempfile.TemporaryDirectory(prefix="cpr-ai-codex-") as temporary_directory:
        snapshot = Path(temporary_directory) / "snapshot.json"
        snapshot.write_text(json.dumps(context, sort_keys=True), encoding="utf-8")
        request = {
            "snapshot_path": str(snapshot),
            "model": kwargs.get("model"),
            "reasoning_effort": kwargs.get("reasoning_effort"),
            "prompt": kwargs.get("prompt"),
            "output_schema": kwargs.get("output_schema"),
        }
        completed = subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=90,
            # A list argv and shell=False disable shell expansion for every model-supplied request field.
            shell=False,  # nosec B603
            cwd=temporary_directory,
            env=safe_subprocess_environment(),
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError("The isolated optional Codex subprocess rejected the turn.")
    try:
        response = json.loads(completed.stdout)
        if not isinstance(response, Mapping) or response.get("ok") is not True:
            raise ValueError("Subprocess did not return a successful structured response.")
        final_response = response["final_response"]
        raw_calls = response["tool_calls"]
        token_usage = response.get("token_usage", {})
        unexpected_actions = response.get("unexpected_actions", [])
        if not isinstance(final_response, str) or not isinstance(raw_calls, list):
            raise TypeError("Subprocess response has an invalid result shape.")
        if not isinstance(token_usage, Mapping) or not isinstance(unexpected_actions, list):
            raise TypeError("Subprocess response has invalid evidence metadata.")
        calls = tuple(
            CPRToolCallRecord(tool=str(call["tool"]), status=str(call["status"]), error=call.get("error"))
            for call in raw_calls
            if isinstance(call, Mapping)
        )
        if len(calls) != len(raw_calls):
            raise TypeError("Subprocess tool evidence must be mappings.")
        return CPRAgentRunResult(
            final_response=final_response,
            tool_calls=calls,
            token_usage={str(key): int(value) for key, value in token_usage.items()},
            unexpected_actions=tuple(str(action) for action in unexpected_actions),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("The isolated Codex subprocess returned malformed evidence.") from error


__all__ = ["build_codex_thread_config", "run_codex_turn", "safe_subprocess_environment"]
