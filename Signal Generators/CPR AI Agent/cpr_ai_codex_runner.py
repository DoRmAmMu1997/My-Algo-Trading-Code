"""Prepare the deliberately narrow child process used for an optional Codex turn.

The configuration below is data, not a trading integration.  It gives a child
only the four frozen MCP reads and removes ambient credentials so an SDK issue
cannot turn into a shell, web, workspace, or order capability.

The subprocess import is intentional: callers and model input cannot choose
the fixed local interpreter or child script passed to it.
"""

from __future__ import annotations

import atexit
import json
import math
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cpr_ai_agent import CPRAgentRunResult, CPRToolCallRecord
from cpr_ai_codex_subprocess import build_isolated_thread_config

_PROCESS_CODEX_HOME_LOCK = threading.Lock()
_PROCESS_CODEX_HOME_TEMPORARY_DIRECTORY: tempfile.TemporaryDirectory[str] | None = None
_PROCESS_CODEX_HOME: Path | None = None


def create_auth_only_codex_home(source_home: Path, runtime_root: Path) -> Path:
    """Copy only subscription auth into a new independent Codex state home.

    The copy happens once per process-level home.  Codex may refresh its own
    token file during later turns, so callers must reuse this destination and
    must never overwrite it from, or synchronize it back to, the operator home.
    """

    source_auth = source_home / "auth.json"
    if (
        not source_home.is_dir()
        or source_home.is_symlink()
        or not source_auth.is_file()
        or source_auth.is_symlink()
    ):
        raise RuntimeError("Codex subscription authentication cannot be isolated safely.")
    runtime_root.mkdir(parents=True, exist_ok=True)
    isolated_home = runtime_root / "codex-home"
    isolated_home.mkdir()
    shutil.copy2(source_auth, isolated_home / "auth.json")
    return isolated_home


def _operator_codex_home() -> Path:
    """Resolve the operator authentication home before constructing child env."""

    configured = os.environ.get("CODEX_HOME")
    return Path(configured) if configured else Path.home() / ".codex"


def _cleanup_process_codex_home() -> None:
    """Remove the process-lifetime isolated home without touching operator state."""

    global _PROCESS_CODEX_HOME, _PROCESS_CODEX_HOME_TEMPORARY_DIRECTORY
    with _PROCESS_CODEX_HOME_LOCK:
        temporary_directory = _PROCESS_CODEX_HOME_TEMPORARY_DIRECTORY
        _PROCESS_CODEX_HOME = None
        _PROCESS_CODEX_HOME_TEMPORARY_DIRECTORY = None
    if temporary_directory is not None:
        temporary_directory.cleanup()


def process_isolated_codex_home() -> Path:
    """Return one auth-only home reused for every serialized turn this process."""

    global _PROCESS_CODEX_HOME, _PROCESS_CODEX_HOME_TEMPORARY_DIRECTORY
    with _PROCESS_CODEX_HOME_LOCK:
        if _PROCESS_CODEX_HOME is not None:
            return _PROCESS_CODEX_HOME
        temporary_directory = tempfile.TemporaryDirectory(prefix="cpr-ai-auth-")
        try:
            isolated_home = create_auth_only_codex_home(
                _operator_codex_home(),
                Path(temporary_directory.name),
            )
        except Exception:
            temporary_directory.cleanup()
            raise
        _PROCESS_CODEX_HOME_TEMPORARY_DIRECTORY = temporary_directory
        _PROCESS_CODEX_HOME = isolated_home
        return isolated_home


atexit.register(_cleanup_process_codex_home)


def safe_subprocess_environment(
    source: Mapping[str, str] | None = None,
    *,
    codex_home: Path,
    profile_home: Path,
) -> dict[str, str]:
    """Build a synthetic profile with only OS plumbing and isolated Codex auth.

    An allowlist is stronger than attempting to recognize every broker's next
    secret name.  HOME, profile, app-data, temp, and CODEX_HOME are replaced,
    not inherited, so ambient config, plugins, MCP servers, skills, and apps are
    outside the child's discovery paths.
    """

    source = os.environ if source is None else source
    environment = {
        key: value for key, value in source.items() if key.upper() in {"SYSTEMROOT", "WINDIR"}
    }
    environment.update(
        {
            "CODEX_HOME": str(codex_home),
            "HOME": str(profile_home),
            "USERPROFILE": str(profile_home),
            "APPDATA": str(profile_home / "AppData" / "Roaming"),
            "LOCALAPPDATA": str(profile_home / "AppData" / "Local"),
            "TEMP": str(profile_home / "Temp"),
            "TMP": str(profile_home / "Temp"),
        }
    )
    return environment


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
    timeout_seconds = float(kwargs.get("timeout_seconds", 90.0))
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("Codex subprocess timeout must be a positive finite number.")
    # Resolve the independent authentication boundary before subprocess launch.
    # Missing/unreadable auth therefore fails closed without ambient fallback.
    isolated_codex_home = process_isolated_codex_home()
    script = Path(__file__).with_name("cpr_ai_codex_subprocess.py")
    with tempfile.TemporaryDirectory(prefix="cpr-ai-codex-") as temporary_directory:
        runtime_directory = Path(temporary_directory)
        profile_home = runtime_directory / "profile"
        for directory in (
            profile_home,
            profile_home / "AppData" / "Roaming",
            profile_home / "AppData" / "Local",
            profile_home / "Temp",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        snapshot = runtime_directory / "snapshot.json"
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
            timeout=timeout_seconds,
            # A list argv and shell=False disable shell expansion for every model-supplied request field.
            shell=False,  # nosec B603
            cwd=temporary_directory,
            env=safe_subprocess_environment(
                codex_home=isolated_codex_home,
                profile_home=profile_home,
            ),
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


__all__ = [
    "build_codex_thread_config",
    "create_auth_only_codex_home",
    "process_isolated_codex_home",
    "run_codex_turn",
    "safe_subprocess_environment",
]
