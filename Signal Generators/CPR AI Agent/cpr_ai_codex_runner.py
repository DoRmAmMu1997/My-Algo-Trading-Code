"""Prepare the deliberately narrow child process used for one Codex turn.

This parent-side adapter creates two layers of temporary isolation.  A single
process-lifetime ``CODEX_HOME`` contains only a copy of subscription
``auth.json`` so SDK token refreshes survive between serialized turns.  Each
turn then receives a fresh synthetic user profile, working directory, frozen
snapshot, and request file; all are deleted when that turn ends.

The configuration is data, not a trading integration.  The child sees only
four frozen MCP reads and cannot inherit broker credentials, operator plugins,
shell tools, web access, workspace files, or order capabilities.  Its output
remains untrusted and must pass the host's independent evidence and policy
checks.  The subprocess import is intentional: neither callers nor model input
can choose the fixed local interpreter or child script.
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
    token file during later turns, so callers reuse this destination under the
    agent's serialization lock.  The temporary copy is never overwritten from,
    symlinked to, or synchronized back to the operator home.
    """

    # Reject symlinks on both the home and file.  Otherwise an apparently
    # temporary boundary could still read or later mutate operator-owned state.
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
    """Locate subscription auth without exposing that home to the child.

    ``CODEX_HOME`` is honored for operators who keep Codex state outside the
    default profile.  Only ``auth.json`` is copied from the resolved directory.
    """

    configured = os.environ.get("CODEX_HOME")
    return Path(configured) if configured else Path.home() / ".codex"


def _cleanup_process_codex_home() -> None:
    """Delete the isolated auth copy at process exit, never operator state."""

    global _PROCESS_CODEX_HOME, _PROCESS_CODEX_HOME_TEMPORARY_DIRECTORY
    with _PROCESS_CODEX_HOME_LOCK:
        temporary_directory = _PROCESS_CODEX_HOME_TEMPORARY_DIRECTORY
        _PROCESS_CODEX_HOME = None
        _PROCESS_CODEX_HOME_TEMPORARY_DIRECTORY = None
    if temporary_directory is not None:
        temporary_directory.cleanup()


def process_isolated_codex_home() -> Path:
    """Return the process-lifetime auth-only home used by serialized turns.

    Reuse is deliberate: a token refresh written by the SDK remains available
    to the next turn, while the source ``auth.json`` is copied only once and is
    never modified.  Construction is lock-guarded in case worker startup races.
    """

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
    """Expose the child's authoritative config builder for tests and callers.

    ``_agent_directory`` remains in this compatibility surface but is ignored:
    the child must not discover tools or configuration from the repository.
    """

    return build_isolated_thread_config(snapshot_path, python_executable)


def run_codex_turn(**kwargs: Any) -> CPRAgentRunResult:
    """Run the optional SDK adapter in a temp-only, sanitized child process.

    The parent writes only the already-frozen public context into an ephemeral
    snapshot file.  The child receives no current workspace, ambient secrets,
    or way to invoke a shell.  Its JSON result is still untrusted and undergoes
    the host's separate four-tool, schema, model/prompt, freshness, and trading
    policy validation.  The configured deadline covers the whole child process,
    not merely the model request inside it.
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
        # Serialize only the already-frozen public mapping.  There is no handle
        # back to the mutable market-data store or the live position object.
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
    # Child stderr is deliberately not copied into the host exception: SDK
    # failures may contain paths or authentication details useful to attackers.
    if completed.returncode != 0:
        raise RuntimeError("The isolated optional Codex subprocess rejected the turn.")
    try:
        # Treat stdout as untrusted input even though the child is local.  A
        # dependency failure must become HOLD, never a partially trusted turn.
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
