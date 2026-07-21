"""Read-only audit of `Dependencies/.env` against the template and the code.

============================== Why this exists ================================
Configuration lives in three places that can silently drift apart:

  1. `Dependencies/.env`        - the real settings (gitignored, per machine)
  2. `Dependencies/env.example` - the committed template / discovery surface
  3. the code's `_env_*` calls  - the in-code default each setting falls back to

A key present in the code and the template but MISSING from `.env` is not an
error - the runner silently uses the in-code default. That is exactly what makes
it dangerous: the operator never sees the knob, and a default they never chose
quietly governs a live-money run. Twelve keys had drifted out this way before
this check existed.

This script reports the drift and changes nothing. Run it after pulling, after
editing `.env`, or before a live session:

    python algo.py check-env

============================== What it reports ================================
  * MISSING  - in `env.example` but not in your `.env` (the in-code default is
               being used; the template's value is shown so you can copy it)
  * UNKNOWN  - in your `.env` but not in `env.example` (usually a typo in a key
               name, or a setting removed from the code -- either way it is
               doing nothing)
  * UNDOCUMENTED - read by the code but absent from `env.example`, so it is
               invisible to anyone reading the template (CI also gates this;
               see `test_repository_policy.py`)

Exit code is 0 when clean and 1 when anything is reported, so it can be used as
a pre-flight gate in a script.

============================== On secrets =====================================
This script NEVER prints a value read from your `.env` -- only key NAMES. The
only values it prints come from `env.example`, which is committed and therefore
contains blank placeholders rather than real credentials. That makes its output
safe to paste into an issue or share when asking for help.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

# The repo root is this file's parent's parent (Dependencies/ -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_ENV = REPO_ROOT / "Dependencies" / ".env"
TEMPLATE_ENV = REPO_ROOT / "Dependencies" / "env.example"

# Helpers that read one setting out of the environment. `_env_*` take
# (key, default); the size-multiplier wrappers take (prefix, key, default), so
# for those the key is the SECOND argument.
ENV_READER_FUNCTIONS = frozenset({"_env_str", "_env_float", "_env_int", "_env_bool"})
SCALED_READER_FUNCTIONS = frozenset({"_scaled_int", "_scaled_float"})


def parse_env_file(path: Path) -> dict[str, str]:
    """Return every ``KEY=value`` pair in an env file, ignoring comments.

    Deliberately simple: `.env` files here are flat ``KEY=value`` lines. A blank
    value is preserved as an empty string because that is meaningful -- the
    `_env_*` helpers treat blank as "use the in-code default".
    """
    pairs: dict[str, str] = {}
    if not path.is_file():
        return pairs
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        pairs[key.strip()] = value.strip()
    return pairs


def env_keys_read_by(path: Path) -> set[str]:
    """Statically collect the env keys one Python module reads.

    Parsed with ``ast`` rather than imported, because the master runner has
    spaces in its filename, loads strategy modules, and configures logging at
    import time -- none of which should happen during a read-only audit.

    Keys assembled from f-strings (the per-strategy ``<PREFIX>_*`` knobs built
    in ``_signal_gen_ops``) are not literals and are skipped: they are
    documented once per strategy family rather than per key.
    """
    keys: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        # A file we cannot parse is not worth failing an audit over.
        return keys
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        function_name = node.func.id
        if function_name in ENV_READER_FUNCTIONS and node.args:
            key_node = node.args[0]
        elif function_name in SCALED_READER_FUNCTIONS and len(node.args) > 1:
            key_node = node.args[1]
        else:
            continue
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            keys.add(key_node.value)
    return keys


def source_files(repo_root: Path) -> list[Path]:
    """The modules whose `_env_*` calls define the configuration surface."""
    files = [repo_root / "Nifty Multi Strategy Front Test - Master File.py"]
    files += sorted((repo_root / "Dependencies").glob("*.py"))
    return [path for path in files if path.is_file()]


def audit(repo_root: Path = REPO_ROOT) -> dict[str, list[str]]:
    """Compare the three sources and return the three finding lists.

    Keys are returned sorted so the output is stable between runs (and therefore
    diffable), and values are never taken from the live `.env`.
    """
    live = parse_env_file(repo_root / "Dependencies" / ".env")
    template = parse_env_file(repo_root / "Dependencies" / "env.example")
    read_by_code: set[str] = set()
    for path in source_files(repo_root):
        read_by_code |= env_keys_read_by(path)

    return {
        "missing": sorted(set(template) - set(live)),
        "unknown": sorted(set(live) - set(template)),
        "undocumented": sorted(read_by_code - set(template)),
    }


def render(findings: dict[str, list[str]], template: dict[str, str]) -> list[str]:
    """Turn the findings into operator-readable lines (no live values)."""
    lines: list[str] = []

    missing = findings["missing"]
    lines.append(f"MISSING from Dependencies/.env ({len(missing)})")
    if missing:
        lines.append("  These exist in env.example; the runner is using its in-code")
        lines.append("  default for each. Copy the line across to make it explicit:")
        lines += [f"    {key}={template.get(key, '')}" for key in missing]
    else:
        lines.append("  none - your .env covers every documented setting.")

    unknown = findings["unknown"]
    lines.append("")
    lines.append(f"UNKNOWN in Dependencies/.env ({len(unknown)})")
    if unknown:
        lines.append("  Not in env.example -- most often a mistyped key name, which")
        lines.append("  means the setting you intended is NOT being applied:")
        lines += [f"    {key}" for key in unknown]
    else:
        lines.append("  none - every key in your .env is a recognised setting.")

    undocumented = findings["undocumented"]
    lines.append("")
    lines.append(f"UNDOCUMENTED in env.example ({len(undocumented)})")
    if undocumented:
        lines.append("  Read by the code but missing from the template, so nobody")
        lines.append("  reading env.example can discover them:")
        lines += [f"    {key}" for key in undocumented]
    else:
        lines.append("  none - every setting the code reads is in the template.")

    return lines


def main(argv: list[str] | None = None) -> int:
    """Print the audit and return 0 when clean, 1 when anything was found."""
    parser = argparse.ArgumentParser(
        prog="check-env",
        description=(
            "Read-only audit of Dependencies/.env against env.example and the "
            "settings the code actually reads. Prints key names only -- never a "
            "value from your .env."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to audit (defaults to this checkout).",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    live_path = repo_root / "Dependencies" / ".env"
    template_path = repo_root / "Dependencies" / "env.example"
    if not template_path.is_file():
        print(f"ERROR: template not found: {template_path}")
        return 2
    if not live_path.is_file():
        print(f"No {live_path} yet - copy env.example to .env and fill it in:")
        print(f"    copy \"{template_path}\" \"{live_path}\"")
        return 1

    findings = audit(repo_root)
    template = parse_env_file(template_path)

    print(f"Auditing {live_path}")
    print(f"  .env keys        : {len(parse_env_file(live_path))}")
    print(f"  env.example keys : {len(template)}")
    print()
    for line in render(findings, template):
        print(line)

    total = sum(len(values) for values in findings.values())
    print()
    if total:
        print(f"{total} finding(s). Nothing was changed - edit .env yourself.")
        return 1
    print("Clean: .env, env.example and the code all agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
