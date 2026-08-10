"""Pytest bootstrap for the shared ``Dependencies/`` suites.

These tests used to live inside ``Dependencies/`` itself, where pytest's default
``prepend`` import mode put that folder on ``sys.path`` for free -- which is how
bare imports such as ``from check_env_config import audit`` and
``from order_splitting import split_order_quantity`` resolved.

Now that the tests live under ``Tests/``, that no longer happens, so the SOURCE
folder is added here instead.  Note it is the source ``Dependencies/`` directory
that goes on the path, NOT this test directory: the modules import each other by
bare name at runtime (the diagnostics run as standalone scripts), and the tests
must exercise that same resolution rather than a test-only arrangement.

Only this one directory is inserted.  Adding the repository root or a wider set
would let a test resolve an import production never performs, which could hide a
missing dependency.
"""

from __future__ import annotations

import os
import sys

# Tests/Dependencies/<this file> -> the repository root is two levels up.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SOURCE_DIR = os.path.join(_REPO_ROOT, "Dependencies")

# Insert only when absent so repeated collection does not grow or reorder
# ``sys.path`` unnecessarily.
if _SOURCE_DIR not in sys.path:
    sys.path.insert(0, _SOURCE_DIR)
