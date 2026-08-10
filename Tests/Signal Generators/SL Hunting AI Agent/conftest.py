"""Pytest bootstrap for the SL Hunting AI Agent.

The agent's folder name contains spaces and its modules import each other by
bare name (`import sl_hunting_tools`, etc.), so that folder goes on ``sys.path``
before the tests import anything.

The path points at the SOURCE agent folder under ``Signal Generators/``, not at
this mirrored test folder -- the tests import the real modules.
"""

from __future__ import annotations

import os
import sys

# Tests/Signal Generators/SL Hunting AI Agent/<this file> -> repository root is
# three levels up.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_AGENT_DIR = os.path.join(_REPO_ROOT, "Signal Generators", "SL Hunting AI Agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)
