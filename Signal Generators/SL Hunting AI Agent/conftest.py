"""Pytest bootstrap for the SL Hunting AI Agent.

The folder name contains spaces and the modules import each other by bare name
(`import sl_hunting_tools`, etc.), so we add this folder to ``sys.path`` before the
tests import anything. pytest auto-loads this ``conftest.py`` because it sits above
the ``tests/`` directory.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
