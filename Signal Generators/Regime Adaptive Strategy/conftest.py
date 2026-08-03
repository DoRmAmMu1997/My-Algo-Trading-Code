"""Pytest bootstrap for the Regime Adaptive strategy folder.

This folder's name contains spaces and its modules import each other by bare
name (`import regime_common`, etc.), so this adds the folder to ``sys.path``
before the tests import anything -- the same pattern as
``SL Hunting AI Agent/conftest.py``.

The parent ("Signal Generators/") is added too, because ``regime_common``
re-exports the shared indicators from ``misc_strategy_common`` which lives one
level up. At runtime ``regime_common`` bootstraps that itself; under pytest the
import can happen through a different entry point, so it is done here as well.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIGNAL_GEN_DIR = os.path.dirname(_HERE)
for _path in (_SIGNAL_GEN_DIR, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)
