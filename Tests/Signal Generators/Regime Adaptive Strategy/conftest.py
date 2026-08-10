"""Pytest bootstrap for the Regime Adaptive strategy tests.

The strategy's folder name contains spaces and its modules import each other by
bare name (`import regime_common`, etc.), so that folder goes on ``sys.path``
before the tests import anything -- the same pattern as
``SL Hunting AI Agent/conftest.py``.

The parent (`Signal Generators/`) is added too, because ``regime_common``
re-exports the shared indicators from ``misc_strategy_common`` which lives one
level up. At runtime ``regime_common`` bootstraps that itself; under pytest the
import can happen through a different entry point, so it is done here as well.

Both paths point at the SOURCE tree, not at this mirrored test folder -- the
tests import the real modules.
"""

from __future__ import annotations

import os
import sys

# Tests/Signal Generators/Regime Adaptive Strategy/<this file> -> repository root
# is three levels up.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SIGNAL_GEN_DIR = os.path.join(_REPO_ROOT, "Signal Generators")
_STRATEGY_DIR = os.path.join(_SIGNAL_GEN_DIR, "Regime Adaptive Strategy")
for _path in (_SIGNAL_GEN_DIR, _STRATEGY_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
