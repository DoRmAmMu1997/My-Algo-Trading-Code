"""Make the independent CPR AI modules importable despite their space-containing folder.

Python cannot use a normal dotted import for a directory named ``CPR AI Agent``.
Pytest loads this file before collecting nearby tests, so it adds only that
independent agent directory to the import search path.  Production uses the
master's existing ``load_module`` helper instead.  Keeping this adjustment in
test configuration avoids packaging or renaming the folder merely for pytest.

Note the path points at the SOURCE agent folder under ``Signal Generators/``,
not at this mirrored test folder -- the tests import the real modules.
"""

from __future__ import annotations

import os
import sys

# Tests/Signal Generators/CPR AI Agent/<this file> -> repository root is three up.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_AGENT_DIR = os.path.join(_REPO_ROOT, "Signal Generators", "CPR AI Agent")
# Only this package root is inserted.  Adding a repository-wide directory would
# make tests pass through imports that production never uses and could hide a
# missing dependency or accidental legacy-CPR coupling.
#
# Insert only when absent so repeated collection does not grow or reorder
# ``sys.path`` unnecessarily.
for _path in (_AGENT_DIR,):
    if _path not in sys.path:
        sys.path.insert(0, _path)
