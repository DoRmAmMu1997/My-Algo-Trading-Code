"""Make the independent CPR AI modules importable despite their space-containing folder.

Python cannot use a normal dotted import for a directory named ``CPR AI Agent``.
Pytest loads this file before collecting nearby tests, so it adds only this
independent agent directory to the import search path.  Production uses the
master's existing ``load_module`` helper instead.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Insert only when absent so repeated pytest collection does not grow or reorder
# ``sys.path`` unnecessarily.
for _path in (_HERE,):
    if _path not in sys.path:
        sys.path.insert(0, _path)
