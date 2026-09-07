"""pytest configuration shared by all tests in this directory.

Sets the matplotlib backend to Agg (non-interactive) before any test module
is imported. This must happen before the first ``import matplotlib`` or
``import bench.plot`` call, and conftest.py is the right place because pytest
loads it before importing test modules.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
