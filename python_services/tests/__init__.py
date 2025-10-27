"""
Test suite for Navidrome's Python services.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PYTHON_SERVICES_ROOT = _THIS_DIR.parent
if str(_PYTHON_SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_SERVICES_ROOT))
