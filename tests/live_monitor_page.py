"""Harness script for the live-monitor render tests.

`AppTest` runs a Streamlit script, so the page under test needs to be reachable
as one. This file is that script — it is not itself a test module.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from guardiancx.database.db import init_engine  # noqa: E402
from guardiancx.ui import components as C  # noqa: E402
from guardiancx.ui.live_monitor import live_monitor  # noqa: E402

init_engine()
C.inject_css()
live_monitor()
