"""Harness script for the render tests — not a test module."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from guardiancx.database.db import init_engine  # noqa: E402
from guardiancx.ui import components as C  # noqa: E402
from guardiancx.ui.guidance import guidance_panel as page  # noqa: E402

init_engine()
C.inject_css()
page()
