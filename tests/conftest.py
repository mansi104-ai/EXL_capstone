import os
import sys
from pathlib import Path

# Keep the test suite offline and deterministic: force the heuristic LLM path
# (env vars take precedence over any .env keys) so tests never call a provider.
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENROUTER_API_KEY"] = ""

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
