# Shim — delegates to the production tts_api package.
# Run with:
#   uvicorn api:app --host 0.0.0.0 --port 8000
import sys
import os

# Ensure tts_api package is importable when running from this directory
sys.path.insert(0, os.path.dirname(__file__))

from tts_api.app.main import app  # noqa: F401  re-export for uvicorn
