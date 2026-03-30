"""
Stub heavy/cloud dependencies that are not installed in the local test
environment so that unit tests for pure-Python helpers (like pick_best_mask)
can run without a full venv.

These stubs are registered in sys.modules before any test module is imported,
so the real imports never execute.
"""
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("CLOUDINARY_URL", "cloudinary://test:test@test")


def _stub(name: str) -> MagicMock:
    mod = MagicMock()
    mod.__name__ = name
    mod.__spec__ = None
    sys.modules[name] = mod
    return mod


# --- modal ---
_stub("modal")

# --- google-generativeai ---
_stub("google")
_stub("google.generativeai")
