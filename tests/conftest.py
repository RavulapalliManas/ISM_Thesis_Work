"""
Shared fixtures for tests.
Adds repo root to sys.path so all imports work.
"""
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
