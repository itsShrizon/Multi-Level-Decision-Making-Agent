"""Test bootstrap.

The DSPy modules read OPENAI_API_KEY at import time (via Settings ->
get_lm). Tests stub the actual LMs but still need the env var set before
any module under test gets imported, otherwise pytest collection fails.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
os.environ.setdefault("LM_MAIN", "openai/gpt-4o-mini")
os.environ.setdefault("LM_FAST", "openai/gpt-4o-mini")
os.environ.setdefault("LM_SUMMARY", "openai/gpt-3.5-turbo")
os.environ.setdefault("LM_REPORT", "openai/gpt-4o-mini")
