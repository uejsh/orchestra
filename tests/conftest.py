# tests/conftest.py
# Handles platform-specific test configuration

import sys
import pytest

# Check if we're on Windows and FAISS might cause issues
def _check_faiss_available():
    """Check if FAISS is available without crashing"""
    try:
        import faiss
        # Try a simple operation to verify it works
        index = faiss.IndexFlatIP(10)
        return True
    except Exception:
        return False

# Mark for skipping tests that require FAISS on problematic platforms
FAISS_AVAILABLE = _check_faiss_available()

skip_if_no_faiss = pytest.mark.skipif(
    not FAISS_AVAILABLE,
    reason="FAISS not available or failed to load on this platform"
)

# Windows-specific handling
IS_WINDOWS = sys.platform == "win32"

skip_on_windows = pytest.mark.skipif(
    IS_WINDOWS,
    reason="Test skipped on Windows due to native library compatibility issues"
)
