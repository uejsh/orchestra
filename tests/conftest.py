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

# Mark for skipping tests that REQUIRE FAISS exclusively
FAISS_AVAILABLE = _check_faiss_available()
skip_if_no_faiss = pytest.mark.skipif(
    not FAISS_AVAILABLE,
    reason="FAISS not available or failed to load on this platform"
)

# New mark: Skip only if NO vector search (FAISS or NumPy) is available
# Since NumPy is a dependency, this is mostly for completeness or if it fails
def _check_vector_search_available():
    try:
        import numpy as np
        return True
    except ImportError:
        return False

VECTOR_SEARCH_AVAILABLE = FAISS_AVAILABLE or _check_vector_search_available()

skip_if_no_vector_search = pytest.mark.skipif(
    not VECTOR_SEARCH_AVAILABLE,
    reason="Neither FAISS nor NumPy fallback available"
)

# Windows handling - we only skip if it's truly platform-incompatible
# (e.g. specifically testing FAISS internals on Windows if it's broken)
IS_WINDOWS = sys.platform == "win32"

skip_on_windows = pytest.mark.skipif(
    IS_WINDOWS,
    reason="Test skipped on Windows due to native library compatibility issues"
)

