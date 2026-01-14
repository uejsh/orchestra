# tests/test_core/test_cache_manager.py

import pytest
import sys
import time
import shutil
import tempfile

from tests.conftest import skip_if_no_vector_search

# We no longer skip on Windows entirely because we have NumpyIndex fallback
pytestmark = skip_if_no_vector_search


from orchestra.core.cache_manager import CacheManager

@pytest.fixture
def temp_cache_dir():
    dirpath = tempfile.mkdtemp()
    yield dirpath
    shutil.rmtree(dirpath)

def test_cache_put_get(temp_cache_dir):
    """Test basic put and get functionality"""
    cm = CacheManager()
    
    query = "Test Query"
    value = {"result": "Success"}
    
    cm.put(query, value)
    cached = cm.get(query)
    
    assert cached == value

def test_cache_miss(temp_cache_dir):
    """Test cache miss"""
    cm = CacheManager()
    assert cm.get("Non-existent") is None

def test_time_window_freshness(temp_cache_dir):
    """Test time window constraint (User Request specific feature)"""
    cm = CacheManager()
    
    query = "Sales Report"
    value = "Old Report"
    
    cm.put(query, value)
    
    # 1. Immediate retrieval should work
    assert cm.get(query, time_window_seconds=10) == value
    
    # 2. Simulate time passing (mocking state timestamp)
    state = cm.store.states[0]
    state.timestamp = time.time() - 100  # Make it 100 seconds old
    
    # 3. Request with small window (should fail)
    assert cm.get(query, time_window_seconds=10) is None  # Too old
    
    # 4. Request with large window (should succeed)
    assert cm.get(query, time_window_seconds=200) == value  # Within window

def test_hierarchical_matching_logic():
    """Test that our matcher logic runs"""
    cm = CacheManager()
    cm.put("Hello World", "Val")
    
    # Exact match should work
    assert cm.get("Hello World") == "Val"

def test_compression_toggle():
    """Test compression enable/disable"""
    # With compression
    cm_compressed = CacheManager(enable_compression=True)
    cm_compressed.put("test", {"large": "data" * 100})
    assert cm_compressed.get("test") == {"large": "data" * 100}
    
    # Without compression
    cm_uncompressed = CacheManager(enable_compression=False)
    cm_uncompressed.put("test", {"large": "data" * 100})
    assert cm_uncompressed.get("test") == {"large": "data" * 100}
