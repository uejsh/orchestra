# tests/test_adapters/test_langchain.py

import pytest
import sys

# Skip entire module on Windows where FAISS/sentence-transformers can crash
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="LangChain tests require FAISS which has issues on Windows"
)

from orchestra import enhance
from orchestra.adapters.langchain import EnhancedLangChain, OrchestraLangChainConfig

class MockRunnable:
    def __init__(self):
        self.invoke_count = 0
        self.ainvoke_count = 0
    
    def invoke(self, input, config=None, **kwargs):
        self.invoke_count += 1
        return f"processed {input}"

    async def ainvoke(self, input, config=None, **kwargs):
        self.ainvoke_count += 1
        return f"processed {input}"

@pytest.fixture
def mock_runnable():
    return MockRunnable()

def test_langchain_sync_caching(mock_runnable):
    enhanced = enhance(mock_runnable)
    
    # 1. First call - miss
    res1 = enhanced.invoke("hello")
    assert res1 == "processed hello"
    assert mock_runnable.invoke_count == 1
    
    # 2. Second call - hit
    res2 = enhanced.invoke("hello")
    assert res2 == "processed hello"
    assert mock_runnable.invoke_count == 1 # Count should not increase

    # 3. New query - miss
    res3 = enhanced.invoke("world")
    assert res3 == "processed world"
    assert mock_runnable.invoke_count == 2

@pytest.mark.asyncio
async def test_langchain_async_caching(mock_runnable):
    enhanced = enhance(mock_runnable)
    
    # 1. First call - miss
    res1 = await enhanced.ainvoke("hello")
    assert res1 == "processed hello"
    assert mock_runnable.ainvoke_count == 1
    
    # 2. Second call - hit
    res2 = await enhanced.ainvoke("hello")
    assert res2 == "processed hello"
    assert mock_runnable.ainvoke_count == 1 # Count should not increase

    # 3. New query - miss
    res3 = await enhanced.ainvoke("world")
    assert res3 == "processed world"
    assert mock_runnable.ainvoke_count == 2

def test_langchain_attribute_delegation(mock_runnable):
    enhanced = enhance(mock_runnable)
    mock_runnable.some_custom_attr = "custom"
    assert enhanced.some_custom_attr == "custom"

def test_langchain_hierarchical_option(mock_runnable):
    """Test hierarchical embeddings option"""
    config = OrchestraLangChainConfig(enable_hierarchical=True)
    enhanced = EnhancedLangChain(mock_runnable, config)
    
    assert enhanced.config.enable_hierarchical == True
    assert enhanced.hierarchical_gen is not None
    
    res = enhanced.invoke("test query")
    assert res == "processed test query"

def test_langchain_compression_option(mock_runnable):
    """Test compression option"""
    config = OrchestraLangChainConfig(enable_compression=True)
    enhanced = EnhancedLangChain(mock_runnable, config)
    
    assert enhanced.config.enable_compression == True
    assert enhanced.compressor is not None
    
    res = enhanced.invoke("test query")
    assert res == "processed test query"

def test_langchain_all_features(mock_runnable):
    """Test with all features enabled"""
    config = OrchestraLangChainConfig(
        enable_hierarchical=True,
        enable_compression=True,
        similarity_threshold=0.95
    )
    enhanced = EnhancedLangChain(mock_runnable, config)
    
    metrics = enhanced.get_metrics()
    assert metrics["hierarchical_enabled"] == True
    assert metrics["compression_enabled"] == True
