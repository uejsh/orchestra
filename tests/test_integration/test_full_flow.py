import pytest
import os
import shutil
from typing import TypedDict
from langgraph.graph import StateGraph, START
from orchestra import enhance, OrchestraConfig

# Define State
class TestState(TypedDict):
    query: str
    result: str

def mock_node(state):
    # Simulate work
    return {"result": f"processed: {state['query']}"}

@pytest.fixture
def temp_cache_dir():
    dir_name = "test_cache_integration"
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    yield dir_name
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

def test_full_flow_integration(temp_cache_dir):
    """
    Test full flow:
    1. Create Graph
    2. Enhance
    3. Invoke (Miss)
    4. Invoke (Hit)
    5. Save/Load
    6. Verify Metrics
    """
    
    # 1. Setup Graph
    graph = StateGraph(TestState)
    graph.add_node("worker", mock_node)
    graph.add_edge(START, "worker")
    compiled = graph.compile()
    
    # 2. Enhance
    config = OrchestraConfig(
        similarity_threshold=0.95,
        enable_cache=True,
        cache_ttl=60
    )
    enhanced = enhance(compiled, config=config)
    
    # 3. First Invocation (Cache Miss)
    res1 = enhanced.invoke({"query": "What is AI?"})
    assert res1["result"] == "processed: What is AI?"
    
    metrics = enhanced.get_metrics()
    assert metrics["cache_hits"] == 0
    assert metrics["cache_misses"] == 1
    
    # 4. Second Invocation (Cache Hit - Exact)
    res2 = enhanced.invoke({"query": "What is AI?"})
    assert res2["result"] == "processed: What is AI?"
    
    metrics = enhanced.get_metrics()
    assert metrics["cache_hits"] == 1
    assert metrics["cache_misses"] == 1
    
    # 5. Semantic Hit (Similar query)
    # Using a slightly different string that should match with high threshold if embedding model is good,
    # but since we might be using random embeddings in fallback or actual model, we need to be careful.
    # If fallback is active (random), this WON'T match.
    # We should check if we are in fallback mode or not.
    # For integration test stability, we'll test Exact Match first or verify module state.
    
    # Let's test Save/Load Persistence
    cache_path = os.path.join(temp_cache_dir, "cache.pkl")
    enhanced.save_cache(cache_path)
    
    assert os.path.exists(cache_path)
    
    # Re-load into new instance
    new_enhanced = enhance(compiled, config=config)
    new_enhanced.load_cache(cache_path)
    
    # Verify loaded state by hitting cache again
    metrics_new = new_enhanced.get_metrics()
    # Metrics reset on new instance but cache size should be preserved
    assert metrics_new["cache_size"] >= 1 
    
    res3 = new_enhanced.invoke({"query": "What is AI?"})
    assert res3["result"] == "processed: What is AI?"
    
    # Should be a hit
    metrics_new_after = new_enhanced.get_metrics()
    # SemanticStore restores stats from disk, so we start with 1 hit and add another
    assert metrics_new_after["cache_hits"] == 2

@pytest.mark.asyncio
async def test_async_integration():
    """Test full async flow"""
    graph = StateGraph(TestState)
    graph.add_node("worker", mock_node)
    graph.add_edge(START, "worker")
    compiled = graph.compile()
    
    enhanced = enhance(compiled)
    
    # Async Invoke
    res = await enhanced.ainvoke({"query": "Async Test"})
    assert res["result"] == "processed: Async Test"
    
    # Async Hit
    res2 = await enhanced.ainvoke({"query": "Async Test"})
    assert res2 == res
    
    metrics = enhanced.get_metrics()
    assert metrics["cache_hits"] == 1
