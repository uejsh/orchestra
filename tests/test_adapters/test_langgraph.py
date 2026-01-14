# tests/test_adapters/test_langgraph.py

import pytest
import sys

from tests.conftest import skip_if_no_vector_search

# We no longer skip on Windows entirely because we have NumpyIndex fallback
pytestmark = skip_if_no_vector_search


from langgraph.graph import StateGraph, START
from typing import TypedDict
from orchestra import enhance, OrchestraConfig
from orchestra.adapters.langgraph import EnhancedLangGraph

class State(TypedDict):
    val: str

def dummy_node(state):
    return {"val": state["val"] + "!"}

def test_langgraph_enhance_wrapper():
    """Test that enhance returns an EnhancedLangGraph"""
    graph = StateGraph(State)
    graph.add_node("node", dummy_node)
    graph.add_edge(START, "node")
    compiled = graph.compile()
    
    enhanced = enhance(compiled)
    assert isinstance(enhanced, EnhancedLangGraph)
    assert enhanced.graph == compiled

def test_langgraph_caching_simulation():
    """Simulate caching behavior with mock components"""
    graph = StateGraph(State)
    graph.add_node("node", dummy_node)
    graph.add_edge(START, "node")
    compiled = graph.compile()
    
    enhanced = enhance(compiled)
    
    input_val = {"val": "test"}
    
    # 1. First Call
    res1 = enhanced.invoke(input_val)
    assert res1["val"] == "test!"
    
    # 2. Second Call (should hit cache)
    res2 = enhanced.invoke(input_val)
    assert res2 == res1
    
    metrics = enhanced.get_metrics()
    assert metrics["cache_hits"] == 1

def test_langgraph_hierarchical_option():
    """Test hierarchical embeddings option"""
    graph = StateGraph(State)
    graph.add_node("node", dummy_node)
    graph.add_edge(START, "node")
    compiled = graph.compile()
    
    config = OrchestraConfig(enable_hierarchical=True)
    enhanced = enhance(compiled, config)
    
    assert enhanced.config.enable_hierarchical == True
    assert enhanced.cache_manager.hierarchical_embedder is not None

    
    input_val = {"val": "test"}
    res = enhanced.invoke(input_val)
    assert res["val"] == "test!"

def test_langgraph_compression_option():
    """Test compression option"""
    graph = StateGraph(State)
    graph.add_node("node", dummy_node)
    graph.add_edge(START, "node")
    compiled = graph.compile()
    
    config = OrchestraConfig(enable_compression=True)
    enhanced = enhance(compiled, config)
    
    assert enhanced.config.enable_compression == True
    assert enhanced.cache_manager.compressor is not None

    
    input_val = {"val": "test"}
    res = enhanced.invoke(input_val)
    assert res["val"] == "test!"

def test_langgraph_all_features():
    """Test with all features enabled"""
    graph = StateGraph(State)
    graph.add_node("node", dummy_node)
    graph.add_edge(START, "node")
    compiled = graph.compile()
    
    config = OrchestraConfig(
        enable_hierarchical=True,
        enable_compression=True,
        similarity_threshold=0.95
    )
    enhanced = enhance(compiled, config)
    
    metrics = enhanced.get_metrics()
    assert metrics["hierarchical_enabled"] == True
    assert metrics["compression_enabled"] == True
